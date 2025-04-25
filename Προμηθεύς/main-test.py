import streamlit as st, torch, torch.nn as nn, torch.optim as optim, numpy as np, os, pandas as pd

# Vocabulary and mappings
vocab = ["wake up", "exercise", "breakfast", "commute", "work", "meeting", "lunch", "dinner", "relax", "sleep", "movie", "friends", "meet with friends"]
vocab2idx = {word: idx for idx, word in enumerate(vocab)}
idx2vocab = {idx: word for idx, word in enumerate(vocab)}
vocab_size = len(vocab)
stri = ""
events = []

os.system('clear' if os.name == 'posix' else 'cls')

# Predefined routines (synthetic human event sequences)
routines = [
    ["wake up", "breakfast", "commute", "work", "lunch", "work", "commute", "dinner", "relax", "sleep"],
    ["wake up", "exercise", "breakfast", "commute", "work", "meeting", "lunch", "work", "commute", "dinner", "sleep"],
    ["wake up", "breakfast", "commute", "work", "lunch", "meeting", "work", "commute", "dinner", "sleep"],
]

# Create training samples with a sliding window (input: window_size events, target: next event)
def generate_samples(window_size=5):
    inputs, targets = [], []
    for routine in routines:
        indices = [vocab2idx[event] for event in routine if event in vocab2idx]
        for i in range(len(indices) - window_size):
            inputs.append(indices[i:i+window_size])
            targets.append(indices[i+window_size])
    return np.array(inputs), np.array(targets)

# LSTM-based event predictor
class EventPredictor(nn.Module):
    def __init__(self, vocab_size, embed_size=16, hidden_size=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

@st.cache_resource(max_entries=1, ttl=3600)
def train_model():
    window_size = 5
    X, y = generate_samples(window_size)
    X = torch.tensor(X, dtype=torch.long)
    y = torch.tensor(y, dtype=torch.long)
    model = EventPredictor(vocab_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    epochs = 1000
    for epoch in range(epochs):
        epoch += 1
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            st.write(f"Epoch {epoch}, Loss: {loss.item():.4f}")
        epoch -= 0.1
    return model, window_size

model, window_size = train_model()

st.title("Human Event Sequence Predictor")
st.table(pd.DataFrame(vocab, columns=["Vocabulary"], dtype=str, index=np.arange(1, vocab_size + 1)))

user_input = st.text_input("Enter a sequence of events (comma-separated):", "wake up, breakfast, commute, work, lunch", on_change=lambda x: x)
predict_length = st.slider("Number of events to predict", 1, 10, 3, on_change=lambda x: x)

def onchange():
    for _ in user_input:
        if stri != ", ":
            stri = stri + _
            events.append(stri)
        else:
            stri = ""

    st.table(pd.DataFrame(events, columns=["Events"], dtype=str, index=np.arange(1, len(events) + 1)))

# Predict next events based on the input sequence
def predict_sequence(model, input_seq, predict_length, window_size):
    sequence = input_seq.copy()
    for _ in range(predict_length):
        window = sequence[-window_size:]
        if len(window) < window_size:
            # Pad with index 0 ("wake up") if input is too short
            window = [0] * (window_size - len(window)) + window
        inp = torch.tensor([window], dtype=torch.long)
        with torch.no_grad():
            logits = model(inp)
            predicted_idx = torch.argmax(logits, dim=1).item()
        sequence.append(predicted_idx)
    return sequence

# Predict next events on the press of a button
if st.button("Predict next events"):
    try:
        events = [event.strip() for event in user_input.split(",")]
        input_indices = [vocab2idx[e] for e in events if e in vocab2idx]
        if not input_indices:
            st.error("Enter at least one valid event from the vocabulary.")
        else:
            full_indices = predict_sequence(model, input_indices, predict_length, window_size)
            predicted_events = [idx2vocab[idx] for idx in full_indices]
            st.table(pd.DataFrame(predicted_events, columns=["Predicted All Events"], dtype=str, index=np.arange(1, len(predicted_events) + 1)))
            st.table(pd.DataFrame(predicted_events[-predict_length:], columns=["Predicted Next Events"], dtype=str, index=np.arange(1, predict_length + 1)))

    except Exception as e:
        st.error(f"Error: {e}")