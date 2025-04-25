import tkinter as tk, numpy as np, os
from tkinter import ttk, messagebox
from torch import *
from torch.nn import *
from torch.optim import *

os.system('clear' if os.name == 'posix' else 'cls')
root = tk.Tk()

# Vocabulary and mappings
vocab = ["wake up", "exercise", "breakfast", "commute", "work", "meeting", "lunch", "dinner", "relax", "sleep", "movie", "friends", "meet with friends"]
vocab2idx = {word: idx for idx, word in enumerate(vocab)}
idx2vocab = {idx: word for idx, word in enumerate(vocab)}
vocab_size = len(vocab)

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
class EventPredictor(Module):
    def __init__(self, vocab_size, embed_size=16, hidden_size=32):
        super().__init__()
        self.embedding = Embedding(vocab_size, embed_size)
        self.lstm = LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = Linear(hidden_size, vocab_size)
    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

def train_model():
    window_size = 5
    X, y = generate_samples(window_size)
    X = tensor(X, dtype=long)
    y = tensor(y, dtype=long)
    model = EventPredictor(vocab_size)
    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.01)
    epochs = 1000
    for epoch in range(epochs):
        epoch += 1
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        loss = loss.item()
        if epoch % 100 == 0:
            tk.Label(root, text=f"Epoch {epoch}, Loss: {loss}")
        epoch -= 1
    return model, window_size

model, window_size = train_model()

def predict_sequence(model, input_seq, predict_length, window_size):
    sequence = input_seq.copy()
    for _ in range(predict_length):
        window = sequence[-window_size:]
        if len(window) < window_size:
            # Pad with index 0 ("wake up") if input is too short
            window = [0] * (window_size - len(window)) + window
        inp = tensor([window], dtype=long)
        with no_grad():
            logits = model(inp)
            predicted_idx = argmax(logits, dim=1).item()
        sequence.append(predicted_idx)
    return sequence

# Predict next events on the press of a button
def on_press_predict():
    try:
        events = [event.strip() for event in user_input.split(",")]
        input_indices = [vocab2idx[e] for e in events if e in vocab2idx]
        if not input_indices:
            messagebox.showerror("Error", "Enter at least one valid event from the vocabulary.")
        else:
            full_indices = predict_sequence(model, input_indices, predict_length, window_size)
            predicted_events = [idx2vocab[idx] for idx in full_indices]
            m.config(text="Full sequence:")
            t.config(text=", ".join(predicted_events))
            e.config(text="Predicted next events:")
            a.config(text=", ".join(predicted_events[-predict_length:]))
    except Exception as e:
        messagebox.showerror("Error", f"Error: {e}")

root.attributes("-topmost", True)

tk.Label(root, text="Enter a sequence of events (comma-separated):").pack()
user_input = ttk.Entry(root)
user_input.pack()

tk.Label(root, text="Number of next events to predict:").pack()
predict_length = tk.Scale(root, orient="horizontal", from_=0, to=10)
predict_length.pack()

tk.Button(root, text="Predict", command=on_press_predict).pack()

m = tk.Label(root)
m.pack()
t = tk.Label(root)
t.pack()
e = tk.Label(root)
e.pack()
a = tk.Label(root)
a.pack()

root.mainloop()