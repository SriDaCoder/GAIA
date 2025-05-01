import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pyttsx3
from cryptography.fernet import Fernet

os.system('cls' if os.name == 'nt' else 'clear')

# Memory setup
memory_dir = "ATHENA_memory"
key_path = os.path.join(memory_dir, "key.key")
history_path = os.path.join(memory_dir, "history.enc")
os.makedirs(memory_dir, exist_ok=True)

# Generate or load AES key
if not os.path.exists(key_path):
    with open(key_path, "wb") as f:
        f.write(Fernet.generate_key())
with open(key_path, "rb") as f:
    key = f.read()
fernet = Fernet(key)

# Load history
def load_history():
    if os.path.exists(history_path):
        with open(history_path, "rb") as f:
            decrypted = fernet.decrypt(f.read()).decode()
            return decrypted.splitlines()
    return []

# Save history
def save_history(lines):
    data = "\n".join(lines).encode()
    encrypted = fernet.encrypt(data)
    with open(history_path, "wb") as f:
        f.write(encrypted)

history = load_history()

# TTS
engine = pyttsx3.init()

# Commands
commands = {
    "status": lambda: "All systems operational.",
    "launch": lambda: "Initiating launch sequence.",
    "abort": lambda: "Abort command acknowledged.",
}

# Data (training seed)
data = (
    "User: Hello\n"
    "AI: Greetings. How can I assist?\n"
    "User: What's your status?\n"
    "AI: All systems operational.\n"
    "User: Launch\n"
    "AI: Initiating launch sequence.\n"
)

# Tokenization
chars = sorted(set(data))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# Hyperparameters
block_size = 64
batch_size = 64
embedding_dim = 64
hidden_size = 128
num_layers = 2
max_iters = 4000
eval_interval = 500
learning_rate = 1e-3

# Batch creation
def get_batch():
    ix = torch.randint(0, len(data) - block_size, (batch_size,))
    x = torch.stack([torch.tensor(encode(data[i:i+block_size])) for i in ix])
    y = torch.stack([torch.tensor(encode(data[i+1:i+block_size+1])) for i in ix])
    return x, y

# Model
class ATHENAAI(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.GRU(embedding_dim, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embed(x)
        out, hidden = self.rnn(x, hidden)
        logits = self.linear(out)
        return logits, hidden

model = ATHENAAI()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for step in range(max_iters):
    x, y = get_batch()
    logits, _ = model(x)
    loss = nn.functional.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % eval_interval == 0:
        print(f"Step {step}, Loss: {loss.item():.4f}")

# Command parser
def check_command(user_input):
    for cmd in commands:
        if cmd in user_input.lower():
            return commands[cmd]()
    return None

# Response generator
def chat(prompt, max_len=200):
    model.eval()
    input_seq = torch.tensor([encode(prompt)], dtype=torch.long)
    hidden = None
    output = prompt

    for _ in range(max_len):
        logits, hidden = model(input_seq[:, -block_size:], hidden)
        probs = torch.softmax(logits[:, -1, :], dim=-1)
        next_char = torch.multinomial(probs, num_samples=1)
        next_idx = next_char.item()
        output += itos[next_idx]
        input_seq = torch.cat([input_seq, next_char], dim=1)
        if output.endswith('\nUser:'):
            break

    reply = output.split('\n')[-2].replace("AI: ", "").strip()
    return reply

os.system('cls' if os.name == 'nt' else 'clear')
print("ATHENA AI - Chatbot with Command Recognition\n")

# Chat loop
while True:
    user_input = input("You: ")
    if user_input.lower() in ['exit', 'quit']:
        break

    history.append(f"User: {user_input}")
    command_response = check_command(user_input)

    if command_response:
        ai_reply = command_response
    else:
        context = "\n".join(history[-block_size:]) + "\nAI:"
        ai_reply = chat(context)

    history.append(f"AI: {ai_reply}")
    print("AI:", ai_reply)
    engine.say(ai_reply)
    engine.runAndWait()

    save_history(history)
