import torch
import torch.nn as nn
import torch.nn.functional as F
import re
from tqdm import tqdm

# Load and tokenize text
with open("your_text_file.txt", "r", encoding="utf-8") as f:
    text = f.read().lower()

# Word-level tokenization
words = re.findall(r'\b\w+\b', text)
vocab = sorted(set(words))
word2idx = {w: i for i, w in enumerate(vocab)}
idx2word = {i: w for w, i in word2idx.items()}
data = [word2idx[w] for w in words]

# Parameters
block_size = 16
batch_size = 32

# Data batch generator
def get_batch():
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([torch.tensor(data[i:i+block_size]) for i in ix])
    y = torch.stack([torch.tensor(data[i+1:i+1+block_size]) for i in ix])
    return x, y

# Model definition
class TransformerLM(nn.Module):
    def __init__(self, vocab_size, n_embed=256, n_heads=4, n_layers=4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, n_embed)
        self.pos_embed = nn.Parameter(torch.zeros(1, block_size, n_embed))
        encoder_layer = nn.TransformerEncoderLayer(d_model=n_embed, nhead=n_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.fc = nn.Linear(n_embed, vocab_size)

    def forward(self, x):
        x = self.embed(x) + self.pos_embed
        x = self.transformer(x)
        return self.fc(x)

# Initialize model
model = TransformerLM(len(vocab))
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

# Training loop
model.train()
for step in tqdm(range(5000)):
    x, y = get_batch()
    logits = model(x)
    loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 500 == 0:
        print(f"Step {step}, loss: {loss.item():.4f}")

# Text generation
def generate(start_words, max_new_words=50):
    model.eval()
    words = start_words.lower().split()
    x = torch.tensor([word2idx.get(w, 0) for w in words[-block_size:]]).unsqueeze(0)
    for _ in range(max_new_words):
        x_cond = x[:, -block_size:]
        logits = model(x_cond)
        probs = F.softmax(logits[:, -1, :], dim=-1)
        next_word_idx = torch.multinomial(probs, num_samples=1)
        x = torch.cat([x, next_word_idx], dim=1)
    return ' '.join(words + [idx2word[i.item()] for i in x[0][len(words):]])

# Example usage
print(generate("once upon a time", 50))