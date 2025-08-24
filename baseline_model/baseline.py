!pip install torch datasets tiktoken --quiet

import torch
import torch.nn as nn
from torch.nn import functional as F
from datasets import load_dataset
import tiktoken # Keep tiktoken for the Transformer model's output head size if needed, but not for data loading/tokenization
import math
import random
import os # Import os for creating directories

# ===== CONFIG =====
DATA_FILE = "training_data.txt" # Using the same data file
BATCH_SIZE = 16 # Set BATCH_SIZE to match the Wave model
MAX_SEQ_LEN = 16  # sequence length - Set MAX_SEQ_LEN to match the Wave model
EMBED_DIM = 64 # Increased EMBED_DIM to make model size more comparable to Wave model total
HEADS = 2 # Reduced number of attention heads
LAYERS = 2 # Reduced number of transformer blocks
DROPOUT = 0.1
LR = 1e-6 # Set LR to match the Wave model's stable rate
# Note: Weight decay and gradient clipping are handled in the training loop if needed.
EPOCHS = 20
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
# ==================


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(EMBED_DIM // HEADS * HEADS, head_size, bias=False) 
        self.query = nn.Linear(EMBED_DIM // HEADS * HEADS, head_size, bias=False)
        self.value = nn.Linear(EMBED_DIM // HEADS * HEADS, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(MAX_SEQ_LEN, MAX_SEQ_LEN))) 
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        B, T, C = x.shape
        assert C == EMBED_DIM, f"Input dimension {C} does not match EMBED_DIM {EMBED_DIM}"
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) / math.sqrt(k.size(-1))
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(EMBED_DIM // HEADS * HEADS, EMBED_DIM)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return self.dropout(out)

class FeedForward(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, 4 * emb_dim),
            nn.ReLU(),
            nn.Linear(4 * emb_dim, emb_dim),
            nn.Dropout(DROPOUT)
        )
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, emb_dim, num_heads):
        super().__init__()
        head_size = emb_dim // num_heads
        self.sa = MultiHeadAttention(num_heads, head_size)
        self.ffwd = FeedForward(emb_dim)
        self.ln1 = nn.LayerNorm(emb_dim)
        self.ln2 = nn.LayerNorm(emb_dim)
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Use the vocabulary size from the initialized tokenizer
        self.token_emb = nn.Embedding(tokenizer.vocab_size, EMBED_DIM)
        self.pos_emb = nn.Embedding(MAX_SEQ_LEN, EMBED_DIM) # Use MAX_SEQ_LEN
        # Ensure EMBED_DIM is a multiple of HEADS for MultiHeadAttention
        assert EMBED_DIM % HEADS == 0, "EMBED_DIM must be a multiple of HEADS"
        self.blocks = nn.Sequential(*[Block(EMBED_DIM, HEADS) for _ in range(LAYERS)])
        self.ln_f = nn.LayerNorm(EMBED_DIM)
        # Use the vocabulary size from the initialized tokenizer for the output head
        self.head = nn.Linear(EMBED_DIM, tokenizer.vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_emb(idx)
        pos_emb = self.pos_emb(torch.arange(T, device=DEVICE))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            # Calculate loss only for the prediction of the next token (at the last time step)
            # Logits shape: (B, T, vocab_size) -> slice to (B, vocab_size) for the last token
            # Targets shape: (B,)
            loss = F.cross_entropy(logits[:, -1, :], targets) # (1, 200) vs (1,) -> OK
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -MAX_SEQ_LEN:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx