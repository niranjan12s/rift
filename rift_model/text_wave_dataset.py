# --- Dataset ---
import torch
from torch.utils.data import Dataset 
import os
import math
import time
import random
from pathlib import Path
from collections import Counter
from typing import List, Tuple
from nltk import word_tokenize


class TextWaveDataset(Dataset):
    """Load file (one sentence per line), tokenize using provided tokenizer instance,
       and provide random windows for a fixed number of samples per epoch."""
    def __init__(self, filepath: str = None, lines: List[str] = None, tokenizer: WordLevelTokenizer = None, max_seq_len: int = 0, num_samples: int = 0):
        if filepath and lines is not None:
             raise ValueError("Provide either filepath or lines, not both.")
        if filepath is None and lines is None:
             raise ValueError("Provide either filepath or lines.")
        if tokenizer is None or max_seq_len == 0 or num_samples == 0:
             raise ValueError("tokenizer, max_seq_len, and num_samples must be provided.")

        if filepath:
            assert os.path.exists(filepath), f"Train file not found: {filepath}"
            with open(filepath, "r", encoding="utf-8") as f:
                self.lines = [l.strip() for l in f if l.strip()]
        else:
            self.lines = lines # Use provided list of lines

        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.num_samples = num_samples # Target number of samples per epoch

        # Pre-tokenize all lines using the correct tokenizer attributes
        # Use tokenizer's internal _preprocess_text to handle cleaning consistently
        self._all_encoded_lines = [
            [self.tokenizer.word_to_id.get(t, self.tokenizer.unk_index)
             for t in self.tokenizer._preprocess_text(ln)] # Use tokenizer's preprocess method
            for ln in self.lines
        ]

        # Create a list of all possible start indices (line_idx, token_idx)
        # A window of size max_seq_len needs at least max_seq_len tokens for input
        self._possible_window_starts = []
        for line_idx, encoded_line in enumerate(self._all_encoded_lines):
            if len(encoded_line) >= max_seq_len + 1:
                 for token_idx in range(len(encoded_line) - max_seq_len):
                    self._possible_window_starts.append((line_idx, token_idx))


        print(f"Total possible windows available for sampling: {len(self._possible_window_starts):,}")
        if num_samples > len(self._possible_window_starts):
             print(f"Warning: Requested {num_samples:,} samples but only {len(self._possible_window_starts):,} possible windows are available. Using {len(self._possible_window_starts):,} samples.")
             self.num_samples = len(self._possible_window_starts)
        else:
             self.num_samples = num_samples

        print(f"Actual number of samples for training per epoch: {self.num_samples:,}")


    def __len__(self):
        # The length of the dataset is the fixed number of samples per epoch
        return self.num_samples

    def __getitem__(self, idx):
        line_idx, token_idx = random.choice(self._possible_window_starts)

        encoded_line = self._all_encoded_lines[line_idx]

        # Extract input window and target token
        # The window is size max_seq_len, starting at token_idx
        inp = encoded_line[token_idx : token_idx + self.max_seq_len]
        # The target is the token immediately following the window
        tgt = encoded_line[token_idx + self.max_seq_len]

        pad_value = self.tokenizer.unk_index if hasattr(self.tokenizer, 'unk_index') else 0 # Use 0 as default padding

        if len(inp) < self.max_seq_len:
             inp = [pad_value] * (self.max_seq_len - len(inp)) + inp


        inp = torch.tensor(inp, dtype=torch.long)
        tgt = torch.tensor(tgt, dtype=torch.long)
        return inp, tgt