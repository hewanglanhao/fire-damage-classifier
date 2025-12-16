import torch
import json
import os
from collections import Counter


class SimpleTokenizer:
    def __init__(
        self, vocab_size=5000, seq_len=50, pad_token="<pad>", unk_token="<unk>"
    ):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.word2idx = {}
        self.idx2word = {}

    def build_vocab(self, texts):
        """Build vocabulary from a list of texts."""
        all_tokens = []
        for text in texts:
            if isinstance(text, str):
                all_tokens.extend(text.lower().split())

        # Count frequencies
        counter = Counter(all_tokens)

        # Most common words
        most_common = counter.most_common(
            self.vocab_size - 2
        )  # Reserve for pad and unk

        self.word2idx = {self.pad_token: 0, self.unk_token: 1}
        self.idx2word = {0: self.pad_token, 1: self.unk_token}

        for idx, (word, _) in enumerate(most_common, start=2):
            self.word2idx[word] = idx
            self.idx2word[idx] = word

    def encode(self, text):
        """Convert text to list of indices."""
        if not isinstance(text, str):
            return [self.word2idx[self.pad_token]] * self.seq_len

        tokens = text.lower().split()
        indices = [
            self.word2idx.get(token, self.word2idx[self.unk_token]) for token in tokens
        ]

        # Pad or truncate
        if len(indices) < self.seq_len:
            indices += [self.word2idx[self.pad_token]] * (self.seq_len - len(indices))
        else:
            indices = indices[: self.seq_len]

        return torch.tensor(indices, dtype=torch.long)

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.word2idx, f)

    def load(self, path):
        with open(path, "r") as f:
            self.word2idx = json.load(f)
        self.idx2word = {v: k for k, v in self.word2idx.items()}
