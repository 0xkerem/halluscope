
import torch
from collections import deque
from typing import List
from utils.config import CFG


class MemoryBank:
    """
    Dynamically maintains last N token embeddings for threshold estimation.
    """
    def __init__(self, max_size: int = None):
        self.max_size = max_size or CFG.clip.memory_bank_size
        self._bank: deque = deque(maxlen=self.max_size)

    def update(self, embeddings: torch.Tensor):
        """embeddings: (seq_len, hidden_dim)"""
        for vec in embeddings:
            self._bank.append(vec.cpu())

    def get_thresholds(self) -> tuple:
        """Returns (h_min, h_max) per neuron using top/bottom p percentile."""
        if len(self._bank) < 10:
            return None, None
        bank_tensor = torch.stack(list(self._bank))  # (N, hidden_dim)
        p = CFG.clip.clip_percentile
        h_min = torch.quantile(bank_tensor, p / 100, dim=0)
        h_max = torch.quantile(bank_tensor, 1 - p / 100, dim=0)
        return h_min, h_max

    def __len__(self):
        return len(self._bank)


# Global memory bank shared across calls
_memory_bank = MemoryBank()