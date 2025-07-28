import torch
from typing import List, Optional


def get_sentence_embedding(
    hidden_states: torch.Tensor,
    layer_index: Optional[int] = None
) -> torch.Tensor:
    """
    Extract sentence embedding from hidden states.
    Uses last token of the middle layer (same as INSIDE paper).

    Args:
        hidden_states : (num_layers, seq_len, hidden_dim)
        layer_index   : override layer selection (None → mid layer)
    Returns:
        embedding : (hidden_dim,)
    """
    num_layers = hidden_states.shape[0]
    idx = layer_index if layer_index is not None else num_layers // 2
    # Last token embedding of the chosen layer
    return hidden_states[idx, -1, :]  # (hidden_dim,)