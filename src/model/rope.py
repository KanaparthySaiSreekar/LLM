"""Rotary Position Embeddings (RoPE)."""

import torch
import torch.nn as nn
from typing import Tuple


class RotaryEmbedding(nn.Module):
    """Rotary Position Embeddings (RoPE).

    RoPE encodes position information by rotating token embeddings in complex space.
    It provides several advantages:
    - No learned parameters
    - Naturally supports relative position information
    - Better length generalization than learned embeddings
    - Used in Llama, DeepSeek, Mistral, Qwen, etc.

    Based on: https://arxiv.org/abs/2104.09864

    Args:
        dim: Dimension per attention head (head_dim)
        max_seq_len: Maximum sequence length
        base: Base for frequency computation (10000 for standard, 500000 for long context)
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 2048,
        base: float = 10000.0,
    ):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute frequency tensor
        # theta_i = base^(-2i/dim) for i in [0, dim/2)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Precompute cos and sin for max_seq_len
        self._set_cos_sin_cache(max_seq_len)

    def _set_cos_sin_cache(self, seq_len: int):
        """Precompute cos and sin values for positions up to seq_len."""
        self.max_seq_len_cached = seq_len

        # Position indices: [0, 1, 2, ..., seq_len-1]
        t = torch.arange(seq_len, device=self.inv_freq.device).type_as(self.inv_freq)

        # Compute frequencies: outer product of positions and inv_freq
        # Shape: [seq_len, dim/2]
        freqs = torch.outer(t, self.inv_freq)

        # Concatenate to get [seq_len, dim]
        # This creates pairs: [freq_0, freq_0, freq_1, freq_1, ...]
        emb = torch.cat([freqs, freqs], dim=-1)

        # Precompute cos and sin
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor (used only for device/dtype, not for computation)
            seq_len: Sequence length

        Returns:
            cos, sin: Cosine and sine embeddings of shape [seq_len, dim]
        """
        # Extend cache if needed
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len)

        return (
            self.cos_cached[:seq_len],
            self.sin_cached[:seq_len],
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dimensions of the input.

    This is used to apply the rotary transformation.

    Args:
        x: Input tensor of shape [..., dim]

    Returns:
        Rotated tensor of same shape
    """
    # Split into two halves
    x1, x2 = x.chunk(2, dim=-1)

    # Rotate: [-x2, x1]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to Q and K.

    Args:
        q: Query tensor of shape [batch, num_heads, seq_len, head_dim]
        k: Key tensor of shape [batch, num_kv_heads, seq_len, head_dim]
        cos: Cosine embeddings of shape [seq_len, head_dim]
        sin: Sine embeddings of shape [seq_len, head_dim]

    Returns:
        Rotated Q and K tensors
    """
    # Reshape cos and sin for broadcasting
    # [seq_len, head_dim] -> [1, 1, seq_len, head_dim]
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)

    # Apply rotation
    # R(x) = x * cos + rotate_half(x) * sin
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed
