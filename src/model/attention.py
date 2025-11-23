"""Attention mechanisms."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math

from .rope import RotaryEmbedding, apply_rotary_pos_emb


class Attention(nn.Module):
    """Multi-Head Attention with optional Grouped Query Attention (GQA).

    Supports:
    - Standard Multi-Head Attention (MHA): n_kv_heads = n_heads
    - Grouped Query Attention (GQA): n_kv_heads < n_heads (e.g., n_heads=32, n_kv_heads=8)
    - Multi-Query Attention (MQA): n_kv_heads = 1

    GQA reduces the KV cache size by sharing key/value heads across multiple query heads,
    significantly improving inference efficiency for long sequences.

    Uses RoPE (Rotary Position Embeddings) for position encoding.

    Args:
        config: Model configuration
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.dropout = config.dropout

        # Number of query heads per KV head (for GQA)
        assert self.n_heads % self.n_kv_heads == 0
        self.n_rep = self.n_heads // self.n_kv_heads

        # Q, K, V projections
        self.q_proj = nn.Linear(self.d_model, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.d_model, self.n_kv_heads * self.head_dim, bias=False)

        # Output projection
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, self.d_model, bias=False)

        # Dropout
        self.attn_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.resid_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # RoPE
        self.rotary_emb = RotaryEmbedding(
            dim=self.head_dim,
            max_seq_len=config.max_seq_len,
            base=config.rope_base,
        )

        # Scaling factor for attention scores
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            x: Input tensor of shape [batch, seq_len, d_model]
            attention_mask: Optional attention mask of shape [batch, 1, seq_len, seq_len]
            use_cache: Whether to return KV cache for inference
            past_key_value: Cached (K, V) from previous forward pass

        Returns:
            output: Output tensor of shape [batch, seq_len, d_model]
            past_key_value: Optional KV cache if use_cache=True
        """
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        # Q: [batch, seq_len, n_heads * head_dim]
        # K, V: [batch, seq_len, n_kv_heads * head_dim]
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape and transpose
        # Q: [batch, n_heads, seq_len, head_dim]
        # K, V: [batch, n_kv_heads, seq_len, head_dim]
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        kv_seq_len = k.shape[2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[2]

        cos, sin = self.rotary_emb(v, seq_len=kv_seq_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Update KV cache if using cache
        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=2)
            v = torch.cat([past_key_value[1], v], dim=2)

        if use_cache:
            past_key_value = (k, v)
        else:
            past_key_value = None

        # Repeat KV heads for GQA
        # K, V: [batch, n_kv_heads, seq_len, head_dim] -> [batch, n_heads, seq_len, head_dim]
        k = self._repeat_kv(k, self.n_rep)
        v = self._repeat_kv(v, self.n_rep)

        # Compute attention scores
        # [batch, n_heads, seq_len, head_dim] @ [batch, n_heads, head_dim, kv_seq_len]
        # -> [batch, n_heads, seq_len, kv_seq_len]
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply causal mask
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # Softmax and dropout
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Apply attention to values
        # [batch, n_heads, seq_len, kv_seq_len] @ [batch, n_heads, kv_seq_len, head_dim]
        # -> [batch, n_heads, seq_len, head_dim]
        attn_output = torch.matmul(attn_weights, v)

        # Reshape and combine heads
        # [batch, n_heads, seq_len, head_dim] -> [batch, seq_len, n_heads * head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.n_heads * self.head_dim)

        # Output projection
        output = self.o_proj(attn_output)
        output = self.resid_dropout(output)

        return output, past_key_value

    def _repeat_kv(self, x: torch.Tensor, n_rep: int) -> torch.Tensor:
        """Repeat KV heads to match number of query heads (for GQA).

        Args:
            x: Tensor of shape [batch, n_kv_heads, seq_len, head_dim]
            n_rep: Number of repetitions

        Returns:
            Tensor of shape [batch, n_heads, seq_len, head_dim]
        """
        if n_rep == 1:
            return x

        batch, n_kv_heads, seq_len, head_dim = x.shape

        # Expand and reshape
        # [batch, n_kv_heads, seq_len, head_dim]
        # -> [batch, n_kv_heads, n_rep, seq_len, head_dim]
        # -> [batch, n_heads, seq_len, head_dim]
        x = x.unsqueeze(2).expand(batch, n_kv_heads, n_rep, seq_len, head_dim)
        x = x.reshape(batch, n_kv_heads * n_rep, seq_len, head_dim)

        return x


def make_causal_mask(
    seq_len: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Create causal attention mask.

    Args:
        seq_len: Sequence length
        device: Device to create mask on
        dtype: Data type for mask

    Returns:
        Causal mask of shape [1, 1, seq_len, seq_len]
        where mask[i, j] = -inf if i < j (cannot attend to future), else 0
    """
    # Create lower triangular matrix
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=dtype), diagonal=1)

    # Convert to additive mask (0 for allowed, -inf for masked)
    mask = mask.masked_fill(mask == 1, float("-inf"))

    # Add batch and head dimensions
    return mask.unsqueeze(0).unsqueeze(0)
