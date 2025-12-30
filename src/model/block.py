"""Transformer block."""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from .attention import Attention
from .ffn import SwiGLU
from .normalization import RMSNorm


class TransformerBlock(nn.Module):
    """Transformer block with pre-normalization.

    Architecture (following Llama, DeepSeek, modern LLMs):
        1. RMSNorm
        2. Multi-Head Attention with RoPE
        3. Residual connection
        4. RMSNorm
        5. SwiGLU FFN
        6. Residual connection

    This is a "pre-norm" architecture where normalization happens before
    each sub-layer (attention and FFN), rather than after. This improves
    training stability for deep networks.

    Args:
        config: Model configuration
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Pre-attention norm
        self.attn_norm = RMSNorm(config.d_model, eps=1e-6)

        # Attention
        self.attn = Attention(config)

        # Pre-FFN norm
        self.ffn_norm = RMSNorm(config.d_model, eps=1e-6)

        # Feed-forward network
        self.ffn = SwiGLU(
            d_model=config.d_model,
            ffn_dim=config.ffn_dim,
            dropout=config.dropout,
            bias=False,
        )

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
            attention_mask: Optional attention mask
            use_cache: Whether to use KV cache
            past_key_value: Cached KV from previous forward pass

        Returns:
            output: Output tensor of shape [batch, seq_len, d_model]
            past_key_value: Optional KV cache if use_cache=True
        """
        # Self-attention with pre-norm and residual
        residual = x
        x = self.attn_norm(x)
        x, past_key_value = self.attn(
            x,
            attention_mask=attention_mask,
            use_cache=use_cache,
            past_key_value=past_key_value,
        )
        x = residual + x

        # Feed-forward with pre-norm and residual
        residual = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = residual + x

        return x, past_key_value
