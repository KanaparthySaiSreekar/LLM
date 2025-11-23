"""Feed-Forward Network modules."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """SwiGLU Feed-Forward Network.

    SwiGLU combines the Swish/SiLU activation with a gating mechanism (GLU).
    It provides better performance than standard ReLU or GELU activations
    and is used in modern LLMs like Llama, DeepSeek, and Mistral.

    Architecture:
        FFN(x) = down(silu(gate(x)) * up(x))

    Where:
        - gate: Linear projection to ffn_dim
        - up: Linear projection to ffn_dim
        - down: Linear projection back to d_model
        - silu: Swish activation (x * sigmoid(x))
        - *: Element-wise multiplication (gating)

    Args:
        d_model: Input/output dimension
        ffn_dim: Hidden dimension (typically 4 * d_model or ~2.67 * d_model for SwiGLU)
        dropout: Dropout probability
        bias: Whether to use bias in linear layers (False for modern LLMs)
    """

    def __init__(
        self,
        d_model: int,
        ffn_dim: int,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()

        self.gate_proj = nn.Linear(d_model, ffn_dim, bias=bias)
        self.up_proj = nn.Linear(d_model, ffn_dim, bias=bias)
        self.down_proj = nn.Linear(ffn_dim, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [batch, seq_len, d_model]

        Returns:
            Output tensor of shape [batch, seq_len, d_model]
        """
        # Gate path with SiLU activation
        gate = F.silu(self.gate_proj(x))

        # Up projection (no activation)
        up = self.up_proj(x)

        # Element-wise multiplication (gating) and down projection
        hidden = gate * up
        output = self.down_proj(hidden)

        return self.dropout(output)


class FeedForward(nn.Module):
    """Standard Feed-Forward Network with configurable activation.

    This is a more general FFN that supports different activation functions.
    For production use with modern LLMs, prefer SwiGLU.

    Args:
        d_model: Input/output dimension
        ffn_dim: Hidden dimension
        activation: Activation function name ("relu", "gelu", "silu")
        dropout: Dropout probability
        bias: Whether to use bias in linear layers
    """

    def __init__(
        self,
        d_model: int,
        ffn_dim: int,
        activation: str = "gelu",
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()

        self.fc1 = nn.Linear(d_model, ffn_dim, bias=bias)
        self.fc2 = nn.Linear(ffn_dim, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Select activation function
        activations = {
            "relu": F.relu,
            "gelu": F.gelu,
            "silu": F.silu,
        }
        if activation not in activations:
            raise ValueError(
                f"Unknown activation: {activation}. Choose from {list(activations.keys())}"
            )
        self.activation = activations[activation]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [batch, seq_len, d_model]

        Returns:
            Output tensor of shape [batch, seq_len, d_model]
        """
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return self.dropout(x)
