"""Normalization layers."""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    RMSNorm is 10-15% faster than LayerNorm and is used in modern LLMs
    like Llama, DeepSeek, and Mistral.

    Unlike LayerNorm, RMSNorm:
    - Does not subtract mean (no re-centering)
    - Does not use bias
    - Only normalizes by RMS

    Args:
        dim: Dimension of the input
        eps: Epsilon for numerical stability
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [..., dim]

        Returns:
            Normalized tensor of same shape
        """
        # Compute RMS
        # rms = sqrt(mean(x^2) + eps)
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)

        # Normalize and scale
        x_normed = x / rms
        return self.weight * x_normed

    def reset_parameters(self):
        """Reset parameters to default values."""
        torch.nn.init.ones_(self.weight)
