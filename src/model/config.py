"""Model configuration."""

from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any
import json


@dataclass
class ModelConfig:
    """Configuration for the Transformer model.

    Designed to scale from 10M to multi-billion parameter models
    without architectural changes.
    """

    # Vocabulary
    vocab_size: int

    # Architecture dimensions
    n_layers: int
    d_model: int
    n_heads: int
    head_dim: int = 64  # Fixed, following modern practice

    # Feed-forward network
    ffn_mult: float = 4.0  # Multiplier for FFN hidden dimension

    # Sequence length
    max_seq_len: int = 2048

    # Regularization
    dropout: float = 0.0

    # Embeddings
    tie_embeddings: bool = True  # Tie input and output embeddings

    # RoPE configuration
    rope_base: float = 10000.0  # 500k for long context (Llama 3.1 style)
    rope_scaling: Optional[Dict[str, Any]] = None  # For context extension

    # Grouped Query Attention (optional)
    n_kv_heads: Optional[int] = None  # If None, uses n_heads (standard MHA)

    # Initialization
    initializer_range: float = 0.02

    def __post_init__(self):
        """Validate configuration."""
        # If n_kv_heads not specified, use MHA (n_kv_heads = n_heads)
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads

        # Validate dimensions
        assert self.d_model % self.n_heads == 0, \
            f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"

        assert self.n_heads % self.n_kv_heads == 0, \
            f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})"

        # Compute derived values
        self.head_dim = self.d_model // self.n_heads
        self.ffn_dim = int(self.ffn_mult * self.d_model)

    @property
    def num_parameters(self) -> int:
        """Estimate number of parameters."""
        # Embeddings
        embed_params = self.vocab_size * self.d_model
        if not self.tie_embeddings:
            embed_params *= 2

        # Per transformer block
        # Attention: Q, K, V, O projections + RMSNorm
        attn_params = (
            self.d_model * self.d_model +  # Q
            self.d_model * (self.n_kv_heads * self.head_dim) +  # K
            self.d_model * (self.n_kv_heads * self.head_dim) +  # V
            self.d_model * self.d_model +  # O
            self.d_model  # RMSNorm
        )

        # FFN: gate, up, down + RMSNorm
        ffn_params = (
            self.d_model * self.ffn_dim +  # gate
            self.d_model * self.ffn_dim +  # up
            self.ffn_dim * self.d_model +  # down
            self.d_model  # RMSNorm
        )

        block_params = attn_params + ffn_params

        # Final norm
        final_norm = self.d_model

        total = embed_params + (block_params * self.n_layers) + final_norm
        return total

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ModelConfig":
        """Create from dictionary."""
        return cls(**config_dict)

    def save(self, path: str):
        """Save configuration to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "ModelConfig":
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    @classmethod
    def from_name(cls, name: str) -> "ModelConfig":
        """Create configuration from preset name."""
        configs = {
            "5M": cls(
                vocab_size=8000,
                n_layers=6,
                d_model=512,
                n_heads=8,
                max_seq_len=512,
                dropout=0.1,
            ),
            "50M": cls(
                vocab_size=8000,
                n_layers=8,
                d_model=768,
                n_heads=12,
                max_seq_len=1024,
                dropout=0.1,
            ),
            "125M": cls(
                vocab_size=32000,
                n_layers=12,
                d_model=768,
                n_heads=12,
                max_seq_len=2048,
                dropout=0.0,
            ),
            "1B": cls(
                vocab_size=32000,
                n_layers=24,
                d_model=2048,
                n_heads=16,
                n_kv_heads=4,  # GQA
                max_seq_len=2048,
                dropout=0.0,
            ),
            "7B": cls(
                vocab_size=32000,
                n_layers=32,
                d_model=4096,
                n_heads=32,
                n_kv_heads=8,  # GQA
                max_seq_len=4096,
                dropout=0.0,
                ffn_mult=3.5,  # ~14336 FFN dim
            ),
        }

        if name not in configs:
            raise ValueError(f"Unknown config name: {name}. Available: {list(configs.keys())}")

        return configs[name]

    def __repr__(self) -> str:
        """String representation."""
        params = self.num_parameters / 1e6
        return (
            f"ModelConfig(\n"
            f"  params={params:.1f}M,\n"
            f"  layers={self.n_layers},\n"
            f"  d_model={self.d_model},\n"
            f"  heads={self.n_heads},\n"
            f"  kv_heads={self.n_kv_heads},\n"
            f"  vocab={self.vocab_size},\n"
            f"  seq_len={self.max_seq_len}\n"
            f")"
        )
