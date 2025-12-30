"""Main Transformer model."""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List
import math

from .config import ModelConfig
from .block import TransformerBlock
from .normalization import RMSNorm
from .attention import make_causal_mask


class Transformer(nn.Module):
    """Decoder-only Transformer model for causal language modeling.

    This is a GPT-style transformer with modern architectural choices:
    - RMSNorm instead of LayerNorm
    - SwiGLU activation instead of GELU
    - RoPE positional embeddings instead of learned
    - Optional Grouped Query Attention (GQA)

    Designed to scale from 10M to multi-billion parameters without
    architectural changes.

    Args:
        config: Model configuration
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Token embeddings
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.d_model)

        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])

        # Final normalization
        self.norm = RMSNorm(config.d_model, eps=1e-6)

        # Output projection (LM head)
        if config.tie_embeddings:
            # Share weights with input embeddings
            self.output = None
        else:
            self.output = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Initialize weights
        self.apply(self._init_weights)

        # Special scaled init for residual projections
        for name, param in self.named_parameters():
            if name.endswith("o_proj.weight") or name.endswith("down_proj.weight"):
                # Scale by 1/sqrt(2*n_layers) for residual paths
                torch.nn.init.normal_(
                    param,
                    mean=0.0,
                    std=config.initializer_range / math.sqrt(2 * config.n_layers)
                )

        # Report number of parameters
        n_params = sum(p.numel() for p in self.parameters())
        print(f"Initialized Transformer with {n_params:,} parameters")
        print(f"Config: {config}")

    def _init_weights(self, module):
        """Initialize weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        return_dict: bool = True,
    ):
        """
        Args:
            input_ids: Token IDs of shape [batch, seq_len]
            attention_mask: Optional attention mask (not typically needed for causal LM)
            use_cache: Whether to use KV cache for generation
            past_key_values: Cached KV from previous forward passes
            return_dict: Whether to return ModelOutput or tuple

        Returns:
            If return_dict=True: ModelOutput with logits and optional past_key_values
            If return_dict=False: Tuple of (logits, past_key_values)
        """
        batch_size, seq_len = input_ids.shape

        # Create causal mask
        if attention_mask is None:
            attention_mask = make_causal_mask(
                seq_len,
                device=input_ids.device,
                dtype=torch.float32,
            )

        # Token embeddings
        hidden_states = self.tok_embeddings(input_ids)

        # Initialize past_key_values if needed
        if past_key_values is None:
            past_key_values = [None] * len(self.layers)

        # Store new KV caches if using cache
        new_past_key_values = [] if use_cache else None

        # Apply transformer blocks
        for i, layer in enumerate(self.layers):
            hidden_states, past_kv = layer(
                hidden_states,
                attention_mask=attention_mask,
                use_cache=use_cache,
                past_key_value=past_key_values[i],
            )

            if use_cache:
                new_past_key_values.append(past_kv)

        # Final normalization
        hidden_states = self.norm(hidden_states)

        # Output projection
        if self.output is not None:
            logits = self.output(hidden_states)
        else:
            # Tied embeddings: use input embedding weights as output projection
            logits = torch.matmul(hidden_states, self.tok_embeddings.weight.t())

        if return_dict:
            return ModelOutput(
                logits=logits,
                past_key_values=new_past_key_values if use_cache else None,
            )
        else:
            return logits, new_past_key_values

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """Generate text autoregressively.

        Args:
            input_ids: Input token IDs of shape [batch, seq_len]
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (1.0 = no change, <1 = more conservative, >1 = more random)
            top_k: If set, only sample from top k tokens
            top_p: If set, sample from smallest set of tokens with cumulative prob >= top_p
            eos_token_id: Stop generation if this token is generated

        Returns:
            Generated token IDs of shape [batch, seq_len + max_new_tokens]
        """
        self.eval()

        for _ in range(max_new_tokens):
            # Forward pass (with context length limit)
            seq_len = input_ids.shape[1]
            if seq_len > self.config.max_seq_len:
                # Truncate to last max_seq_len tokens
                input_ids_cond = input_ids[:, -self.config.max_seq_len:]
            else:
                input_ids_cond = input_ids

            with torch.no_grad():
                outputs = self.forward(input_ids_cond, return_dict=True)
                logits = outputs.logits

            # Get logits for last token
            logits = logits[:, -1, :]  # [batch, vocab_size]

            # Apply temperature
            logits = logits / temperature

            # Optional: top-k sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            # Optional: nucleus (top-p) sampling
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(
                    torch.softmax(sorted_logits, dim=-1), dim=-1
                )

                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Keep at least one token
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = False

                # Scatter back to original indexing
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float('-inf')

            # Sample from distribution
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)

            # Check for EOS
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        return input_ids

    @torch.no_grad()
    def estimate_mfu(self, fwdbwd_per_iter: int, dt: float, num_tokens: int) -> float:
        """Estimate model FLOPs utilization (MFU).

        MFU = (actual FLOPs/sec) / (peak FLOPs/sec)

        Args:
            fwdbwd_per_iter: Number of forward-backward passes per iteration
            dt: Time per iteration in seconds
            num_tokens: Number of tokens processed per iteration

        Returns:
            MFU as a fraction (e.g., 0.5 = 50% utilization)
        """
        # Estimate FLOPs per token (forward pass)
        # For decoder-only: ~6N FLOPs per token
        N = sum(p.numel() for p in self.parameters())
        flops_per_token = 6 * N
        flops_per_fwdbwd = flops_per_token * num_tokens

        # Total FLOPs for iteration (forward + backward = 2x forward)
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter * 3  # 3x for backward

        # FLOPs achieved
        flops_achieved = flops_per_iter / dt

        # Peak FLOPs (example values, adjust for your GPU)
        # A100: 312 TFLOPS (BF16)
        # V100: 125 TFLOPS (FP16)
        # H100: 1000 TFLOPS (FP8)
        # For now, assume A100
        flops_peak = 312e12

        mfu = flops_achieved / flops_peak
        return mfu


class ModelOutput:
    """Container for model outputs."""

    def __init__(self, logits, past_key_values=None):
        self.logits = logits
        self.past_key_values = past_key_values
