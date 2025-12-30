"""Test script to verify model architecture."""

import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from model.config import ModelConfig
from model.transformer import Transformer


def test_model_forward():
    """Test model forward pass."""
    print("=" * 60)
    print("Testing Transformer Model")
    print("=" * 60)

    # Create small config for testing
    config = ModelConfig.from_name("5M")
    print(f"\nConfiguration:\n{config}")

    # Create model
    print("\nInitializing model...")
    model = Transformer(config)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,} ({n_params/1e6:.2f}M)")

    # Create dummy input
    batch_size = 2
    seq_len = 128
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    print(f"\nInput shape: {input_ids.shape}")

    # Forward pass
    print("Running forward pass...")
    with torch.no_grad():
        outputs = model(input_ids, return_dict=True)

    print(f"Output logits shape: {outputs.logits.shape}")
    print(f"Expected shape: ({batch_size}, {seq_len}, {config.vocab_size})")

    assert outputs.logits.shape == (batch_size, seq_len, config.vocab_size), \
        "Output shape mismatch!"

    print("✓ Forward pass successful!")

    # Test generation
    print("\nTesting generation...")
    model.eval()
    prompt = torch.randint(0, config.vocab_size, (1, 10))
    print(f"Prompt shape: {prompt.shape}")

    generated = model.generate(
        prompt,
        max_new_tokens=20,
        temperature=1.0,
    )

    print(f"Generated shape: {generated.shape}")
    print(f"Generated {generated.shape[1] - prompt.shape[1]} new tokens")
    print("✓ Generation successful!")

    # Test with different model sizes
    print("\n" + "=" * 60)
    print("Testing different model sizes")
    print("=" * 60)

    for name in ["5M", "50M", "125M"]:
        config = ModelConfig.from_name(name)
        model = Transformer(config)
        n_params = sum(p.numel() for p in model.parameters())

        print(f"\n{name}: {n_params:,} parameters ({n_params/1e6:.1f}M)")
        print(f"  Config: L={config.n_layers}, d={config.d_model}, "
              f"heads={config.n_heads}, kv_heads={config.n_kv_heads}")

        # Quick forward pass
        input_ids = torch.randint(0, config.vocab_size, (1, 64))
        with torch.no_grad():
            outputs = model(input_ids, return_dict=True)

        print(f"  ✓ Forward pass OK: {outputs.logits.shape}")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_model_forward()
