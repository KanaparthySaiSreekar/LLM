"""Train SentencePiece tokenizer."""

import argparse
import sentencepiece as spm
from pathlib import Path
import json


def train_tokenizer(
    input_files: list,
    model_prefix: str,
    vocab_size: int = 8000,
    model_type: str = "bpe",
    character_coverage: float = 1.0,
    byte_fallback: bool = True,
):
    """Train SentencePiece tokenizer.

    Args:
        input_files: List of input text files
        model_prefix: Prefix for output files (will create .model and .vocab)
        vocab_size: Target vocabulary size
        model_type: Model type ('bpe' or 'unigram')
        character_coverage: Character coverage (1.0 for English, 0.9995 for CJK)
        byte_fallback: Enable byte fallback for unknown characters
    """
    # Join input files
    input_str = ",".join(input_files)

    # SentencePiece training arguments
    train_args = {
        "input": input_str,
        "model_prefix": model_prefix,
        "model_type": model_type,
        "vocab_size": vocab_size,
        "character_coverage": character_coverage,
        "byte_fallback": byte_fallback,
        "normalization_rule_name": "nfkc",
        "remove_extra_whitespaces": False,  # Preserve whitespace for code
        "add_dummy_prefix": True,
        "split_by_unicode_script": True,
        "split_by_whitespace": True,
        # Special tokens
        "unk_id": 0,
        "bos_id": 1,
        "eos_id": 2,
        "pad_id": 3,
        # Reserve some IDs for future special tokens
        "control_symbols": [f"<reserved_{i}>" for i in range(4, 16)],
    }

    # Convert args to command line format
    train_cmd = " ".join([f"--{k}={v}" for k, v in train_args.items()])

    print(f"Training SentencePiece tokenizer...")
    print(f"Input files: {input_files}")
    print(f"Vocab size: {vocab_size}")
    print(f"Model type: {model_type}")
    print(f"Output prefix: {model_prefix}")

    # Train tokenizer
    spm.SentencePieceTrainer.train(train_cmd)

    print(f"✓ Tokenizer training complete!")
    print(f"  Model saved to: {model_prefix}.model")
    print(f"  Vocab saved to: {model_prefix}.vocab")

    # Save configuration
    config = {
        "vocab_size": vocab_size,
        "model_type": model_type,
        "character_coverage": character_coverage,
        "byte_fallback": byte_fallback,
        "special_tokens": {
            "unk_id": 0,
            "bos_id": 1,
            "eos_id": 2,
            "pad_id": 3,
        }
    }

    config_path = f"{model_prefix}_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"  Config saved to: {config_path}")

    # Test tokenizer
    print("\nTesting tokenizer...")
    sp = spm.SentencePieceProcessor()
    sp.load(f"{model_prefix}.model")

    test_texts = [
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "Neural networks are awesome!",
    ]

    for text in test_texts:
        tokens = sp.encode(text)
        decoded = sp.decode(tokens)
        print(f"  '{text}' -> {len(tokens)} tokens -> '{decoded}'")


def main():
    parser = argparse.ArgumentParser(description="Train SentencePiece tokenizer")

    parser.add_argument(
        "--input",
        type=str,
        nargs="+",
        required=True,
        help="Input text files (can specify multiple)"
    )
    parser.add_argument(
        "--model-prefix",
        type=str,
        default="tokenizer",
        help="Output model prefix (default: tokenizer)"
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=8000,
        help="Vocabulary size (default: 8000)"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="bpe",
        choices=["bpe", "unigram"],
        help="Model type (default: bpe)"
    )
    parser.add_argument(
        "--character-coverage",
        type=float,
        default=1.0,
        help="Character coverage (default: 1.0 for English)"
    )
    parser.add_argument(
        "--no-byte-fallback",
        action="store_true",
        help="Disable byte fallback"
    )

    args = parser.parse_args()

    # Validate input files exist
    for input_file in args.input:
        if not Path(input_file).exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")

    # Train tokenizer
    train_tokenizer(
        input_files=args.input,
        model_prefix=args.model_prefix,
        vocab_size=args.vocab_size,
        model_type=args.model_type,
        character_coverage=args.character_coverage,
        byte_fallback=not args.no_byte_fallback,
    )


if __name__ == "__main__":
    main()
