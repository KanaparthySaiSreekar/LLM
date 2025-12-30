"""Tokenizer wrapper for SentencePiece."""

import sentencepiece as spm
from typing import List, Union
from pathlib import Path
import json


class Tokenizer:
    """SentencePiece tokenizer wrapper.

    Provides a simple interface for encoding/decoding text using SentencePiece BPE.

    Args:
        model_path: Path to trained SentencePiece model (.model file)
    """

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)

        # Special token IDs (standard configuration)
        self.unk_id = self.sp.unk_id()
        self.bos_id = self.sp.bos_id()
        self.eos_id = self.sp.eos_id()
        self.pad_id = self.sp.pad_id()

        # Vocab size
        self.vocab_size = self.sp.vocab_size()

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> List[int]:
        """Encode text to token IDs.

        Args:
            text: Text to encode
            add_bos: Whether to add BOS token at start
            add_eos: Whether to add EOS token at end

        Returns:
            List of token IDs
        """
        tokens = self.sp.encode(text)

        if add_bos:
            tokens = [self.bos_id] + tokens
        if add_eos:
            tokens = tokens + [self.eos_id]

        return tokens

    def decode(self, tokens: Union[List[int], int]) -> str:
        """Decode token IDs to text.

        Args:
            tokens: Token ID or list of token IDs

        Returns:
            Decoded text
        """
        if isinstance(tokens, int):
            tokens = [tokens]

        return self.sp.decode(tokens)

    def encode_batch(
        self,
        texts: List[str],
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> List[List[int]]:
        """Encode batch of texts.

        Args:
            texts: List of texts to encode
            add_bos: Whether to add BOS token at start
            add_eos: Whether to add EOS token at end

        Returns:
            List of token ID lists
        """
        return [self.encode(text, add_bos=add_bos, add_eos=add_eos) for text in texts]

    def decode_batch(self, token_lists: List[List[int]]) -> List[str]:
        """Decode batch of token ID lists.

        Args:
            token_lists: List of token ID lists

        Returns:
            List of decoded texts
        """
        return [self.decode(tokens) for tokens in token_lists]

    def token_to_id(self, token: str) -> int:
        """Get ID for a token string."""
        return self.sp.piece_to_id(token)

    def id_to_token(self, id: int) -> str:
        """Get token string for an ID."""
        return self.sp.id_to_piece(id)

    def save_config(self, path: str):
        """Save tokenizer configuration.

        Args:
            path: Path to save config JSON
        """
        config = {
            "model_path": str(self.model_path),
            "vocab_size": self.vocab_size,
            "bos_id": self.bos_id,
            "eos_id": self.eos_id,
            "pad_id": self.pad_id,
            "unk_id": self.unk_id,
        }

        with open(path, 'w') as f:
            json.dump(config, f, indent=2)

    @classmethod
    def from_pretrained(cls, tokenizer_dir: str) -> "Tokenizer":
        """Load tokenizer from directory.

        Args:
            tokenizer_dir: Directory containing tokenizer.model

        Returns:
            Tokenizer instance
        """
        tokenizer_dir = Path(tokenizer_dir)
        model_path = tokenizer_dir / "tokenizer.model"

        if not model_path.exists():
            raise FileNotFoundError(f"Tokenizer model not found at {model_path}")

        return cls(str(model_path))

    def __len__(self) -> int:
        """Return vocabulary size."""
        return self.vocab_size

    def __repr__(self) -> str:
        return f"Tokenizer(vocab_size={self.vocab_size}, model={self.model_path})"
