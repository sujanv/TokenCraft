"""
Base tokenizer abstractions, data structures, and serialization interfaces.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Union
import json
import os


@dataclass
class Token:
    """Represents a single emitted token with metadata."""
    id: int
    text: str
    start_char: int = -1
    end_char: int = -1
    is_special: bool = False
    byte_repr: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TokenizationResult:
    """Result of tokenizing a piece of text."""
    text: str
    tokens: List[Token]
    token_ids: List[int] = field(default_factory=list)
    token_strings: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.token_ids and self.tokens:
            self.token_ids = [t.id for t in self.tokens]
        if not self.token_strings and self.tokens:
            self.token_strings = [t.text for t in self.tokens]

    @property
    def token_count(self) -> int:
        return len(self.tokens)

    @property
    def char_count(self) -> int:
        return len(self.text)

    @property
    def byte_count(self) -> int:
        return len(self.text.encode("utf-8"))

    @property
    def word_count(self) -> int:
        # Whitespace separated word count for fertility calculation
        words = self.text.strip().split()
        return max(1, len(words))

    @property
    def compression_ratio(self) -> float:
        """Bytes per token (higher = better compression of raw bytes into fewer tokens)."""
        if self.token_count == 0:
            return 0.0
        return round(self.byte_count / self.token_count, 3)

    @property
    def fertility(self) -> float:
        """Tokens per word (lower = closer to 1 token per word)."""
        if self.word_count == 0:
            return 0.0
        return round(self.token_count / self.word_count, 3)

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "token_ids": self.token_ids,
            "token_strings": self.token_strings,
            "token_count": self.token_count,
            "char_count": self.char_count,
            "byte_count": self.byte_count,
            "word_count": self.word_count,
            "compression_ratio": self.compression_ratio,
            "fertility": self.fertility,
            "tokens": [t.to_dict() for t in self.tokens],
        }


@dataclass
class TokenizerMetrics:
    """Quantitative performance and statistical metrics for a tokenizer."""
    name: str
    vocab_size: int
    token_count: int
    char_count: int
    byte_count: int
    word_count: int
    compression_ratio: float  # bytes / token
    fertility: float          # tokens / word
    oov_count: int = 0
    oov_rate: float = 0.0     # percentage of tokens that are [UNK]

    def to_dict(self) -> dict:
        return asdict(self)


class BaseTokenizer(ABC):
    """Abstract base class for all tokenizers in TokenCraft."""

    def __init__(self, name: str = "BaseTokenizer"):
        self.name = name
        self.vocab: Dict[str, int] = {}
        self.inverse_vocab: Dict[int, str] = {}
        self.special_tokens: Dict[str, int] = {}
        self.is_trained: bool = False

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def _sync_inverse_vocab(self):
        """Rebuild inverse vocabulary mapping integer IDs to string tokens."""
        self.inverse_vocab = {idx: token for token, idx in self.vocab.items()}

    def add_special_token(self, token: str) -> int:
        """Register a special token, assigning a new ID if not already present."""
        if token in self.vocab:
            idx = self.vocab[token]
        else:
            idx = len(self.vocab)
            self.vocab[token] = idx
            self.inverse_vocab[idx] = token
        self.special_tokens[token] = idx
        return idx

    @abstractmethod
    def train(self, corpus: Union[str, List[str]], vocab_size: int, **kwargs) -> None:
        """Train the tokenizer on a given text corpus until reaching target vocab_size."""
        pass

    @abstractmethod
    def encode(self, text: str) -> TokenizationResult:
        """Tokenize text into a TokenizationResult containing Tokens and IDs."""
        pass

    @abstractmethod
    def decode(self, token_ids: List[int]) -> str:
        """Reconstruct original text from a sequence of token IDs."""
        pass

    def tokenize(self, text: str) -> List[str]:
        """Convenience method returning just the token strings."""
        return self.encode(text).token_strings

    def to_dict(self) -> Dict[str, Any]:
        """Serialize tokenizer configuration and vocabulary state."""
        return {
            "name": self.name,
            "type": self.__class__.__name__,
            "vocab": self.vocab,
            "special_tokens": self.special_tokens,
            "is_trained": self.is_trained,
        }

    def save(self, file_path: str) -> None:
        """Save the tokenizer state to a JSON file."""
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        data = self.to_dict()
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaseTokenizer":
        """Reconstruct tokenizer instance from serialized dictionary."""
        pass

    @classmethod
    def load(cls, file_path: str) -> "BaseTokenizer":
        """Load tokenizer from JSON file."""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def compute_metrics(self, text: str) -> TokenizerMetrics:
        """Compute metrics for the given text."""
        result = self.encode(text)
        unk_id = self.special_tokens.get("<unk>", self.special_tokens.get("[UNK]", None))
        oov_count = sum(1 for tid in result.token_ids if tid == unk_id) if unk_id is not None else 0
        oov_rate = round((oov_count / max(1, result.token_count)) * 100, 2)

        return TokenizerMetrics(
            name=self.name,
            vocab_size=self.vocab_size,
            token_count=result.token_count,
            char_count=result.char_count,
            byte_count=result.byte_count,
            word_count=result.word_count,
            compression_ratio=result.compression_ratio,
            fertility=result.fertility,
            oov_count=oov_count,
            oov_rate=oov_rate,
        )
