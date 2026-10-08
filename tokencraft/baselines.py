"""
Baseline tokenizers: ByteTokenizer (pure 256-byte stream) and WordTokenizer (whitespace/word level).
"""

from __future__ import annotations
from typing import Dict, List, Optional, Union, Any
from collections import Counter
import re

from tokencraft.base import BaseTokenizer, Token, TokenizationResult


class ByteTokenizer(BaseTokenizer):
    """
    Byte-level Tokenizer mapping directly to UTF-8 bytes (0-255).
    Guarantees 100% vocabulary coverage, no <unk> ever, fixed vocab size 256.
    """

    def __init__(self, name: str = "Byte-Tokenizer"):
        super().__init__(name=name)
        self._init_vocab()

    def _init_vocab(self):
        self.vocab = {}
        for b in range(256):
            self.vocab[f"<byte_0x{b:02x}>"] = b
        self._sync_inverse_vocab()
        self.is_trained = True

    def train(self, corpus: Union[str, List[str]], vocab_size: int = 256, **kwargs) -> None:
        # Byte vocabulary is fixed to 256 byte values
        self._init_vocab()

    def encode(self, text: str) -> TokenizationResult:
        raw_bytes = text.encode("utf-8")
        tokens: List[Token] = []
        for i, b in enumerate(raw_bytes):
            tokens.append(
                Token(
                    id=b,
                    text=f"0x{b:02x}",
                    start_char=-1,
                    end_char=-1,
                    is_special=False,
                    byte_repr=f"0x{b:02x}",
                )
            )
        return TokenizationResult(text=text, tokens=tokens)

    def decode(self, token_ids: List[int]) -> str:
        valid_bytes = bytes([tid for tid in token_ids if 0 <= tid < 256])
        return valid_bytes.decode("utf-8", errors="replace")

    def to_dict(self) -> Dict[str, Any]:
        return super().to_dict()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ByteTokenizer":
        tok = cls(name=data.get("name", "Byte-Tokenizer"))
        return tok


class WordTokenizer(BaseTokenizer):
    """
    Word-level Tokenizer (splits on whitespace and punctuation).
    High vocabulary requirements; falls back to [UNK] on unseen words.
    """

    def __init__(
        self,
        name: str = "Word-Tokenizer",
        unk_token: str = "[UNK]",
        pad_token: str = "[PAD]",
        lowercase: bool = True,
    ):
        super().__init__(name=name)
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.lowercase = lowercase
        self.regex = re.compile(r"""\w+|[^\w\s]""")
        self._init_special_tokens()

    def _init_special_tokens(self):
        self.vocab = {}
        self.add_special_token(self.pad_token)
        self.add_special_token(self.unk_token)
        self._sync_inverse_vocab()

    def train(
        self,
        corpus: Union[str, List[str]],
        vocab_size: int = 5000,
        min_frequency: int = 1,
        **kwargs,
    ) -> None:
        if isinstance(corpus, str):
            corpus = [corpus]

        self._init_special_tokens()
        word_counts: Counter[str] = Counter()

        for doc in corpus:
            if self.lowercase:
                doc = doc.lower()
            words = self.regex.findall(doc)
            for w in words:
                word_counts[w] += 1

        for w, count in word_counts.most_common():
            if len(self.vocab) >= vocab_size:
                break
            if count >= min_frequency and w not in self.vocab:
                idx = len(self.vocab)
                self.vocab[w] = idx

        self._sync_inverse_vocab()
        self.is_trained = True

    def encode(self, text: str) -> TokenizationResult:
        processed_text = text.lower() if self.lowercase else text
        tokens: List[Token] = []
        unk_id = self.special_tokens[self.unk_token]

        for match in self.regex.finditer(processed_text):
            word_str = match.group(0)
            tok_id = self.vocab.get(word_str, unk_id)
            tokens.append(
                Token(
                    id=tok_id,
                    text=word_str,
                    start_char=match.start(),
                    end_char=match.end(),
                    is_special=(tok_id == unk_id),
                )
            )

        return TokenizationResult(text=text, tokens=tokens)

    def decode(self, token_ids: List[int]) -> str:
        words = []
        for tid in token_ids:
            if tid in self.inverse_vocab:
                w = self.inverse_vocab[tid]
                if w == self.pad_token:
                    continue
                words.append(w)
        text = " ".join(words)
        return re.sub(r"\s+([.,!?;:')}\]])", r"\1", text)

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({
            "unk_token": self.unk_token,
            "pad_token": self.pad_token,
            "lowercase": self.lowercase,
        })
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WordTokenizer":
        tok = cls(
            name=data.get("name", "Word-Tokenizer"),
            unk_token=data.get("unk_token", "[UNK]"),
            pad_token=data.get("pad_token", "[PAD]"),
            lowercase=data.get("lowercase", True),
        )
        tok.vocab = data["vocab"]
        tok.special_tokens = data.get("special_tokens", {})
        tok.is_trained = data.get("is_trained", True)
        tok._sync_inverse_vocab()
        return tok
