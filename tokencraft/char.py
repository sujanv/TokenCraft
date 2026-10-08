"""
Character-level Tokenizer implemented from scratch.
Serves as an essential baseline for comparing subword models (BPE, WordPiece)
against granular character sequence representations.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Union, Any
from collections import Counter

from tokencraft.base import BaseTokenizer, Token, TokenizationResult


class CharacterTokenizer(BaseTokenizer):
    """
    Character-level Tokenizer.

    Each individual character is treated as an independent token.
    Properties:
    - Small vocabulary size (typically 100-500 characters).
    - High fertility rate (many tokens per word).
    - Low compression ratio (~1 byte per token).
    - Perfect representation of morphology and character typos.
    """

    def __init__(
        self,
        name: str = "Character-Tokenizer",
        unk_token: str = "<unk>",
        pad_token: str = "<pad>",
        bos_token: str = "<bos>",
        eos_token: str = "<eos>",
    ):
        super().__init__(name=name)
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self._init_special_tokens()

    def _init_special_tokens(self):
        self.vocab = {}
        self.add_special_token(self.pad_token)
        self.add_special_token(self.unk_token)
        self.add_special_token(self.bos_token)
        self.add_special_token(self.eos_token)
        self._sync_inverse_vocab()

    def train(
        self,
        corpus: Union[str, List[str]],
        vocab_size: int = 500,
        min_frequency: int = 1,
        **kwargs,
    ) -> None:
        """
        Train character vocabulary by scanning all characters in the corpus.
        """
        if isinstance(corpus, str):
            corpus = [corpus]

        self._init_special_tokens()
        char_counts: Counter[str] = Counter()

        for doc in corpus:
            for ch in doc:
                char_counts[ch] += 1

        # Keep characters meeting min_frequency up to vocab_size limit
        available_slots = vocab_size - len(self.vocab)
        most_common = char_counts.most_common()

        for ch, count in most_common:
            if len(self.vocab) >= vocab_size:
                break
            if count >= min_frequency and ch not in self.vocab:
                idx = len(self.vocab)
                self.vocab[ch] = idx

        self._sync_inverse_vocab()
        self.is_trained = True

    def encode(self, text: str) -> TokenizationResult:
        """Tokenize text into character tokens."""
        tokens: List[Token] = []
        unk_id = self.special_tokens[self.unk_token]

        for i, ch in enumerate(text):
            tok_id = self.vocab.get(ch, unk_id)
            tokens.append(
                Token(
                    id=tok_id,
                    text=ch,
                    start_char=i,
                    end_char=i + 1,
                    is_special=False,
                )
            )

        return TokenizationResult(text=text, tokens=tokens)

    def decode(self, token_ids: List[int]) -> str:
        """Reconstruct string by joining character tokens."""
        chars = []
        for tid in token_ids:
            if tid in self.inverse_vocab:
                ch = self.inverse_vocab[tid]
                if ch in self.special_tokens:
                    if ch == self.unk_token:
                        chars.append("")
                    continue
                chars.append(ch)
        return "".join(chars)

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({
            "unk_token": self.unk_token,
            "pad_token": self.pad_token,
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
        })
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CharacterTokenizer":
        tok = cls(
            name=data.get("name", "Character-Tokenizer"),
            unk_token=data.get("unk_token", "<unk>"),
            pad_token=data.get("pad_token", "<pad>"),
            bos_token=data.get("bos_token", "<bos>"),
            eos_token=data.get("eos_token", "<eos>"),
        )
        tok.vocab = data["vocab"]
        tok.special_tokens = data.get("special_tokens", {})
        tok.is_trained = data.get("is_trained", True)
        tok._sync_inverse_vocab()
        return tok
