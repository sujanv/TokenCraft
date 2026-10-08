"""
WordPiece Tokenizer implemented from scratch.
Uses likelihood ratio scoring during training and greedy longest-match prefix
segmentation during encoding (BERT style).
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Set, Optional, Union, Any
from collections import Counter, defaultdict
import re

from tokencraft.base import BaseTokenizer, Token, TokenizationResult
from tokencraft.pretokenizers import RegexPreTokenizer, Normalizer


class WordPieceTokenizer(BaseTokenizer):
    """
    WordPiece Tokenizer (used by BERT).

    Key properties:
    - Continuation subwords are prefixed with '##' (e.g. 'un', '##aff', '##able').
    - Training scores pairs by Likelihood Ratio: score = count(u, v) / (count(u) * count(v)).
    - Encoding uses greedy longest-match prefix search.
    - If a character cannot be matched, the whole word falls back to [UNK].
    """

    def __init__(
        self,
        name: str = "WordPiece-Tokenizer",
        prefix: str = "##",
        unk_token: str = "[UNK]",
        pad_token: str = "[PAD]",
        cls_token: str = "[CLS]",
        sep_token: str = "[SEP]",
        mask_token: str = "[MASK]",
        lowercase: bool = True,
    ):
        super().__init__(name=name)
        self.prefix = prefix
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.cls_token = cls_token
        self.sep_token = sep_token
        self.mask_token = mask_token
        self.normalizer = Normalizer(lowercase=lowercase, clean_whitespace=False)
        self.pre_tokenizer = RegexPreTokenizer(pattern="basic")
        self._init_special_tokens()

    def _init_special_tokens(self):
        """Register BERT-standard special tokens."""
        self.vocab = {}
        self.add_special_token(self.pad_token)
        self.add_special_token(self.unk_token)
        self.add_special_token(self.cls_token)
        self.add_special_token(self.sep_token)
        self.add_special_token(self.mask_token)
        self._sync_inverse_vocab()

    def train(
        self,
        corpus: Union[str, List[str]],
        vocab_size: int = 1000,
        min_frequency: int = 2,
        show_progress: bool = False,
    ) -> None:
        """
        Train WordPiece vocabulary using likelihood ratio pair selection.
        """
        if isinstance(corpus, str):
            corpus = [corpus]

        self._init_special_tokens()

        # Step 1: Pre-tokenize corpus and count word frequencies
        word_counts: Counter[str] = Counter()
        char_counts: Counter[str] = Counter()

        for doc in corpus:
            doc = self.normalizer.normalize(doc)
            words = self.pre_tokenizer.split_text(doc)
            for w in words:
                if w:
                    word_counts[w] += 1
                    for i, ch in enumerate(w):
                        tok = ch if i == 0 else f"{self.prefix}{ch}"
                        char_counts[tok] += 1

        # Step 2: Initialize vocab with all seen characters
        for ch_token in sorted(char_counts.keys()):
            if ch_token not in self.vocab:
                idx = len(self.vocab)
                self.vocab[ch_token] = idx

        self._sync_inverse_vocab()

        # Represent each word as a list of subwords
        splits: Dict[str, List[str]] = {}
        for w in word_counts.keys():
            splits[w] = [w[0]] + [f"{self.prefix}{c}" for c in w[1:]]

        # Step 3: Iteratively merge best pair based on WordPiece score
        while len(self.vocab) < vocab_size:
            # Count pair frequencies and individual token frequencies
            pair_counts: Counter[Tuple[str, str]] = Counter()
            token_counts: Counter[str] = Counter()

            for word, freq in word_counts.items():
                split = splits[word]
                if len(split) == 1:
                    token_counts[split[0]] += freq
                    continue
                for i in range(len(split) - 1):
                    pair = (split[i], split[i + 1])
                    pair_counts[pair] += freq
                    token_counts[split[i]] += freq
                token_counts[split[-1]] += freq

            if not pair_counts:
                break

            # Calculate score = count(u, v) / (count(u) * count(v))
            best_pair = None
            best_score = -1.0
            for pair, count in pair_counts.items():
                if count < min_frequency:
                    continue
                score = count / (token_counts[pair[0]] * token_counts[pair[1]])
                if score > best_score:
                    best_score = score
                    best_pair = pair

            if best_pair is None:
                break

            # Construct new merged token
            u, v = best_pair
            merged_token = u + (v[len(self.prefix):] if v.startswith(self.prefix) else v)
            if merged_token not in self.vocab:
                idx = len(self.vocab)
                self.vocab[merged_token] = idx

            # Update splits
            for word in word_counts.keys():
                split = splits[word]
                if len(split) <= 1:
                    continue
                i = 0
                new_split = []
                while i < len(split):
                    if i < len(split) - 1 and split[i] == u and split[i + 1] == v:
                        new_split.append(merged_token)
                        i += 2
                    else:
                        new_split.append(split[i])
                        i += 1
                splits[word] = new_split

            if show_progress and len(self.vocab) % 100 == 0:
                print(f"[WordPiece] Vocab size: {len(self.vocab)}/{vocab_size} (merged '{merged_token}')")

        self._sync_inverse_vocab()
        self.is_trained = True

    def _tokenize_word(self, word: str) -> List[str]:
        """
        Greedy longest-match prefix tokenization for a single word.
        If any character cannot be matched, emits [UNK].
        """
        if not word:
            return []

        tokens: List[str] = []
        start = 0
        word_len = len(word)

        while start < word_len:
            end = word_len
            cur_substr = None

            while start < end:
                sub = word[start:end]
                if start > 0:
                    sub = f"{self.prefix}{sub}"
                if sub in self.vocab:
                    cur_substr = sub
                    break
                end -= 1

            if cur_substr is None:
                return [self.unk_token]

            tokens.append(cur_substr)
            start = end

        return tokens

    def encode(self, text: str) -> TokenizationResult:
        """Tokenize text into WordPiece tokens."""
        normalized_text = self.normalizer.normalize(text)
        tokens: List[Token] = []

        chunks = self.pre_tokenizer.pre_tokenize(normalized_text)
        for chunk_str, start_idx, end_idx in chunks:
            sub_tokens = self._tokenize_word(chunk_str)

            # Map to Token dataclasses with approximate spans
            curr_pos = start_idx
            chunk_len = max(1, end_idx - start_idx)
            step_len = max(1, chunk_len // max(1, len(sub_tokens)))

            for tok_str in sub_tokens:
                is_unk = (tok_str == self.unk_token)
                tok_id = self.vocab.get(tok_str, self.special_tokens[self.unk_token])
                next_pos = min(end_idx, curr_pos + step_len)
                tokens.append(
                    Token(
                        id=tok_id,
                        text=tok_str,
                        start_char=curr_pos,
                        end_char=next_pos,
                        is_special=(tok_str in self.special_tokens),
                    )
                )
                curr_pos = next_pos

        return TokenizationResult(text=text, tokens=tokens)

    def decode(self, token_ids: List[int]) -> str:
        """
        Reconstruct text from WordPiece token IDs by stripping '##' prefixes
        and formatting punctuation.
        """
        out_tokens: List[str] = []
        for tid in token_ids:
            if tid not in self.inverse_vocab:
                continue
            tok_str = self.inverse_vocab[tid]
            if tok_str in self.special_tokens:
                if tok_str == self.unk_token:
                    out_tokens.append(tok_str)
                continue

            if tok_str.startswith(self.prefix):
                # Continuation token: attach to previous token without space
                suffix = tok_str[len(self.prefix):]
                if out_tokens:
                    out_tokens[-1] = out_tokens[-1] + suffix
                else:
                    out_tokens.append(suffix)
            else:
                out_tokens.append(tok_str)

        # Join tokens with space, preserving standard punctuation attachment
        text = " ".join(out_tokens)
        text = re.sub(r"\s+([.,!?;:')}\]])", r"\1", text)
        text = re.sub(r"([({[])\s+", r"\1", text)
        return text

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({
            "prefix": self.prefix,
            "unk_token": self.unk_token,
            "pad_token": self.pad_token,
            "cls_token": self.cls_token,
            "sep_token": self.sep_token,
            "mask_token": self.mask_token,
            "lowercase": self.normalizer.lowercase,
        })
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WordPieceTokenizer":
        tok = cls(
            name=data.get("name", "WordPiece-Tokenizer"),
            prefix=data.get("prefix", "##"),
            unk_token=data.get("unk_token", "[UNK]"),
            pad_token=data.get("pad_token", "[PAD]"),
            cls_token=data.get("cls_token", "[CLS]"),
            sep_token=data.get("sep_token", "[SEP]"),
            mask_token=data.get("mask_token", "[MASK]"),
            lowercase=data.get("lowercase", True),
        )
        tok.vocab = data["vocab"]
        tok.special_tokens = data.get("special_tokens", {})
        tok.is_trained = data.get("is_trained", True)
        tok._sync_inverse_vocab()
        return tok
