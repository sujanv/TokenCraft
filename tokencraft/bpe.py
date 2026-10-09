"""
Byte-Pair Encoding (BPE) Tokenizer implemented from scratch.
Supports Byte-level BPE (GPT-2/GPT-4 style) with lossless UTF-8 roundtrip,
regex pre-tokenization, and special token handling.
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Set, Optional, Union, Any
from collections import Counter, defaultdict
import re

from tokencraft.base import BaseTokenizer, Token, TokenizationResult
from tokencraft.pretokenizers import (
    bytes_to_unicode,
    unicode_to_bytes,
    RegexPreTokenizer,
    Normalizer,
)


class BPETokenizer(BaseTokenizer):
    """
    Byte-level Byte-Pair Encoding (BPE) Tokenizer.

    Key properties:
    - Byte-level base vocabulary (256 initial byte tokens) guarantees no OOV (<unk>) errors.
    - Regex pre-tokenization prevents merges crossing punctuation/whitespace boundaries.
    - Exact merge ranking enables deterministic, fast subword encoding.
    - Lossless UTF-8 decoding.
    """

    def __init__(
        self,
        name: str = "BPE-Tokenizer",
        pre_tokenizer_pattern: str = "gpt2",
        lowercase: bool = False,
    ):
        super().__init__(name=name)
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = unicode_to_bytes()
        self.pre_tokenizer = RegexPreTokenizer(pattern=pre_tokenizer_pattern)
        self.normalizer = Normalizer(lowercase=lowercase)
        self.merges: List[Tuple[str, str]] = []
        self.bpe_ranks: Dict[Tuple[str, str], int] = {}
        self._init_base_vocab()

    def _init_base_vocab(self):
        """Initialize vocabulary with the 256 unique byte-unicode tokens."""
        self.vocab = {}
        # Special tokens first
        self.add_special_token("<|endoftext|>")
        self.add_special_token("<pad>")
        self.add_special_token("<unk>")

        # Add 256 byte tokens
        for b in sorted(self.byte_encoder.keys()):
            u_char = self.byte_encoder[b]
            if u_char not in self.vocab:
                idx = len(self.vocab)
                self.vocab[u_char] = idx

        self._sync_inverse_vocab()

    def _get_pairs(self, word: Tuple[str, ...]) -> Set[Tuple[str, str]]:
        """Return the set of symbol pairs in a word tuple."""
        pairs = set()
        for i in range(len(word) - 1):
            pairs.add((word[i], word[i + 1]))
        return pairs

    def train(
        self,
        corpus: Union[str, List[str]],
        vocab_size: int = 1000,
        min_frequency: int = 2,
        show_progress: bool = False,
    ) -> None:
        """
        Train BPE merge rules on the given text corpus.

        Args:
            corpus: A single string or list of text documents.
            vocab_size: Target vocabulary size (including base bytes and special tokens).
            min_frequency: Minimum pair frequency required to be merged.
            show_progress: Whether to log training iterations.
        """
        if isinstance(corpus, str):
            corpus = [corpus]

        # Reset merges and rebuild base vocabulary
        self.merges = []
        self.bpe_ranks = {}
        self._init_base_vocab()

        target_merges = vocab_size - len(self.vocab)
        if target_merges <= 0:
            self.is_trained = True
            return

        # Pre-tokenize corpus and count word frequencies
        word_freqs: Counter[Tuple[str, ...]] = Counter()
        for doc in corpus:
            doc = self.normalizer.normalize(doc)
            chunks = self.pre_tokenizer.split_text(doc)
            for chunk in chunks:
                # Convert chunk utf-8 bytes into byte-unicode characters
                b_chars = tuple(self.byte_encoder[b] for b in chunk.encode("utf-8"))
                if b_chars:
                    word_freqs[b_chars] += 1

        # Iterative BPE merges
        for step in range(target_merges):
            pair_counts: Counter[Tuple[str, str]] = Counter()
            for word, freq in word_freqs.items():
                for i in range(len(word) - 1):
                    pair_counts[(word[i], word[i + 1])] += freq

            if not pair_counts:
                break

            best_pair, best_count = pair_counts.most_common(1)[0]
            if best_count < min_frequency:
                break

            # Register new merge rule
            self.merges.append(best_pair)
            self.bpe_ranks[best_pair] = step

            # Register merged token in vocabulary
            new_token = best_pair[0] + best_pair[1]
            if new_token not in self.vocab:
                idx = len(self.vocab)
                self.vocab[new_token] = idx

            # Update word representations
            new_word_freqs: Counter[Tuple[str, ...]] = Counter()
            first, second = best_pair
            for word, freq in word_freqs.items():
                i = 0
                new_word = []
                while i < len(word):
                    if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                        new_word.append(first + second)
                        i += 2
                    else:
                        new_word.append(word[i])
                        i += 1
                new_word_freqs[tuple(new_word)] = freq
            word_freqs = new_word_freqs

            if show_progress and (step + 1) % 100 == 0:
                print(f"[BPE] Iteration {step + 1}/{target_merges}: merged {best_pair} (freq: {best_count})")

        self._sync_inverse_vocab()
        self.is_trained = True

    def _bpe_encode_word(self, word: Tuple[str, ...]) -> List[str]:
        """Apply learned BPE merge rules greedily to a single word tuple."""
        if len(word) <= 1:
            return list(word)

        word_list = list(word)
        while len(word_list) > 1:
            pairs = [(word_list[i], word_list[i + 1]) for i in range(len(word_list) - 1)]
            # Find the pair with the lowest rank (highest priority) in learned merges
            min_pair = None
            min_rank = float("inf")
            for pair in pairs:
                rank = self.bpe_ranks.get(pair, float("inf"))
                if rank < min_rank:
                    min_rank = rank
                    min_pair = pair

            if min_pair is None or min_rank == float("inf"):
                break

            # Merge the best pair
            first, second = min_pair
            new_word = []
            i = 0
            while i < len(word_list):
                if i < len(word_list) - 1 and word_list[i] == first and word_list[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word_list[i])
                    i += 1
            word_list = new_word

        return word_list

    def encode(self, text: str) -> TokenizationResult:
        """
        Tokenize input text using byte-level BPE.
        Preserves character span offsets.
        """
        normalized_text = self.normalizer.normalize(text)
        tokens: List[Token] = []

        # Find special tokens first if present in the text
        special_regex_parts = [re.escape(tok) for tok in self.special_tokens.keys()]
        special_pattern = re.compile("|".join(special_regex_parts)) if special_regex_parts else None

        current_pos = 0
        while current_pos < len(normalized_text):
            # Check for special token match at current position
            if special_pattern:
                spec_match = special_pattern.match(normalized_text, current_pos)
                if spec_match:
                    spec_tok = spec_match.group(0)
                    tok_id = self.special_tokens[spec_tok]
                    tokens.append(
                        Token(
                            id=tok_id,
                            text=spec_tok,
                            start_char=current_pos,
                            end_char=current_pos + len(spec_tok),
                            is_special=True,
                        )
                    )
                    current_pos += len(spec_tok)
                    continue

            # Find next special token or process until end
            next_special_start = len(normalized_text)
            if special_pattern:
                m = special_pattern.search(normalized_text, current_pos)
                if m:
                    next_special_start = m.start()

            subtext = normalized_text[current_pos:next_special_start]
            subtext_offset = current_pos

            # Pre-tokenize the regular segment
            chunks = self.pre_tokenizer.pre_tokenize(subtext)
            for chunk_str, start_idx, end_idx in chunks:
                b_chars = tuple(self.byte_encoder[b] for b in chunk_str.encode("utf-8"))
                bpe_tokens = self._bpe_encode_word(b_chars)

                # Map subwords back to tokens
                abs_start = subtext_offset + start_idx
                # Calculate approximate character offsets for sub-tokens
                total_len = end_idx - start_idx
                tok_offset = abs_start

                for tok_str in bpe_tokens:
                    tok_id = self.vocab.get(tok_str, self.special_tokens.get("<unk>", 0))
                    # Raw byte representation
                    raw_bytes = bytes(self.byte_decoder[c] for c in tok_str if c in self.byte_decoder)
                    tok_text_display = raw_bytes.decode("utf-8", errors="replace")

                    sub_len = max(1, int(len(tok_str) * (total_len / max(1, len(b_chars)))))
                    sub_end = min(subtext_offset + end_idx, tok_offset + sub_len)

                    tokens.append(
                        Token(
                            id=tok_id,
                            text=tok_text_display,
                            start_char=tok_offset,
                            end_char=sub_end,
                            is_special=False,
                            byte_repr=" ".join(f"0x{b:02x}" for b in raw_bytes),
                        )
                    )
                    tok_offset = sub_end

            current_pos = next_special_start

        return TokenizationResult(text=text, tokens=tokens)

    def decode(self, token_ids: List[int]) -> str:
        """
        Losslessly reconstruct string from token IDs.
        """
        byte_stream = bytearray()
        for tid in token_ids:
            if tid in self.inverse_vocab:
                tok_str = self.inverse_vocab[tid]
                # Check if this token is a special token
                if tok_str in self.special_tokens:
                    byte_stream.extend(tok_str.encode("utf-8"))
                else:
                    for ch in tok_str:
                        if ch in self.byte_decoder:
                            byte_stream.append(self.byte_decoder[ch])
                        else:
                            byte_stream.extend(ch.encode("utf-8"))
        return byte_stream.decode("utf-8", errors="replace")

    def get_provenance_tracker(self) -> "BPEProvenanceTracker":
        """Returns a provenance tracker for inspecting merge derivation trees."""
        from tokencraft.provenance import BPEProvenanceTracker
        return BPEProvenanceTracker(self.merges)

    def trace_token(self, token: str) -> str:
        """Returns ASCII derivation tree showing how the subword was assembled."""
        tracker = self.get_provenance_tracker()
        return tracker.trace_token(token)

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({
            "merges": self.merges,
            "pre_tokenizer_pattern": self.pre_tokenizer.pattern_name,
            "lowercase": self.normalizer.lowercase,
        })
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BPETokenizer":
        tok = cls(
            name=data.get("name", "BPE-Tokenizer"),
            pre_tokenizer_pattern=data.get("pre_tokenizer_pattern", "gpt2"),
            lowercase=data.get("lowercase", False),
        )
        tok.vocab = data["vocab"]
        tok.special_tokens = data.get("special_tokens", {})
        tok.merges = [tuple(m) for m in data.get("merges", [])]
        tok.bpe_ranks = {tuple(m): i for i, m in enumerate(tok.merges)}
        tok.is_trained = data.get("is_trained", True)
        tok._sync_inverse_vocab()
        return tok
