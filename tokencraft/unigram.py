"""
Unigram Subword Tokenizer (SentencePiece / Kudo 2018 style) implemented from scratch.
Uses a probabilistic language model with Viterbi dynamic programming segmentation
and iterative vocabulary pruning.
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Set, Optional, Union, Any
from collections import Counter
import math
import re

from tokencraft.base import BaseTokenizer, Token, TokenizationResult
from tokencraft.pretokenizers import RegexPreTokenizer, Normalizer


class UnigramTokenizer(BaseTokenizer):
    """
    Unigram Subword Tokenizer.

    Key properties:
    - Top-down pruning: starts with a large seed vocabulary and iteratively
      removes tokens that minimize the increase in corpus negative log-likelihood.
    - Optimal segmentation: uses Viterbi dynamic programming to select the
      subword segmentation with highest probability.
    - Used in SentencePiece, T5, ALBERT, and LLaMA tokenizers.
    """

    def __init__(
        self,
        name: str = "Unigram-Tokenizer",
        unk_token: str = "<unk>",
        pad_token: str = "<pad>",
        bos_token: str = "<s>",
        eos_token: str = "</s>",
        lowercase: bool = True,
    ):
        super().__init__(name=name)
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.normalizer = Normalizer(lowercase=lowercase)
        self.pre_tokenizer = RegexPreTokenizer(pattern="whitespace")
        self.token_probs: Dict[str, float] = {}
        self.token_log_probs: Dict[str, float] = {}
        self._init_special_tokens()

    def _init_special_tokens(self):
        self.vocab = {}
        self.add_special_token(self.pad_token)
        self.add_special_token(self.unk_token)
        self.add_special_token(self.bos_token)
        self.add_special_token(self.eos_token)
        self._sync_inverse_vocab()

    def _extract_seed_vocab(self, words: Counter[str], max_seed_size: int = 2000) -> Dict[str, float]:
        """Generate candidate substrings from the corpus to form the initial seed vocabulary."""
        substring_counts: Counter[str] = Counter()
        char_counts: Counter[str] = Counter()

        for word, freq in words.items():
            # Ensure every single character is present so tokenization cannot fail
            for ch in word:
                char_counts[ch] += freq

            word_len = len(word)
            # Collect common substrings up to length 12
            for i in range(word_len):
                for j in range(i + 1, min(word_len + 1, i + 13)):
                    sub = word[i:j]
                    substring_counts[sub] += freq

        # Base seed includes all characters
        seed_vocab: Dict[str, float] = {}
        for ch, count in char_counts.items():
            seed_vocab[ch] = count

        # Add top substrings until max_seed_size
        remaining_slots = max_seed_size - len(seed_vocab)
        for sub, count in substring_counts.most_common():
            if len(seed_vocab) >= max_seed_size:
                break
            if sub not in seed_vocab:
                seed_vocab[sub] = count

        total_count = sum(seed_vocab.values())
        return {tok: count / total_count for tok, count in seed_vocab.items()}

    def _viterbi_segment(self, word: str, log_probs: Dict[str, float]) -> Tuple[List[str], float]:
        """
        Viterbi dynamic programming search to find the optimal subword segmentation
        maximizing the log-probability of the word.
        """
        n = len(word)
        # dp[i] = (best_neg_log_prob, best_prev_index, best_token)
        dp = [float("inf")] * (n + 1)
        prev = [-1] * (n + 1)
        best_tok = [""] * (n + 1)
        dp[0] = 0.0

        for i in range(n):
            if dp[i] == float("inf"):
                continue
            for j in range(i + 1, n + 1):
                sub = word[i:j]
                if sub in log_probs:
                    score = dp[i] - log_probs[sub]
                    if score < dp[j]:
                        dp[j] = score
                        prev[j] = i
                        best_tok[j] = sub

        if dp[n] == float("inf"):
            # Fallback if unknown characters occur
            return [self.unk_token], float("inf")

        # Backtrack
        tokens = []
        curr = n
        while curr > 0:
            tokens.append(best_tok[curr])
            curr = prev[curr]
        tokens.reverse()
        return tokens, dp[n]

    def train(
        self,
        corpus: Union[str, List[str]],
        vocab_size: int = 500,
        shrink_factor: float = 0.8,
        show_progress: bool = False,
    ) -> None:
        """
        Train Unigram model using iterative expectation-maximization pruning.
        """
        if isinstance(corpus, str):
            corpus = [corpus]

        self._init_special_tokens()

        # Step 1: Count word frequencies with whitespace marker (SentencePiece '_' prefix)
        word_counts: Counter[str] = Counter()
        for doc in corpus:
            doc = self.normalizer.normalize(doc)
            for w in doc.split():
                if w:
                    # Prepend SentencePiece-style underscore for whitespace preservation
                    word_counts[" " + w] += 1

        # Step 2: Build initial seed vocabulary
        current_probs = self._extract_seed_vocab(word_counts, max_seed_size=max(vocab_size * 3, 1000))
        log_probs = {tok: math.log(max(1e-12, p)) for tok, p in current_probs.items()}

        # Always keep individual single characters in the vocabulary
        mandatory_chars = set()
        for w in word_counts.keys():
            for c in w:
                mandatory_chars.add(c)

        # Step 3: Iteratively prune vocabulary towards target vocab_size
        iteration = 0
        target_token_slots = vocab_size - len(self.special_tokens)

        while len(current_probs) > target_token_slots:
            iteration += 1
            # Segment corpus with current model and compute token usage counts
            token_counts: Counter[str] = Counter()
            total_loss = 0.0

            for word, freq in word_counts.items():
                toks, loss = self._viterbi_segment(word, log_probs)
                if loss != float("inf"):
                    total_loss += loss * freq
                    for t in toks:
                        token_counts[t] += freq

            # Estimate new probabilities from usage
            total_used = sum(token_counts.values()) or 1
            new_probs = {t: token_counts[t] / total_used for t in current_probs if token_counts[t] > 0}
            for c in mandatory_chars:
                if c not in new_probs:
                    new_probs[c] = 1e-6

            # Compute score of removing each token (frequency-based approximation)
            sorted_tokens = sorted(
                new_probs.keys(),
                key=lambda t: (t in mandatory_chars, new_probs[t]),
                reverse=True
            )

            # Determine next target size for this step
            next_size = max(target_token_slots, int(len(new_probs) * shrink_factor))
            keep_tokens = set(sorted_tokens[:next_size]) | mandatory_chars

            current_probs = {t: new_probs[t] for t in keep_tokens}
            total_p = sum(current_probs.values())
            current_probs = {t: p / total_p for t, p in current_probs.items()}
            log_probs = {t: math.log(max(1e-12, p)) for t, p in current_probs.items()}

            if show_progress:
                print(f"[Unigram] Iteration {iteration}: Vocab pruned to {len(current_probs)} tokens")

            if len(current_probs) <= target_token_slots:
                break

        # Finalize vocabulary
        self.vocab = {}
        self._init_special_tokens()
        for tok in sorted(current_probs.keys()):
            if tok not in self.vocab:
                self.vocab[tok] = len(self.vocab)

        self.token_probs = current_probs
        self.token_log_probs = log_probs
        self._sync_inverse_vocab()
        self.is_trained = True

    def encode(self, text: str) -> TokenizationResult:
        """Tokenize text into Unigram subword tokens using Viterbi search."""
        normalized_text = self.normalizer.normalize(text)
        tokens: List[Token] = []
        words = normalized_text.split()
        curr_char = 0

        for w in words:
            sp_word = " " + w
            tok_strs, _ = self._viterbi_segment(sp_word, self.token_log_probs)

            # Map to Tokens
            w_start = normalized_text.find(w, curr_char)
            if w_start == -1:
                w_start = curr_char
            w_end = w_start + len(w)
            curr_char = w_end

            tok_step = max(1, len(w) // max(1, len(tok_strs)))
            tok_offset = w_start

            for t_str in tok_strs:
                t_id = self.vocab.get(t_str, self.special_tokens[self.unk_token])
                t_end = min(w_end, tok_offset + tok_step)
                tokens.append(
                    Token(
                        id=t_id,
                        text=t_str,
                        start_char=tok_offset,
                        end_char=t_end,
                        is_special=(t_str in self.special_tokens),
                    )
                )
                tok_offset = t_end

        return TokenizationResult(text=text, tokens=tokens)

    def decode(self, token_ids: List[int]) -> str:
        """Decode Unigram token IDs back into string, stripping ' ' markers."""
        pieces = []
        for tid in token_ids:
            if tid in self.inverse_vocab:
                tok_str = self.inverse_vocab[tid]
                if tok_str in self.special_tokens:
                    continue
                pieces.append(tok_str)
        text = "".join(pieces).replace(" ", " ").strip()
        return text

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({
            "token_probs": self.token_probs,
            "unk_token": self.unk_token,
            "pad_token": self.pad_token,
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
            "lowercase": self.normalizer.lowercase,
        })
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UnigramTokenizer":
        tok = cls(
            name=data.get("name", "Unigram-Tokenizer"),
            unk_token=data.get("unk_token", "<unk>"),
            pad_token=data.get("pad_token", "<pad>"),
            bos_token=data.get("bos_token", "<s>"),
            eos_token=data.get("eos_token", "</s>"),
            lowercase=data.get("lowercase", True),
        )
        tok.vocab = data["vocab"]
        tok.special_tokens = data.get("special_tokens", {})
        tok.token_probs = data.get("token_probs", {})
        tok.token_log_probs = {t: math.log(max(1e-12, p)) for t, p in tok.token_probs.items()}
        tok.is_trained = data.get("is_trained", True)
        tok._sync_inverse_vocab()
        return tok
