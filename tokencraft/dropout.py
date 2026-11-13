"""
BPE-Dropout (Subword Regularization) for Byte-Pair Encoding.
Reference: Provilkov et al. (2020), "BPE-Dropout: Simple and Effective Subword Regularization"
https://arxiv.org/abs/1910.13267

Allows stochastic tokenization during training to improve model robustness,
morphological awareness, and out-of-domain generalization.
"""

from __future__ import annotations
import random
from typing import Dict, List, Tuple, Optional, Union, Any


def bpe_encode_word_dropout(
    word: Tuple[str, ...],
    bpe_ranks: Dict[Tuple[str, str], int],
    dropout_rate: float = 0.0,
    rng: Optional[random.Random] = None,
) -> List[str]:
    """
    Applies learned BPE merges to a word tuple with dropout probability.
    
    If dropout_rate == 0.0, this is standard deterministic greedy BPE.
    If dropout_rate > 0.0, each candidate merge has a probability `dropout_rate`
    of being skipped in that step, yielding alternate valid subword segmentations.
    """
    if len(word) <= 1:
        return list(word)

    if rng is None:
        rng = random.Random()

    word_list = list(word)

    while len(word_list) > 1:
        pairs = [(word_list[i], word_list[i + 1]) for i in range(len(word_list) - 1)]

        # Find all available merges present in bpe_ranks
        available_pairs = []
        for pair in pairs:
            if pair in bpe_ranks:
                available_pairs.append((bpe_ranks[pair], pair))

        if not available_pairs:
            break

        # Sort available candidate pairs by rank (lowest rank = highest priority)
        available_pairs.sort(key=lambda x: x[0])

        # Provilkov et al. formulation:
        # Filter all available pairs in the current word: each is kept with probability (1 - dropout_rate).
        # From the surviving candidates, pick the one with lowest rank (highest priority).
        if dropout_rate > 0.0:
            surviving = [p for rank, p in available_pairs if rng.random() >= dropout_rate]
            if surviving:
                # Pick surviving pair with lowest rank
                selected_pair = min(surviving, key=lambda p: bpe_ranks[p])
            else:
                # If all were dropped by chance, take the highest priority one to ensure convergence
                selected_pair = available_pairs[0][1]
        else:
            selected_pair = available_pairs[0][1]

        if selected_pair is None:
            break

        # Merge occurrences of selected_pair
        first, second = selected_pair
        new_word = []
        i = 0
        merged_any = False
        while i < len(word_list):
            if i < len(word_list) - 1 and word_list[i] == first and word_list[i + 1] == second:
                new_word.append(first + second)
                i += 2
                merged_any = True
            else:
                new_word.append(word_list[i])
                i += 1

        if not merged_any:
            break

        word_list = new_word

    return word_list


class BPEDropoutSampler:
    """
    Generates multiple stochastic tokenizations for a sentence using BPE-Dropout.
    """

    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer

    def sample(
        self,
        text: str,
        n: int = 5,
        dropout_rate: float = 0.2,
        seed: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Samples `n` distinct stochastic token segmentations of `text`.
        """
        rng = random.Random(seed)
        samples = []
        seen_strings = set()

        for _ in range(n * 3):
            if len(samples) >= n:
                break
            res = self.tokenizer.encode(text, dropout_rate=dropout_rate, rng=rng)
            sig = tuple(res.token_strings)
            if sig not in seen_strings:
                seen_strings.add(sig)
                samples.append({
                    "token_strings": res.token_strings,
                    "token_ids": res.token_ids,
                    "token_count": res.token_count,
                    "compression_ratio": res.compression_ratio,
                    "fertility": res.fertility,
                })

        # If diversity didn't fill all slots, fill remaining with repeats
        while len(samples) < n:
            res = self.tokenizer.encode(text, dropout_rate=dropout_rate, rng=rng)
            samples.append({
                "token_strings": res.token_strings,
                "token_ids": res.token_ids,
                "token_count": res.token_count,
                "compression_ratio": res.compression_ratio,
                "fertility": res.fertility,
            })

        return samples
