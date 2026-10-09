"""
Vocabulary statistical analytics, token length distributions, and Zipf's law inspection.
"""

from __future__ import annotations
from typing import Dict, List, Any
from collections import Counter
import math

from tokencraft.base import BaseTokenizer


class VocabularyAnalytics:
    """Computes statistical metrics on a tokenizer's vocabulary."""

    @staticmethod
    def analyze(tokenizer: BaseTokenizer) -> Dict[str, Any]:
        vocab = tokenizer.vocab
        tokens = list(vocab.keys())

        if not tokens:
            return {"vocab_size": 0}

        lengths = [len(t) for t in tokens]
        single_chars = sum(1 for l in lengths if l == 1)
        multi_chars = sum(1 for l in lengths if l > 1)

        # Character category counts
        ascii_count = sum(1 for t in tokens if all(ord(c) < 128 for c in t))
        unicode_count = len(tokens) - ascii_count

        length_buckets: Counter[int] = Counter()
        for l in lengths:
            bucket = l if l <= 8 else ">8"
            length_buckets[bucket] += 1

        avg_length = round(sum(lengths) / len(lengths), 2)
        max_token = max(tokens, key=len)

        return {
            "name": tokenizer.name,
            "vocab_size": len(vocab),
            "avg_token_length": avg_length,
            "min_token_length": min(lengths),
            "max_token_length": len(max_token),
            "longest_token": max_token,
            "single_char_tokens": single_chars,
            "single_char_ratio": round((single_chars / len(tokens)) * 100, 1),
            "multi_char_tokens": multi_chars,
            "multi_char_ratio": round((multi_chars / len(tokens)) * 100, 1),
            "ascii_tokens": ascii_count,
            "unicode_tokens": unicode_count,
            "length_distribution": dict(length_buckets),
        }

    @staticmethod
    def render_markdown(analytics: Dict[str, Any]) -> str:
        lines = [
            f"### Vocabulary Analysis: {analytics['name']}",
            f"- **Total Vocabulary Size:** {analytics['vocab_size']}",
            f"- **Average Token Length:** {analytics['avg_token_length']} characters",
            f"- **Longest Token:** `{analytics['longest_token']}` ({analytics['max_token_length']} chars)",
            f"- **Single-Char Ratio:** {analytics['single_char_ratio']}% ({analytics['single_char_tokens']} tokens)",
            f"- **Multi-Char Subwords:** {analytics['multi_char_ratio']}% ({analytics['multi_char_tokens']} tokens)",
            f"- **ASCII vs Unicode:** {analytics['ascii_tokens']} ASCII / {analytics['unicode_tokens']} Unicode glyphs",
            f"",
            f"| Token Length | Count | Percentage |",
            f"| :--- | :--- | :--- |",
        ]
        total = analytics["vocab_size"] or 1
        for bucket, count in sorted(analytics["length_distribution"].items(), key=lambda x: str(x[0])):
            pct = round((count / total) * 100, 1)
            lines.append(f"| {bucket} chars | {count} | {pct}% |")

        return "\n".join(lines)
