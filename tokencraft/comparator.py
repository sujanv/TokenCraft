"""
Comparison engine for evaluating BPE, WordPiece, Character, Byte, and Word tokenizers.
Provides side-by-side metrics, character span alignment, and visual reporting.
"""

from __future__ import annotations
from typing import Dict, List, Any, Optional
import time
import math

from tokencraft.base import BaseTokenizer, TokenizationResult, TokenizerMetrics


class TokenizerComparator:
    """
    Compares multiple tokenizers on arbitrary input text or test suites.
    """

    def __init__(self, tokenizers: Optional[List[BaseTokenizer]] = None):
        self.tokenizers: Dict[str, BaseTokenizer] = {}
        if tokenizers:
            for tok in tokenizers:
                self.add_tokenizer(tok)

    def add_tokenizer(self, tokenizer: BaseTokenizer) -> None:
        self.tokenizers[tokenizer.name] = tokenizer

    def compare_sentence(self, sentence: str) -> Dict[str, Any]:
        """
        Tokenizes the input sentence across all registered tokenizers,
        returning side-by-side results, metrics, and character-level alignment.
        """
        results: Dict[str, TokenizationResult] = {}
        metrics: Dict[str, TokenizerMetrics] = {}
        splits: Dict[str, List[Dict[str, Any]]] = {}

        for name, tok in self.tokenizers.items():
            res = tok.encode(sentence)
            results[name] = res
            metrics[name] = tok.compute_metrics(sentence)

            splits[name] = [
                {
                    "id": t.id,
                    "text": t.text,
                    "is_special": t.is_special,
                    "start": t.start_char,
                    "end": t.end_char,
                    "byte_repr": t.byte_repr,
                }
                for t in res.tokens
            ]

        # Comparative analysis summary
        comparison_table = []
        for name, m in metrics.items():
            comparison_table.append({
                "Tokenizer": name,
                "Vocab Size": m.vocab_size,
                "Token Count": m.token_count,
                "Compression Ratio": f"{m.compression_ratio:.2f} B/tok",
                "Fertility": f"{m.fertility:.2f} tok/wd",
                "OOV Rate": f"{m.oov_rate:.1f}%",
            })

        return {
            "sentence": sentence,
            "metrics": {k: v.to_dict() for k, v in metrics.items()},
            "splits": splits,
            "table": comparison_table,
        }

    def benchmark_speed(self, sentences: List[str], iterations: int = 10) -> Dict[str, Any]:
        """
        Benchmark encoding and decoding latency / throughput across tokenizers.
        """
        results = {}
        total_chars = sum(len(s) for s in sentences) * iterations
        total_bytes = sum(len(s.encode("utf-8")) for s in sentences) * iterations

        for name, tok in self.tokenizers.items():
            # Warm up
            for s in sentences[:2]:
                _ = tok.decode(tok.encode(s).token_ids)

            # Benchmark Encode
            start = time.perf_counter()
            total_tokens = 0
            all_ids = []
            for _ in range(iterations):
                for s in sentences:
                    res = tok.encode(s)
                    total_tokens += res.token_count
                    all_ids.append(res.token_ids)
            encode_time = time.perf_counter() - start

            # Benchmark Decode
            start = time.perf_counter()
            for t_ids in all_ids:
                _ = tok.decode(t_ids)
            decode_time = time.perf_counter() - start

            encode_speed_kb = (total_bytes / 1024) / max(0.0001, encode_time)
            results[name] = {
                "vocab_size": tok.vocab_size,
                "encode_latency_ms": round((encode_time / (len(sentences) * iterations)) * 1000, 3),
                "decode_latency_ms": round((decode_time / (len(sentences) * iterations)) * 1000, 3),
                "encode_throughput_tokens_sec": round(total_tokens / max(0.0001, encode_time), 1),
                "encode_throughput_kb_sec": round(encode_speed_kb, 1),
            }

        return results

    def generate_markdown_report(self, sentence: str) -> str:
        """Generates a comprehensive markdown comparison report."""
        comp = self.compare_sentence(sentence)
        lines = [
            f"# TokenCraft Comparison Report",
            f"",
            f"**Input Sentence:**",
            f"> `{sentence}`",
            f"",
            f"## Tokenizer Metrics",
            f"",
            f"| Tokenizer | Vocab Size | Tokens | Compression (Bytes/Token) | Fertility (Tokens/Word) | OOV Rate |",
            f"| :--- | :--- | :--- | :--- | :--- | :--- |",
        ]

        for row in comp["table"]:
            lines.append(
                f"| **{row['Tokenizer']}** | {row['Vocab Size']} | {row['Token Count']} | {row['Compression Ratio']} | {row['Fertility']} | {row['OOV Rate']} |"
            )

        lines.extend([
            f"",
            f"## Sentence Tokenization Split Breakdown",
            f"",
        ])

        for name, tokens in comp["splits"].items():
            token_display = " ".join([f"`[{t['text']}]`" for t in tokens])
            lines.extend([
                f"### {name}",
                f"- **Tokens emitted:** {len(tokens)}",
                f"- **Split visualization:** {token_display}",
                f"- **Token IDs:** `{[t['id'] for t in tokens]}`",
                f"",
            ])

        return "\n".join(lines)
