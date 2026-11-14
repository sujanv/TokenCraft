"""
Cross-Tokenizer Boundary Alignment and Diff Visualizer for TokenCraft.
Analyzes exact token boundary agreement, boundary Jaccard overlap,
and highlights where two tokenization algorithms agree or diverge on the same text.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, List, Set, Tuple, Any

from tokencraft.base import BaseTokenizer, TokenizationResult


@dataclass
class TokenizerDiffResult:
    """Detailed boundary diff comparing two tokenizers on the same text."""
    tokenizer1_name: str
    tokenizer2_name: str
    text: str
    tokens1: List[str]
    tokens2: List[str]
    token_count1: int
    token_count2: int
    boundaries1: List[int]
    boundaries2: List[int]
    shared_boundaries: List[int]
    exclusive_boundaries1: List[int]
    exclusive_boundaries2: List[int]
    boundary_overlap_score: float  # Jaccard index: shared / union
    aligned_spans: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def render_cli(self) -> str:
        """Formats the diff report for ANSI terminal output."""
        bold = "\033[1m"
        reset = "\033[0m"
        dim = "\033[2m"
        green = "\033[32m"
        cyan = "\033[36m"
        yellow = "\033[33m"

        lines = [
            f"{bold}======================================================================{reset}",
            f"{bold} Cross-Tokenizer Diff: {self.tokenizer1_name} vs {self.tokenizer2_name}{reset}",
            f"{bold}======================================================================{reset}",
            f"{dim}Input Text:{reset} \"{self.text}\"\n",
            f"{cyan}{self.tokenizer1_name}{reset} ({self.token_count1} tokens):",
            f"  " + " | ".join(f"[{t}]" for t in self.tokens1),
            f"\n{yellow}{self.tokenizer2_name}{reset} ({self.token_count2} tokens):",
            f"  " + " | ".join(f"[{t}]" for t in self.tokens2),
            f"\n{bold}Boundary Analysis:{reset}",
            f"  Total Shared Cut Points:       {green}{len(self.shared_boundaries)}{reset}",
            f"  Cuts unique to {self.tokenizer1_name}: {cyan}{len(self.exclusive_boundaries1)}{reset}",
            f"  Cuts unique to {self.tokenizer2_name}: {yellow}{len(self.exclusive_boundaries2)}{reset}",
            f"  Boundary Jaccard Overlap:       {bold}{self.boundary_overlap_score * 100:.1f}%{reset}",
        ]
        return "\n".join(lines)


class TokenizerDiffEngine:
    """Computes alignment diffs and boundary overlap between two tokenizers."""

    @staticmethod
    def compare(
        tokenizer1: BaseTokenizer,
        tokenizer2: BaseTokenizer,
        text: str,
    ) -> TokenizerDiffResult:
        res1 = tokenizer1.encode(text)
        res2 = tokenizer2.encode(text)

        # Compute cumulative character boundaries from emitted tokens
        # Boundary is character offset after each token (except the last end-of-text)
        b1: List[int] = []
        cur1 = 0
        for tok in res1.token_strings[:-1]:
            cur1 += len(tok)
            b1.append(cur1)

        b2: List[int] = []
        cur2 = 0
        for tok in res2.token_strings[:-1]:
            cur2 += len(tok)
            b2.append(cur2)

        set1 = set(b1)
        set2 = set(b2)

        shared = sorted(list(set1 & set2))
        excl1 = sorted(list(set1 - set2))
        excl2 = sorted(list(set2 - set1))
        union = set1 | set2

        overlap = (len(shared) / len(union)) if union else 1.0

        # Construct unified span segments
        all_cuts = sorted(list(union | {0, len(text)}))
        aligned_spans = []
        for i in range(len(all_cuts) - 1):
            s = all_cuts[i]
            e = all_cuts[i + 1]
            seg_text = text[s:e]
            is_cut1 = e in set1 or e == len(text)
            is_cut2 = e in set2 or e == len(text)
            aligned_spans.append({
                "start": s,
                "end": e,
                "text": seg_text,
                "cut_tok1": is_cut1,
                "cut_tok2": is_cut2,
                "agreed_boundary": is_cut1 and is_cut2,
            })

        return TokenizerDiffResult(
            tokenizer1_name=tokenizer1.name,
            tokenizer2_name=tokenizer2.name,
            text=text,
            tokens1=res1.token_strings,
            tokens2=res2.token_strings,
            token_count1=res1.token_count,
            token_count2=res2.token_count,
            boundaries1=b1,
            boundaries2=b2,
            shared_boundaries=shared,
            exclusive_boundaries1=excl1,
            exclusive_boundaries2=excl2,
            boundary_overlap_score=round(overlap, 3),
            aligned_spans=aligned_spans,
        )
