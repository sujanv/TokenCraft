"""
Batch Encoding and Processing Pipeline for TokenCraft.
Provides padding, truncation, attention mask generation, and uniform tensor-ready batches.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Union, Dict, Any


@dataclass
class BatchEncoding:
    """
    Container representing the result of batch tokenization.
    Tensors are 2D arrays ready for consumption by deep learning frameworks (PyTorch, TensorFlow, JAX).
    """
    input_ids: List[List[int]]
    attention_mask: List[List[int]]
    token_strings: List[List[str]]
    lengths: List[int]
    max_length: int
    padding_side: str = "right"
    truncated: List[bool] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "input_ids": self.input_ids,
            "attention_mask": self.attention_mask,
            "token_strings": self.token_strings,
            "lengths": self.lengths,
            "max_length": self.max_length,
            "padding_side": self.padding_side,
            "truncated": self.truncated,
        }

    def __getitem__(self, item: str) -> Any:
        if item == "input_ids":
            return self.input_ids
        elif item == "attention_mask":
            return self.attention_mask
        elif item == "token_strings":
            return self.token_strings
        elif item == "lengths":
            return self.lengths
        raise KeyError(f"Key {item} not found in BatchEncoding")

    def __len__(self) -> int:
        return len(self.input_ids)


class BatchProcessor:
    """
    Processes collections of text sequences into uniformly padded and truncated batches.
    """

    @staticmethod
    def encode_batch(
        tokenizer: Any,
        texts: List[str],
        padding: Union[bool, str] = "longest",  # "longest", "max_length", or False
        max_length: Optional[int] = None,
        truncation: bool = True,
        padding_side: str = "right",  # "right" or "left"
        return_attention_mask: bool = True,
        **kwargs,
    ) -> BatchEncoding:
        """
        Tokenizes a list of strings into a BatchEncoding.

        Args:
            tokenizer: Any BaseTokenizer instance.
            texts: List of input strings.
            padding: Padding strategy ("longest", "max_length", or False/None).
            max_length: Maximum sequence length.
            truncation: Whether to truncate sequences exceeding max_length.
            padding_side: Direction to pad ("right" for GPT/BERT, "left" for decoder-only generation).
            return_attention_mask: Whether to compute 1/0 attention masks.
        """
        # Determine pad token ID
        pad_id = None
        for cand in ["<pad>", "[PAD]", "<|pad|>"]:
            if cand in tokenizer.special_tokens:
                pad_id = tokenizer.special_tokens[cand]
                break
        if pad_id is None:
            # Fall back to 0
            pad_id = tokenizer.vocab.get("<pad>", tokenizer.vocab.get("[PAD]", 0))

        raw_results = [tokenizer.encode(t, **kwargs) for t in texts]
        raw_ids = [r.token_ids[:] for r in raw_results]
        raw_strings = [r.token_strings[:] for r in raw_results]

        # Calculate target max length
        longest_len = max((len(ids) for ids in raw_ids), default=0)

        if padding == "max_length":
            target_len = max_length if max_length is not None else longest_len
        elif padding == "longest" or padding is True:
            target_len = longest_len
            if max_length is not None and truncation:
                target_len = min(target_len, max_length)
        else:
            target_len = longest_len
            if max_length is not None and truncation:
                target_len = min(target_len, max_length)

        final_ids: List[List[int]] = []
        final_strings: List[List[str]] = []
        attention_masks: List[List[int]] = []
        lengths: List[int] = []
        truncated_flags: List[bool] = []

        for ids, strings in zip(raw_ids, raw_strings):
            was_truncated = False
            # Truncation
            if truncation and max_length is not None and len(ids) > max_length:
                if padding_side == "left":
                    ids = ids[-max_length:]
                    strings = strings[-max_length:]
                else:
                    ids = ids[:max_length]
                    strings = strings[:max_length]
                was_truncated = True

            actual_len = len(ids)
            lengths.append(actual_len)
            truncated_flags.append(was_truncated)

            pad_count = max(0, target_len - actual_len)
            if padding:
                if padding_side == "left":
                    padded_ids = [pad_id] * pad_count + ids
                    padded_strings = ["<pad>"] * pad_count + strings
                    mask = [0] * pad_count + [1] * actual_len
                else:
                    padded_ids = ids + [pad_id] * pad_count
                    padded_strings = strings + ["<pad>"] * pad_count
                    mask = [1] * actual_len + [0] * pad_count
            else:
                padded_ids = ids
                padded_strings = strings
                mask = [1] * actual_len

            final_ids.append(padded_ids)
            final_strings.append(padded_strings)
            attention_masks.append(mask if return_attention_mask else [])

        return BatchEncoding(
            input_ids=final_ids,
            attention_mask=attention_masks,
            token_strings=final_strings,
            lengths=lengths,
            max_length=target_len,
            padding_side=padding_side,
            truncated=truncated_flags,
        )
