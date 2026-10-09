"""
Exporter utilities converting TokenCraft models to Hugging Face tokenizers and OpenAI formats.
"""

from __future__ import annotations
import json
import os
from typing import Dict, Any

from tokencraft.bpe import BPETokenizer


class HuggingFaceExporter:
    """Exports BPETokenizer to Hugging Face tokenizer.json format."""

    @staticmethod
    def export_tokenizer_json(tokenizer: BPETokenizer, output_file: str) -> None:
        """
        Converts BPETokenizer to standard Hugging Face tokenizers format (tokenizer.json).
        Compatible with AutoTokenizer from Hugging Face Transformers.
        """
        # Convert merges to list of strings "u v"
        hf_merges = [f"{u} {v}" for u, v in tokenizer.merges]

        hf_data = {
            "version": "1.0",
            "truncation": None,
            "padding": None,
            "added_tokens": [
                {
                    "id": idx,
                    "content": tok,
                    "single_word": False,
                    "lstrip": False,
                    "rstrip": False,
                    "normalized": False,
                    "special": True,
                }
                for tok, idx in tokenizer.special_tokens.items()
            ],
            "normalizer": {
                "type": "Sequence",
                "normalizers": [
                    {"type": "NFKC"}
                ]
            },
            "pre_tokenizer": {
                "type": "ByteLevel",
                "add_prefix_space": False,
                "trim_offsets": True,
                "use_regex": True,
            },
            "post_processor": None,
            "decoder": {
                "type": "ByteLevel",
                "add_prefix_space": False,
                "trim_offsets": True,
                "use_regex": True,
            },
            "model": {
                "type": "BPE",
                "dropout": None,
                "unk_token": "<unk>",
                "continuing_subword_prefix": None,
                "end_of_word_suffix": None,
                "fuse_unk": False,
                "byte_fallback": True,
                "vocab": tokenizer.vocab,
                "merges": hf_merges,
            }
        }

        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(hf_data, f, indent=2, ensure_ascii=False)


class OpenAIGPT2Exporter:
    """Exports BPETokenizer to classic GPT-2 vocab.json and merges.txt."""

    @staticmethod
    def export(tokenizer: BPETokenizer, output_dir: str) -> None:
        """
        Writes:
          1. vocab.json - mapping token string to ID
          2. merges.txt - newline-separated merge pairs
        """
        os.makedirs(output_dir, exist_ok=True)
        vocab_path = os.path.join(output_dir, "vocab.json")
        merges_path = os.path.join(output_dir, "merges.txt")

        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(tokenizer.vocab, f, indent=2, ensure_ascii=False)

        with open(merges_path, "w", encoding="utf-8") as f:
            f.write("#version: 0.2\n")
            for u, v in tokenizer.merges:
                f.write(f"{u} {v}\n")
