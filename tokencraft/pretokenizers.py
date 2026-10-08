"""
Pre-tokenizers, normalizers, and byte-to-unicode mappings for TokenCraft.
"""

from __future__ import annotations
import re
import unicodedata
from typing import List, Tuple, Dict


def bytes_to_unicode() -> Dict[int, str]:
    """
    Returns a dictionary mapping each byte (0..255) to a distinct printable Unicode character.
    This is the exact reversible mapping popularized by GPT-2 (Radford et al.) and Hugging Face.
    It guarantees:
      1. Every single byte (including nulls, control chars, UTF-8 continuation bytes) maps to a unique character.
      2. No character is standard whitespace (avoiding regex split bugs).
      3. Tokenization is 100% lossless for any arbitrary byte sequence.
    """
    # Standard printable ASCII ranges (excluding control characters and whitespace)
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


def unicode_to_bytes() -> Dict[str, int]:
    """Inverse of bytes_to_unicode()."""
    b2u = bytes_to_unicode()
    return {v: k for k, v in b2u.items()}


class Normalizer:
    """Configurable text normalizer for cleaning and standardizing text."""

    def __init__(
        self,
        lowercase: bool = False,
        strip_accents: bool = False,
        unicode_normal: str = "NFKC",  # "NFC", "NFD", "NFKC", "NFKD", or None
        clean_whitespace: bool = False,
    ):
        self.lowercase = lowercase
        self.strip_accents = strip_accents
        self.unicode_normal = unicode_normal
        self.clean_whitespace = clean_whitespace

    def normalize(self, text: str) -> str:
        """Apply normalization pipeline to input string."""
        if self.unicode_normal:
            text = unicodedata.normalize(self.unicode_normal, text)

        if self.strip_accents:
            # Decompose into NFD and strip non-spacing mark characters (category 'Mn')
            nfd = unicodedata.normalize("NFD", text)
            text = "".join(c for c in nfd if unicodedata.category(c) != "Mn")
            if self.unicode_normal:
                text = unicodedata.normalize(self.unicode_normal, text)

        if self.lowercase:
            text = text.lower()

        if self.clean_whitespace:
            text = re.sub(r"\s+", " ", text).strip()

        return text


class RegexPreTokenizer:
    """
    Regex-based pre-tokenizer splitting text into initial word/symbol units
    before subword algorithms (BPE / WordPiece) process them.
    """

    # GPT-2 style pre-tokenization regex pattern (Unicode-aware without external regex package)
    GPT2_PATTERN = r"""'s|'t|'re|'ve|'m|'ll|'d| ?[^\W\d_]+| ?\d+| ?[^\s\w]+|\s+(?!\S)|\s+"""

    # GPT-4 style pattern (case-insensitive contractions, punctuation clusters, numbers)
    GPT4_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)| ?[^\W\d_]+| ?\d{1,3}| ?[^\s\w]+|\s+(?!\S)|\s+"""

    # Basic whitespace & punctuation split pattern (BERT / WordPiece friendly)
    BASIC_PATTERN = r"""\w+|[^\w\s]"""

    def __init__(self, pattern: str = "gpt2"):
        if pattern == "gpt2":
            self.regex = re.compile(self.GPT2_PATTERN)
        elif pattern == "basic":
            self.regex = re.compile(self.BASIC_PATTERN)
        elif pattern == "whitespace":
            self.regex = re.compile(r"""\S+""")
        else:
            self.regex = re.compile(pattern)
        self.pattern_name = pattern

    def pre_tokenize(self, text: str) -> List[Tuple[str, int, int]]:
        """
        Splits text into chunks, returning a list of tuples: (substring, start_index, end_index).
        """
        results = []
        for match in self.regex.finditer(text):
            results.append((match.group(0), match.start(), match.end()))
        return results

    def split_text(self, text: str) -> List[str]:
        """Convenience function returning just the substrings."""
        return [chunk for chunk, _, _ in self.pre_tokenize(text)]
