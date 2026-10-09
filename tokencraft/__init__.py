"""
TokenCraft - Build, train, compare, and visualize tokenizers from scratch.

Supports BPE (Byte-Pair Encoding), WordPiece, Character-level, Byte-level,
and Word-level tokenization with zero external dependencies.
"""

from tokencraft.base import (
    BaseTokenizer,
    Token,
    TokenizationResult,
    TokenizerMetrics,
)
from tokencraft.bpe import BPETokenizer
from tokencraft.wordpiece import WordPieceTokenizer
from tokencraft.unigram import UnigramTokenizer
from tokencraft.char import CharacterTokenizer
from tokencraft.baselines import ByteTokenizer, WordTokenizer
from tokencraft.comparator import TokenizerComparator

__version__ = "1.1.0"
__all__ = [
    "BaseTokenizer",
    "Token",
    "TokenizationResult",
    "TokenizerMetrics",
    "BPETokenizer",
    "WordPieceTokenizer",
    "UnigramTokenizer",
    "CharacterTokenizer",
    "ByteTokenizer",
    "WordTokenizer",
    "TokenizerComparator",
]
