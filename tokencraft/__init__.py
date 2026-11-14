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
from tokencraft.dropout import BPEDropoutSampler, bpe_encode_word_dropout
from tokencraft.batch import BatchEncoding, BatchProcessor
from tokencraft.trie import Trie, TrieNode
from tokencraft.diff import TokenizerDiffEngine, TokenizerDiffResult
from tokencraft.provenance import BPEProvenanceTracker
from tokencraft.exporter import HuggingFaceExporter, OpenAIGPT2Exporter
from tokencraft.analytics import VocabularyAnalytics

__version__ = "1.2.0"
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
    "BPEDropoutSampler",
    "bpe_encode_word_dropout",
    "BatchEncoding",
    "BatchProcessor",
    "Trie",
    "TrieNode",
    "TokenizerDiffEngine",
    "TokenizerDiffResult",
    "BPEProvenanceTracker",
    "HuggingFaceExporter",
    "OpenAIGPT2Exporter",
    "VocabularyAnalytics",
]
