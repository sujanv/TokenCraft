"""
Tests for TokenizerComparator and Metrics.
"""

import unittest
from tokencraft.bpe import BPETokenizer
from tokencraft.wordpiece import WordPieceTokenizer
from tokencraft.char import CharacterTokenizer
from tokencraft.comparator import TokenizerComparator


class TestTokenizerComparator(unittest.TestCase):
    def setUp(self):
        corpus = ["the quick brown fox jumps over the lazy dog"]
        self.bpe = BPETokenizer()
        self.bpe.train(corpus, vocab_size=280)

        self.wp = WordPieceTokenizer()
        self.wp.train(corpus, vocab_size=60)

        self.ch = CharacterTokenizer()
        self.ch.train(corpus, vocab_size=40)

        self.comparator = TokenizerComparator([self.bpe, self.wp, self.ch])

    def test_compare_sentence_structure(self):
        comp = self.comparator.compare_sentence("the quick brown fox")
        self.assertIn("sentence", comp)
        self.assertIn("metrics", comp)
        self.assertIn("splits", comp)
        self.assertIn("table", comp)

        self.assertEqual(len(comp["splits"]), 3)
        self.assertEqual(len(comp["table"]), 3)

    def test_markdown_report_generation(self):
        report = self.comparator.generate_markdown_report("the quick brown fox")
        self.assertIn("# TokenCraft Comparison Report", report)
        self.assertIn("| Tokenizer |", report)

    def test_benchmark_speed(self):
        bench = self.comparator.benchmark_speed(["the quick fox"], iterations=5)
        self.assertIn(self.bpe.name, bench)
        self.assertIn("encode_throughput_tokens_sec", bench[self.bpe.name])


if __name__ == "__main__":
    unittest.main()
