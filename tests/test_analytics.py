"""
Tests for VocabularyAnalytics.
"""

import unittest
from tokencraft.bpe import BPETokenizer
from tokencraft.analytics import VocabularyAnalytics


class TestAnalytics(unittest.TestCase):
    def test_vocabulary_analytics(self):
        corpus = ["the quick brown fox jumps over the lazy dog"]
        bpe = BPETokenizer()
        bpe.train(corpus, vocab_size=275)

        stats = VocabularyAnalytics.analyze(bpe)
        self.assertGreater(stats["vocab_size"], 0)
        self.assertIn("avg_token_length", stats)
        self.assertIn("single_char_ratio", stats)
        self.assertIn("multi_char_ratio", stats)

        md = VocabularyAnalytics.render_markdown(stats)
        self.assertIn("Vocabulary Analysis", md)
        self.assertIn("Average Token Length", md)


if __name__ == "__main__":
    unittest.main()
