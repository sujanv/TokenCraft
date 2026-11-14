"""
Unit tests for cross-tokenizer boundary alignment and diff engine.
"""

import unittest
from tokencraft.bpe import BPETokenizer
from tokencraft.wordpiece import WordPieceTokenizer
from tokencraft.char import CharacterTokenizer
from tokencraft.diff import TokenizerDiffEngine, TokenizerDiffResult


class TestTokenizerDiff(unittest.TestCase):
    def setUp(self):
        corpus = ["the quick brown fox jumps over the lazy dog"]
        self.bpe = BPETokenizer(name="BPE")
        self.bpe.train(corpus, vocab_size=280)

        self.wp = WordPieceTokenizer(name="WordPiece")
        self.wp.train(corpus, vocab_size=80)

        self.char_tok = CharacterTokenizer(name="Char")
        self.char_tok.train(corpus, vocab_size=50)

    def test_identical_tokenizer_diff(self):
        text = "the quick brown"
        diff = TokenizerDiffEngine.compare(self.bpe, self.bpe, text)
        self.assertEqual(diff.boundary_overlap_score, 1.0)
        self.assertEqual(len(diff.exclusive_boundaries1), 0)
        self.assertEqual(len(diff.exclusive_boundaries2), 0)

    def test_cross_tokenizer_diff(self):
        text = "the quick brown fox"
        diff = TokenizerDiffEngine.compare(self.bpe, self.wp, text)
        self.assertIsInstance(diff, TokenizerDiffResult)
        self.assertGreaterEqual(diff.boundary_overlap_score, 0.0)
        self.assertLessEqual(diff.boundary_overlap_score, 1.0)
        self.assertTrue(len(diff.aligned_spans) > 0)

        cli_output = diff.render_cli()
        self.assertIn("Cross-Tokenizer Diff", cli_output)
        self.assertIn("Boundary Jaccard Overlap", cli_output)


if __name__ == "__main__":
    unittest.main()
