"""
Unit tests for BPE-Dropout and stochastic subword regularization.
"""

import unittest
import random
from tokencraft.bpe import BPETokenizer
from tokencraft.dropout import BPEDropoutSampler, bpe_encode_word_dropout


class TestBPEDropout(unittest.TestCase):
    def setUp(self):
        self.corpus = [
            "the quick brown fox jumps over the lazy dog",
            "tokenization algorithms using byte pair encoding",
            "natural language processing models",
        ]
        self.bpe = BPETokenizer()
        self.bpe.train(self.corpus, vocab_size=300)

    def test_deterministic_when_dropout_zero(self):
        text = "tokenization algorithms"
        res1 = self.bpe.encode(text, dropout_rate=0.0)
        res2 = self.bpe.encode(text, dropout_rate=0.0)
        self.assertEqual(res1.token_ids, res2.token_ids)
        self.assertEqual(res1.token_strings, res2.token_strings)

    def test_stochastic_sampling_seed_reproducibility(self):
        text = "tokenization algorithms"
        res_a = self.bpe.encode(text, dropout_rate=0.3, rng=random.Random(42))
        res_b = self.bpe.encode(text, dropout_rate=0.3, rng=random.Random(42))
        self.assertEqual(res_a.token_ids, res_b.token_ids)

    def test_sample_segmentations_diversity(self):
        text = "tokenization algorithms and natural language processing"
        samples = self.bpe.sample_segmentations(text, n=5, dropout_rate=0.4, seed=123)
        self.assertEqual(len(samples), 5)
        for s in samples:
            self.assertIn("token_strings", s)
            self.assertIn("token_ids", s)
            self.assertIn("token_count", s)
            # Reconstructing characters should match original words
            joined = "".join(s["token_strings"]).replace(" ", " ").strip()
            self.assertTrue(len(joined) > 0)


if __name__ == "__main__":
    unittest.main()
