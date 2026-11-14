"""
Unit tests for batch encoding, padding, truncation, and attention mask collation.
"""

import unittest
from tokencraft.bpe import BPETokenizer
from tokencraft.batch import BatchProcessor, BatchEncoding


class TestBatchProcessor(unittest.TestCase):
    def setUp(self):
        corpus = ["short text", "a much longer piece of text for batch testing purposes"]
        self.bpe = BPETokenizer()
        self.bpe.train(corpus, vocab_size=280)

    def test_longest_padding(self):
        texts = ["hello", "hello world testing batch"]
        batch = self.bpe.encode_batch(texts, padding="longest")
        self.assertEqual(len(batch), 2)
        # Both sequences should have same length
        self.assertEqual(len(batch.input_ids[0]), len(batch.input_ids[1]))
        self.assertEqual(batch.max_length, len(batch.input_ids[0]))

        # Attention mask verification: first should have 0s at the end
        mask0 = batch.attention_mask[0]
        self.assertIn(0, mask0)
        self.assertEqual(mask0[:batch.lengths[0]], [1] * batch.lengths[0])

    def test_max_length_truncation(self):
        texts = ["this is a very long sequence that definitely exceeds five tokens"]
        batch = self.bpe.encode_batch(texts, max_length=5, truncation=True, padding="max_length")
        self.assertEqual(len(batch.input_ids[0]), 5)
        self.assertEqual(batch.max_length, 5)
        self.assertTrue(batch.truncated[0])

    def test_left_padding(self):
        texts = ["short", "longer text"]
        batch = self.bpe.encode_batch(texts, padding="longest", padding_side="left")
        self.assertEqual(batch.padding_side, "left")
        # In left padding, padding zeros appear at the start
        mask0 = batch.attention_mask[0]
        pad_count = batch.max_length - batch.lengths[0]
        if pad_count > 0:
            self.assertEqual(mask0[:pad_count], [0] * pad_count)


if __name__ == "__main__":
    unittest.main()
