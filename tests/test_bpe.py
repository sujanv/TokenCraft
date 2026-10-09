"""
Tests for Byte-Pair Encoding (BPE) Tokenizer.
"""

import unittest
import os
import tempfile
from tokencraft.bpe import BPETokenizer


class TestBPETokenizer(unittest.TestCase):
    def setUp(self):
        self.corpus = [
            "the quick brown fox jumps over the lazy dog",
            "tokenization using byte pair encoding is powerful",
            "low lower lowest widely wider widest",
            "hello world hello there",
        ]
        self.tokenizer = BPETokenizer(name="TestBPE")
        self.tokenizer.train(self.corpus, vocab_size=320)

    def test_vocab_size(self):
        # Initial base bytes (256) + 3 special tokens + learned merges
        self.assertGreaterEqual(self.tokenizer.vocab_size, 259)

    def test_lossless_roundtrip_ascii(self):
        sentence = "the quick brown fox jumps over the lazy dog"
        res = self.tokenizer.encode(sentence)
        decoded = self.tokenizer.decode(res.token_ids)
        self.assertEqual(sentence, decoded)

    def test_lossless_roundtrip_unicode_and_emojis(self):
        sentence = "Testing Unicode: こんにちは 🚀✨ and Accents: café, naïve!"
        res = self.tokenizer.encode(sentence)
        decoded = self.tokenizer.decode(res.token_ids)
        self.assertEqual(sentence, decoded)

    def test_special_tokens(self):
        self.tokenizer.add_special_token("<|custom|>")
        text = "Hello <|custom|> World"
        res = self.tokenizer.encode(text)
        self.assertIn("<|custom|>", res.token_strings)
        decoded = self.tokenizer.decode(res.token_ids)
        self.assertEqual(text, decoded)

    def test_save_and_load(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            self.tokenizer.save(tmp_path)
            loaded = BPETokenizer.load(tmp_path)
            self.assertEqual(self.tokenizer.vocab_size, loaded.vocab_size)
            self.assertEqual(self.tokenizer.merges, loaded.merges)

            text = "Testing loaded BPE model"
            self.assertEqual(
                self.tokenizer.encode(text).token_ids,
                loaded.encode(text).token_ids
            )
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


if __name__ == "__main__":
    unittest.main()
