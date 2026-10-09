"""
Tests for Unigram Subword Tokenizer.
"""

import unittest
import os
import tempfile
from tokencraft.unigram import UnigramTokenizer


class TestUnigramTokenizer(unittest.TestCase):
    def setUp(self):
        self.corpus = [
            "the quick brown fox jumps over the lazy dog",
            "tokenization algorithms using subwords are effective",
            "unigram language modeling searches for optimal paths",
        ]
        self.tokenizer = UnigramTokenizer(name="TestUnigram", lowercase=True)
        self.tokenizer.train(self.corpus, vocab_size=60)

    def test_vocab_presence(self):
        self.assertGreater(self.tokenizer.vocab_size, 4)
        self.assertTrue(self.tokenizer.is_trained)

    def test_encode_and_decode(self):
        text = "the quick brown fox"
        res = self.tokenizer.encode(text)
        self.assertGreater(len(res.tokens), 0)
        decoded = self.tokenizer.decode(res.token_ids)
        self.assertEqual(text, decoded)

    def test_save_and_load(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            self.tokenizer.save(tmp_path)
            loaded = UnigramTokenizer.load(tmp_path)
            self.assertEqual(self.tokenizer.vocab_size, loaded.vocab_size)

            text = "tokenization"
            self.assertEqual(
                self.tokenizer.encode(text).token_ids,
                loaded.encode(text).token_ids
            )
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


if __name__ == "__main__":
    unittest.main()
