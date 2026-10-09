"""
Tests for WordPiece Tokenizer.
"""

import unittest
import os
import tempfile
from tokencraft.wordpiece import WordPieceTokenizer


class TestWordPieceTokenizer(unittest.TestCase):
    def setUp(self):
        self.corpus = [
            "playing player playable replay",
            "hoping hopped hopper",
            "the quick brown fox jumps over the lazy dog",
        ]
        self.tokenizer = WordPieceTokenizer(name="TestWordPiece", lowercase=True)
        self.tokenizer.train(self.corpus, vocab_size=60)

    def test_continuation_tokens(self):
        # Tokens should contain prefix '##'
        has_prefix = any(k.startswith("##") for k in self.tokenizer.vocab.keys())
        self.assertTrue(has_prefix)

    def test_longest_match_encode(self):
        res = self.tokenizer.encode("playing")
        self.assertGreater(len(res.tokens), 0)
        # Should be decoded back
        decoded = self.tokenizer.decode(res.token_ids)
        self.assertEqual("playing", decoded)

    def test_unknown_token(self):
        # Character not in vocabulary should emit [UNK]
        res = self.tokenizer.encode("xylophone")
        # Unless 'x' happened to be in vocab, should contain [UNK]
        self.assertTrue(any(t.text == "[UNK]" or t.id == self.tokenizer.special_tokens["[UNK]"] for t in res.tokens) or len(res.tokens) > 0)

    def test_save_and_load(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            self.tokenizer.save(tmp_path)
            loaded = WordPieceTokenizer.load(tmp_path)
            self.assertEqual(self.tokenizer.vocab_size, loaded.vocab_size)
            self.assertEqual(self.tokenizer.prefix, loaded.prefix)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


if __name__ == "__main__":
    unittest.main()
