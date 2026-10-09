"""
Tests for Character and Baseline Tokenizers.
"""

import unittest
from tokencraft.char import CharacterTokenizer
from tokencraft.baselines import ByteTokenizer, WordTokenizer


class TestCharacterAndBaselines(unittest.TestCase):
    def test_character_tokenizer(self):
        corpus = ["hello world"]
        tok = CharacterTokenizer()
        tok.train(corpus, vocab_size=50)

        sentence = "hello"
        res = tok.encode(sentence)
        self.assertEqual(res.token_count, len(sentence))
        decoded = tok.decode(res.token_ids)
        self.assertEqual(sentence, decoded)

    def test_byte_tokenizer(self):
        tok = ByteTokenizer()
        self.assertEqual(tok.vocab_size, 256)

        sentence = "Byte tokenizer test: 🚀"
        res = tok.encode(sentence)
        self.assertEqual(res.token_count, len(sentence.encode("utf-8")))
        decoded = tok.decode(res.token_ids)
        self.assertEqual(sentence, decoded)

    def test_word_tokenizer(self):
        corpus = ["the quick brown fox jumps over the lazy dog"]
        tok = WordTokenizer()
        tok.train(corpus, vocab_size=50)

        res = tok.encode("the quick brown fox")
        self.assertEqual(res.token_count, 4)
        decoded = tok.decode(res.token_ids)
        self.assertEqual("the quick brown fox", decoded)


if __name__ == "__main__":
    unittest.main()
