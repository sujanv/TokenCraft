"""
Tests for pre-tokenizers, normalizers, and byte-to-unicode mappings.
"""

import unittest
from tokencraft.pretokenizers import (
    bytes_to_unicode,
    unicode_to_bytes,
    RegexPreTokenizer,
    Normalizer,
)


class TestPreTokenizers(unittest.TestCase):
    def test_bytes_to_unicode_bijection(self):
        b2u = bytes_to_unicode()
        u2b = unicode_to_bytes()

        self.assertEqual(len(b2u), 256)
        self.assertEqual(len(u2b), 256)

        # Ensure bijection
        for b in range(256):
            u_char = b2u[b]
            self.assertEqual(u2b[u_char], b)

    def test_normalizer(self):
        norm = Normalizer(lowercase=True, clean_whitespace=True, strip_accents=True)
        text = "  HÉLLÔ   WÖRLD!  "
        res = norm.normalize(text)
        self.assertEqual(res, "hello world!")

    def test_regex_pretokenizer_gpt2(self):
        tok = RegexPreTokenizer("gpt2")
        text = "Hello world! You're awesome."
        chunks = tok.split_text(text)
        self.assertIn("Hello", chunks)
        self.assertIn(" world", chunks)
        self.assertIn("'re", chunks)

    def test_regex_pretokenizer_spans(self):
        tok = RegexPreTokenizer("basic")
        text = "hello world"
        spans = tok.pre_tokenize(text)
        self.assertEqual(len(spans), 2)
        self.assertEqual(spans[0], ("hello", 0, 5))
        self.assertEqual(spans[1], ("world", 6, 11))


if __name__ == "__main__":
    unittest.main()
