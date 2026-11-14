"""
Unit tests for prefix Trie data structure and greedy tokenization.
"""

import unittest
from tokencraft.trie import Trie


class TestTrie(unittest.TestCase):
    def setUp(self):
        self.vocab = {
            "a": 1,
            "an": 2,
            "and": 3,
            "the": 4,
            "then": 5,
            "there": 6,
        }
        self.trie = Trie.from_vocab(self.vocab)

    def test_membership(self):
        self.assertEqual(len(self.trie), 6)
        self.assertIn("the", self.trie)
        self.assertIn("and", self.trie)
        self.assertNotIn("xyz", self.trie)
        self.assertNotIn("th", self.trie)

    def test_longest_prefix_match(self):
        match = self.trie.longest_prefix_match("therefore")
        self.assertIsNotNone(match)
        tok_str, tok_id = match
        self.assertEqual(tok_str, "there")
        self.assertEqual(tok_id, 6)

    def test_common_prefix_search(self):
        prefixes = self.trie.common_prefix_search("andiron")
        # Should match 'a', 'an', 'and'
        prefix_strings = [p[0] for p in prefixes]
        self.assertEqual(prefix_strings, ["a", "an", "and"])

    def test_greedy_tokenize(self):
        tokens = self.trie.tokenize_greedy("thentheand")
        # should greedy split: 'then' (5), 'the' (4), 'and' (3)
        tok_strings = [t[0] for t in tokens]
        self.assertEqual(tok_strings, ["then", "the", "and"])


if __name__ == "__main__":
    unittest.main()
