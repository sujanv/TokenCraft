"""
Tests for BPE Provenance Tracker and Derivation Trees.
"""

import unittest
from tokencraft.bpe import BPETokenizer
from tokencraft.provenance import BPEProvenanceTracker


class TestProvenanceTracker(unittest.TestCase):
    def setUp(self):
        self.corpus = [
            "playing players playable played",
            "working worker workable worked",
        ]
        self.bpe = BPETokenizer()
        self.bpe.train(self.corpus, vocab_size=280)

    def test_derivation_tree_generation(self):
        tracker = BPEProvenanceTracker(self.bpe.merges)
        if self.bpe.merges:
            first_merge = self.bpe.merges[0]
            merged_tok = first_merge[0] + first_merge[1]

            tree = tracker.build_tree(merged_tok)
            self.assertFalse(tree.is_leaf)
            self.assertEqual(tree.token, merged_tok)
            self.assertIsNotNone(tree.left)
            self.assertIsNotNone(tree.right)

            ascii_tree = tracker.trace_token(merged_tok)
            self.assertIn(merged_tok, ascii_tree)
            self.assertIn("Derivation Tree for:", ascii_tree)

    def test_leaf_token_tree(self):
        tracker = BPEProvenanceTracker(self.bpe.merges)
        leaf = tracker.build_tree("z")
        self.assertTrue(leaf.is_leaf)
        self.assertIn("Leaf Token", tracker.trace_token("z"))


if __name__ == "__main__":
    unittest.main()
