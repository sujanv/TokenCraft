"""
Fast Trie Data Structure for Prefix Token Matching, Vocabulary Indexing, and Search.
Used for sub-linear greedy subword matching and token exploration.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Any


class TrieNode:
    """Node in the prefix Trie."""
    __slots__ = ("children", "is_terminal", "token", "token_id")

    def __init__(self):
        self.children: Dict[str, TrieNode] = {}
        self.is_terminal: bool = False
        self.token: Optional[str] = None
        self.token_id: Optional[int] = None


class Trie:
    """
    Prefix Trie supporting fast longest-prefix matching, common-prefix search,
    and greedy dictionary tokenization.
    """

    def __init__(self):
        self.root = TrieNode()
        self.size = 0

    def insert(self, token: str, token_id: int) -> None:
        """Insert a token into the Trie with its assigned integer token ID."""
        node = self.root
        for char in token:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        if not node.is_terminal:
            self.size += 1
        node.is_terminal = True
        node.token = token
        node.token_id = token_id

    def longest_prefix_match(self, text: str, start: int = 0) -> Optional[Tuple[str, int]]:
        """
        Finds the longest token in the Trie that forms a prefix of text[start:].
        Returns (token_string, token_id) or None if no prefix matches.
        """
        node = self.root
        best_token = None
        best_id = None
        n = len(text)

        for i in range(start, n):
            char = text[i]
            if char not in node.children:
                break
            node = node.children[char]
            if node.is_terminal:
                best_token = node.token
                best_id = node.token_id

        if best_token is not None:
            return best_token, best_id
        return None

    def common_prefix_search(self, text: str, start: int = 0) -> List[Tuple[str, int]]:
        """
        Returns all tokens in the Trie that form prefixes of text[start:].
        Ordered from shortest prefix to longest prefix.
        """
        node = self.root
        results = []
        n = len(text)

        for i in range(start, n):
            char = text[i]
            if char not in node.children:
                break
            node = node.children[char]
            if node.is_terminal:
                results.append((node.token, node.token_id))

        return results

    def tokenize_greedy(self, text: str, unk_id: int = 1) -> List[Tuple[str, int]]:
        """
        Greedily segments text using longest prefix match.
        If a character has no match, emits the single character with unk_id.
        """
        tokens = []
        i = 0
        n = len(text)

        while i < n:
            match = self.longest_prefix_match(text, start=i)
            if match:
                tok_str, tok_id = match
                tokens.append((tok_str, tok_id))
                i += len(tok_str)
            else:
                tokens.append((text[i], unk_id))
                i += 1

        return tokens

    @classmethod
    def from_vocab(cls, vocab: Dict[str, int]) -> Trie:
        """Constructs a Trie from a dictionary mapping token string to ID."""
        trie = cls()
        for tok, idx in vocab.items():
            trie.insert(tok, idx)
        return trie

    def __len__(self) -> int:
        return self.size

    def __contains__(self, token: str) -> bool:
        node = self.root
        for char in token:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_terminal
