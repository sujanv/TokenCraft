"""
Token provenance and merge tree visualization for Byte-Pair Encoding.
Enables tracing how complex subword tokens were assembled from elemental bytes.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any


@dataclass
class MergeNode:
    """A node in the BPE derivation / merge tree."""
    token: str
    rank: int = -1
    left: Optional["MergeNode"] = None
    right: Optional["MergeNode"] = None
    is_leaf: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "token": self.token,
            "rank": self.rank,
            "is_leaf": self.is_leaf,
            "left": self.left.to_dict() if self.left else None,
            "right": self.right.to_dict() if self.right else None,
        }

    def render_ascii(self, prefix: str = "", is_last: bool = True) -> str:
        """Renders the derivation tree as an ASCII hierarchical graph."""
        connector = "└── " if is_last else "├── "
        node_label = f"'{self.token}'" if self.is_leaf else f"[{self.token}] (Merge #{self.rank})"
        result = prefix + connector + node_label + "\n"

        child_prefix = prefix + ("    " if is_last else "│   ")
        children = []
        if self.left:
            children.append(self.left)
        if self.right:
            children.append(self.right)

        for i, child in enumerate(children):
            result += child.render_ascii(child_prefix, is_last=(i == len(children) - 1))

        return result


class BPEProvenanceTracker:
    """
    Builds and queries merge derivation trees from learned BPE merge rules.
    """

    def __init__(self, merges: List[Tuple[str, str]]):
        self.merges = merges
        self.merge_map: Dict[str, Tuple[Tuple[str, str], int]] = {}
        for rank, (u, v) in enumerate(merges):
            merged = u + v
            self.merge_map[merged] = ((u, v), rank)

    def build_tree(self, token: str) -> MergeNode:
        """Recursively decomposes a subword token into its constituent merges."""
        if token in self.merge_map:
            (u, v), rank = self.merge_map[token]
            left_node = self.build_tree(u)
            right_node = self.build_tree(v)
            return MergeNode(token=token, rank=rank, left=left_node, right=right_node, is_leaf=False)
        else:
            return MergeNode(token=token, is_leaf=True)

    def trace_token(self, token: str) -> str:
        """Returns the ASCII tree string representation for a given token."""
        root = self.build_tree(token)
        # Render starting with root
        tree_str = f"Derivation Tree for: '{token}'\n"
        if root.is_leaf:
            tree_str += f"└── '{token}' (Base Character / Leaf Token)\n"
        else:
            tree_str += f"[{token}] (Merge #{root.rank})\n"
            if root.left:
                tree_str += root.left.render_ascii("", is_last=False)
            if root.right:
                tree_str += root.right.render_ascii("", is_last=True)
        return tree_str
