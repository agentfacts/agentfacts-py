"""
Merkle tree implementation for verifiable data structures.

Provides a binary hash tree that enables efficient and secure
verification of data integrity.
"""

import hashlib
from dataclasses import dataclass, field
from typing import Any


def _hash_leaf(data: bytes) -> bytes:
    """Hash a leaf node with a 0x00 prefix to prevent second-preimage attacks."""
    return hashlib.sha256(b"\x00" + data).digest()


def _hash_node(left: bytes, right: bytes) -> bytes:
    """Hash an internal node with a 0x01 prefix."""
    return hashlib.sha256(b"\x01" + left + right).digest()


@dataclass
class MerkleProof:
    """
    A proof that a leaf exists in the Merkle tree.

    Contains the sibling hashes needed to reconstruct the root.
    """

    leaf_index: int
    leaf_hash: str
    siblings: list[
        tuple[str, str]
    ]  # List of (hash, position) where position is 'left' or 'right'
    root: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize proof to a dictionary."""
        return {
            "leaf_index": self.leaf_index,
            "leaf_hash": self.leaf_hash,
            "siblings": self.siblings,
            "root": self.root,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MerkleProof":
        """Deserialize proof from a dictionary."""
        return cls(
            leaf_index=data["leaf_index"],
            leaf_hash=data["leaf_hash"],
            siblings=[(s[0], s[1]) for s in data["siblings"]],
            root=data["root"],
        )


@dataclass
class MerkleTree:
    """
    A Merkle tree for verifiable data integrity.

    Supports append-only operations for transparency logs.
    """

    leaves: list[bytes] = field(default_factory=list)
    _tree: list[list[bytes]] = field(default_factory=list)
    _root: bytes | None = field(default=None)

    def append(self, data: bytes) -> int:
        """
        Append data to the tree and return the leaf index.

        Args:
            data: The data to append

        Returns:
            Index of the new leaf
        """
        leaf_hash = _hash_leaf(data)
        self.leaves.append(leaf_hash)
        self._rebuild_tree()
        return len(self.leaves) - 1

    def append_hash(self, leaf_hash: bytes) -> int:
        """
        Append a pre-computed leaf hash to the tree.

        Args:
            leaf_hash: The hash to append (32 bytes)

        Returns:
            Index of the new leaf
        """
        if len(leaf_hash) != 32:
            raise ValueError(f"Invalid hash length: expected 32, got {len(leaf_hash)}")
        self.leaves.append(leaf_hash)
        self._rebuild_tree()
        return len(self.leaves) - 1

    def _rebuild_tree(self) -> None:
        """Rebuild the tree structure from leaves."""
        if not self.leaves:
            self._tree = []
            self._root = None
            return

        # Start with leaf layer
        current_layer = list(self.leaves)
        self._tree = [current_layer]

        # Build up the tree
        while len(current_layer) > 1:
            next_layer = []
            for i in range(0, len(current_layer), 2):
                left = current_layer[i]
                # If odd number of nodes, duplicate the last one
                right = current_layer[i + 1] if i + 1 < len(current_layer) else left
                next_layer.append(_hash_node(left, right))
            current_layer = next_layer
            self._tree.append(current_layer)

        self._root = current_layer[0] if current_layer else None

    @property
    def root(self) -> bytes | None:
        """Get the Merkle root hash."""
        return self._root

    @property
    def root_hex(self) -> str | None:
        """Get the Merkle root as a hex string."""
        return self._root.hex() if self._root else None

    def __len__(self) -> int:
        """Return the number of leaves in the tree."""
        return len(self.leaves)

    def get_proof(self, index: int) -> MerkleProof:
        """
        Get a proof of inclusion for a leaf at the given index.

        Args:
            index: The leaf index

        Returns:
            MerkleProof that can be used to verify inclusion

        Raises:
            IndexError: If index is out of range
        """
        if index < 0 or index >= len(self.leaves):
            raise IndexError(f"Leaf index {index} out of range")

        siblings: list[tuple[str, str]] = []
        current_index = index

        for layer in self._tree[:-1]:  # Exclude root layer
            # Find sibling
            if current_index % 2 == 0:
                # Current is left, sibling is right
                sibling_index = current_index + 1
                position = "right"
            else:
                # Current is right, sibling is left
                sibling_index = current_index - 1
                position = "left"

            # If sibling exists, add to proof
            if sibling_index < len(layer):
                siblings.append((layer[sibling_index].hex(), position))
            else:
                # Odd layer, sibling is self
                siblings.append((layer[current_index].hex(), position))

            # Move up the tree
            current_index //= 2

        return MerkleProof(
            leaf_index=index,
            leaf_hash=self.leaves[index].hex(),
            siblings=siblings,
            root=self.root_hex or "",
        )

    @staticmethod
    def verify_proof(proof: MerkleProof, leaf_data: bytes | None = None) -> bool:
        """
        Verify a Merkle proof.

        Args:
            proof: The proof to verify
            leaf_data: Optional raw leaf data (if provided, will hash it first)

        Returns:
            True if the proof is valid
        """
        if leaf_data is not None:
            computed_hash = _hash_leaf(leaf_data)
        else:
            try:
                computed_hash = bytes.fromhex(proof.leaf_hash)
            except ValueError:
                return False

        for sibling_hex, position in proof.siblings:
            try:
                sibling = bytes.fromhex(sibling_hex)
            except ValueError:
                return False
            if position == "right":
                computed_hash = _hash_node(computed_hash, sibling)
            else:
                computed_hash = _hash_node(sibling, computed_hash)

        return computed_hash.hex() == proof.root

    def to_dict(self) -> dict[str, Any]:
        """Serialize tree to a dictionary (for storage/transmission)."""
        return {
            "leaves": [leaf.hex() for leaf in self.leaves],
            "root": self.root_hex,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MerkleTree":
        """Deserialize tree from a dictionary."""
        tree = cls()
        for leaf_hex in data.get("leaves", []):
            tree.append_hash(bytes.fromhex(leaf_hex))
        return tree

    @classmethod
    def from_leaves(cls, leaves: list[bytes]) -> "MerkleTree":
        """Create a tree from a list of leaf data."""
        tree = cls()
        for leaf in leaves:
            tree.append(leaf)
        return tree
