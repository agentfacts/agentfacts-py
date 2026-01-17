"""Tests for Merkle tree and transparency log."""

import pytest

from agentfacts.crypto.keys import KeyPair
from agentfacts.merkle.log import TransparencyLog
from agentfacts.merkle.tree import MerkleTree


class TestMerkleTree:
    """Tests for the Merkle tree implementation."""

    def test_empty_tree(self):
        """Test empty tree state."""
        tree = MerkleTree()

        assert len(tree) == 0
        assert tree.root is None
        assert tree.root_hex is None

    def test_single_leaf(self):
        """Test tree with single leaf."""
        tree = MerkleTree()
        tree.append(b"leaf1")

        assert len(tree) == 1
        assert tree.root is not None
        assert tree.root_hex is not None

    def test_multiple_leaves(self):
        """Test tree with multiple leaves."""
        tree = MerkleTree()
        tree.append(b"leaf1")
        tree.append(b"leaf2")
        tree.append(b"leaf3")

        assert len(tree) == 3
        assert tree.root is not None

    def test_root_changes_with_append(self):
        """Test that root changes when leaves are added."""
        tree = MerkleTree()

        tree.append(b"leaf1")
        root1 = tree.root_hex

        tree.append(b"leaf2")
        root2 = tree.root_hex

        assert root1 != root2

    def test_deterministic_root(self):
        """Test that same leaves produce same root."""
        tree1 = MerkleTree()
        tree1.append(b"a")
        tree1.append(b"b")
        tree1.append(b"c")

        tree2 = MerkleTree()
        tree2.append(b"a")
        tree2.append(b"b")
        tree2.append(b"c")

        assert tree1.root_hex == tree2.root_hex

    def test_get_proof(self):
        """Test generating inclusion proofs."""
        tree = MerkleTree()
        tree.append(b"leaf1")
        tree.append(b"leaf2")
        tree.append(b"leaf3")

        proof = tree.get_proof(1)

        assert proof.leaf_index == 1
        assert proof.root == tree.root_hex

    def test_verify_proof(self):
        """Test verifying inclusion proofs."""
        tree = MerkleTree()
        tree.append(b"leaf1")
        tree.append(b"leaf2")
        tree.append(b"leaf3")

        proof = tree.get_proof(1)

        # Should verify with original data
        assert MerkleTree.verify_proof(proof, b"leaf2")

        # Should fail with wrong data
        assert not MerkleTree.verify_proof(proof, b"wrong")

    def test_proof_for_all_leaves(self):
        """Test proofs work for all leaves."""
        tree = MerkleTree()
        data = [b"a", b"b", b"c", b"d", b"e"]

        for d in data:
            tree.append(d)

        for i, d in enumerate(data):
            proof = tree.get_proof(i)
            assert MerkleTree.verify_proof(proof, d)

    def test_serialization(self):
        """Test tree serialization and deserialization."""
        tree1 = MerkleTree()
        tree1.append(b"leaf1")
        tree1.append(b"leaf2")

        data = tree1.to_dict()
        tree2 = MerkleTree.from_dict(data)

        assert tree2.root_hex == tree1.root_hex
        assert len(tree2) == len(tree1)

    def test_from_leaves(self):
        """Test creating tree from list of leaves."""
        leaves = [b"a", b"b", b"c"]
        tree = MerkleTree.from_leaves(leaves)

        assert len(tree) == 3
        assert tree.root is not None


class TestTransparencyLog:
    """Tests for the transparency log."""

    def test_create_log(self):
        """Test creating a transparency log."""
        log = TransparencyLog(agent_did="did:key:z123")

        assert len(log) == 0
        assert log.root is None

    def test_append_entry(self):
        """Test appending entries to the log."""
        log = TransparencyLog(agent_did="did:key:z123")
        kp = KeyPair.generate()

        entry = log.append(
            entry_type="test",
            data={"key": "value"},
            issuer_did="did:key:z456",
            key_pair=kp,
        )

        assert len(log) == 1
        assert entry.type == "test"
        assert entry.index == 0
        assert entry.signature is not None

    def test_log_evidence(self):
        """Test convenience method for logging evidence."""
        log = TransparencyLog(agent_did="did:key:z123")
        kp = KeyPair.generate()

        entry = log.log_evidence("audit", {"result": "pass"}, kp)

        assert entry.type == "audit"
        assert entry.issuer_did == "did:key:z123"

    def test_verify_entry(self):
        """Test verifying log entries."""
        log = TransparencyLog(agent_did="did:key:z123")
        kp = KeyPair.generate()

        log.append("test", {"data": 1}, "did:key:z123", kp)

        # Should verify Merkle inclusion
        assert log.verify_entry(0)

        # Should also verify signature with key
        assert log.verify_entry(0, kp)

    def test_verify_entry_requires_signature_with_key(self):
        """Test unsigned entries fail when a key is provided."""
        log = TransparencyLog(agent_did="did:key:z123")
        kp = KeyPair.generate()

        log.append("test", {"data": 1}, "did:key:z123")

        assert log.verify_entry(0)
        assert not log.verify_entry(0, kp)

    def test_get_proof(self):
        """Test getting inclusion proof for entry."""
        log = TransparencyLog(agent_did="did:key:z123")
        kp = KeyPair.generate()

        log.append("test", {"data": 1}, "did:key:z123", kp)
        log.append("test", {"data": 2}, "did:key:z123", kp)

        proof = log.get_proof(1)

        assert proof.leaf_index == 1
        assert proof.root == log.root

    def test_serialization(self):
        """Test log serialization and deserialization."""
        log1 = TransparencyLog(agent_did="did:key:z123")
        kp = KeyPair.generate()

        log1.append("event1", {"x": 1}, "did:key:z123", kp)
        log1.append("event2", {"x": 2}, "did:key:z123", kp)

        # Serialize
        data = log1.to_dict()
        json_str = log1.export_json()

        # Deserialize
        log2 = TransparencyLog.from_dict(data)
        log3 = TransparencyLog.import_json(json_str)

        assert log2.root == log1.root
        assert log3.root == log1.root
        assert len(log2) == 2

    def test_serialization_detects_root_mismatch(self):
        """Test tampered merkle_root is rejected on deserialization."""
        log1 = TransparencyLog(agent_did="did:key:z123")
        kp = KeyPair.generate()

        log1.append("event1", {"x": 1}, "did:key:z123", kp)
        log1.append("event2", {"x": 2}, "did:key:z123", kp)

        data = log1.to_dict()
        data["merkle_root"] = "deadbeef"

        with pytest.raises(ValueError):
            TransparencyLog.from_dict(data)
