"""
Transparency log for recording verifiable evidence.

Provides an append-only log structure backed by a Merkle tree,
enabling efficient verification of agent attestations and events.
"""

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from agentfacts.crypto.canonicalization import canonicalize_json
from agentfacts.crypto.keys import KeyPair
from agentfacts.merkle.tree import MerkleProof, MerkleTree


def _format_timestamp(timestamp: datetime) -> str:
    """Format timestamp as RFC 3339 with Z suffix."""
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    timestamp = timestamp.astimezone(timezone.utc)
    return timestamp.isoformat().replace("+00:00", "Z")


def _parse_timestamp(value: str) -> datetime:
    """Parse RFC 3339 timestamp with optional Z suffix."""
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    return datetime.fromisoformat(value)


@dataclass
class LogEntry:
    """
    A single entry in the transparency log.

    Each entry is signed and can be independently verified.
    """

    id: str
    type: str
    timestamp: datetime
    data: dict[str, Any]
    issuer_did: str
    signature: str | None = None
    index: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize entry to a dictionary."""
        return {
            "id": self.id,
            "type": self.type,
            "timestamp": _format_timestamp(self.timestamp),
            "data": self.data,
            "issuer_did": self.issuer_did,
            "signature": self.signature,
            "index": self.index,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LogEntry":
        """Deserialize entry from a dictionary."""
        timestamp = data["timestamp"]
        if isinstance(timestamp, str):
            timestamp = _parse_timestamp(timestamp)

        return cls(
            id=data["id"],
            type=data["type"],
            timestamp=timestamp,
            data=data["data"],
            issuer_did=data["issuer_did"],
            signature=data.get("signature"),
            index=data.get("index"),
        )

    def canonical_bytes(self) -> bytes:
        """Get the canonical bytes for signing/hashing."""
        signable = {
            "id": self.id,
            "type": self.type,
            "timestamp": _format_timestamp(self.timestamp),
            "data": self.data,
            "issuer_did": self.issuer_did,
        }
        return canonicalize_json(signable)

    def sign(self, key_pair: KeyPair) -> None:
        """Sign this entry with the given key pair."""
        self.signature = key_pair.sign_base64(self.canonical_bytes())

    def verify(self, key_pair: KeyPair) -> bool:
        """Verify this entry's signature."""
        if not self.signature:
            return False
        return key_pair.verify_base64(self.canonical_bytes(), self.signature)


@dataclass
class TransparencyLog:
    """
    An append-only transparency log backed by a Merkle tree.

    Used to record verifiable attestations, audit events, and
    evidence about an agent's behavior.
    """

    agent_did: str
    entries: list[LogEntry] = field(default_factory=list)
    _tree: MerkleTree = field(default_factory=MerkleTree)

    def append(
        self,
        entry_type: str,
        data: dict[str, Any],
        issuer_did: str,
        key_pair: KeyPair | None = None,
    ) -> LogEntry:
        """
        Append a new entry to the log.

        Args:
            entry_type: Type of the entry (e.g., 'security_audit', 'capability_invocation')
            data: The entry data/claims
            issuer_did: DID of the entity creating this entry
            key_pair: Optional key pair for signing the entry

        Returns:
            The created LogEntry with its index in the tree
        """
        entry = LogEntry(
            id=str(uuid.uuid4()),
            type=entry_type,
            timestamp=datetime.now(timezone.utc),
            data=data,
            issuer_did=issuer_did,
        )

        # Sign if key pair provided
        if key_pair:
            entry.sign(key_pair)

        # Add to Merkle tree
        entry_bytes = entry.canonical_bytes()
        index = self._tree.append(entry_bytes)
        entry.index = index

        self.entries.append(entry)
        return entry

    def log_evidence(
        self,
        evidence_type: str,
        data: dict[str, Any],
        key_pair: KeyPair,
    ) -> LogEntry:
        """
        Log evidence about the agent (convenience wrapper for append).

        Args:
            evidence_type: Type of evidence (e.g., 'security_test', 'compliance_check')
            data: Evidence data
            key_pair: Key pair for signing

        Returns:
            The created LogEntry
        """
        return self.append(
            entry_type=evidence_type,
            data=data,
            issuer_did=self.agent_did,
            key_pair=key_pair,
        )

    @property
    def root(self) -> str | None:
        """Get the current Merkle root."""
        return self._tree.root_hex

    def __len__(self) -> int:
        """Return the number of entries in the log."""
        return len(self.entries)

    def get_proof(self, index: int) -> MerkleProof:
        """
        Get a proof of inclusion for an entry.

        Args:
            index: The entry index

        Returns:
            MerkleProof for the entry
        """
        return self._tree.get_proof(index)

    def verify_entry(self, index: int, key_pair: KeyPair | None = None) -> bool:
        """
        Verify an entry exists in the log and optionally verify its signature.

        Args:
            index: The entry index
            key_pair: Optional key pair for signature verification

        Returns:
            True if verification passes
        """
        if index < 0 or index >= len(self.entries):
            return False

        entry = self.entries[index]

        # Verify Merkle inclusion
        proof = self.get_proof(index)
        if not MerkleTree.verify_proof(proof, entry.canonical_bytes()):
            return False

        # Optionally verify signature
        if key_pair:
            if not entry.signature:
                return False
            if not entry.verify(key_pair):
                return False

        return True

    def to_dict(self) -> dict[str, Any]:
        """Serialize the log to a dictionary."""
        return {
            "agent_did": self.agent_did,
            "entries": [e.to_dict() for e in self.entries],
            "merkle_root": self.root,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TransparencyLog":
        """Deserialize a log from a dictionary."""
        log = cls(agent_did=data["agent_did"])
        for entry_data in data.get("entries", []):
            entry = LogEntry.from_dict(entry_data)
            # Re-add to tree to rebuild Merkle structure
            entry_bytes = entry.canonical_bytes()
            index = log._tree.append(entry_bytes)
            entry.index = index
            log.entries.append(entry)
        expected_root = data.get("merkle_root")
        if expected_root and log.root != expected_root:
            raise ValueError("Transparency log merkle_root mismatch")
        return log

    def export_json(self) -> str:
        """Export log as JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def import_json(cls, json_str: str) -> "TransparencyLog":
        """Import log from JSON string."""
        return cls.from_dict(json.loads(json_str))
