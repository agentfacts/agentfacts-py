"""
Decentralised Identifier (DID) implementation for AgentFacts SDK.

Implements the did:key method using Ed25519 public keys, providing
a self-certifying identifier for agents.

DID Format: did:key:z6Mk<base58-multibase-encoded-public-key>
"""

import hashlib
import re
from dataclasses import dataclass

from agentfacts.crypto.keys import KeyPair

# Multicodec prefix for Ed25519 public key (0xed01)
ED25519_MULTICODEC_PREFIX = bytes([0xED, 0x01])

# Base58 Bitcoin alphabet
BASE58_ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"


def _base58_encode(data: bytes) -> str:
    """Encode bytes to base58 (Bitcoin alphabet)."""
    num = int.from_bytes(data, "big")
    if num == 0:
        return BASE58_ALPHABET[0]

    result = []
    while num > 0:
        num, remainder = divmod(num, 58)
        result.append(BASE58_ALPHABET[remainder])

    # Handle leading zeros
    for byte in data:
        if byte == 0:
            result.append(BASE58_ALPHABET[0])
        else:
            break

    return "".join(reversed(result))


def _base58_decode(encoded: str) -> bytes:
    """Decode base58 string to bytes."""
    num = 0
    for char in encoded:
        num = num * 58 + BASE58_ALPHABET.index(char)

    # Calculate required byte length
    byte_length = (num.bit_length() + 7) // 8

    # Handle leading '1's (zeros)
    leading_zeros = 0
    for char in encoded:
        if char == BASE58_ALPHABET[0]:
            leading_zeros += 1
        else:
            break

    result = num.to_bytes(byte_length, "big") if num > 0 else b""
    return b"\x00" * leading_zeros + result


@dataclass
class DID:
    """
    Decentralised Identifier using the did:key method.

    A DID is a globally unique, self-certifying identifier derived
    from the agent's Ed25519 public key.
    """

    method: str
    identifier: str

    # Regex pattern for did:key format
    DID_KEY_PATTERN = re.compile(r"^did:key:z([a-km-zA-HJ-NP-Z1-9]+)$")

    @classmethod
    def from_public_key(cls, public_key_bytes: bytes) -> "DID":
        """
        Create a DID from Ed25519 public key bytes.

        The identifier is created by:
        1. Prepending the Ed25519 multicodec prefix (0xed01)
        2. Base58-encoding the result
        3. Prepending 'z' (multibase prefix for base58btc)
        """
        # Prepend multicodec prefix
        multicodec_key = ED25519_MULTICODEC_PREFIX + public_key_bytes

        # Base58 encode
        encoded = _base58_encode(multicodec_key)

        # Add multibase prefix 'z' for base58btc
        identifier = f"z{encoded}"

        return cls(method="key", identifier=identifier)

    @classmethod
    def from_key_pair(cls, key_pair: KeyPair) -> "DID":
        """Create a DID from a KeyPair."""
        return cls.from_public_key(key_pair.public_key_bytes)

    @classmethod
    def generate(cls) -> tuple["DID", KeyPair]:
        """Generate a new DID with a fresh key pair."""
        key_pair = KeyPair.generate()
        did = cls.from_key_pair(key_pair)
        return did, key_pair

    @classmethod
    def parse(cls, did_string: str) -> "DID":
        """
        Parse a DID string.

        Args:
            did_string: A DID in format "did:key:z<base58>"

        Raises:
            ValueError: If the DID format is invalid
        """
        if not did_string.startswith("did:"):
            raise ValueError(
                f"Invalid DID format: must start with 'did:' - got {did_string}"
            )

        parts = did_string.split(":", 2)
        if len(parts) != 3:
            raise ValueError(
                f"Invalid DID format: expected 'did:method:identifier' - got {did_string}"
            )

        method = parts[1]
        identifier = parts[2]

        if method == "key" and not cls.DID_KEY_PATTERN.match(did_string):
            raise ValueError(f"Invalid did:key format: {did_string}")

        return cls(method=method, identifier=identifier)

    def extract_public_key(self) -> bytes:
        """
        Extract the Ed25519 public key from a did:key DID.

        Raises:
            ValueError: If this is not a did:key or the format is invalid
        """
        if self.method != "key":
            raise ValueError(f"Cannot extract public key from did:{self.method}")

        if not self.identifier.startswith("z"):
            raise ValueError("Invalid multibase prefix (expected 'z' for base58btc)")

        # Decode base58 (skip the 'z' prefix)
        decoded = _base58_decode(self.identifier[1:])

        # Verify and strip multicodec prefix
        if not decoded.startswith(ED25519_MULTICODEC_PREFIX):
            raise ValueError("Invalid multicodec prefix (expected Ed25519)")

        public_key = decoded[len(ED25519_MULTICODEC_PREFIX) :]

        if len(public_key) != 32:
            raise ValueError(
                f"Invalid public key length: expected 32 bytes, got {len(public_key)}"
            )

        return public_key

    def to_key_pair(self) -> KeyPair:
        """
        Create a verification-only KeyPair from this DID.

        Note: The returned KeyPair can only verify, not sign.
        """
        public_key = self.extract_public_key()
        return KeyPair.from_public_key_bytes(public_key)

    @property
    def uri(self) -> str:
        """Get the full DID URI."""
        return f"did:{self.method}:{self.identifier}"

    def __str__(self) -> str:
        return self.uri

    def __repr__(self) -> str:
        return f"DID({self.uri!r})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, DID):
            return self.uri == other.uri
        if isinstance(other, str):
            return self.uri == other
        return False

    def __hash__(self) -> int:
        return hash(self.uri)

    def short_id(self, length: int = 8) -> str:
        """Get a shortened identifier for display purposes."""
        return (
            self.identifier[:length]
            if len(self.identifier) > length
            else self.identifier
        )

    def fingerprint(self) -> str:
        """Get a SHA-256 fingerprint of the DID."""
        digest = hashlib.sha256(self.uri.encode()).hexdigest()
        return digest[:16]
