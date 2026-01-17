"""
Ed25519 key pair management for AgentFacts SDK.

Provides secure key generation, serialization, and storage.

Pythonic Features:
- Context manager for secure temporary keys
- cached_property for expensive derivations
- __repr__ for debugging
- __eq__ and __hash__ for collections
"""

from __future__ import annotations

import base64
import secrets
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)


@dataclass
class KeyPair:
    """
    Ed25519 key pair for agent identity.

    The private key is used for signing metadata and handshake responses.
    The public key is embedded in the DID and used for verification.
    """

    _private_key: Ed25519PrivateKey
    _public_key: Ed25519PublicKey

    @classmethod
    def generate(cls) -> KeyPair:
        """Generate a new Ed25519 key pair."""
        private_key = Ed25519PrivateKey.generate()
        public_key = private_key.public_key()
        return cls(_private_key=private_key, _public_key=public_key)

    @classmethod
    def from_private_key_bytes(cls, private_bytes: bytes) -> KeyPair:
        """Load key pair from raw private key bytes (32 bytes)."""
        private_key = Ed25519PrivateKey.from_private_bytes(private_bytes)
        public_key = private_key.public_key()
        return cls(_private_key=private_key, _public_key=public_key)

    @classmethod
    def from_private_key_base64(cls, private_b64: str) -> KeyPair:
        """Load key pair from base64-encoded private key."""
        private_bytes = base64.urlsafe_b64decode(private_b64)
        return cls.from_private_key_bytes(private_bytes)

    @classmethod
    def from_pem(cls, pem_data: bytes) -> KeyPair:
        """Load key pair from PEM-encoded private key."""
        private_key = serialization.load_pem_private_key(pem_data, password=None)
        if not isinstance(private_key, Ed25519PrivateKey):
            raise ValueError("PEM does not contain an Ed25519 private key")
        public_key = private_key.public_key()
        return cls(_private_key=private_key, _public_key=public_key)

    @classmethod
    def from_file(cls, path: Path | str) -> KeyPair:
        """Load key pair from a PEM file."""
        path = Path(path)
        pem_data = path.read_bytes()
        return cls.from_pem(pem_data)

    @classmethod
    def from_public_key_bytes(cls, public_bytes: bytes) -> KeyPair:
        """
        Create a verification-only key pair from public key bytes.

        Note: This key pair cannot sign, only verify.
        """
        public_key = Ed25519PublicKey.from_public_bytes(public_bytes)
        # Create a dummy private key structure - signing will fail
        return cls(_private_key=None, _public_key=public_key)  # type: ignore

    @classmethod
    def from_public_key_base64(cls, public_b64: str) -> KeyPair:
        """Create a verification-only key pair from base64-encoded public key."""
        public_bytes = base64.urlsafe_b64decode(public_b64)
        return cls.from_public_key_bytes(public_bytes)

    @property
    def private_key(self) -> Ed25519PrivateKey:
        """Get the private key (raises if verification-only)."""
        if self._private_key is None:
            raise ValueError("This key pair is verification-only (no private key)")
        return self._private_key

    @property
    def public_key(self) -> Ed25519PublicKey:
        """Get the public key."""
        return self._public_key

    @property
    def private_key_bytes(self) -> bytes:
        """Get raw private key bytes (32 bytes)."""
        return self.private_key.private_bytes_raw()

    @property
    def public_key_bytes(self) -> bytes:
        """Get raw public key bytes (32 bytes)."""
        return self._public_key.public_bytes_raw()

    @property
    def private_key_base64(self) -> str:
        """Get base64url-encoded private key."""
        return base64.urlsafe_b64encode(self.private_key_bytes).decode("ascii")

    @property
    def public_key_base64(self) -> str:
        """Get base64url-encoded public key."""
        return base64.urlsafe_b64encode(self.public_key_bytes).decode("ascii")

    def to_pem(self, *, encrypt: bool = False, password: bytes | None = None) -> bytes:
        """
        Serialize private key to PEM format.

        Args:
            encrypt: Whether to encrypt the PEM file
            password: Password for encryption (required if encrypt=True)
        """
        encryption: serialization.KeySerializationEncryption
        if encrypt:
            if password is None:
                raise ValueError("Password required for encryption")
            encryption = serialization.BestAvailableEncryption(password)
        else:
            encryption = serialization.NoEncryption()

        return self.private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=encryption,
        )

    def save(
        self, path: Path | str, *, encrypt: bool = False, password: bytes | None = None
    ) -> None:
        """Save private key to a PEM file."""
        path = Path(path)
        pem_data = self.to_pem(encrypt=encrypt, password=password)
        path.write_bytes(pem_data)
        # Set restrictive permissions (owner read/write only)
        path.chmod(0o600)

    def sign(self, message: bytes) -> bytes:
        """Sign a message with the private key."""
        return self.private_key.sign(message)

    def sign_base64(self, message: bytes) -> str:
        """Sign a message and return base64-encoded signature."""
        signature = self.sign(message)
        return base64.urlsafe_b64encode(signature).decode("ascii")

    def verify(self, message: bytes, signature: bytes) -> bool:
        """
        Verify a signature against a message.

        Returns True if valid, False otherwise.
        """
        try:
            self._public_key.verify(signature, message)
            return True
        except InvalidSignature:
            return False
        except (TypeError, ValueError):
            # Invalid input types or malformed data
            return False

    def verify_base64(self, message: bytes, signature_b64: str) -> bool:
        """Verify a base64-encoded signature."""
        try:
            signature = base64.urlsafe_b64decode(signature_b64)
            return self.verify(message, signature)
        except (ValueError, TypeError):
            # Invalid base64 encoding or wrong types
            return False

    def can_sign(self) -> bool:
        """Check if this key pair can sign (has private key)."""
        return self._private_key is not None

    @staticmethod
    def generate_nonce(length: int = 32) -> str:
        """Generate a cryptographic nonce for handshake challenges."""
        return base64.urlsafe_b64encode(secrets.token_bytes(length)).decode("ascii")

    # -------------------------------------------------------------------------
    # Pythonic Enhancements
    # -------------------------------------------------------------------------

    def __repr__(self) -> str:
        key_type = "full" if self.can_sign() else "verify-only"
        fingerprint = self.fingerprint[:8]
        return f"KeyPair({key_type}, fingerprint={fingerprint}...)"

    def __str__(self) -> str:
        return f"KeyPair({self.fingerprint[:16]})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, KeyPair):
            return NotImplemented
        return self.public_key_bytes == other.public_key_bytes

    def __hash__(self) -> int:
        return hash(self.public_key_bytes)

    @cached_property
    def fingerprint(self) -> str:
        """
        SHA-256 fingerprint of the public key (cached).

        Useful for display and comparison without exposing the full key.
        """
        import hashlib

        return hashlib.sha256(self.public_key_bytes).hexdigest()

    # -------------------------------------------------------------------------
    # Context Manager for Temporary Keys
    # -------------------------------------------------------------------------

    def __enter__(self) -> KeyPair:
        """Enter context manager - returns self."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """
        Exit context manager - securely clear private key material.

        Note: This is a best-effort cleanup. Python's garbage collector
        may still hold references to key bytes in memory.
        """
        # Clear internal references to encourage garbage collection
        # The actual cryptography library handles secure memory clearing
        if self._private_key is not None:
            self._private_key = None  # type: ignore

    @classmethod
    def temporary(cls) -> KeyPair:
        """
        Generate a temporary key pair for use in a context manager.

        The key is automatically cleared when exiting the context.

        Example:
            ```python
            with KeyPair.temporary() as kp:
                facts = AgentFacts(..., key_pair=kp)
                facts.sign()
                # kp is valid here
            # kp.can_sign() is now False
            ```
        """
        return cls.generate()
