"""Tests for cryptographic primitives."""

import pytest

from agentfacts.crypto.did import DID
from agentfacts.crypto.keys import KeyPair


class TestKeyPair:
    """Tests for Ed25519 key pair management."""

    def test_generate_key_pair(self):
        """Test key pair generation."""
        kp = KeyPair.generate()

        assert kp.can_sign()
        assert len(kp.private_key_bytes) == 32
        assert len(kp.public_key_bytes) == 32

    def test_sign_and_verify(self):
        """Test signing and verification."""
        kp = KeyPair.generate()
        message = b"Hello, AgentFacts!"

        signature = kp.sign(message)
        assert kp.verify(message, signature)

        # Wrong message should fail
        assert not kp.verify(b"Wrong message", signature)

    def test_sign_and_verify_base64(self):
        """Test base64 signing and verification."""
        kp = KeyPair.generate()
        message = b"Hello, AgentFacts!"

        signature_b64 = kp.sign_base64(message)
        assert kp.verify_base64(message, signature_b64)

    def test_from_private_key_bytes(self):
        """Test loading key pair from private key bytes."""
        kp1 = KeyPair.generate()
        kp2 = KeyPair.from_private_key_bytes(kp1.private_key_bytes)

        assert kp1.public_key_bytes == kp2.public_key_bytes

    def test_from_public_key_bytes(self):
        """Test creating verification-only key pair."""
        kp1 = KeyPair.generate()
        kp2 = KeyPair.from_public_key_bytes(kp1.public_key_bytes)

        # Should be able to verify
        message = b"test"
        signature = kp1.sign(message)
        assert kp2.verify(message, signature)

        # Should not be able to sign
        assert not kp2.can_sign()
        with pytest.raises(ValueError):
            kp2.sign(message)

    def test_generate_nonce(self):
        """Test nonce generation."""
        nonce1 = KeyPair.generate_nonce()
        nonce2 = KeyPair.generate_nonce()

        assert nonce1 != nonce2
        assert len(nonce1) > 0


class TestDID:
    """Tests for Decentralised Identifiers."""

    def test_from_public_key(self):
        """Test DID generation from public key."""
        kp = KeyPair.generate()
        did = DID.from_public_key(kp.public_key_bytes)

        assert did.method == "key"
        assert did.identifier.startswith("z")
        assert did.uri.startswith("did:key:z")

    def test_from_key_pair(self):
        """Test DID generation from key pair."""
        kp = KeyPair.generate()
        did = DID.from_key_pair(kp)

        assert did.method == "key"
        assert "did:key:z" in did.uri

    def test_generate(self):
        """Test DID and key pair generation."""
        did, kp = DID.generate()

        assert did.method == "key"
        assert kp.can_sign()

    def test_parse_did_key(self):
        """Test parsing a did:key string."""
        kp = KeyPair.generate()
        original_did = DID.from_key_pair(kp)

        parsed = DID.parse(original_did.uri)

        assert parsed.method == original_did.method
        assert parsed.identifier == original_did.identifier
        assert parsed.uri == original_did.uri

    def test_parse_invalid_did(self):
        """Test parsing invalid DID strings."""
        with pytest.raises(ValueError):
            DID.parse("not-a-did")

        with pytest.raises(ValueError):
            DID.parse("did:invalid")

    def test_extract_public_key(self):
        """Test extracting public key from DID."""
        kp = KeyPair.generate()
        did = DID.from_key_pair(kp)

        extracted_pk = did.extract_public_key()

        assert extracted_pk == kp.public_key_bytes

    def test_to_key_pair(self):
        """Test creating verification key pair from DID."""
        kp1 = KeyPair.generate()
        did = DID.from_key_pair(kp1)

        kp2 = did.to_key_pair()

        # Should be able to verify signatures
        message = b"test"
        signature = kp1.sign(message)
        assert kp2.verify(message, signature)

    def test_did_equality(self):
        """Test DID equality comparison."""
        kp = KeyPair.generate()
        did1 = DID.from_key_pair(kp)
        did2 = DID.from_key_pair(kp)
        did3 = DID.from_key_pair(KeyPair.generate())

        assert did1 == did2
        assert did1 == did1.uri
        assert did1 != did3

    def test_short_id(self):
        """Test short ID generation."""
        kp = KeyPair.generate()
        did = DID.from_key_pair(kp)

        short = did.short_id(8)

        assert len(short) == 8
        assert did.identifier.startswith(short)

    def test_fingerprint(self):
        """Test fingerprint generation."""
        kp = KeyPair.generate()
        did = DID.from_key_pair(kp)

        fp = did.fingerprint()

        assert len(fp) == 16  # SHA-256 truncated to 16 hex chars
