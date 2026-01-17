"""Tests for core AgentFacts functionality."""

import json

import pytest
from pydantic import ValidationError

from agentfacts.core import AgentFacts
from agentfacts.crypto.canonicalization import compute_hash
from agentfacts.crypto.keys import KeyPair
from agentfacts.models import BaselineModel, Capability, LogProofEntry, ModelProvider
from agentfacts.plugins import VerificationContext


class TestAgentFacts:
    """Tests for the main AgentFacts class."""

    def test_create_basic(self):
        """Test basic AgentFacts creation."""
        facts = AgentFacts(
            name="Test Agent",
            description="A test agent",
        )

        assert facts.name == "Test Agent"
        assert facts.did.startswith("did:key:z")
        assert not facts.is_signed

    def test_create_with_model(self):
        """Test AgentFacts creation with baseline model."""
        facts = AgentFacts(
            name="GPT Agent",
            baseline_model=BaselineModel(
                name="gpt-4",
                provider=ModelProvider.OPENAI,
                temperature=0.7,
            ),
        )

        assert facts.metadata.agent.model.name == "gpt-4"
        assert facts.metadata.agent.model.provider == ModelProvider.OPENAI
        assert facts.metadata.agent.model.temperature == 0.7

    def test_create_with_capabilities(self):
        """Test AgentFacts creation with capabilities."""
        facts = AgentFacts(
            name="Tool Agent",
            capabilities=[
                Capability(name="search", description="Search the web"),
                Capability(name="calculator", description="Do math"),
            ],
        )

        assert len(facts.metadata.agent.capabilities) == 2
        assert facts.metadata.agent.capabilities[0].name == "search"

    def test_sign_and_verify(self):
        """Test signing and verifying metadata."""
        facts = AgentFacts(name="Signed Agent")

        assert not facts.is_signed

        facts.sign()

        assert facts.is_signed
        assert facts.metadata.signature is not None
        assert facts.metadata.log_proof is not None

        result = facts.verify()

        assert result.valid
        assert result.did == facts.did

    def test_verify_unsigned_fails(self):
        """Test that verifying unsigned metadata fails."""
        facts = AgentFacts(name="Unsigned Agent")

        result = facts.verify()

        assert not result.valid
        assert "not signed" in result.errors[0].lower()

    def test_sign_with_different_key(self):
        """Test signing with a different key pair."""
        facts = AgentFacts(name="Agent")
        other_key = KeyPair.generate()

        facts.sign(other_key)

        # Should be signed even when using a different key pair
        assert facts.is_signed

    def test_strict_publisher_did_mismatch_fails(self):
        """Test strict publisher DID mismatch enforcement."""
        facts = AgentFacts(name="Agent")
        other_key = KeyPair.generate()
        facts.sign(other_key)

        context = VerificationContext(strict_publisher_did_match=True)
        result = facts.verify(context=context)

        assert not result.valid
        assert any("Publisher DID mismatch" in error for error in result.errors)

    def test_invalid_inclusion_hash_reports_error(self):
        """Test invalid inclusion hashes return errors instead of raising."""
        facts = AgentFacts(name="Agent")
        facts.sign()

        facts.metadata.log_proof.inclusion = [LogProofEntry(hash="zz", position="left")]

        result = facts.verify()

        assert not result.valid
        assert any("Invalid inclusion entry hash" in error for error in result.errors)

    def test_log_checkpoint_mismatch_fails(self):
        """Test external log checkpoint mismatch detection."""
        facts = AgentFacts(name="Agent")
        facts.sign()

        context = VerificationContext(
            log_root_provider=lambda log_id: "deadbeef",
            strict_log_checkpoint=True,
        )
        result = facts.verify(context=context)

        assert not result.valid
        assert any("Log checkpoint root mismatch" in error for error in result.errors)

    def test_serialization_roundtrip(self):
        """Test JSON serialization and deserialization."""
        original = AgentFacts(
            name="Serialized Agent",
            description="Test serialization",
            baseline_model=BaselineModel(
                name="claude-3",
                provider=ModelProvider.ANTHROPIC,
            ),
            capabilities=[
                Capability(name="tool1", description="First tool"),
            ],
        )
        original.sign()

        # Serialize
        json_str = original.to_json()
        data = json.loads(json_str)

        # Deserialize
        restored = AgentFacts.from_dict(data)

        assert restored.name == original.name
        assert restored.did == original.did
        assert restored.is_signed
        assert restored.metadata.agent.model.name == "claude-3"

    def test_from_json(self):
        """Test loading from JSON string."""
        original = AgentFacts(name="JSON Agent")
        original.sign()

        json_str = original.to_json()
        restored = AgentFacts.from_json(json_str)

        assert restored.name == original.name
        assert restored.did == original.did

    def test_from_dict_rejects_wrong_spec_version(self):
        """Test invalid spec_version is rejected."""
        original = AgentFacts(name="Spec Agent")
        original.sign()

        data = original.to_dict()
        data["spec_version"] = "v0.0"

        with pytest.raises(ValidationError):
            AgentFacts.from_dict(data)

    def test_from_dict_missing_public_key(self):
        """Test loading metadata without publisher keys cannot verify."""
        original = AgentFacts(name="Keyless Agent")
        original.sign()

        data = original.to_dict()
        data["publisher"]["keys"] = []
        data["publisher"]["id"] = "did:web:example.com"

        restored = AgentFacts.from_dict(data)

        result = restored.verify()
        assert not result.valid
        assert "No public key available for verification" in result.errors


class TestHandshakeProtocol:
    """Tests for the verified handshake protocol."""

    def test_create_challenge(self):
        """Test creating a handshake challenge."""
        facts = AgentFacts(name="Challenger")
        challenge = facts.create_challenge()

        assert challenge.nonce is not None
        assert challenge.challenger_did == facts.did
        assert challenge.expires_at > challenge.timestamp

    def test_respond_to_challenge(self):
        """Test responding to a handshake challenge."""
        challenger = AgentFacts(name="Challenger")
        responder = AgentFacts(name="Responder")

        challenge = challenger.create_challenge()
        response = responder.respond_to_challenge(challenge)

        assert response.nonce == challenge.nonce
        assert response.responder_did == responder.did
        assert response.signature is not None

    def test_verify_response(self):
        """Test verifying a handshake response."""
        challenger = AgentFacts(name="Challenger")
        responder = AgentFacts(name="Responder")

        challenge = challenger.create_challenge()
        response = responder.respond_to_challenge(challenge)

        result = challenger.verify_response(challenge, response)

        assert result.valid
        assert result.did == responder.did

    def test_verify_response_with_metadata_hash(self):
        """Test verifying a handshake response with metadata hash check."""
        challenger = AgentFacts(name="Challenger")
        responder = AgentFacts(name="Responder")

        challenge = challenger.create_challenge()
        response = responder.respond_to_challenge(challenge)
        expected_hash = compute_hash(responder._get_signable_data())

        result = challenger.verify_response(
            challenge,
            response,
            expected_metadata_hash=expected_hash,
        )

        assert result.valid
        assert result.did == responder.did

    def test_verify_response_metadata_hash_mismatch(self):
        """Test that metadata hash mismatch fails verification."""
        challenger = AgentFacts(name="Challenger")
        responder = AgentFacts(name="Responder")

        challenge = challenger.create_challenge()
        response = responder.respond_to_challenge(challenge)

        result = challenger.verify_response(
            challenge,
            response,
            expected_metadata_hash="deadbeef",
        )

        assert not result.valid

    def test_expired_challenge_fails(self):
        """Test that expired challenges are rejected."""
        challenger = AgentFacts(name="Challenger")
        responder = AgentFacts(name="Responder")

        # Create an already-expired challenge
        challenge = challenger.create_challenge(ttl_seconds=-1)

        with pytest.raises(ValueError, match="expired"):
            responder.respond_to_challenge(challenge)


class TestEvidenceLogging:
    """Tests for the transparency log."""

    def test_log_evidence(self):
        """Test logging evidence."""
        facts = AgentFacts(name="Logging Agent")

        facts.log_evidence("test_event", {"key": "value"})

        assert facts.merkle_root is not None

    def test_multiple_evidence_entries(self):
        """Test logging multiple evidence entries."""
        facts = AgentFacts(name="Logging Agent")

        facts.log_evidence("event1", {"data": 1})
        root1 = facts.merkle_root

        facts.log_evidence("event2", {"data": 2})
        root2 = facts.merkle_root

        # Root should change with each entry
        assert root1 != root2
