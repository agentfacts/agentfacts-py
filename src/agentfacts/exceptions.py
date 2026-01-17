"""
Custom exceptions for AgentFacts SDK.

Provides a structured exception hierarchy for better error handling.
"""


class AgentFactsError(Exception):
    """Base exception for all AgentFacts errors."""

    pass


class CryptoError(AgentFactsError):
    """Base exception for cryptographic errors."""

    pass


class SignatureError(CryptoError):
    """Error during signature creation or verification."""

    pass


class SignatureVerificationError(SignatureError):
    """Signature verification failed."""

    pass


class SignatureCreationError(SignatureError):
    """Failed to create signature."""

    pass


class KeyError(CryptoError):
    """Error related to key operations."""

    pass


class InvalidKeyError(KeyError):
    """Invalid or malformed key."""

    pass


class KeyNotFoundError(KeyError):
    """Key file not found."""

    pass


class VerificationOnlyKeyError(KeyError):
    """Attempted signing operation with verification-only key."""

    pass


class DIDError(AgentFactsError):
    """Error related to DID operations."""

    pass


class InvalidDIDError(DIDError):
    """Invalid or malformed DID."""

    pass


class DIDResolutionError(DIDError):
    """Failed to resolve DID."""

    pass


class PolicyError(AgentFactsError):
    """Base exception for policy-related errors."""

    pass


class PolicyViolationError(PolicyError):
    """Agent violated one or more policy rules."""

    def __init__(self, message: str, violations: list[str] | None = None):
        super().__init__(message)
        self.violations = violations or []


class PolicyNotFoundError(PolicyError):
    """Requested policy was not found."""

    pass


class HandshakeError(AgentFactsError):
    """Base exception for handshake protocol errors."""

    pass


class ChallengeExpiredError(HandshakeError):
    """Handshake challenge has expired."""

    pass


class InvalidChallengeError(HandshakeError):
    """Invalid or malformed challenge."""

    pass


class InvalidResponseError(HandshakeError):
    """Invalid handshake response."""

    pass


class IntrospectionError(AgentFactsError):
    """Error during agent introspection."""

    pass


class UnsupportedAgentTypeError(IntrospectionError):
    """Agent type is not supported for introspection."""

    pass


class SerializationError(AgentFactsError):
    """Error during serialization or deserialization."""

    pass


class CanonicalizationError(SerializationError):
    """Error during JSON canonicalization."""

    pass


class MerkleError(AgentFactsError):
    """Error related to Merkle tree operations."""

    pass


class InvalidProofError(MerkleError):
    """Merkle proof is invalid."""

    pass


class MiddlewareError(AgentFactsError):
    """Error in HTTP middleware."""

    pass


class AgentIdentityRequiredError(MiddlewareError):
    """Agent identity headers are required but missing."""

    pass
