"""
Attestation Verification plugin interfaces.

Defines protocols and data classes for verifying attestation formats.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from agentfacts.plugins.context import VerificationContext


@dataclass
class AttestationPayload:
    """
    Parsed attestation payload with claims.

    Attributes:
        format: The credential format (sd-jwt-vc, anoncreds, json-ld, mdoc)
        issuer: DID of the issuer
        subject: DID of the subject
        claims: Extracted claims from the credential
        raw: The original raw payload
    """

    format: str
    issuer: str
    subject: str
    claims: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] | str | None = None


@dataclass
class AttestationVerificationResult:
    """
    Result of verifying an attestation.

    Attributes:
        valid: Whether the attestation signature is valid
        payload: Parsed payload if valid
        errors: List of error messages if invalid
        warnings: Non-fatal issues detected
    """

    valid: bool
    payload: AttestationPayload | None = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@runtime_checkable
class AttestationVerifier(Protocol):
    """
    Protocol for verifying external attestation formats.

    Implement this to support Verifiable Credential formats like
    SD-JWT-VC, AnonCreds, JSON-LD credentials, or mDocs.

    Example:
        ```python
        class SdJwtVcVerifier:
            @property
            def formats(self) -> set[str]:
                return {"sd-jwt-vc"}

            def verify(self, attestation, context) -> AttestationVerificationResult:
                # Decode and verify SD-JWT
                ...
        ```
    """

    @property
    def formats(self) -> set[str]:
        """
        Set of attestation formats this verifier supports.

        Common formats: "sd-jwt-vc", "anoncreds", "json-ld", "mdoc"
        """
        ...

    def verify(
        self,
        attestation: Any,  # Attestation model
        context: VerificationContext,
    ) -> AttestationVerificationResult:
        """
        Verify an attestation's cryptographic proof.

        Args:
            attestation: The Attestation object to verify
            context: Verification context with resolvers and config

        Returns:
            AttestationVerificationResult with validation status
        """
        ...
