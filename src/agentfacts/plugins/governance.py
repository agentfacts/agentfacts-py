"""
Governance Adapter plugin interfaces.

Defines protocols and data classes for policy intermediate representation
and governance document conversion.
"""

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

# -----------------------------------------------------------------------------
# Policy Intermediate Representation (IR)
# -----------------------------------------------------------------------------


@dataclass
class RequireIssuerIR:
    """Policy IR: Require attestation from specific issuer."""

    did: str


@dataclass
class RequireAttestationIR:
    """Policy IR: Require attestation of specific type/format."""

    format: str | None = None
    attestation_type: str | None = None
    issuer: str | None = None
    max_age_days: int | None = None


@dataclass
class RequireCapabilityIR:
    """Policy IR: Require specific capability with max risk level."""

    capability: str | None = None
    max_risk_level: str | None = None  # low, medium, high


@dataclass
class RequireStatusNotRevokedIR:
    """Policy IR: Require that credential status is not revoked."""

    pass


@dataclass
class RequireComplianceIR:
    """Policy IR: Require compliance with specific framework."""

    framework: str


@dataclass
class DenyCapabilityIR:
    """Policy IR: Deny agents with specific capability."""

    capability: str


# Union type for all policy IR types
PolicyIR = (
    RequireIssuerIR
    | RequireAttestationIR
    | RequireCapabilityIR
    | RequireStatusNotRevokedIR
    | RequireComplianceIR
    | DenyCapabilityIR
)


# -----------------------------------------------------------------------------
# Governance Adapter Protocol
# -----------------------------------------------------------------------------


@runtime_checkable
class GovernanceAdapter(Protocol):
    """
    Protocol for converting governance documents to policy IR.

    Implement this to support governance frameworks like DEGov,
    OID4VC trust frameworks, or organizational policies.

    Example:
        ```python
        class DEGovAdapter:
            def to_policy_ir(self, doc: dict) -> list[PolicyIR]:
                # Parse DEGov document and extract rules
                rules = []
                if "required_issuers" in doc:
                    for issuer in doc["required_issuers"]:
                        rules.append(RequireIssuerIR(did=issuer))
                return rules
        ```
    """

    def to_policy_ir(self, doc: dict[str, Any]) -> list[PolicyIR]:
        """
        Convert a governance document to policy IR.

        Args:
            doc: The governance document (parsed JSON/YAML)

        Returns:
            List of PolicyIR rules to evaluate
        """
        ...
