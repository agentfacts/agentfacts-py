"""
Verification Context for enhanced verification with plugins.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from agentfacts.plugins.attestation import AttestationVerifier
from agentfacts.plugins.did import DIDResolver
from agentfacts.plugins.governance import GovernanceAdapter, PolicyIR
from agentfacts.plugins.status import StatusChecker


@dataclass
class VerificationContext:
    """
    Context for enhanced verification with plugins.

    Holds optional resolvers, verifiers, and configuration for
    the verification pipeline. When passed to verify(), enables
    external DID resolution, VC verification, and status checks.

    All fields are optional - None values use default behavior.

    Example:
        ```python
        from agentfacts import AgentFacts
        from agentfacts.plugins import VerificationContext

        # Create context with custom resolver
        context = VerificationContext(
            did_resolver=DidWebResolver(),
            attestation_verifiers=[SdJwtVcVerifier()],
        )

        # Use context in verification
        facts = AgentFacts.from_json(data)
        result = facts.verify(context=context)
        ```

    Attributes:
        did_resolver: Custom DID resolver (uses built-in did:key if None)
        attestation_verifiers: List of VC format verifiers
        status_checkers: List of revocation status checkers
        governance_adapter: Adapter for converting governance docs to policy
        governance_doc: Governance document to apply (requires adapter)
        clock: Custom clock function for testing (returns current UTC time)
        cache: Optional cache dict for resolver results
        strict_did_verification: If True, fail on DID resolution errors
        strict_publisher_did_match: If True, fail when publisher DID mismatches signature key
        log_root_provider: Optional callable returning expected log root for a log_id
        strict_log_checkpoint: If True, fail when log checkpoint cannot be verified
        verify_attestation_signatures: If True, verify VC signatures
        check_revocation_status: If True, check status lists
    """

    did_resolver: DIDResolver | None = None
    attestation_verifiers: list[AttestationVerifier] = field(default_factory=list)
    status_checkers: list[StatusChecker] = field(default_factory=list)
    governance_adapter: GovernanceAdapter | None = None
    governance_doc: dict[str, Any] | None = None
    clock: Callable[[], datetime] | None = None
    cache: dict[str, Any] | None = None

    # Verification flags
    strict_did_verification: bool = False
    strict_publisher_did_match: bool = False
    log_root_provider: Callable[[str], str | None] | None = None
    strict_log_checkpoint: bool = False
    verify_attestation_signatures: bool = True
    check_revocation_status: bool = True

    def get_time(self) -> datetime:
        """Get current time using custom clock or default."""
        if self.clock:
            return self.clock()
        from agentfacts.utils import utcnow

        return utcnow()

    def get_did_resolver(self) -> DIDResolver | None:
        """Get the DID resolver, if any."""
        return self.did_resolver

    def get_verifier_for_format(self, format: str) -> AttestationVerifier | None:
        """Find a verifier that supports the given format."""
        for verifier in self.attestation_verifiers:
            if format in verifier.formats:
                return verifier
        return None

    def get_status_checker(self, status_ref: str) -> StatusChecker | None:
        """Find a status checker that supports the given reference."""
        for checker in self.status_checkers:
            if checker.supports(status_ref):
                return checker
        return None

    def get_policy_ir(self) -> list[PolicyIR]:
        """Convert governance doc to policy IR using adapter."""
        if self.governance_adapter and self.governance_doc:
            return self.governance_adapter.to_policy_ir(self.governance_doc)
        return []
