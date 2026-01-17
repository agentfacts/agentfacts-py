"""
Plugin interfaces for AgentFacts extensibility.

Defines protocols for DID resolution, attestation verification,
status checking, and governance adaptation. All interfaces are
optional - the default AgentFacts behavior requires no plugins.

These interfaces enable integration with:
- External DID methods (did:web, did:ion, did:ethr, etc.)
- Verifiable Credential formats (sd-jwt-vc, anoncreds, json-ld, mdoc)
- Revocation/status list services
- Governance frameworks (DEGov, OID4VC policies)
"""

# DID Resolution
# Attestation Verification
from agentfacts.plugins.attestation import (
    AttestationPayload,
    AttestationVerificationResult,
    AttestationVerifier,
)

# Verification Context
from agentfacts.plugins.context import (
    VerificationContext,
)
from agentfacts.plugins.did import (
    DIDResolver,
    ResolvedDID,
)

# Policy IR and Governance
from agentfacts.plugins.governance import (
    DenyCapabilityIR,
    GovernanceAdapter,
    PolicyIR,
    RequireAttestationIR,
    RequireCapabilityIR,
    RequireComplianceIR,
    RequireIssuerIR,
    RequireStatusNotRevokedIR,
)

# Metadata Provider
from agentfacts.plugins.metadata import (
    MetadataProvider,
)

# Plugin Registry
from agentfacts.plugins.registry import (
    PluginRegistry,
    get_plugin_registry,
    reset_plugin_registry,
)

# Status/Revocation Checking
from agentfacts.plugins.status import (
    StatusChecker,
    StatusCheckResult,
)

__all__ = [
    # DID Resolution
    "DIDResolver",
    "ResolvedDID",
    # Attestation Verification
    "AttestationPayload",
    "AttestationVerificationResult",
    "AttestationVerifier",
    # Status/Revocation Checking
    "StatusCheckResult",
    "StatusChecker",
    # Policy IR
    "DenyCapabilityIR",
    "PolicyIR",
    "RequireAttestationIR",
    "RequireCapabilityIR",
    "RequireComplianceIR",
    "RequireIssuerIR",
    "RequireStatusNotRevokedIR",
    # Governance
    "GovernanceAdapter",
    # Metadata Provider
    "MetadataProvider",
    # Verification Context
    "VerificationContext",
    # Plugin Registry
    "PluginRegistry",
    "get_plugin_registry",
    "reset_plugin_registry",
]
