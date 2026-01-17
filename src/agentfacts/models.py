"""
Core data models for AgentFacts SDK.

These models represent the AgentFacts Card (v0.1) - structured metadata
about an agent's identity, capabilities, publisher, and policy.

Pythonic Features:
- Rich comparison operators for Capability (sortable by risk)
- __repr__ for debugging in REPL
- __hash__ for use in sets/dicts (where applicable)
- Pydantic v2 with frozen models where appropriate
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from functools import total_ordering
from typing import Any, Literal

from pydantic import BaseModel, Field

from agentfacts.utils import utcnow as _utcnow

# Risk level ordering for comparison operators
_RISK_ORDER: dict[str | None, int] = {
    None: 0,
    "low": 1,
    "medium": 2,
    "high": 3,
}


class ModelProvider(str, Enum):
    """Known LLM providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    COHERE = "cohere"
    MISTRAL = "mistral"
    META = "meta"
    HUGGINGFACE = "huggingface"
    LOCAL = "local"
    UNKNOWN = "unknown"


class BaselineModel(BaseModel):
    """
    Model metadata - describes the underlying LLM.

    This is the "engine" of the agent, capturing model lineage
    for compliance and capability assessment.
    """

    name: str = Field(..., description="Model name (e.g., 'gpt-4', 'claude-3-opus')")
    provider: ModelProvider = Field(
        default=ModelProvider.UNKNOWN, description="Model provider"
    )
    version: str | None = Field(default=None, description="Model version if available")
    temperature: float | None = Field(
        default=None, ge=0.0, le=2.0, description="Sampling temperature"
    )
    max_tokens: int | None = Field(
        default=None, gt=0, description="Maximum output tokens"
    )
    context_window: int | None = Field(
        default=None, gt=0, description="Context window size"
    )
    extra_params: dict[str, Any] = Field(
        default_factory=dict, description="Additional model parameters"
    )

    def __repr__(self) -> str:
        return f"BaselineModel(name={self.name!r}, provider={self.provider.value!r})"

    def __str__(self) -> str:
        return f"{self.name} ({self.provider.value})"


@total_ordering
class Capability(BaseModel):
    """
    Agent capability - describes a tool or action the agent can perform.

    Captures tool descriptions for trust assessment and policy evaluation.

    Supports rich comparison operators for sorting by risk level:
        >>> sorted(capabilities)  # sorts low -> medium -> high
        >>> max(capabilities)     # returns highest risk capability
    """

    name: str = Field(..., description="Tool/capability name")
    description: str = Field(default="", description="Human-readable description")
    parameters: dict[str, Any] = Field(
        default_factory=dict, description="Tool parameter schema"
    )
    risk_level: str | None = Field(
        default=None, description="Assessed risk level (low/medium/high)"
    )
    requires_approval: bool = Field(
        default=False, description="Whether this capability requires human approval"
    )
    # Multi-agent delegation support
    delegatable: bool = Field(
        default=False,
        description="Whether this capability can be delegated to other agents",
    )
    delegatable_to: list[str] = Field(
        default_factory=list, description="DIDs of agents this can be delegated to"
    )

    def __repr__(self) -> str:
        risk = self.risk_level or "unassessed"
        return f"Capability(name={self.name!r}, risk={risk!r})"

    def __str__(self) -> str:
        return self.name

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Capability):
            return NotImplemented
        return self.name == other.name and self.risk_level == other.risk_level

    def __lt__(self, other: object) -> bool:
        """Compare by risk level (low < medium < high)."""
        if not isinstance(other, Capability):
            return NotImplemented
        return _RISK_ORDER.get(self.risk_level, 0) < _RISK_ORDER.get(
            other.risk_level, 0
        )

    def __hash__(self) -> int:
        return hash((self.name, self.risk_level))

    @property
    def is_high_risk(self) -> bool:
        """Check if this capability is high risk."""
        return self.risk_level == "high"

    @property
    def is_dangerous(self) -> bool:
        """Alias for is_high_risk - more readable in conditionals."""
        return self.is_high_risk


class Tool(BaseModel):
    """Tool metadata for agent integrations."""

    id: str = Field(..., description="Tool identifier")
    name: str | None = Field(default=None, description="Human-readable tool name")
    description: str | None = Field(default=None, description="Tool description")
    type: str | None = Field(
        default=None, description="Tool type (api, http, python, mcp)"
    )
    parameters: dict[str, Any] = Field(
        default_factory=dict, description="Tool parameter schema"
    )


class AgentRole(BaseModel):
    """
    Role information for multi-agent systems like CrewAI.

    Captures the agent's role, goal, and backstory which define
    its persona and behavior in a crew.
    """

    role_name: str = Field(
        ..., description="The agent's role (e.g., 'Senior Researcher')"
    )
    goal: str = Field(default="", description="The agent's primary goal")
    backstory: str = Field(
        default="", description="Background context for the agent's persona"
    )
    hierarchy_level: int = Field(
        default=0, description="0 = worker, 1 = manager, 2+ = higher"
    )


class DelegationPolicy(BaseModel):
    """
    Delegation settings for multi-agent systems.

    Controls whether and how an agent can delegate work to other agents.
    """

    can_delegate: bool = Field(
        default=False, description="Whether agent can delegate tasks"
    )
    can_receive_delegation: bool = Field(
        default=True, description="Whether agent can receive delegated tasks"
    )
    delegatable_to: list[str] = Field(
        default_factory=list, description="DIDs of agents this agent can delegate to"
    )
    requires_approval_to_delegate: bool = Field(
        default=True, description="Requires human approval before delegating"
    )
    max_delegation_depth: int = Field(
        default=1, description="Max depth of delegation chain"
    )


class Attestation(BaseModel):
    """
    A signed attestation - evidence appended to the agent's transparency log.

    Used to record security audits, compliance checks, or runtime events.
    Supports both native AgentFacts attestations and external Verifiable
    Credential formats (SD-JWT-VC, AnonCreds, JSON-LD, mDoc).

    For external VC formats, use the format/payload/payload_ref fields.
    The native signature field is used for AgentFacts-native attestations.
    """

    id: str = Field(..., description="Unique attestation ID")
    type: str = Field(
        ..., description="Attestation type (e.g., 'security_audit', 'compliance_check')"
    )
    issuer: str = Field(..., description="DID of the attestation issuer")
    subject: str = Field(..., description="DID of the agent being attested")
    issued_at: datetime = Field(
        default_factory=_utcnow, description="Issuance timestamp"
    )
    expires_at: datetime | None = Field(
        default=None, description="Expiration timestamp"
    )
    claims: dict[str, Any] = Field(
        default_factory=dict, description="Attestation claims/data"
    )
    signature: str | None = Field(
        default=None, description="Ed25519 signature of the attestation"
    )

    # External VC format support (all optional)
    format: str | None = Field(
        default=None,
        description="External credential format: 'sd-jwt-vc', 'anoncreds', 'json-ld', 'mdoc'",
    )
    payload: dict[str, Any] | str | None = Field(
        default=None, description="Opaque VC/VP payload (format-specific)"
    )
    payload_ref: str | None = Field(
        default=None, description="URL or reference ID to fetch the payload"
    )
    status_ref: str | None = Field(
        default=None, description="Reference to revocation/status list entry"
    )
    proof_type: str | None = Field(
        default=None,
        description="Proof type descriptor (e.g., 'Ed25519Signature2020', 'BbsBlsSignature2020')",
    )


class OperationalConstraints(BaseModel):
    """
    Operational constraints and limits for the agent.

    Extended to cover modern agentic patterns including code execution,
    MCP servers, sandboxing, and resource limits.
    """

    # Iteration and time limits
    max_iterations: int | None = Field(
        default=None, description="Maximum agent iterations"
    )
    timeout_seconds: int | None = Field(default=None, description="Operation timeout")

    # Network constraints
    allowed_domains: list[str] = Field(
        default_factory=list, description="Allowed network domains"
    )
    blocked_domains: list[str] = Field(
        default_factory=list, description="Blocked network domains"
    )

    # Action constraints
    blocked_actions: list[str] = Field(
        default_factory=list, description="Explicitly blocked actions"
    )
    allowed_actions: list[str] = Field(
        default_factory=list, description="Explicitly allowed actions (allowlist mode)"
    )

    # Human oversight
    requires_human_approval: bool = Field(
        default=False, description="Requires human-in-the-loop"
    )
    approval_required_for: list[str] = Field(
        default_factory=list, description="Specific actions requiring approval"
    )

    # Code execution
    code_execution_allowed: bool = Field(
        default=False, description="Whether code execution is permitted"
    )
    code_sandbox_type: str | None = Field(
        default=None, description="Sandbox type: 'docker', 'e2b', 'pyodide', 'none'"
    )

    # MCP servers
    mcp_servers_allowed: bool = Field(
        default=True, description="Whether MCP server connections allowed"
    )
    mcp_server_allowlist: list[str] = Field(
        default_factory=list, description="Allowed MCP server names/commands"
    )

    # Resource limits
    max_tokens_per_request: int | None = Field(
        default=None, description="Max tokens per LLM request"
    )
    max_tool_calls_per_turn: int | None = Field(
        default=None, description="Max tool calls per agent turn"
    )
    max_memory_mb: int | None = Field(
        default=None, description="Maximum memory usage in MB"
    )

    # Rate limiting
    rate_limit_requests_per_minute: int | None = Field(
        default=None, description="Maximum requests per minute"
    )


class ComplianceInfo(BaseModel):
    """
    Compliance and regulatory information.

    Extended to cover modern AI regulations including EU AI Act,
    NIST AI RMF, and organizational policies.
    """

    # Regulatory frameworks
    frameworks: list[str] = Field(
        default_factory=list,
        description="Compliance frameworks (e.g., 'EU_AI_ACT', 'NIST_AI_RMF', 'ISO_42001')",
    )
    risk_category: str | None = Field(
        default=None,
        description="Risk category: 'minimal', 'limited', 'high', 'unacceptable'",
    )

    # EU AI Act specific
    eu_ai_act_compliant: bool | None = Field(
        default=None, description="Explicit EU AI Act compliance status"
    )
    eu_ai_act_category: str | None = Field(
        default=None,
        description="EU AI Act category: 'prohibited', 'high_risk', 'limited_risk', 'minimal_risk'",
    )

    # Audit information
    last_audit: datetime | None = Field(
        default=None, description="Last compliance audit date"
    )
    auditor: str | None = Field(default=None, description="DID of the auditor")
    audit_report_url: str | None = Field(
        default=None, description="URL to audit report"
    )

    # Data governance
    data_retention_days: int | None = Field(
        default=None, description="Data retention period"
    )
    pii_handling: str | None = Field(
        default=None,
        description="PII handling policy: 'none', 'anonymized', 'encrypted', 'raw'",
    )
    gdpr_compliant: bool | None = Field(
        default=None, description="GDPR compliance status"
    )

    # Transparency
    transparency_report_url: str | None = Field(
        default=None, description="URL to transparency/model card documentation"
    )
    explainability_level: str | None = Field(
        default=None, description="Explainability: 'black_box', 'partial', 'full'"
    )

    # Certifications
    certifications: list[str] = Field(
        default_factory=list, description="List of certifications held"
    )

    # Organizational policies
    organization_policy_url: str | None = Field(
        default=None, description="URL to organization's AI policy"
    )


class Policy(BaseModel):
    """Policy metadata for compliance and constraints."""

    compliance: ComplianceInfo = Field(default_factory=ComplianceInfo)
    constraints: OperationalConstraints = Field(default_factory=OperationalConstraints)


class PublisherKey(BaseModel):
    """Public key metadata for publisher verification."""

    id: str = Field(..., description="Key identifier (DID URL fragment)")
    type: str = Field(..., description="Key type (e.g., Ed25519VerificationKey2020)")
    public_key: str = Field(
        ..., description="Public key material (base64 or multibase)"
    )
    controller: str | None = Field(
        default=None, description="Key controller identifier"
    )
    purpose: str | None = Field(default=None, description="Key purpose")


class Publisher(BaseModel):
    """Publisher identity information."""

    id: str = Field(..., description="Publisher identifier (DID or DNS-based)")
    name: str | None = Field(default=None, description="Publisher name")
    contact: str | None = Field(default=None, description="Security contact")
    keys: list[PublisherKey] = Field(
        default_factory=list, description="Publisher public keys"
    )


class AgentInfo(BaseModel):
    """
    Agent identity and metadata.

    This is the core payload describing the agent.
    """

    id: str = Field(..., description="Decentralised Identifier (DID)")
    name: str = Field(..., description="Human-readable agent name")
    description: str = Field(default="", description="Agent description")
    version: str = Field(default="1.0.0", description="Agent version")
    model: BaselineModel = Field(..., description="Underlying LLM information")
    capabilities: list[Capability] = Field(
        default_factory=list, description="Agent capabilities/tools"
    )
    tools: list[Tool] = Field(default_factory=list, description="Tool inventory")
    framework: str | None = Field(
        default=None, description="Agent framework (langchain, crewai, autogen, etc.)"
    )
    role: AgentRole | None = Field(
        default=None, description="Role info for multi-agent systems"
    )
    delegation: DelegationPolicy = Field(
        default_factory=DelegationPolicy, description="Delegation settings"
    )
    group_memberships: list[str] = Field(
        default_factory=list, description="DIDs of groups/crews this agent belongs to"
    )
    context: dict[str, Any] = Field(
        default_factory=dict, description="Additional context"
    )


class SignatureBlock(BaseModel):
    """Signature block for AgentFacts Cards."""

    alg: str = Field(..., description="Signature algorithm (e.g., 'ed25519')")
    key_id: str = Field(..., description="Key identifier or DID URL fragment")
    value: str = Field(..., description="Signature value (base64)")


class LogProofEntry(BaseModel):
    """Inclusion proof entry with explicit position."""

    hash: str = Field(..., description="Sibling hash (hex)")
    position: Literal["left", "right"] = Field(
        ..., description="Sibling position relative to the leaf"
    )


class LogProof(BaseModel):
    """Log inclusion proof for AgentFacts Cards."""

    log_id: str = Field(..., description="Transparency log identifier")
    leaf_hash: str = Field(..., description="Hash of the signed card")
    root_hash: str = Field(..., description="Merkle root hash")
    inclusion: list[LogProofEntry] = Field(
        default_factory=list,
        description="Inclusion proof entries with sibling positions",
    )


class AgentFactsCard(BaseModel):
    """
    AgentFacts Card v0.1.

    This is the canonical identity document for an AI agent.
    """

    spec_version: Literal["v0.1"] = Field(default="v0.1", description="Spec version")
    agent: AgentInfo = Field(..., description="Agent identity and metadata")
    publisher: Publisher = Field(..., description="Publisher identity")
    policy: Policy = Field(
        default_factory=Policy, description="Compliance and constraints"
    )
    issued_at: datetime = Field(
        default_factory=_utcnow, description="Card issuance time"
    )
    signature: SignatureBlock = Field(..., description="Signature block")
    log_proof: LogProof = Field(..., description="Transparency log inclusion proof")
    attestations: list[Attestation] = Field(
        default_factory=list, description="Attestations"
    )
    extensions: dict[str, Any] = Field(
        default_factory=dict, description="Namespaced extensions"
    )


class VerificationResult(BaseModel):
    """Result of verifying an agent's identity or metadata."""

    valid: bool = Field(..., description="Whether verification passed")
    did: str | None = Field(default=None, description="Verified DID")
    errors: list[str] = Field(default_factory=list, description="Verification errors")
    warnings: list[str] = Field(
        default_factory=list, description="Verification warnings"
    )
    verified_at: datetime = Field(default_factory=_utcnow)
    policy_violations: list[str] = Field(
        default_factory=list, description="Policy violations found"
    )

    @property
    def passed(self) -> bool:
        """Alias for valid."""
        return self.valid

    def __repr__(self) -> str:
        status = "✓ VALID" if self.valid else "✗ INVALID"
        return (
            f"VerificationResult({status}, errors={len(self.errors)}, "
            f"warnings={len(self.warnings)})"
        )

    def __bool__(self) -> bool:
        """Allow using result directly in conditionals: if result: ..."""
        return self.valid


class HandshakeChallenge(BaseModel):
    """Challenge for the verified handshake protocol."""

    nonce: str = Field(..., description="Cryptographic nonce (base64)")
    timestamp: datetime = Field(default_factory=_utcnow)
    challenger_did: str = Field(..., description="DID of the challenger")
    expires_at: datetime = Field(..., description="Challenge expiration")


class HandshakeResponse(BaseModel):
    """Response to a handshake challenge."""

    nonce: str = Field(..., description="Original nonce")
    responder_did: str = Field(..., description="DID of the responder")
    signature: str = Field(..., description="Signature of the nonce")
    public_key: str = Field(..., description="Public key for verification")
    metadata_hash: str | None = Field(
        default=None, description="Hash of agent metadata"
    )


class ProcessType(str, Enum):
    """Process types for multi-agent workflows."""

    SEQUENTIAL = "sequential"
    HIERARCHICAL = "hierarchical"
    PARALLEL = "parallel"
    EVENT_DRIVEN = "event_driven"
    CUSTOM = "custom"


class GroupMetadata(BaseModel):
    """
    Metadata for agent groups/crews.

    Represents a collection of agents working together, such as
    a CrewAI Crew or AutoGen GroupChat.
    """

    # Identity
    did: str = Field(..., description="Decentralised Identifier for the group")
    name: str = Field(..., description="Human-readable group name")
    description: str = Field(default="", description="Group description")
    version: str = Field(default="1.0.0", description="Group version")

    # Timestamps
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)

    # Members
    member_dids: list[str] = Field(
        default_factory=list, description="DIDs of member agents"
    )
    entry_agent_did: str | None = Field(
        default=None, description="DID of the entry point agent"
    )

    # Process/workflow
    process_type: ProcessType = Field(
        default=ProcessType.SEQUENTIAL, description="Execution pattern"
    )
    topology: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Adjacency list: agent DID -> list of DIDs it can communicate with",
    )
    max_rounds: int | None = Field(
        default=None, description="Maximum conversation rounds"
    )

    # Shared resources
    shared_memory: bool = Field(
        default=False, description="Whether agents share memory/context"
    )
    shared_tools: list[str] = Field(
        default_factory=list, description="Tools shared across all agents"
    )

    # Framework
    framework: str | None = Field(
        default=None, description="Source framework (crewai, autogen, etc.)"
    )

    # Cryptographic proof
    public_key: str | None = Field(
        default=None, description="Ed25519 public key (base64)"
    )
    signature: str | None = Field(default=None, description="Signature of the metadata")

    # Context
    context: dict[str, Any] = Field(
        default_factory=dict, description="Additional context"
    )
