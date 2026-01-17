"""
AgentFacts SDK - The SSL of the Agentic Web

Provides cryptographically verifiable identity for AI agents,
enabling trust through DIDs, signed metadata, and transparency logs.

Supports multiple agent frameworks:
- LangChain / LangGraph
- CrewAI (multi-agent crews)
- AutoGen (conversational agents)
- And more through pluggable introspection

Quick Start:
    ```python
    from agentfacts import AgentFacts

    facts = AgentFacts.from_agent_signed(my_agent, "My Agent")
    print(facts.to_json())

    from agentfacts import enable_default_warnings
    enable_default_warnings(True)
    ```
"""

from agentfacts.core import AgentFacts
from agentfacts.crypto.did import DID
from agentfacts.crypto.keys import KeyPair
from agentfacts.exceptions import (
    AgentFactsError,
    AgentIdentityRequiredError,
    CanonicalizationError,
    ChallengeExpiredError,
    CryptoError,
    DIDError,
    DIDResolutionError,
    HandshakeError,
    IntrospectionError,
    InvalidChallengeError,
    InvalidDIDError,
    InvalidKeyError,
    InvalidProofError,
    InvalidResponseError,
    KeyNotFoundError,
    MerkleError,
    MiddlewareError,
    PolicyError,
    PolicyNotFoundError,
    PolicyViolationError,
    SerializationError,
    SignatureCreationError,
    SignatureError,
    SignatureVerificationError,
    UnsupportedAgentTypeError,
    VerificationOnlyKeyError,
)
from agentfacts.group import GroupFacts
from agentfacts.logging import (
    AgentFactsLogger,
    configure_logging,
    enable_default_warnings,
    get_logger,
)
from agentfacts.models import (
    AgentFactsCard,
    AgentInfo,
    AgentRole,
    Attestation,
    BaselineModel,
    Capability,
    ComplianceInfo,
    DelegationPolicy,
    GroupMetadata,
    LogProof,
    LogProofEntry,
    ModelProvider,
    OperationalConstraints,
    Policy,
    ProcessType,
    Publisher,
    PublisherKey,
    SignatureBlock,
    Tool,
    VerificationResult,
)
from agentfacts.plugins import (
    VerificationContext,
    get_plugin_registry,
)
from agentfacts.schema import (
    SCHEMA_VERSION,
    get_agent_card_schema,
    get_agent_schema,
    get_all_schemas,
    get_capability_schema,
    get_group_metadata_schema,
    get_policy_schema,
    get_publisher_schema,
    get_tool_schema,
)

__version__ = "0.1.0"

__all__ = [
    # Core classes
    "AgentFacts",
    "GroupFacts",
    # Models
    "AgentFactsCard",
    "AgentInfo",
    "AgentRole",
    "Attestation",
    "BaselineModel",
    "Capability",
    "ComplianceInfo",
    "DelegationPolicy",
    "GroupMetadata",
    "LogProof",
    "LogProofEntry",
    "ModelProvider",
    "OperationalConstraints",
    "Policy",
    "ProcessType",
    "Publisher",
    "PublisherKey",
    "SignatureBlock",
    "Tool",
    "VerificationResult",
    # Crypto
    "DID",
    "KeyPair",
    # Logging
    "AgentFactsLogger",
    "configure_logging",
    "enable_default_warnings",
    "get_logger",
    # Schema
    "SCHEMA_VERSION",
    "get_agent_card_schema",
    "get_agent_schema",
    "get_all_schemas",
    "get_capability_schema",
    "get_group_metadata_schema",
    "get_policy_schema",
    "get_publisher_schema",
    "get_tool_schema",
    # Exceptions
    "AgentFactsError",
    "AgentIdentityRequiredError",
    "CanonicalizationError",
    "ChallengeExpiredError",
    "CryptoError",
    "DIDError",
    "DIDResolutionError",
    "HandshakeError",
    "IntrospectionError",
    "InvalidChallengeError",
    "InvalidDIDError",
    "InvalidKeyError",
    "InvalidProofError",
    "InvalidResponseError",
    "KeyNotFoundError",
    "MerkleError",
    "MiddlewareError",
    "PolicyError",
    "PolicyNotFoundError",
    "PolicyViolationError",
    "SerializationError",
    "SignatureCreationError",
    "SignatureError",
    "SignatureVerificationError",
    "UnsupportedAgentTypeError",
    "VerificationOnlyKeyError",
    # Plugins
    "VerificationContext",
    "get_plugin_registry",
]
