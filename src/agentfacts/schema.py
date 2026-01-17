"""
JSON Schema exports for AgentFacts data models.

Provides easy access to JSON schemas for all AgentFacts models,
enabling framework developers to understand the data contracts
without reading source code.

Example:
    ```python
    from agentfacts.schema import get_all_schemas, SCHEMA_VERSION

    schemas = get_all_schemas()
    print(f"Schema version: {schemas['version']}")

    from agentfacts.schema import get_agent_card_schema
    schema = get_agent_card_schema()
    print(schema)
    ```
"""

from typing import Any

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
    HandshakeChallenge,
    HandshakeResponse,
    LogProof,
    LogProofEntry,
    OperationalConstraints,
    Policy,
    Publisher,
    PublisherKey,
    SignatureBlock,
    Tool,
    VerificationResult,
)

# Schema version tracks breaking changes to the data models
SCHEMA_VERSION = "v0.1"


def get_agent_card_schema() -> dict[str, Any]:
    """Get JSON Schema for AgentFactsCard."""
    return AgentFactsCard.model_json_schema()


def get_agent_schema() -> dict[str, Any]:
    """Get JSON Schema for AgentInfo."""
    return AgentInfo.model_json_schema()


def get_publisher_schema() -> dict[str, Any]:
    """Get JSON Schema for Publisher."""
    return Publisher.model_json_schema()


def get_policy_schema() -> dict[str, Any]:
    """Get JSON Schema for Policy."""
    return Policy.model_json_schema()


def get_capability_schema() -> dict[str, Any]:
    """Get JSON Schema for Capability."""
    return Capability.model_json_schema()


def get_tool_schema() -> dict[str, Any]:
    """Get JSON Schema for Tool."""
    return Tool.model_json_schema()


def get_baseline_model_schema() -> dict[str, Any]:
    """Get JSON Schema for BaselineModel."""
    return BaselineModel.model_json_schema()


def get_operational_constraints_schema() -> dict[str, Any]:
    """Get JSON Schema for OperationalConstraints."""
    return OperationalConstraints.model_json_schema()


def get_compliance_info_schema() -> dict[str, Any]:
    """Get JSON Schema for ComplianceInfo."""
    return ComplianceInfo.model_json_schema()


def get_signature_schema() -> dict[str, Any]:
    """Get JSON Schema for SignatureBlock."""
    return SignatureBlock.model_json_schema()


def get_log_proof_schema() -> dict[str, Any]:
    """Get JSON Schema for LogProof."""
    return LogProof.model_json_schema()


def get_log_proof_entry_schema() -> dict[str, Any]:
    """Get JSON Schema for LogProofEntry."""
    return LogProofEntry.model_json_schema()


def get_publisher_key_schema() -> dict[str, Any]:
    """Get JSON Schema for PublisherKey."""
    return PublisherKey.model_json_schema()


def get_attestation_schema() -> dict[str, Any]:
    """Get JSON Schema for Attestation."""
    return Attestation.model_json_schema()


def get_agent_role_schema() -> dict[str, Any]:
    """Get JSON Schema for AgentRole."""
    return AgentRole.model_json_schema()


def get_delegation_policy_schema() -> dict[str, Any]:
    """Get JSON Schema for DelegationPolicy."""
    return DelegationPolicy.model_json_schema()


def get_handshake_challenge_schema() -> dict[str, Any]:
    """Get JSON Schema for HandshakeChallenge."""
    return HandshakeChallenge.model_json_schema()


def get_handshake_response_schema() -> dict[str, Any]:
    """Get JSON Schema for HandshakeResponse."""
    return HandshakeResponse.model_json_schema()


def get_group_metadata_schema() -> dict[str, Any]:
    """Get JSON Schema for GroupMetadata."""
    return GroupMetadata.model_json_schema()


def get_all_schemas() -> dict[str, Any]:
    """
    Get all schemas in one dict.

    Returns a comprehensive dict containing JSON schemas
    for all AgentFacts data models.
    """
    return {
        "version": SCHEMA_VERSION,
        "agent_card": get_agent_card_schema(),
        "agent": get_agent_schema(),
        "publisher": get_publisher_schema(),
        "publisher_key": get_publisher_key_schema(),
        "policy": get_policy_schema(),
        "capability": get_capability_schema(),
        "tool": get_tool_schema(),
        "baseline_model": get_baseline_model_schema(),
        "operational_constraints": get_operational_constraints_schema(),
        "compliance_info": get_compliance_info_schema(),
        "signature": get_signature_schema(),
        "log_proof": get_log_proof_schema(),
        "log_proof_entry": get_log_proof_entry_schema(),
        "attestation": get_attestation_schema(),
        "agent_role": get_agent_role_schema(),
        "delegation_policy": get_delegation_policy_schema(),
        "handshake_challenge": get_handshake_challenge_schema(),
        "handshake_response": get_handshake_response_schema(),
        "group_metadata": get_group_metadata_schema(),
        "verification_result": VerificationResult.model_json_schema(),
    }


def validate_against_schema(
    data: dict[str, Any], schema_name: str
) -> tuple[bool, list[str]]:
    """
    Validate data against a schema.

    Uses jsonschema for validation if available, otherwise
    performs basic structural validation.
    """
    schemas = get_all_schemas()
    if schema_name not in schemas:
        return False, [f"Unknown schema: {schema_name}"]

    schema = schemas[schema_name]

    try:
        import jsonschema

        validator = jsonschema.Draft7Validator(schema)
        errors = list(validator.iter_errors(data))
        if errors:
            return False, [str(e.message) for e in errors]
        return True, []
    except ImportError:
        required = schema.get("required", [])
        missing = [f for f in required if f not in data]
        if missing:
            return False, [f"Missing required fields: {missing}"]
        return True, []
