"""
Example 10: Structured Logging and Audit Trail

This example demonstrates:
- Structured logging (human or JSON)
- Contextual log fields for auditability
- Signature, verification, and policy logging
- Evidence log export for audit trails
"""

import logging
import uuid

from agentfacts import AgentFacts
from agentfacts.logging import configure_logging, enable_default_warnings
from agentfacts.models import BaselineModel, Capability, ModelProvider
from agentfacts.policy import Policy


def main():
    print("=" * 60)
    print("AgentFacts SDK - Logging and Audit Example")
    print("=" * 60)

    run_id = uuid.uuid4().hex[:8]

    # Configure structured logging (set json_output=False for human-friendly logs)
    logger = configure_logging(level=logging.INFO, json_output=True)
    logger.set_context(run_id=run_id, environment="local")

    # Enable warnings when defaults are used (helps during introspection)
    enable_default_warnings(True)

    # Create a signed AgentFacts profile
    facts = AgentFacts(
        name="AuditReady Agent",
        description="Demonstrates structured logging and audit trails",
        baseline_model=BaselineModel(
            name="gpt-4-turbo",
            provider=ModelProvider.OPENAI,
            temperature=0.2,
            max_tokens=2048,
        ),
        capabilities=[
            Capability(name="web_search", description="Search the web", risk_level="medium"),
            Capability(name="calculator", description="Perform calculations", risk_level="low"),
        ],
    )

    logger.info(
        "Created AgentFacts profile",
        event_type="agent_created",
        agent_name=facts.name,
        agent_did=facts.did,
    )

    facts.sign()
    logger.log_signature_created(facts.did)

    verification = facts.verify()
    logger.log_signature_verified(
        facts.did,
        verification.valid,
        errors=verification.errors,
    )

    policy = Policy.basic_trust()
    policy_result = policy.evaluate(facts.metadata)
    logger.log_policy_evaluation(
        "basic_trust",
        facts.did,
        policy_result.passed,
        [str(v) for v in policy_result.violations],
    )

    # Record auditable evidence
    facts.log_evidence("security_audit", {
        "auditor": "security-team",
        "result": "passed",
        "scope": ["prompt_injection", "data_exfiltration"],
    })
    facts.log_evidence("runtime_metrics", {
        "window": "last_24h",
        "requests": 421,
        "error_rate": 0.002,
        "p95_latency_ms": 210,
    })

    logger.info(
        "Recorded audit evidence",
        event_type="audit_evidence",
        merkle_root=facts.merkle_root,
        evidence_entries=len(facts.evidence_log.entries),
    )

    # Verify the evidence log and export it for auditors
    evidence_valid = all(
        facts.evidence_log.verify_entry(i, facts.key_pair)
        for i in range(len(facts.evidence_log))
    )

    logger.info(
        "Audit log verification complete",
        event_type="audit_verify",
        valid=evidence_valid,
    )

    audit_json = facts.evidence_log.export_json()
    with open("agent_transparency_log.json", "w") as f:
        f.write(audit_json)

    logger.info(
        "Audit log exported",
        event_type="audit_export",
        path="agent_transparency_log.json",
        bytes=len(audit_json),
    )

    print("\n" + "=" * 60)
    print("Logging and audit example complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
