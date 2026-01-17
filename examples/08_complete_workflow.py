"""
Example 8: Complete End-to-End Workflow

This example brings together all the features in a realistic scenario:

Scenario: A company deploys an AI agent that needs to:
1. Be registered with verifiable identity
2. Pass security audit
3. Connect to a protected data service
4. Log all actions for compliance

This demonstrates the full "SSL of the Agentic Web" concept.
"""

import json
from datetime import datetime, timedelta, timezone

from agentfacts import AgentFacts, KeyPair
from agentfacts.models import (
    BaselineModel,
    Capability,
    ModelProvider,
    Attestation,
    ComplianceInfo,
)
from agentfacts.policy import (
    Policy,
    PolicyEngine,
    RequireSignature,
    RequireProvider,
    RequireAttestation,
    RequireCompliance,
    DenyCapability,
)


def step_1_create_agent():
    """Step 1: Create and configure the agent."""
    print("\n" + "=" * 60)
    print("STEP 1: Create Agent Identity")
    print("=" * 60)

    # In production, you would load this from secure storage
    key_pair = KeyPair.generate()

    # Create the agent with full metadata
    agent = AgentFacts(
        name="Enterprise Data Analyst v2.0",
        description="Production AI agent for financial data analysis",
        version="2.0.0",
        key_pair=key_pair,
        baseline_model=BaselineModel(
            name="gpt-4-turbo-2024-04-09",
            provider=ModelProvider.OPENAI,
            temperature=0.3,  # Low temperature for consistency
            max_tokens=4096,
        ),
        capabilities=[
            Capability(
                name="sql_query",
                description="Execute read-only SQL queries against approved databases",
                risk_level="medium",
                parameters={"allowed_tables": ["transactions", "accounts", "reports"]},
            ),
            Capability(
                name="report_generator",
                description="Generate PDF and Excel reports from query results",
                risk_level="low",
            ),
            Capability(
                name="email_sender",
                description="Send reports to approved recipients",
                risk_level="medium",
                requires_approval=True,
            ),
        ],
    )

    # Set compliance info
    agent.metadata.policy.compliance = ComplianceInfo(
        frameworks=["SOC2", "GDPR", "EU_AI_ACT"],
        risk_category="limited",
    )

    print(f"  Agent Created: {agent.name}")
    print(f"  DID: {agent.did}")
    print(f"  Model: {agent.metadata.agent.model.name}")
    print(f"  Capabilities: {len(agent.metadata.agent.capabilities)}")
    print(f"  Compliance: {', '.join(agent.metadata.policy.compliance.frameworks)}")

    return agent, key_pair


def step_2_security_audit(agent, auditor_key):
    """Step 2: Pass security audit and get attestation."""
    print("\n" + "=" * 60)
    print("STEP 2: Security Audit & Attestation")
    print("=" * 60)

    from agentfacts.crypto.did import DID

    # Auditor creates their identity
    auditor_did = DID.from_key_pair(auditor_key)

    print(f"  Auditor DID: {auditor_did.uri}")

    # Simulate security audit checks
    audit_checks = [
        ("Input Validation", True),
        ("Output Filtering", True),
        ("Rate Limiting", True),
        ("Prompt Injection Protection", True),
        ("Data Leakage Prevention", True),
    ]

    print("\n  Security Audit Results:")
    all_passed = True
    for check, passed in audit_checks:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"    {check}: {status}")
        if not passed:
            all_passed = False

    if all_passed:
        # Create attestation data for signing
        from agentfacts.crypto import canonicalize_json

        attestation_data = {
            "id": f"audit-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
            "type": "security_audit",
            "issuer": auditor_did.uri,
            "subject": agent.did,
            "issued_at": datetime.now(timezone.utc).isoformat() + "Z",
            "expires_at": (datetime.now(timezone.utc) + timedelta(days=90)).isoformat() + "Z",
            "claims": {
                "audit_type": "comprehensive_security_review",
                "checks_performed": len(audit_checks),
                "checks_passed": sum(1 for _, p in audit_checks if p),
                "auditor_certification": "ISO27001",
                "next_audit_due": (datetime.now(timezone.utc) + timedelta(days=90)).isoformat(),
            },
        }

        # Auditor signs the attestation
        canonical = canonicalize_json(attestation_data)
        signature = auditor_key.sign_base64(canonical)

        # Create attestation with signature
        attestation = Attestation(
            id=attestation_data["id"],
            type="security_audit",
            issuer=auditor_did.uri,
            subject=agent.did,
            issued_at=datetime.now(timezone.utc),
            expires_at=datetime.now(timezone.utc) + timedelta(days=90),
            claims=attestation_data["claims"],
            signature=signature,
        )

        # Add attestation to agent
        agent.add_attestation(attestation)

        print(f"\n  ✓ Security Audit Passed")
        print(f"  Attestation ID: {attestation.id}")
        print(f"  Valid until: {attestation.expires_at}")

    return agent


def step_3_sign_metadata(agent, key_pair):
    """Step 3: Sign the complete metadata."""
    print("\n" + "=" * 60)
    print("STEP 3: Sign Agent Metadata")
    print("=" * 60)

    # Log evidence of signing event
    agent.log_evidence("metadata_signed", {
        "version": agent.metadata.agent.version,
        "capabilities_count": len(agent.metadata.agent.capabilities),
        "attestations_count": len(agent.metadata.attestations),
    })

    # Sign the metadata
    agent.sign(key_pair)

    print(f"  Metadata signed: {agent.is_signed}")
    print(f"  Signature: {agent.signature[:50]}...")
    print(f"  Card Log Root: {agent.metadata.log_proof.root_hash}")

    return agent


def step_4_service_verification(agent):
    """Step 4: Service verifies the agent before granting access."""
    print("\n" + "=" * 60)
    print("STEP 4: Service-Side Verification")
    print("=" * 60)

    # Service defines its access policy
    service_policy = Policy(
        name="data_service_access",
        rules=[
            RequireSignature(),
            RequireProvider([ModelProvider.OPENAI, ModelProvider.ANTHROPIC]),
            RequireAttestation("security_audit", max_age_days=90),
            RequireCompliance(["SOC2"]),
            DenyCapability(["shell", "code_executor", "sudo"]),
        ],
    )

    print("  Service Policy: data_service_access")
    print("  Requirements:")
    print("    • Valid signature")
    print("    • Provider: OpenAI or Anthropic")
    print("    • Security audit within 90 days")
    print("    • SOC2 compliance")
    print("    • No shell/code execution capabilities")

    # Verify the agent
    print("\n  Verifying agent...")

    # First, verify signature
    sig_result = agent.verify()
    print(f"    Signature: {'✓ Valid' if sig_result.valid else '✗ Invalid'}")

    # Then, evaluate policy
    policy_result = service_policy.evaluate(agent.metadata)
    print(f"    Policy: {'✓ Passed' if policy_result.passed else '✗ Failed'}")

    if policy_result.violations:
        for v in policy_result.violations:
            print(f"      └─ {v}")

    if sig_result.valid and policy_result.passed:
        print("\n  ✓ Agent APPROVED for data service access")
        return True
    else:
        print("\n  ✗ Agent DENIED")
        return False


def step_5_authenticated_session(agent, key_pair):
    """Step 5: Establish authenticated session and perform work."""
    print("\n" + "=" * 60)
    print("STEP 5: Authenticated Work Session")
    print("=" * 60)

    # Create handshake with service
    from agentfacts.models import HandshakeChallenge

    # Service sends challenge
    print("  [Service] Sending handshake challenge...")
    service = AgentFacts(name="Data Service")
    challenge = service.create_challenge()
    print(f"    Challenge nonce: {challenge.nonce[:30]}...")

    # Agent responds
    print("\n  [Agent] Responding to challenge...")
    response = agent.respond_to_challenge(challenge)
    print(f"    Response signature: {response.signature[:30]}...")

    # Service verifies
    print("\n  [Service] Verifying response...")
    verify_result = service.verify_response(challenge, response)
    print(f"    Verification: {'✓ Valid' if verify_result.valid else '✗ Invalid'}")

    if verify_result.valid:
        print("\n  ✓ Authenticated session established")

        # Simulate work with audit logging
        print("\n  Performing work (with audit logging)...")

        # Work item 1
        agent.log_evidence("tool_invocation", {
            "tool": "sql_query",
            "query_hash": "abc123",  # Hash, not actual query for privacy
            "result_rows": 150,
            "duration_ms": 234,
        })
        print("    • Executed SQL query (150 rows)")

        # Work item 2
        agent.log_evidence("tool_invocation", {
            "tool": "report_generator",
            "format": "pdf",
            "pages": 12,
            "duration_ms": 1567,
        })
        print("    • Generated PDF report (12 pages)")

        # Work item 3
        agent.log_evidence("action_approval_requested", {
            "action": "email_sender",
            "recipient_count": 3,
            "approved_by": "admin@company.com",
        })
        print("    • Email sending approved by admin")

        agent.log_evidence("tool_invocation", {
            "tool": "email_sender",
            "recipients": 3,
            "status": "sent",
        })
        print("    • Sent report to 3 recipients")

        print(f"\n  Session complete. Merkle root: {agent.merkle_root[:32]}...")

    return agent


def step_6_export_audit_trail(agent):
    """Step 6: Export complete audit trail."""
    print("\n" + "=" * 60)
    print("STEP 6: Export Audit Trail")
    print("=" * 60)

    # Re-sign to capture final merkle root
    agent.sign()

    # Export metadata
    metadata_json = agent.to_json()

    # Export transparency log
    log_json = agent.evidence_log.export_json()

    print(f"  Agent Metadata: {len(metadata_json)} bytes")
    print(f"  Transparency Log: {len(log_json)} bytes")
    print(f"  Total Evidence Entries: {len(agent.evidence_log.entries)}")

    # Show evidence summary
    print("\n  Evidence Summary:")
    event_types = {}
    for entry in agent.evidence_log.entries:
        event_types[entry.type] = event_types.get(entry.type, 0) + 1

    for event_type, count in sorted(event_types.items()):
        print(f"    • {event_type}: {count}")

    # Save files
    with open("agent_final_metadata.json", "w") as f:
        f.write(metadata_json)
    print("\n  Saved: agent_final_metadata.json")

    with open("agent_transparency_log.json", "w") as f:
        f.write(log_json)
    print("  Saved: agent_transparency_log.json")

    return metadata_json, log_json


def step_7_third_party_verification(metadata_json):
    """Step 7: Third party can verify the agent independently."""
    print("\n" + "=" * 60)
    print("STEP 7: Third-Party Verification")
    print("=" * 60)

    print("  A third party receives agent_final_metadata.json")
    print("  They can verify without any shared secrets:\n")

    # Load from JSON (simulating third party)
    restored_agent = AgentFacts.from_json(metadata_json)

    # Verify signature
    print("  1. Verify cryptographic signature...")
    result = restored_agent.verify()
    print(f"     Result: {'✓ Valid' if result.valid else '✗ Invalid'}")
    print(f"     DID: {result.did}")

    # Verify DID derivation
    print("\n  2. Verify DID matches public key...")
    from agentfacts.crypto.did import DID
    from agentfacts.crypto.keys import KeyPair

    pk = KeyPair.from_public_key_base64(restored_agent.public_key)
    expected_did = DID.from_key_pair(pk)
    matches = expected_did.uri == restored_agent.did
    print(f"     Result: {'✓ Matches' if matches else '✗ Mismatch'}")

    # Check attestations
    print("\n  3. Check attestations...")
    for att in restored_agent.metadata.attestations:
        print(f"     • {att.type} from {att.issuer[:30]}...")
        print(f"       Issued: {att.issued_at.strftime('%Y-%m-%d')}")
        expired = datetime.now(timezone.utc) > att.expires_at if att.expires_at else False
        print(f"       Status: {'✗ Expired' if expired else '✓ Valid'}")

    # Evaluate against policy
    print("\n  4. Evaluate against standard policy...")
    engine = PolicyEngine.with_defaults()
    policy_result = engine.evaluate(restored_agent.metadata, "basic_trust")
    print(f"     Basic Trust: {'✓ Passed' if policy_result.passed else '✗ Failed'}")

    print("\n  ✓ Third-party verification complete")
    print("    The agent's identity and history are cryptographically verified")


def main():
    print("=" * 80)
    print("AgentFacts SDK - Complete End-to-End Workflow")
    print("=" * 80)
    print("""
Scenario: Enterprise AI Agent Deployment

An organization is deploying an AI agent that needs:
• Verifiable identity (DID)
• Security audit certification
• Access to protected data services
• Complete audit trail for compliance
""")

    # Create auditor key (in real scenario, this is a trusted third party)
    auditor_key = KeyPair.generate()

    # Execute workflow
    agent, key_pair = step_1_create_agent()
    agent = step_2_security_audit(agent, auditor_key)
    agent = step_3_sign_metadata(agent, key_pair)
    approved = step_4_service_verification(agent)

    if approved:
        agent = step_5_authenticated_session(agent, key_pair)
        metadata_json, log_json = step_6_export_audit_trail(agent)
        step_7_third_party_verification(metadata_json)

    print("\n" + "=" * 80)
    print("Workflow Complete!")
    print("=" * 80)
    print("""
Summary:
• Agent identity created with DID: did:key:z6Mk...
• Security audit passed and attested
• Metadata cryptographically signed
• Service access granted via policy verification
• Authenticated session with mutual handshake
• All actions logged to Merkle tree
• Audit trail exportable and independently verifiable

This demonstrates the complete "SSL of the Agentic Web" concept:
Trust through cryptographic verification, not blind faith.
""")


if __name__ == "__main__":
    main()
