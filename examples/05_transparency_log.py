"""
Example 5: Transparency Log and Evidence Logging

This example demonstrates:
- Logging evidence to the Merkle tree
- Generating and verifying inclusion proofs
- Building audit trails
- Using the transparency log for compliance
"""

from datetime import datetime, timezone

from agentfacts import AgentFacts
from agentfacts.merkle import MerkleTree, TransparencyLog
from agentfacts.crypto.keys import KeyPair
from agentfacts.models import BaselineModel, ModelProvider, Attestation


def demonstrate_merkle_tree():
    """Show basic Merkle tree operations."""
    print("\n" + "=" * 60)
    print("Merkle Tree Basics")
    print("=" * 60)

    # Create a tree
    tree = MerkleTree()
    print(f"\nEmpty tree root: {tree.root_hex}")

    # Add some data
    print("\nAdding data to tree...")
    data_items = [
        b"Transaction 1: Agent initialized",
        b"Transaction 2: Tool invoked - web_search",
        b"Transaction 3: Tool invoked - calculator",
        b"Transaction 4: Response generated",
        b"Transaction 5: Session completed",
    ]

    for i, data in enumerate(data_items):
        index = tree.append(data)
        print(f"  [{index}] Added: {data.decode()[:40]}...")
        print(f"      New root: {tree.root_hex[:32]}...")

    # Generate and verify proofs
    print("\n--- Inclusion Proofs ---")
    for i in range(len(tree)):
        proof = tree.get_proof(i)
        is_valid = MerkleTree.verify_proof(proof, data_items[i])
        print(f"  Proof for item {i}: {'✓ Valid' if is_valid else '✗ Invalid'}")

    # Demonstrate tamper detection
    print("\n--- Tamper Detection ---")
    proof = tree.get_proof(2)
    tampered_data = b"TAMPERED: This is not the original data"
    is_valid = MerkleTree.verify_proof(proof, tampered_data)
    print(f"  Original data: ✓ Valid")
    print(f"  Tampered data: {'✓ Valid' if is_valid else '✗ DETECTED - Invalid proof'}")


def demonstrate_transparency_log():
    """Show the transparency log for evidence tracking."""
    print("\n" + "=" * 60)
    print("Transparency Log")
    print("=" * 60)

    # Create a key pair for signing
    key_pair = KeyPair.generate()
    agent_did = "did:key:zAgent123"

    # Create log
    log = TransparencyLog(agent_did=agent_did)
    print(f"\nCreated log for agent: {agent_did}")

    # Log various events
    print("\nLogging events...")

    # Event 1: Initialization
    entry1 = log.append(
        entry_type="agent_initialized",
        data={
            "model": "gpt-4",
            "capabilities": ["search", "calculator"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        issuer_did=agent_did,
        key_pair=key_pair,
    )
    print(f"  [{entry1.index}] agent_initialized")
    print(f"      Merkle root: {log.root[:32]}...")

    # Event 2: Tool invocation
    entry2 = log.append(
        entry_type="tool_invocation",
        data={
            "tool": "web_search",
            "query": "latest AI news",
            "duration_ms": 1234,
        },
        issuer_did=agent_did,
        key_pair=key_pair,
    )
    print(f"  [{entry2.index}] tool_invocation")
    print(f"      Merkle root: {log.root[:32]}...")

    # Event 3: Security check
    entry3 = log.append(
        entry_type="security_check",
        data={
            "check_type": "output_filtering",
            "result": "passed",
            "blocked_patterns": 0,
        },
        issuer_did=agent_did,
        key_pair=key_pair,
    )
    print(f"  [{entry3.index}] security_check")
    print(f"      Merkle root: {log.root[:32]}...")

    # Event 4: External attestation
    entry4 = log.append(
        entry_type="external_attestation",
        data={
            "auditor": "did:key:zAuditorXYZ",
            "audit_type": "compliance_review",
            "result": "compliant",
            "frameworks": ["SOC2", "GDPR"],
        },
        issuer_did="did:key:zAuditorXYZ",  # External issuer
        key_pair=key_pair,
    )
    print(f"  [{entry4.index}] external_attestation")
    print(f"      Merkle root: {log.root[:32]}...")

    # Verify entries
    print("\n--- Entry Verification ---")
    for i in range(len(log)):
        entry = log.entries[i]
        is_valid = log.verify_entry(i, key_pair)
        print(f"  Entry {i} ({entry.type}): {'✓ Valid' if is_valid else '✗ Invalid'}")

    # Export and import
    print("\n--- Serialization ---")
    json_export = log.export_json()
    print(f"  Exported to JSON: {len(json_export)} bytes")

    restored_log = TransparencyLog.import_json(json_export)
    print(f"  Restored entries: {len(restored_log)}")
    print(f"  Roots match: {restored_log.root == log.root}")


def demonstrate_agent_evidence_logging():
    """Show evidence logging integrated with AgentFacts."""
    print("\n" + "=" * 60)
    print("AgentFacts Evidence Logging")
    print("=" * 60)

    # Create agent
    facts = AgentFacts(
        name="Auditable Agent",
        description="An agent with full audit trail",
        baseline_model=BaselineModel(name="gpt-4", provider=ModelProvider.OPENAI),
    )

    print(f"\nAgent: {facts.name}")
    print(f"DID:   {facts.did}")
    print(f"Initial Merkle root: {facts.merkle_root}")

    # Log various evidence
    print("\nLogging evidence...")

    # 1. Security scan
    facts.log_evidence("security_scan", {
        "scanner": "OWASP ZAP",
        "vulnerabilities_found": 0,
        "scan_duration_seconds": 45,
    })
    print(f"  Logged: security_scan")
    print(f"  Merkle root: {facts.merkle_root[:32]}...")

    # 2. Performance test
    facts.log_evidence("performance_test", {
        "test_type": "load_test",
        "requests_per_second": 1000,
        "p99_latency_ms": 150,
        "error_rate": 0.001,
    })
    print(f"  Logged: performance_test")
    print(f"  Merkle root: {facts.merkle_root[:32]}...")

    # 3. Compliance check
    facts.log_evidence("compliance_check", {
        "framework": "EU_AI_ACT",
        "category": "limited_risk",
        "requirements_met": True,
        "documentation_complete": True,
    })
    print(f"  Logged: compliance_check")
    print(f"  Merkle root: {facts.merkle_root[:32]}...")

    # 4. Runtime monitoring
    facts.log_evidence("runtime_metrics", {
        "period": "2024-01-15T00:00:00Z/2024-01-15T23:59:59Z",
        "total_requests": 15423,
        "successful_requests": 15398,
        "blocked_requests": 25,
        "average_response_time_ms": 89,
    })
    print(f"  Logged: runtime_metrics")
    print(f"  Merkle root: {facts.merkle_root[:32]}...")

    # Sign the agent metadata (adds log proof)
    facts.sign()
    print(f"\nCard signed with log root: {facts.metadata.log_proof.root_hash[:32]}...")

    # Export final state
    print("\n--- Final Agent State ---")
    metadata = facts.metadata
    print(f"  Name:           {metadata.agent.name}")
    print(f"  DID:            {metadata.agent.id}")
    print(f"  Signed:         {facts.is_signed}")
    print(f"  Card Log Root:  {metadata.log_proof.root_hash[:32]}...")
    print(f"  Evidence Count: {len(facts.evidence_log.entries)} entries")


def demonstrate_audit_scenario():
    """Show a realistic audit scenario."""
    print("\n" + "=" * 60)
    print("Audit Scenario: Compliance Review")
    print("=" * 60)

    # Setup: Production agent with history
    facts = AgentFacts(
        name="Production Agent v2.1",
        baseline_model=BaselineModel(name="gpt-4-turbo", provider=ModelProvider.OPENAI),
    )

    # Simulate historical evidence (in production, this builds up over time)
    historical_events = [
        ("deployment", {"version": "2.1.0", "environment": "production"}),
        ("security_audit", {"auditor": "SecurityCorp", "passed": True}),
        ("incident_response", {"incident_id": "INC-001", "resolved": True, "root_cause": "rate_limit"}),
        ("capability_added", {"capability": "database_query", "approval": "admin@company.com"}),
        ("compliance_review", {"framework": "SOC2", "status": "compliant"}),
    ]

    print("\nHistorical evidence trail:")
    for event_type, data in historical_events:
        facts.log_evidence(event_type, data)
        print(f"  • {event_type}")

    facts.sign()

    # Auditor verification
    print("\n--- Auditor Verification Process ---")

    # 1. Verify signature
    result = facts.verify()
    print(f"1. Signature verification: {'✓ Valid' if result.valid else '✗ Invalid'}")

    # 2. Check log proof is included
    print(f"2. Log proof in card: {'✓ Present' if facts.metadata.log_proof else '✗ Missing'}")

    # 3. Verify evidence integrity
    all_valid = all(
        facts.evidence_log.verify_entry(i)
        for i in range(len(facts.evidence_log))
    )
    print(f"3. Evidence integrity: {'✓ All entries valid' if all_valid else '✗ Tampered entries detected'}")

    # 4. Generate audit report
    print("\n--- Audit Report ---")
    print(f"  Agent:          {facts.name}")
    print(f"  DID:            {facts.did}")
    print(f"  Model:          {facts.metadata.agent.model.name}")
    print(f"  Evidence Count: {len(facts.evidence_log.entries)}")
    print(f"  Merkle Root:    {facts.merkle_root}")
    print(f"  Signature:      {facts.signature[:40]}...")
    print(f"\n  Verdict: ✓ Agent passes audit requirements")


def main():
    print("=" * 60)
    print("AgentFacts SDK - Transparency Log Example")
    print("=" * 60)

    demonstrate_merkle_tree()
    demonstrate_transparency_log()
    demonstrate_agent_evidence_logging()
    demonstrate_audit_scenario()

    print("\n" + "=" * 60)
    print("Transparency log example complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
