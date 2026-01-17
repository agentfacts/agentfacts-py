"""
Example 4: Policy-Based Verification

This example demonstrates Zero Trust policies for evaluating agents:
- Built-in policy rules
- Custom policy creation
- Policy engine usage
- Enterprise security scenarios
"""

from datetime import datetime, timedelta, timezone

from agentfacts import AgentFacts
from agentfacts.models import (
    Attestation,
    BaselineModel,
    Capability,
    ComplianceInfo,
    ModelProvider,
)
from agentfacts.policy import (
    Policy,
    PolicyEngine,
    RequireSignature,
    RequireProvider,
    RequireModel,
    RequireAttestation,
    RequireCapability,
    DenyCapability,
    RequireCompliance,
)
from agentfacts.policy.rules import RequireRiskLevel


def create_sample_agents():
    """Create sample agents with varying trust levels."""

    # Agent 1: Fully compliant enterprise agent
    enterprise_agent = AgentFacts(
        name="Enterprise Data Analyst",
        description="SOC2 compliant data analysis agent",
        baseline_model=BaselineModel(
            name="gpt-4-turbo",
            provider=ModelProvider.OPENAI,
            temperature=0.3,
        ),
        capabilities=[
            Capability(name="sql_query", description="Query databases", risk_level="medium"),
            Capability(name="report_generator", description="Generate reports", risk_level="low"),
        ],
    )
    enterprise_agent.sign()

    # Add compliance info
    enterprise_agent.metadata.policy.compliance = ComplianceInfo(
        frameworks=["SOC2", "GDPR", "EU_AI_ACT"],
        risk_category="limited",
    )

    # Add security attestation
    enterprise_agent.add_attestation(Attestation(
        id="att-001",
        type="security_audit",
        issuer="did:key:zAuditorXYZ",
        subject=enterprise_agent.did,
        issued_at=datetime.now(timezone.utc) - timedelta(days=30),
        claims={"result": "passed", "findings": 0},
    ))
    enterprise_agent.sign()  # Re-sign after adding attestation

    # Agent 2: Basic agent without attestations
    basic_agent = AgentFacts(
        name="Simple Helper",
        description="A basic assistant agent",
        baseline_model=BaselineModel(
            name="gpt-3.5-turbo",
            provider=ModelProvider.OPENAI,
        ),
        capabilities=[
            Capability(name="chat", description="Chat with users", risk_level="low"),
        ],
    )
    basic_agent.sign()

    # Agent 3: High-risk agent with shell access
    risky_agent = AgentFacts(
        name="System Administrator",
        description="Agent with system-level access",
        baseline_model=BaselineModel(
            name="claude-3-opus",
            provider=ModelProvider.ANTHROPIC,
        ),
        capabilities=[
            Capability(name="shell_executor", description="Execute shell commands", risk_level="high"),
            Capability(name="file_manager", description="Manage files", risk_level="high"),
            Capability(name="process_manager", description="Manage processes", risk_level="high"),
        ],
    )
    risky_agent.sign()

    # Agent 4: Unsigned agent
    unsigned_agent = AgentFacts(
        name="Untrusted Agent",
        description="This agent has not been signed",
    )
    # Note: Not signed!

    return {
        "enterprise": enterprise_agent,
        "basic": basic_agent,
        "risky": risky_agent,
        "unsigned": unsigned_agent,
    }


def demonstrate_individual_rules(agents):
    """Show how individual policy rules work."""
    print("\n" + "=" * 60)
    print("Individual Policy Rules")
    print("=" * 60)

    # Rule 1: RequireSignature
    print("\n--- RequireSignature ---")
    rule = RequireSignature()
    for name, agent in agents.items():
        result = rule.evaluate(agent.metadata)
        status = "✓ PASS" if result.passed else "✗ FAIL"
        print(f"  {name:12} : {status}")

    # Rule 2: RequireProvider
    print("\n--- RequireProvider (OpenAI only) ---")
    rule = RequireProvider([ModelProvider.OPENAI])
    for name, agent in agents.items():
        result = rule.evaluate(agent.metadata)
        status = "✓ PASS" if result.passed else "✗ FAIL"
        provider = agent.metadata.agent.model.provider.value
        print(f"  {name:12} : {status} (provider: {provider})")

    # Rule 3: RequireModel
    print("\n--- RequireModel (gpt-4 variants) ---")
    rule = RequireModel(["gpt-4"], allow_prefix_match=True)
    for name, agent in agents.items():
        result = rule.evaluate(agent.metadata)
        status = "✓ PASS" if result.passed else "✗ FAIL"
        model = agent.metadata.agent.model.name
        print(f"  {name:12} : {status} (model: {model})")

    # Rule 4: DenyCapability
    print("\n--- DenyCapability (no shell access) ---")
    rule = DenyCapability(["shell_executor", "shell", "bash"])
    for name, agent in agents.items():
        result = rule.evaluate(agent.metadata)
        status = "✓ PASS" if result.passed else "✗ FAIL"
        caps = [c.name for c in agent.metadata.agent.capabilities]
        print(f"  {name:12} : {status} (capabilities: {caps})")

    # Rule 5: RequireRiskLevel
    print("\n--- RequireRiskLevel (max: medium) ---")
    rule = RequireRiskLevel("medium")
    for name, agent in agents.items():
        result = rule.evaluate(agent.metadata)
        status = "✓ PASS" if result.passed else "✗ FAIL"
        risks = [c.risk_level for c in agent.metadata.agent.capabilities]
        print(f"  {name:12} : {status} (risk levels: {risks})")


def demonstrate_composite_policies(agents):
    """Show how to combine rules into policies."""
    print("\n" + "=" * 60)
    print("Composite Policies")
    print("=" * 60)

    # Policy 1: Basic Trust (just needs signature)
    print("\n--- Policy: Basic Trust ---")
    print("    Rules: RequireSignature")
    policy = Policy.basic_trust()
    for name, agent in agents.items():
        result = policy.evaluate(agent.metadata)
        status = "✓ PASS" if result.passed else "✗ FAIL"
        print(f"  {name:12} : {status}")

    # Policy 2: Enterprise (signature + provider + recent audit)
    print("\n--- Policy: Strict Enterprise ---")
    print("    Rules: RequireSignature + RequireProvider(OpenAI/Anthropic) + RequireAttestation(security_audit, 90 days)")
    policy = Policy.strict_enterprise()
    for name, agent in agents.items():
        result = policy.evaluate(agent.metadata)
        status = "✓ PASS" if result.passed else "✗ FAIL"
        violations = [str(v) for v in result.violations]
        print(f"  {name:12} : {status}")
        if violations:
            for v in violations[:2]:  # Show first 2 violations
                print(f"               └─ {v}")

    # Policy 3: Custom low-risk policy
    print("\n--- Policy: Low Risk Only ---")
    print("    Rules: RequireSignature + RequireRiskLevel(low) + DenyCapability(shell)")
    custom_policy = Policy(
        name="low_risk_only",
        rules=[
            RequireSignature(),
            RequireRiskLevel("low"),
            DenyCapability(["shell_executor", "shell", "sudo"]),
        ],
    )
    for name, agent in agents.items():
        result = custom_policy.evaluate(agent.metadata)
        status = "✓ PASS" if result.passed else "✗ FAIL"
        print(f"  {name:12} : {status}")


def demonstrate_policy_engine(agents):
    """Show the policy engine for centralized policy management."""
    print("\n" + "=" * 60)
    print("Policy Engine")
    print("=" * 60)

    # Create engine with multiple policies
    engine = PolicyEngine()

    # Register policies
    engine.register_policy(Policy.basic_trust())
    engine.register_policy(Policy.strict_enterprise())
    engine.register_policy(Policy(
        name="data_access",
        rules=[
            RequireSignature(),
            RequireProvider([ModelProvider.OPENAI]),
            RequireCompliance(["SOC2"]),
        ],
    ))

    # Set default
    engine.set_default_policy(Policy.basic_trust())

    print("\n--- Registered Policies ---")
    for name in engine.policies:
        print(f"  • {name}")

    # Evaluate enterprise agent against all policies
    print("\n--- Enterprise Agent vs All Policies ---")
    enterprise = agents["enterprise"]
    results = engine.evaluate_all(enterprise.metadata)
    for policy_name, result in results.items():
        status = "✓ PASS" if result.passed else "✗ FAIL"
        print(f"  {policy_name:20} : {status}")

    # Quick trust check
    print("\n--- Quick Trust Checks ---")
    for name, agent in agents.items():
        is_trusted = engine.is_trusted(agent.metadata)
        status = "✓ Trusted" if is_trusted else "✗ Untrusted"
        print(f"  {name:12} : {status}")

    # Generate 403 response for rejected agent
    print("\n--- 403 Response Example ---")
    result = engine.evaluate(agents["unsigned"].metadata, "basic_trust")
    if not result.passed:
        response = engine.create_403_response(result)
        print(f"  Error:   {response['error']}")
        print(f"  Code:    {response['code']}")
        print(f"  Message: {response['message'][:60]}...")


def main():
    print("=" * 60)
    print("AgentFacts SDK - Policy Verification Example")
    print("=" * 60)

    # Create sample agents
    print("\nCreating sample agents...")
    agents = create_sample_agents()
    for name, agent in agents.items():
        signed = "✓ signed" if agent.is_signed else "✗ unsigned"
        print(f"  • {name}: {agent.name} ({signed})")

    # Demonstrate features
    demonstrate_individual_rules(agents)
    demonstrate_composite_policies(agents)
    demonstrate_policy_engine(agents)

    print("\n" + "=" * 60)
    print("Policy verification example complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
