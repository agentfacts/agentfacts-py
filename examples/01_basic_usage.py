"""
Example 1: Basic AgentFacts Usage

This example demonstrates the fundamental operations:
- Creating an agent identity
- Signing metadata (including fluent API pattern)
- Verifying signatures
- Exporting/importing JSON
- Using convenience factory methods
"""

from agentfacts import AgentFacts, KeyPair, enable_default_warnings
from agentfacts.models import BaselineModel, Capability, ModelProvider

# Optionally enable warnings when defaults are used
# enable_default_warnings(True)


def main():
    print("=" * 60)
    print("AgentFacts SDK - Basic Usage Example")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # 1. Create a simple agent identity
    # -------------------------------------------------------------------------
    print("\n1. Creating agent identity...")

    facts = AgentFacts(
        name="Research Assistant",
        description="An AI agent that helps with research and analysis tasks",
        version="1.0.0",
    )

    print(f"   Agent Name: {facts.name}")
    print(f"   Agent DID:  {facts.did}")
    print(f"   Signed:     {facts.is_signed}")

    # -------------------------------------------------------------------------
    # 2. Create an agent with detailed metadata
    # -------------------------------------------------------------------------
    print("\n2. Creating agent with detailed metadata...")

    detailed_facts = AgentFacts(
        name="Data Analyst Agent",
        description="Analyzes datasets and generates insights",
        baseline_model=BaselineModel(
            name="gpt-4-turbo",
            provider=ModelProvider.OPENAI,
            temperature=0.7,
            max_tokens=4096,
        ),
        capabilities=[
            Capability(
                name="sql_query",
                description="Execute SQL queries against databases",
                risk_level="medium",
            ),
            Capability(
                name="chart_generator",
                description="Generate visualizations from data",
                risk_level="low",
            ),
            Capability(
                name="file_reader",
                description="Read CSV and Excel files",
                risk_level="low",
            ),
        ],
    )

    print(f"   Agent Name:  {detailed_facts.name}")
    print(f"   Model:       {detailed_facts.metadata.agent.model.name}")
    print(f"   Provider:    {detailed_facts.metadata.agent.model.provider.value}")
    print(f"   Capabilities: {len(detailed_facts.metadata.agent.capabilities)}")
    for cap in detailed_facts.metadata.agent.capabilities:
        print(f"     - {cap.name} (risk: {cap.risk_level})")

    # -------------------------------------------------------------------------
    # 3. Sign the metadata (fluent API)
    # -------------------------------------------------------------------------
    print("\n3. Signing agent metadata...")

    # sign() returns self for fluent chaining
    detailed_facts.sign()

    print(f"   Signed:    {detailed_facts.is_signed}")
    print(f"   Signature: {detailed_facts.signature[:50]}...")  # New .signature property

    # -------------------------------------------------------------------------
    # 3b. Fluent API pattern - create, sign, and verify in one chain
    # -------------------------------------------------------------------------
    print("\n3b. Fluent API example (create, sign, verify in one chain)...")

    chained_result = AgentFacts(
        name="Chained Agent",
        baseline_model=BaselineModel(name="gpt-4", provider=ModelProvider.OPENAI),
    ).sign().verify()

    print(f"   Chained verification: {'✓ Valid' if chained_result.valid else '✗ Invalid'}")

    # -------------------------------------------------------------------------
    # 4. Verify the signature
    # -------------------------------------------------------------------------
    print("\n4. Verifying signature...")

    result = detailed_facts.verify()

    print(f"   Valid:    {result.valid}")
    print(f"   DID:      {result.did}")
    if result.errors:
        print(f"   Errors:   {result.errors}")
    if result.warnings:
        print(f"   Warnings: {result.warnings}")

    # -------------------------------------------------------------------------
    # 5. Export to JSON
    # -------------------------------------------------------------------------
    print("\n5. Exporting to JSON...")

    json_output = detailed_facts.to_json()
    print(f"   JSON length: {len(json_output)} characters")

    # Save to file
    with open("agent_metadata.json", "w") as f:
        f.write(json_output)
    print("   Saved to: agent_metadata.json")

    # -------------------------------------------------------------------------
    # 6. Import from JSON
    # -------------------------------------------------------------------------
    print("\n6. Importing from JSON...")

    with open("agent_metadata.json", "r") as f:
        json_data = f.read()

    restored_facts = AgentFacts.from_json(json_data)

    print(f"   Restored Agent: {restored_facts.name}")
    print(f"   DID Match:      {restored_facts.did == detailed_facts.did}")
    print(f"   Still Signed:   {restored_facts.is_signed}")

    # Verify the restored agent
    restore_result = restored_facts.verify()
    print(f"   Verification:   {'✓ Valid' if restore_result.valid else '✗ Invalid'}")

    # -------------------------------------------------------------------------
    # 7. Using a custom key pair
    # -------------------------------------------------------------------------
    print("\n7. Using a custom key pair...")

    # Generate a key pair separately
    key_pair = KeyPair.generate()
    print(f"   Generated Key Pair")
    print(f"   Public Key: {key_pair.public_key_base64[:40]}...")

    # Create agent with this key pair
    custom_agent = AgentFacts(
        name="Custom Key Agent",
        key_pair=key_pair,
    )

    print(f"   Agent DID: {custom_agent.did}")

    # The DID is derived from the public key
    from agentfacts.crypto.did import DID
    expected_did = DID.from_key_pair(key_pair)
    print(f"   DID matches key: {custom_agent.did == expected_did.uri}")

    print("\n" + "=" * 60)
    print("Basic usage example complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
