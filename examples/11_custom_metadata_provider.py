"""
Example 11: Custom Metadata Provider

This example demonstrates:
- Implementing a metadata provider that resolves AgentFacts by DID
- Storing signed metadata in-memory
- Using the provider to rehydrate and verify an agent profile

The same provider can be passed into the FastAPI/Flask middleware
for full verification in production.
"""

from __future__ import annotations

from typing import Any

from agentfacts import AgentFacts
from agentfacts.plugins import MetadataProvider
from agentfacts.models import BaselineModel, Capability, ModelProvider


class InMemoryMetadataProvider(MetadataProvider):
    """Simple in-memory metadata provider for demos and tests."""

    def __init__(self) -> None:
        self._store: dict[str, dict[str, Any]] = {}

    def upsert(self, facts: AgentFacts) -> None:
        """Store a signed AgentFacts profile by DID."""
        self._store[facts.did] = facts.to_dict()

    def resolve(self, did: str, *_args: Any, **_kwargs: Any) -> dict[str, Any] | None:
        """Return metadata for the given DID or None if missing."""
        return self._store.get(did)


def main() -> None:
    print("=" * 60)
    print("AgentFacts SDK - Custom Metadata Provider Example")
    print("=" * 60)

    # Create and sign an AgentFacts profile
    facts = AgentFacts(
        name="Metadata Provider Agent",
        description="Stored and resolved via a custom provider",
        baseline_model=BaselineModel(
            name="gpt-4-turbo",
            provider=ModelProvider.OPENAI,
        ),
        capabilities=[
            Capability(name="search", description="Search the web", risk_level="medium"),
            Capability(name="summarize", description="Summarize results", risk_level="low"),
        ],
    )
    facts.sign()

    provider = InMemoryMetadataProvider()
    provider.upsert(facts)

    # Resolve metadata via the provider
    raw_metadata = provider.resolve(facts.did)
    if not raw_metadata:
        raise RuntimeError("Metadata missing from provider")

    restored = AgentFacts.from_dict(raw_metadata)
    verification = restored.verify()

    print(f"Agent DID: {restored.did}")
    print(f"Verified:  {verification.valid}")
    print(f"Model:     {restored.metadata.agent.model.name}")
    print(f"Tools:     {len(restored.metadata.agent.capabilities)}")

    print("\n" + "=" * 60)
    print("Custom metadata provider example complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
