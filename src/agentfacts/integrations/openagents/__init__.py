"""
OpenAgents integration for AgentFacts SDK.

Provides introspection for OpenAgents agents, networks, and configurations.

Example:
    ```python
    from agentfacts.integrations.openagents import from_agent

    # Create AgentFacts from an OpenAgents agent
    facts = from_agent(my_agent, "My Agent", sign=True)
    ```
"""

from typing import Any

# Re-export main classes for convenience
from agentfacts.core import AgentFacts
from agentfacts.group import GroupFacts
from agentfacts.integrations.openagents.factory import create_group_from_openagents
from agentfacts.integrations.openagents.introspector import (
    # Main integration class
    OpenAgentsIntegration,
    # Introspection functions
    introspect_agent_config,
    introspect_any,
    introspect_network,
    introspect_network_config,
    introspect_tools,
    introspect_worker_agent,
)
from agentfacts.models import (
    AgentFactsCard,
    BaselineModel,
    Capability,
    GroupMetadata,
    OperationalConstraints,
    VerificationResult,
)


def from_agent(agent: Any, name: str, sign: bool = False, **kwargs: Any) -> AgentFacts:
    """
    Convenience function to create AgentFacts from an OpenAgents agent.

    Args:
        agent: OpenAgents agent, config, or network
        name: Agent name
        sign: If True, sign the agent facts
        **kwargs: Additional arguments

    Returns:
        AgentFacts instance

    Example:
        ```python
        from agentfacts.integrations.openagents import from_agent

        facts = from_agent(my_agent, "Research Agent", sign=True)
        ```
    """
    if sign:
        return AgentFacts.from_agent_signed(
            agent, name, framework="openagents", **kwargs
        )
    return AgentFacts.from_agent(agent, name, framework="openagents", **kwargs)


__all__ = [
    # Integration class
    "OpenAgentsIntegration",
    # Introspection functions
    "introspect_agent_config",
    "introspect_any",
    "introspect_network",
    "introspect_network_config",
    "introspect_tools",
    "introspect_worker_agent",
    # Main classes
    "AgentFacts",
    "GroupFacts",
    # Models
    "AgentFactsCard",
    "BaselineModel",
    "Capability",
    "GroupMetadata",
    "OperationalConstraints",
    "VerificationResult",
    # Convenience functions
    "from_agent",
    # Factory function
    "create_group_from_openagents",
]


# Auto-register with global registry
def _register() -> None:
    try:
        from agentfacts.integrations import get_registry

        registry = get_registry()
        registry.register(OpenAgentsIntegration())
    except Exception:
        pass


_register()
