"""
CrewAI integration for AgentFacts SDK.

Provides introspection for CrewAI Agent and Crew objects.

Example:
    ```python
    from agentfacts.integrations.crewai import from_agent, from_crew

    # Create AgentFacts for a single agent
    facts = from_agent(my_agent, "My Agent", sign=True)

    # Create GroupFacts for an entire crew
    group = from_crew(my_crew, "My Crew", sign=True)
    ```
"""

from typing import Any

# Re-export main classes for convenience
from agentfacts.core import AgentFacts
from agentfacts.group import GroupFacts
from agentfacts.integrations.crewai.factory import create_group_from_crew
from agentfacts.integrations.crewai.introspector import (
    # Main integration class
    CrewAIIntegration,
    # Introspection functions
    introspect_agent,
    introspect_any,
    introspect_crew,
    introspect_llm,
    introspect_tools,
)
from agentfacts.models import (
    AgentFactsCard,
    AgentRole,
    BaselineModel,
    Capability,
    DelegationPolicy,
    GroupMetadata,
    OperationalConstraints,
    VerificationResult,
)


def from_agent(agent: Any, name: str, sign: bool = False, **kwargs: Any) -> AgentFacts:
    """
    Convenience function to create AgentFacts from a CrewAI Agent.

    Args:
        agent: CrewAI Agent instance
        name: Agent name
        sign: If True, sign the agent facts
        **kwargs: Additional arguments

    Returns:
        AgentFacts instance

    Example:
        ```python
        from agentfacts.integrations.crewai import from_agent

        facts = from_agent(my_agent, "Researcher", sign=True)
        ```
    """
    if sign:
        return AgentFacts.from_agent_signed(agent, name, framework="crewai", **kwargs)
    return AgentFacts.from_agent(agent, name, framework="crewai", **kwargs)


def from_crew(
    crew: Any, name: str | None = None, sign: bool = False, **kwargs: Any
) -> GroupFacts:
    """
    Convenience function to create GroupFacts from a CrewAI Crew.

    Args:
        crew: CrewAI Crew instance
        name: Optional group name
        sign: If True, sign the group and all members
        **kwargs: Additional arguments

    Returns:
        GroupFacts instance

    Example:
        ```python
        from agentfacts.integrations.crewai import from_crew

        group = from_crew(my_crew, sign=True)
        ```
    """
    if sign:
        return GroupFacts.from_crewai_signed(crew, name, **kwargs)
    return GroupFacts.from_crewai(crew, name, **kwargs)


__all__ = [
    # Integration class
    "CrewAIIntegration",
    # Introspection functions
    "introspect_agent",
    "introspect_any",
    "introspect_crew",
    "introspect_llm",
    "introspect_tools",
    # Main classes
    "AgentFacts",
    "GroupFacts",
    # Models
    "AgentFactsCard",
    "AgentRole",
    "BaselineModel",
    "Capability",
    "DelegationPolicy",
    "GroupMetadata",
    "OperationalConstraints",
    "VerificationResult",
    # Convenience functions
    "from_agent",
    "from_crew",
    # Factory function
    "create_group_from_crew",
]


# Auto-register with global registry
def _register() -> None:
    try:
        from agentfacts.integrations import get_registry

        registry = get_registry()
        registry.register(CrewAIIntegration())
    except Exception:
        pass


_register()
