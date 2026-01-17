"""
LlamaIndex integration for AgentFacts SDK.

Provides introspection for LlamaIndex query engines, agents, and indexes.

Example:
    ```python
    from agentfacts.integrations.llamaindex import from_agent

    # Create AgentFacts from a LlamaIndex agent
    facts = from_agent(my_agent, "Research Agent", sign=True)
    ```
"""

from typing import Any

# Re-export main classes for convenience
from agentfacts.core import AgentFacts
from agentfacts.integrations.llamaindex.factory import create_group_from_llamaindex
from agentfacts.integrations.llamaindex.introspector import (
    # Main integration class
    LlamaIndexIntegration,
    # Introspection functions
    introspect_agent,
    introspect_any,
    introspect_index,
    introspect_llm,
    introspect_query_engine,
    introspect_tools,
)
from agentfacts.models import (
    AgentFactsCard,
    BaselineModel,
    Capability,
    OperationalConstraints,
    VerificationResult,
)


def from_agent(agent: Any, name: str, sign: bool = False, **kwargs: Any) -> AgentFacts:
    """
    Convenience function to create AgentFacts from a LlamaIndex agent.

    Args:
        agent: LlamaIndex Agent, QueryEngine, or Index
        name: Agent name
        sign: If True, sign the agent facts
        **kwargs: Additional arguments

    Returns:
        AgentFacts instance

    Example:
        ```python
        from agentfacts.integrations.llamaindex import from_agent

        facts = from_agent(my_agent, "Research Agent", sign=True)
        ```
    """
    if sign:
        return AgentFacts.from_agent_signed(
            agent, name, framework="llamaindex", **kwargs
        )
    return AgentFacts.from_agent(agent, name, framework="llamaindex", **kwargs)


__all__ = [
    # Integration class
    "LlamaIndexIntegration",
    # Introspection functions
    "introspect_agent",
    "introspect_any",
    "introspect_index",
    "introspect_llm",
    "introspect_query_engine",
    "introspect_tools",
    # Main classes
    "AgentFacts",
    # Models
    "AgentFactsCard",
    "BaselineModel",
    "Capability",
    "OperationalConstraints",
    "VerificationResult",
    # Convenience functions
    "from_agent",
    # Factory function
    "create_group_from_llamaindex",
]


# Auto-register with global registry
def _register() -> None:
    try:
        from agentfacts.integrations import get_registry

        registry = get_registry()
        registry.register(LlamaIndexIntegration())
    except Exception:
        pass


_register()
