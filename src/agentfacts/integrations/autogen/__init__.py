"""
AutoGen integration for AgentFacts SDK.

Provides introspection for Microsoft AutoGen agents and group chats.

Example:
    ```python
    from agentfacts.integrations.autogen import from_agent

    # Create AgentFacts for an AutoGen agent
    facts = from_agent(my_agent, "Assistant", sign=True)

    # Works with GroupChat and GroupChatManager too
    facts = from_agent(my_group_chat, "Team Chat", sign=True)
    ```
"""

from typing import Any

# Re-export main classes for convenience
from agentfacts.core import AgentFacts
from agentfacts.group import GroupFacts
from agentfacts.integrations.autogen.factory import create_group_from_autogen
from agentfacts.integrations.autogen.introspector import (
    # Main integration class
    AutoGenIntegration,
    # Introspection functions
    introspect_agent,
    introspect_any,
    introspect_group_chat,
    introspect_group_chat_manager,
    introspect_llm_config,
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
    Convenience function to create AgentFacts from an AutoGen agent.

    Args:
        agent: AutoGen agent, GroupChat, or GroupChatManager
        name: Agent name
        sign: If True, sign the agent facts
        **kwargs: Additional arguments

    Returns:
        AgentFacts instance

    Example:
        ```python
        from agentfacts.integrations.autogen import from_agent

        facts = from_agent(assistant, "Research Agent", sign=True)
        ```
    """
    if sign:
        return AgentFacts.from_agent_signed(agent, name, framework="autogen", **kwargs)
    return AgentFacts.from_agent(agent, name, framework="autogen", **kwargs)


__all__ = [
    # Integration class
    "AutoGenIntegration",
    # Introspection functions
    "introspect_agent",
    "introspect_any",
    "introspect_group_chat",
    "introspect_group_chat_manager",
    "introspect_llm_config",
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
    "create_group_from_autogen",
]


# Auto-register with global registry
def _register() -> None:
    try:
        from agentfacts.integrations import get_registry

        registry = get_registry()
        registry.register(AutoGenIntegration())
    except Exception:
        pass


_register()
