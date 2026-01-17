"""
Hugging Face integration for AgentFacts SDK.

Provides introspection for Hugging Face agent frameworks:
- smolagents: CodeAgent, ToolCallingAgent, MultiStepAgent
- tiny-agents: agent.json configuration files
- Hub agents: agents loaded from/pushed to HF Hub

Example:
    ```python
    from agentfacts.integrations.huggingface import from_agent

    # Create AgentFacts from a smolagents agent
    facts = from_agent(my_agent, "Code Agent", sign=True)

    # Load and introspect a tiny-agents config
    from agentfacts.integrations.huggingface import load_tiny_agents_config
    config = load_tiny_agents_config("./my-agent")
    ```
"""

from typing import Any

# Re-export main classes for convenience
from agentfacts.core import AgentFacts
from agentfacts.integrations.huggingface.factory import create_group_from_huggingface
from agentfacts.integrations.huggingface.introspector import (
    # Main integration class
    HuggingFaceIntegration,
    # Introspection functions
    introspect_any,
    introspect_smolagent,
    introspect_tiny_agents_config,
    introspect_tools,
    # Utility functions
    load_tiny_agents_config,
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
    Convenience function to create AgentFacts from a Hugging Face agent.

    Args:
        agent: Hugging Face agent, config dict, or path to agent.json
        name: Agent name
        sign: If True, sign the agent facts
        **kwargs: Additional arguments

    Returns:
        AgentFacts instance

    Example:
        ```python
        from agentfacts.integrations.huggingface import from_agent

        facts = from_agent(my_agent, "Code Agent", sign=True)
        ```
    """
    if sign:
        return AgentFacts.from_agent_signed(
            agent, name, framework="huggingface", **kwargs
        )
    return AgentFacts.from_agent(agent, name, framework="huggingface", **kwargs)


__all__ = [
    # Integration class
    "HuggingFaceIntegration",
    # Introspection functions
    "introspect_any",
    "introspect_smolagent",
    "introspect_tiny_agents_config",
    "introspect_tools",
    # Utility functions
    "load_tiny_agents_config",
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
    "create_group_from_huggingface",
]


# Auto-register with global registry
def _register() -> None:
    try:
        from agentfacts.integrations import get_registry

        registry = get_registry()
        registry.register(HuggingFaceIntegration())
    except Exception:
        pass


_register()
