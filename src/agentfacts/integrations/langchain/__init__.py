"""
LangChain integration for AgentFacts SDK.

Provides automatic introspection of LangChain agents, chains, tools,
LCEL runnables, and LangGraph graphs to extract metadata.

Example:
    ```python
    from agentfacts.integrations.langchain import from_agent

    # Create AgentFacts from a LangChain agent
    facts = from_agent(my_agent, "Research Agent", sign=True)
    print(facts.to_json())

    # Use the callback handler for runtime tracking
    from agentfacts.integrations.langchain import AgentFactsCallbackHandler

    handler = AgentFactsCallbackHandler(agent_facts=facts)
    result = agent.invoke({"input": "query"}, callbacks=[handler])
    ```
"""

from typing import Any

# Re-export main classes for convenience
from agentfacts.core import AgentFacts
from agentfacts.integrations.langchain.callback import AgentFactsCallbackHandler
from agentfacts.integrations.langchain.introspector import (
    LangChainIntegration,
    introspect_agent,
    introspect_any,
    introspect_chain,
    introspect_langgraph,
    introspect_llm,
    introspect_runnable,
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
    Convenience function to create AgentFacts from a LangChain agent.

    Args:
        agent: LangChain Agent, AgentExecutor, Chain, or Runnable
        name: Agent name
        sign: If True, sign the agent facts
        **kwargs: Additional arguments

    Returns:
        AgentFacts instance

    Example:
        ```python
        from agentfacts.integrations.langchain import from_agent

        facts = from_agent(my_agent, "Research Agent", sign=True)
        print(facts.to_json())
        ```
    """
    if sign:
        return AgentFacts.from_langchain_signed(agent, name, **kwargs)
    return AgentFacts.from_langchain(agent, name, **kwargs)


__all__ = [
    # Introspection functions
    "introspect_agent",
    "introspect_any",
    "introspect_chain",
    "introspect_langgraph",
    "introspect_llm",
    "introspect_runnable",
    "introspect_tools",
    # Integration class
    "LangChainIntegration",
    # Callback handler
    "AgentFactsCallbackHandler",
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
]


# Auto-register with global registry
def _register() -> None:
    try:
        from agentfacts.integrations import get_registry

        registry = get_registry()
        registry.register(LangChainIntegration())
    except Exception:
        pass


_register()
