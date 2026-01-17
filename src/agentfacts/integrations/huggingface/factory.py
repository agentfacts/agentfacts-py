"""
Factory functions for creating AgentFacts/GroupFacts from HuggingFace objects.

This module contains the actual factory logic, separate from introspection.
Supports smolagents and tiny-agents formats.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from agentfacts.crypto.keys import KeyPair
    from agentfacts.group import GroupFacts


def create_group_from_huggingface(
    agent_or_config: Any,
    name: str | None = None,
    key_pair: Optional["KeyPair"] = None,
) -> "GroupFacts":
    """
    Create GroupFacts from a Hugging Face agent or config.

    Supports multiple HuggingFace agent formats:
    - smolagents: CodeAgent, ToolCallingAgent, MultiStepAgent
    - tiny-agents: agent.json config dict or file path
    - Hub agents: agents loaded from HF Hub

    For smolagents with managed_agents, this creates a multi-member GroupFacts
    with HIERARCHICAL process type. For other cases, creates a single-member group.

    Args:
        agent_or_config: One of the following:
            - smolagents agent instance (CodeAgent, ToolCallingAgent, etc.)
            - tiny-agents config dict with model, provider, servers
            - Path to directory containing agent.json
            - Path to agent.json file directly
        name: Group name (auto-generated if not provided).
        key_pair: Optional key pair for the group.

    Returns:
        GroupFacts instance with extracted capabilities and metadata.

    Expected smolagents Attributes:
        - model: Model instance (InferenceClientModel, LiteLLMModel, etc.)
        - tools (list): List of tool instances (optional).
        - max_steps (int): Maximum execution steps (optional).
        - managed_agents (list): Sub-agents for hierarchical setup (optional).
        - name (str): Agent name (optional).

    Expected tiny-agents Config Keys:
        - model (str): Model identifier, e.g., "gpt-4o" (required).
        - provider (str): Model provider, e.g., "openai" (optional).
        - endpointUrl (str): Custom API endpoint (optional).
        - servers (list): MCP server configurations (optional).

    Example:
        ```python
        from smolagents import CodeAgent, InferenceClientModel
        from smolagents.tools import DuckDuckGoSearchTool
        from agentfacts import GroupFacts

        # Example 1: smolagents CodeAgent with tools
        model = InferenceClientModel("Qwen/Qwen2.5-72B-Instruct")
        agent = CodeAgent(
            tools=[DuckDuckGoSearchTool()],
            model=model,
            max_steps=10,
        )
        group = GroupFacts.from_huggingface(agent, name="Research Agent")
        group.sign_all()

        # Example 2: tiny-agents config with MCP servers
        config = {
            "model": "gpt-4o",
            "provider": "openai",
            "servers": [
                {
                    "type": "stdio",
                    "command": "npx",
                    "args": ["-y", "@playwright/mcp@latest"]
                }
            ]
        }
        group = GroupFacts.from_huggingface(config, name="Browser Agent")
        ```
    """
    from agentfacts.core import AgentFacts
    from agentfacts.group import GroupFacts
    from agentfacts.integrations.huggingface.introspector import HuggingFaceIntegration
    from agentfacts.models import ProcessType

    introspector = HuggingFaceIntegration()

    class_name = type(agent_or_config).__name__

    # Check for multi-agent smolagent with managed_agents
    if hasattr(agent_or_config, "managed_agents"):
        managed = getattr(agent_or_config, "managed_agents", None)
        if managed and len(managed) > 0:
            return _create_from_smolagents_multi(
                agent_or_config, name, key_pair, introspector
            )

    # Introspect the object
    result = introspector.introspect(agent_or_config)

    # Determine name
    if name is None:
        if isinstance(agent_or_config, dict):
            model_id = agent_or_config.get("model", "Tiny Agent")
            name = f"Tiny Agent ({model_id})"
        elif isinstance(agent_or_config, (str, Path)):
            name = f"Tiny Agent ({Path(agent_or_config).name})"
        elif "agent_type" in result.context:
            name = f"smolagents {result.context['agent_type']}"
        elif "agent_name" in result.context:
            name = result.context["agent_name"]
        else:
            name = f"HuggingFace {class_name}"

    # Determine description
    description = ""
    if "description" in result.context:
        description = result.context["description"]
    elif "instructions" in result.context:
        description = result.context["instructions"][:200]
    elif "execution_mode" in result.context:
        if result.context["execution_mode"] == "code":
            description = "Code-executing agent using Python interpreter"
        else:
            description = "Tool-calling agent using JSON format"

    # Create AgentFacts
    agent_facts = AgentFacts(
        name=name,
        description=description,
        baseline_model=result.baseline_model,
        capabilities=result.capabilities,
        constraints=result.constraints,
    )
    agent_facts.metadata.agent.framework = "huggingface"
    agent_facts.metadata.agent.context.update(result.context)

    # Determine process type
    process_type = ProcessType.SEQUENTIAL
    if "has_managed_agents" in result.context:
        process_type = ProcessType.HIERARCHICAL

    # Create the group
    group = GroupFacts(
        name=name,
        members=[agent_facts],
        process_type=process_type,
        key_pair=key_pair,
        framework="huggingface",
    )

    # Add HuggingFace-specific context
    group._metadata.context.update(
        {
            "component_type": (
                class_name
                if not isinstance(agent_or_config, (dict, str, Path))
                else "config"
            ),
        }
    )

    # Copy relevant context
    for key in ["execution_mode", "model_class", "config_type", "mcp_server_count"]:
        if key in result.context:
            group._metadata.context[key] = result.context[key]

    return group


def _create_from_smolagents_multi(
    agent: Any,
    name: str | None,
    key_pair: Optional["KeyPair"],
    introspector: Any,
) -> "GroupFacts":
    """Create GroupFacts from a smolagents agent with managed_agents."""
    from agentfacts.core import AgentFacts
    from agentfacts.group import GroupFacts
    from agentfacts.models import ProcessType

    # Introspect main agent
    result = introspector.introspect(agent)

    class_name = type(agent).__name__

    # Determine group name
    group_name = name or result.context.get("agent_name", f"smolagents {class_name}")

    # Create AgentFacts for main agent
    main_facts = AgentFacts(
        name=result.context.get("agent_name", "Main Agent"),
        description=result.context.get(
            "description", result.context.get("instructions", "")[:200]
        ),
        baseline_model=result.baseline_model,
        capabilities=result.capabilities,
        constraints=result.constraints,
    )
    main_facts.metadata.agent.framework = "huggingface"
    main_facts.metadata.agent.context.update(result.context)

    member_facts: list[AgentFacts] = [main_facts]

    # Process managed agents
    managed_agents = getattr(agent, "managed_agents", [])
    for managed in managed_agents:
        m_result = introspector.introspect(managed)

        m_name = m_result.context.get("agent_name", type(managed).__name__)
        m_description = m_result.context.get("description", "")[:200]

        m_facts = AgentFacts(
            name=m_name,
            description=m_description,
            baseline_model=m_result.baseline_model,
            capabilities=m_result.capabilities,
            constraints=m_result.constraints,
        )
        m_facts.metadata.agent.framework = "huggingface"
        m_facts.metadata.agent.context.update(m_result.context)
        member_facts.append(m_facts)

    # Create the group
    group = GroupFacts(
        name=group_name,
        members=member_facts,
        process_type=ProcessType.HIERARCHICAL,
        key_pair=key_pair,
        framework="huggingface",
    )

    # Add context
    group._metadata.context.update(
        {
            "component_type": class_name,
            "managed_agent_count": len(managed_agents),
        }
    )

    return group
