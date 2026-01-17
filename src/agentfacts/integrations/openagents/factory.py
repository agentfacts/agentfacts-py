"""
Factory functions for creating AgentFacts/GroupFacts from OpenAgents objects.

This module contains the actual factory logic, separate from introspection.
"""

from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from agentfacts.crypto.keys import KeyPair
    from agentfacts.group import GroupFacts


def create_group_from_openagents(
    network_or_agent: Any,
    name: str | None = None,
    key_pair: Optional["KeyPair"] = None,
) -> "GroupFacts":
    """
    Create GroupFacts from an OpenAgents network or agent.

    OpenAgents is an infrastructure platform for AI agent networks.
    This factory supports:
    - Network instances (creates a group with all registered agents)
    - AgentConfig (creates a single-member group)
    - WorkerAgent implementations

    Args:
        network_or_agent: OpenAgents Network, AgentConfig, or WorkerAgent
        name: Group name (defaults based on the component type)
        key_pair: Optional key pair for the group

    Returns:
        GroupFacts instance

    Example:
        ```python
        from openagents.core.network import create_network
        from openagents.models.agent_config import AgentConfig
        from agentfacts import GroupFacts

        # From a network
        network = create_network(config)
        group = GroupFacts.from_openagents(network, name="Agent Network")

        # From an agent config
        config = AgentConfig(
            model_name="gpt-4",
            instruction="You are a helpful assistant.",
            provider="openai"
        )
        group = GroupFacts.from_openagents(config, name="Assistant Agent")
        group.sign_all()
        ```
    """
    from agentfacts.core import AgentFacts
    from agentfacts.group import GroupFacts
    from agentfacts.integrations.openagents.introspector import OpenAgentsIntegration
    from agentfacts.models import ProcessType

    introspector = OpenAgentsIntegration()

    class_name = type(network_or_agent).__name__

    # Check if it's a Network with multiple agents
    if class_name == "Network" or (
        hasattr(network_or_agent, "register_agent")
        and hasattr(network_or_agent, "agents")
    ):
        return _create_from_network(network_or_agent, name, key_pair, introspector)

    # Single agent/config case
    result = introspector.introspect(network_or_agent)

    # Determine name
    if name is None:
        if "agent_id" in result.context:
            name = f"OpenAgents {result.context['agent_id']}"
        elif "network_name" in result.context:
            name = result.context["network_name"]
        else:
            name = f"OpenAgents {class_name}"

    # Determine description
    description = ""
    if "instruction" in result.context:
        description = result.context["instruction"][:200]
    elif "config_type" in result.context:
        description = f"OpenAgents {result.context['config_type']}"

    # Create AgentFacts
    agent_facts = AgentFacts(
        name=name,
        description=description,
        baseline_model=result.baseline_model,
        capabilities=result.capabilities,
        constraints=result.constraints,
    )
    agent_facts.metadata.agent.framework = "openagents"
    agent_facts.metadata.agent.context.update(result.context)

    # Determine process type
    process_type = ProcessType.EVENT_DRIVEN
    if "process_type" in result.context:
        pt_str = result.context["process_type"]
        if pt_str == ProcessType.HIERARCHICAL.value:
            process_type = ProcessType.HIERARCHICAL

    # Create the group
    group = GroupFacts(
        name=name,
        members=[agent_facts],
        process_type=process_type,
        key_pair=key_pair,
        framework="openagents",
    )

    # Add OpenAgents-specific context
    group._metadata.context.update(
        {
            "component_type": class_name,
        }
    )

    return group


def _create_from_network(
    network: Any,
    name: str | None,
    key_pair: Optional["KeyPair"],
    introspector: Any,
) -> "GroupFacts":
    """Create GroupFacts from an OpenAgents Network with multiple agents."""
    from agentfacts.core import AgentFacts
    from agentfacts.group import GroupFacts
    from agentfacts.models import ProcessType

    # Get network-level info
    network_result = introspector.introspect(network)
    network_context = network_result.context

    # Determine group name
    group_name = name or network_context.get("network_name", "OpenAgents Network")

    # Get agents from the network
    agents = getattr(network, "agents", None) or getattr(network, "_agents", None) or {}

    member_facts: list[AgentFacts] = []

    if isinstance(agents, dict):
        # Agents stored as dict with agent_id keys
        for agent_id, agent in agents.items():
            result = introspector.introspect(agent)

            facts = AgentFacts(
                name=str(agent_id),
                description=(
                    result.context.get("instruction", "")[:200]
                    if "instruction" in result.context
                    else ""
                ),
                baseline_model=result.baseline_model,
                capabilities=result.capabilities,
                constraints=result.constraints,
            )
            facts.metadata.agent.framework = "openagents"
            facts.metadata.agent.context.update(result.context)
            member_facts.append(facts)
    elif hasattr(agents, "__iter__"):
        # Agents as a list
        for agent in agents:
            agent_id = getattr(agent, "agent_id", None) or getattr(
                agent, "default_agent_id", type(agent).__name__
            )
            result = introspector.introspect(agent)

            facts = AgentFacts(
                name=str(agent_id),
                description=(
                    result.context.get("instruction", "")[:200]
                    if "instruction" in result.context
                    else ""
                ),
                baseline_model=result.baseline_model,
                capabilities=result.capabilities,
                constraints=result.constraints,
            )
            facts.metadata.agent.framework = "openagents"
            facts.metadata.agent.context.update(result.context)
            member_facts.append(facts)

    # Determine process type
    process_type = ProcessType.EVENT_DRIVEN
    if network_context.get("network_mode") == "centralized":
        process_type = ProcessType.HIERARCHICAL

    # Create the group
    group = GroupFacts(
        name=group_name,
        members=member_facts,
        process_type=process_type,
        key_pair=key_pair,
        framework="openagents",
    )

    # Add network context
    group._metadata.context.update(
        {
            "component_type": "Network",
            "transport": network_context.get("transport"),
            "network_mode": network_context.get("network_mode"),
        }
    )

    return group
