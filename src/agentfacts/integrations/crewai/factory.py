"""
Factory functions for creating AgentFacts/GroupFacts from CrewAI objects.

This module contains the actual factory logic, separate from introspection.
"""

from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from agentfacts.crypto.keys import KeyPair
    from agentfacts.group import GroupFacts


def create_group_from_crew(
    crew: Any,
    name: str | None = None,
    key_pair: Optional["KeyPair"] = None,
) -> "GroupFacts":
    """
    Create GroupFacts from a CrewAI Crew.

    Automatically creates AgentFacts for each agent in the crew
    and establishes the group relationships. Extracts role, goal,
    backstory, tools, LLM config, and delegation settings.

    Args:
        crew: CrewAI Crew instance with agents and tasks.
        name: Group name (defaults to "CrewAI Crew").
        key_pair: Optional key pair for the group. If not provided,
            a new key pair is generated.

    Returns:
        GroupFacts instance with AgentFacts for each crew agent.

    Expected Crew Attributes:
        - agents (list): List of CrewAI Agent objects (required).
        - process (Process): Execution pattern (optional, defaults to sequential).
        - tasks (list): List of Task objects (optional).
        - memory (bool): Shared memory setting (optional).

    Expected Agent Attributes:
        - role (str): Agent's role name (required).
        - goal (str): Agent's goal (optional).
        - backstory (str): Agent's backstory (optional).
        - tools (list): List of tools (optional).
        - llm (LLM): LLM configuration (optional).
        - allow_delegation (bool): Delegation setting (optional).

    Example:
        ```python
        from crewai import Crew, Agent, Task
        from agentfacts import GroupFacts

        crew = Crew(agents=[researcher, writer], tasks=[task])
        group = GroupFacts.from_crewai(crew, name="Research Team")
        group.sign_all()
        ```
    """
    from agentfacts.core import AgentFacts
    from agentfacts.group import GroupFacts
    from agentfacts.integrations.crewai.introspector import CrewAIIntegration
    from agentfacts.models import AgentRole, DelegationPolicy, ProcessType

    introspector = CrewAIIntegration()

    # Get crew-level info
    agents = getattr(crew, "agents", [])
    process = getattr(crew, "process", None)

    # Determine process type
    process_type = ProcessType.SEQUENTIAL
    if process:
        process_str = str(process).lower()
        if "hierarchical" in process_str:
            process_type = ProcessType.HIERARCHICAL

    # Create AgentFacts for each crew agent
    member_facts: list[AgentFacts] = []
    for agent in agents:
        agent_name = getattr(agent, "role", None) or getattr(agent, "name", "Agent")
        result = introspector.introspect(agent)

        facts = AgentFacts(
            name=str(agent_name),
            description=str(getattr(agent, "goal", "")),
            baseline_model=result.baseline_model,
            capabilities=result.capabilities,
            constraints=result.constraints,
        )

        # Set role if available
        if "role" in result.context:
            facts.metadata.agent.role = AgentRole.model_validate(result.context["role"])

        # Set delegation policy
        if "delegation" in result.context:
            facts.metadata.agent.delegation = DelegationPolicy.model_validate(
                result.context["delegation"]
            )

        facts.metadata.agent.framework = "crewai"
        facts.metadata.agent.context.update(result.context)

        member_facts.append(facts)

    # Create the group
    group_name = name or "CrewAI Crew"
    group = GroupFacts(
        name=group_name,
        members=member_facts,
        process_type=process_type,
        key_pair=key_pair,
        framework="crewai",
    )

    # Add crew-specific context
    tasks = getattr(crew, "tasks", [])
    group._metadata.context["task_count"] = len(tasks)
    if getattr(crew, "memory", None):
        group._metadata.shared_memory = True

    return group
