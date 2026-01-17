"""
Factory functions for creating AgentFacts/GroupFacts from AutoGen objects.

This module contains the actual factory logic, separate from introspection.
"""

from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from agentfacts.crypto.keys import KeyPair
    from agentfacts.group import GroupFacts


def create_group_from_autogen(
    group_chat: Any,
    name: str | None = None,
    key_pair: Optional["KeyPair"] = None,
) -> "GroupFacts":
    """
    Create GroupFacts from an AutoGen GroupChat.

    Automatically creates AgentFacts for each agent in the GroupChat
    and establishes the group relationships. Extracts LLM config,
    code execution settings, and conversation parameters.

    Args:
        group_chat: AutoGen GroupChat or GroupChatManager instance.
        name: Group name (defaults to "AutoGen GroupChat").
        key_pair: Optional key pair for the group. If not provided,
            a new key pair is generated.

    Returns:
        GroupFacts instance with AgentFacts for each agent.

    Expected GroupChat Attributes:
        - agents (list): List of AutoGen agent objects (required).
        - max_round (int): Maximum conversation rounds (optional).
        - speaker_selection_method (str): How next speaker is chosen (optional).

    Expected Agent Attributes:
        - name (str): Agent name (required).
        - description (str): Agent description (optional).
        - system_message (str): System prompt (optional).
        - llm_config (dict): LLM configuration with 'model' key (optional).
        - code_execution_config (dict): Code execution settings (optional).
        - human_input_mode (str): "ALWAYS", "NEVER", or "TERMINATE" (optional).

    Example:
        ```python
        from autogen import GroupChat, AssistantAgent, UserProxyAgent
        from agentfacts import GroupFacts

        group_chat = GroupChat(agents=[user_proxy, assistant, coder], max_round=12)
        group = GroupFacts.from_autogen(group_chat, name="Dev Team")
        group.sign_all()
        ```
    """
    from agentfacts.core import AgentFacts
    from agentfacts.group import GroupFacts
    from agentfacts.integrations.autogen.introspector import AutoGenIntegration
    from agentfacts.models import ProcessType

    introspector = AutoGenIntegration()

    # Handle GroupChatManager
    actual_group_chat = group_chat
    if hasattr(group_chat, "groupchat"):
        actual_group_chat = group_chat.groupchat

    # Get agents
    agents = getattr(actual_group_chat, "agents", [])

    # Create AgentFacts for each agent
    member_facts: list[AgentFacts] = []
    for agent in agents:
        agent_name = getattr(agent, "name", None) or type(agent).__name__
        result = introspector.introspect(agent)

        description = getattr(agent, "description", "") or ""

        facts = AgentFacts(
            name=str(agent_name),
            description=str(description),
            baseline_model=result.baseline_model,
            capabilities=result.capabilities,
            constraints=result.constraints,
        )

        facts.metadata.agent.framework = "autogen"
        facts.metadata.agent.context.update(result.context)

        member_facts.append(facts)

    # Create the group
    group_name = name or "AutoGen GroupChat"
    group = GroupFacts(
        name=group_name,
        members=member_facts,
        process_type=ProcessType.EVENT_DRIVEN,
        key_pair=key_pair,
        framework="autogen",
    )

    # Add AutoGen-specific context
    max_round = getattr(actual_group_chat, "max_round", None)
    if max_round:
        group._metadata.max_rounds = int(max_round)

    speaker_method = getattr(actual_group_chat, "speaker_selection_method", None)
    if speaker_method:
        group._metadata.context["speaker_selection_method"] = str(speaker_method)

    return group
