"""
Factory functions for creating AgentFacts/GroupFacts from LlamaIndex objects.

This module contains the actual factory logic, separate from introspection.
"""

from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from agentfacts.crypto.keys import KeyPair
    from agentfacts.group import GroupFacts


def create_group_from_llamaindex(
    query_engine_or_agent: Any,
    name: str | None = None,
    key_pair: Optional["KeyPair"] = None,
) -> "GroupFacts":
    """
    Create GroupFacts from a LlamaIndex query engine or agent.

    For LlamaIndex, a "group" typically represents a query engine with
    multiple components (retriever, response synthesizer) or an agent
    with multiple tools. This factory creates a single-member GroupFacts
    with the engine/agent as the primary member.

    For more complex multi-agent setups, you can manually create
    GroupFacts with multiple AgentFacts members.

    Args:
        query_engine_or_agent: LlamaIndex QueryEngine, AgentRunner, or Index
        name: Group name (defaults based on the component type)
        key_pair: Optional key pair for the group

    Returns:
        GroupFacts instance

    Example:
        ```python
        from llama_index.core import VectorStoreIndex
        from agentfacts import GroupFacts

        index = VectorStoreIndex.from_documents(documents)
        query_engine = index.as_query_engine()

        group = GroupFacts.from_llamaindex(query_engine, name="RAG System")
        group.sign_all()
        ```
    """
    from agentfacts.core import AgentFacts
    from agentfacts.group import GroupFacts
    from agentfacts.integrations.llamaindex.introspector import LlamaIndexIntegration
    from agentfacts.models import ProcessType

    introspector = LlamaIndexIntegration()

    # Introspect the object
    result = introspector.introspect(query_engine_or_agent)

    # Determine name from context
    class_name = type(query_engine_or_agent).__name__
    if name is None:
        if (
            "Agent" in class_name
            or class_name == "AgentRunner"
            or "Index" in class_name
        ):
            name = f"LlamaIndex {class_name}"
        elif "QueryEngine" in class_name:
            name = "LlamaIndex Query Engine"
        else:
            name = f"LlamaIndex {class_name}"

    # Determine description
    description = ""
    if "engine_type" in result.context:
        description = (
            f"{result.context['engine_type']} for document retrieval and synthesis"
        )
    elif "agent_type" in result.context:
        description = f"{result.context['agent_type']} for tool-augmented reasoning"
    elif "index_type" in result.context:
        description = f"{result.context['index_type']} document index"

    # Create AgentFacts for the main component
    agent_facts = AgentFacts(
        name=name,
        description=description,
        baseline_model=result.baseline_model,
        capabilities=result.capabilities,
        constraints=result.constraints,
    )
    agent_facts.metadata.agent.framework = "llamaindex"
    agent_facts.metadata.agent.context.update(result.context)

    # Create the group
    group = GroupFacts(
        name=name,
        members=[agent_facts],
        process_type=ProcessType.SEQUENTIAL,
        key_pair=key_pair,
        framework="llamaindex",
    )

    # Add LlamaIndex-specific context
    group._metadata.context.update(
        {
            "component_type": class_name,
        }
    )

    # Copy relevant context fields
    for key in ["retriever_type", "index_type", "vector_store_type", "embed_model"]:
        if key in result.context:
            group._metadata.context[key] = result.context[key]

    return group
