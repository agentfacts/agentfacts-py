"""
LlamaIndex introspection for AgentFacts SDK.

Extracts metadata from LlamaIndex query engines, agents, and indexes.
"""

from dataclasses import dataclass
from typing import Any

from agentfacts.integrations.base import (
    BaseIntrospector,
    ContextKey,
    IntrospectionResult,
)
from agentfacts.integrations.utils import RiskAssessor
from agentfacts.models import (
    BaselineModel,
    Capability,
    ModelProvider,
    OperationalConstraints,
)

# Mapping of LlamaIndex LLM class names to providers
LLAMAINDEX_LLM_PROVIDERS: dict[str, ModelProvider] = {
    "OpenAI": ModelProvider.OPENAI,
    "ChatOpenAI": ModelProvider.OPENAI,
    "AzureOpenAI": ModelProvider.OPENAI,
    "Anthropic": ModelProvider.ANTHROPIC,
    "Claude": ModelProvider.ANTHROPIC,
    "Gemini": ModelProvider.GOOGLE,
    "GoogleGenerativeAI": ModelProvider.GOOGLE,
    "VertexAI": ModelProvider.GOOGLE,
    "Ollama": ModelProvider.LOCAL,
    "LlamaCPP": ModelProvider.LOCAL,
    "HuggingFaceLLM": ModelProvider.LOCAL,
    "MistralAI": ModelProvider.MISTRAL,
    "Cohere": ModelProvider.UNKNOWN,
    "Replicate": ModelProvider.UNKNOWN,
}


def _extract_llm(obj: Any) -> Any:
    """Extract LLM from various LlamaIndex objects."""
    # Direct LLM attribute
    llm = getattr(obj, "llm", None) or getattr(obj, "_llm", None)
    if llm:
        return llm

    # From agent_worker
    agent_worker = getattr(obj, "agent_worker", None)
    if agent_worker:
        llm = getattr(agent_worker, "llm", None) or getattr(agent_worker, "_llm", None)
        if llm:
            return llm

    # From response synthesizer
    response_synth = getattr(obj, "_response_synthesizer", None)
    if response_synth:
        llm = getattr(response_synth, "_llm", None) or getattr(
            response_synth, "llm", None
        )
        if llm:
            return llm

    # From service context (older versions)
    service_context = getattr(obj, "service_context", None) or getattr(
        obj, "_service_context", None
    )
    if service_context:
        llm = getattr(service_context, "llm", None)
        if llm:
            return llm

    return None


def introspect_llm(llm: Any) -> BaselineModel:
    """Extract LLM metadata."""
    class_name = type(llm).__name__

    # Detect provider
    provider = LLAMAINDEX_LLM_PROVIDERS.get(class_name, ModelProvider.UNKNOWN)

    # Additional provider detection from module
    if provider == ModelProvider.UNKNOWN:
        module = getattr(type(llm), "__module__", "")
        if "openai" in module.lower():
            provider = ModelProvider.OPENAI
        elif "anthropic" in module.lower():
            provider = ModelProvider.ANTHROPIC
        elif "google" in module.lower() or "gemini" in module.lower():
            provider = ModelProvider.GOOGLE

    # Get model name
    model_name = "unknown"
    for attr in ["model", "model_name", "model_id", "_model"]:
        if hasattr(llm, attr):
            value = getattr(llm, attr)
            if value and isinstance(value, str):
                model_name = value
                break

    # Get temperature
    temperature = None
    temp = getattr(llm, "temperature", None)
    if isinstance(temp, (int, float)):
        temperature = float(temp)

    # Get max tokens
    max_tokens = None
    for attr in ["max_tokens", "max_output_tokens", "_max_tokens"]:
        if hasattr(llm, attr):
            value = getattr(llm, attr)
            if isinstance(value, int) and value > 0:
                max_tokens = value
                break

    return BaselineModel(
        name=model_name,
        provider=provider,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def introspect_tools(tools: list[Any]) -> list[Capability]:
    """Extract tool capabilities."""
    capabilities = []

    for tool in tools:
        # Get tool metadata
        metadata = getattr(tool, "metadata", None)
        if metadata:
            name = getattr(metadata, "name", None) or type(tool).__name__
            description = getattr(metadata, "description", "") or ""
        else:
            name = getattr(tool, "name", None) or type(tool).__name__
            description = getattr(tool, "description", "") or ""

        # Extract parameter schema
        parameters: dict[str, Any] = {}
        fn_schema = getattr(tool, "fn_schema", None)
        if fn_schema:
            try:
                if hasattr(fn_schema, "model_json_schema"):
                    parameters = fn_schema.model_json_schema()
                elif hasattr(fn_schema, "schema"):
                    parameters = fn_schema.schema()
            except Exception:
                pass

        # Assess risk level
        risk_level = RiskAssessor.assess_tool_risk(name, description)

        capabilities.append(
            Capability(
                name=name,
                description=str(description)[:200],
                parameters=parameters,
                risk_level=risk_level,
            )
        )

    return capabilities


def introspect_agent(
    agent: Any,
) -> tuple[
    BaselineModel | None, list[Capability], OperationalConstraints, dict[str, Any]
]:
    """Introspect a LlamaIndex agent (AgentRunner, ReActAgent, etc.)."""
    baseline_model = None
    capabilities: list[Capability] = []
    constraints = OperationalConstraints()
    context: dict[str, Any] = {}

    class_name = type(agent).__name__
    context["agent_type"] = class_name

    # Extract tools
    tools = getattr(agent, "tools", None)
    if tools:
        capabilities = introspect_tools(tools)
        context["tool_count"] = len(tools)

    # Extract LLM from agent_worker or directly
    llm = _extract_llm(agent)
    if llm:
        baseline_model = introspect_llm(llm)

    # Extract agent-specific settings
    agent_worker = getattr(agent, "agent_worker", None)
    if agent_worker:
        context["worker_type"] = type(agent_worker).__name__

        # Check for max iterations
        max_iterations = getattr(agent_worker, "max_iterations", None)
        if max_iterations:
            constraints.max_iterations = int(max_iterations)

        # Check for verbose mode
        verbose = getattr(agent_worker, "verbose", False)
        context["verbose"] = verbose

    # Memory/chat history
    memory = getattr(agent, "memory", None)
    if memory:
        context["has_memory"] = True
        memory_type = type(memory).__name__
        context["memory_type"] = memory_type

    # ReAct specific
    if "ReAct" in class_name:
        context["reasoning_loop"] = True
        if not any(c.name == "reasoning" for c in capabilities):
            capabilities.append(
                Capability(
                    name="reasoning",
                    description="ReAct reasoning loop for step-by-step problem solving",
                    risk_level="low",
                )
            )

    # OpenAI function calling agents
    if "OpenAI" in class_name or "FunctionCalling" in class_name:
        context["function_calling"] = True

    return baseline_model, capabilities, constraints, context


def introspect_query_engine(
    engine: Any,
) -> tuple[
    BaselineModel | None, list[Capability], OperationalConstraints, dict[str, Any]
]:
    """Introspect a LlamaIndex query engine."""
    baseline_model = None
    capabilities: list[Capability] = []
    constraints = OperationalConstraints()
    context: dict[str, Any] = {}

    class_name = type(engine).__name__
    context["engine_type"] = class_name

    # Query capability
    capabilities.append(
        Capability(
            name="query",
            description="Execute queries against indexed documents",
            risk_level="low",
        )
    )

    # Extract retriever info
    retriever = getattr(engine, "_retriever", None) or getattr(
        engine, "retriever", None
    )
    if retriever:
        context["retriever_type"] = type(retriever).__name__

        similarity_top_k = getattr(retriever, "_similarity_top_k", None) or getattr(
            retriever, "similarity_top_k", None
        )
        if similarity_top_k:
            context["similarity_top_k"] = similarity_top_k

    # Extract LLM
    llm = _extract_llm(engine)
    if llm:
        baseline_model = introspect_llm(llm)

    # Sub-question engine specific
    if "SubQuestion" in class_name:
        context["multi_step"] = True
        capabilities.append(
            Capability(
                name="sub_question_decomposition",
                description="Breaks complex queries into sub-questions",
                risk_level="low",
            )
        )

    # Router engine
    if "Router" in class_name:
        context["routing"] = True
        query_engines = getattr(engine, "query_engines", None)
        if query_engines:
            context["routed_engine_count"] = len(query_engines)

    # Citation engine
    if "Citation" in class_name:
        context["citations"] = True
        capabilities.append(
            Capability(
                name="citations",
                description="Provides source citations for responses",
                risk_level="low",
            )
        )

    return baseline_model, capabilities, constraints, context


def introspect_index(
    index: Any,
) -> tuple[
    BaselineModel | None, list[Capability], OperationalConstraints, dict[str, Any]
]:
    """Introspect a LlamaIndex index."""
    baseline_model = None
    capabilities: list[Capability] = []
    constraints = OperationalConstraints()
    context: dict[str, Any] = {}

    class_name = type(index).__name__
    context["index_type"] = class_name

    # Index capability
    capabilities.append(
        Capability(
            name="document_indexing",
            description=f"Index and retrieve documents using {class_name}",
            risk_level="low",
        )
    )

    # Extract docstore info
    docstore = getattr(index, "docstore", None)
    if docstore:
        context["has_docstore"] = True
        docs = getattr(docstore, "docs", None)
        if docs and isinstance(docs, dict):
            context["document_count"] = len(docs)

    # Extract vector store info
    vector_store = getattr(index, "_vector_store", None) or getattr(
        index, "vector_store", None
    )
    if vector_store:
        context["vector_store_type"] = type(vector_store).__name__

    # Knowledge Graph specific
    if "KnowledgeGraph" in class_name:
        context["knowledge_graph"] = True

    return baseline_model, capabilities, constraints, context


def introspect_any(
    obj: Any,
) -> tuple[
    BaselineModel | None, list[Capability], OperationalConstraints, dict[str, Any]
]:
    """Introspect any LlamaIndex object to extract metadata."""
    class_name = type(obj).__name__

    if "Agent" in class_name or class_name == "AgentRunner":
        return introspect_agent(obj)
    elif "Index" in class_name:
        return introspect_index(obj)
    elif "QueryEngine" in class_name or hasattr(obj, "query"):
        return introspect_query_engine(obj)
    else:
        return introspect_query_engine(obj)


@dataclass
class LlamaIndexIntegration(BaseIntrospector):
    """
    LlamaIndex integration for AgentFacts.

    Supports:
    - QueryEngine (VectorStoreIndex, SummaryIndex, etc.)
    - AgentRunner / ReActAgent / OpenAIAgent
    - VectorStoreIndex, DocumentSummaryIndex, KnowledgeGraphIndex
    - Retrievers
    """

    @property
    def framework_name(self) -> str:
        return "llamaindex"

    def can_introspect(self, obj: Any) -> bool:
        """Check if object is a LlamaIndex component."""
        module = getattr(type(obj), "__module__", "")

        # Check module name
        if "llama_index" in module.lower() or "llamaindex" in module.lower():
            return True

        # Check class name patterns
        class_name = type(obj).__name__
        llamaindex_classes = {
            "QueryEngine",
            "RetrieverQueryEngine",
            "CitationQueryEngine",
            "AgentRunner",
            "ReActAgent",
            "OpenAIAgent",
            "FunctionCallingAgent",
            "VectorStoreIndex",
            "SummaryIndex",
            "DocumentSummaryIndex",
            "KnowledgeGraphIndex",
            "TreeIndex",
            "VectorIndexRetriever",
            "ListIndexRetriever",
            "ChatEngine",
            "SimpleChatEngine",
            "ContextChatEngine",
        }
        if class_name in llamaindex_classes:
            return True

        # Check for LlamaIndex-specific attributes
        if hasattr(obj, "query") and hasattr(obj, "_retriever"):
            return True
        if hasattr(obj, "index_struct") and hasattr(obj, "docstore"):
            return True
        return hasattr(obj, "agent_worker") and hasattr(obj, "tools")

    # Alias for Integration protocol

    def introspect(self, obj: Any) -> IntrospectionResult:
        """Introspect a LlamaIndex object."""
        baseline, capabilities, constraints, extra_context = introspect_any(obj)

        context: dict[ContextKey | str, Any] = {
            ContextKey.FRAMEWORK: "llamaindex",
            ContextKey.TOOL_COUNT: len(capabilities),
        }

        class_name = type(obj).__name__
        if "Agent" in class_name or class_name == "AgentRunner":
            if "ReAct" in class_name:
                context[ContextKey.AGENT_TYPE] = "LLAMAINDEX_REACT_AGENT"
            elif "OpenAI" in class_name:
                context[ContextKey.AGENT_TYPE] = "LLAMAINDEX_OPENAI_AGENT"
            else:
                context[ContextKey.AGENT_TYPE] = "LLAMAINDEX_AGENT"
        elif "Index" in class_name:
            context[ContextKey.AGENT_TYPE] = f"LLAMAINDEX_{class_name.upper()}"
        elif "QueryEngine" in class_name:
            context[ContextKey.AGENT_TYPE] = "LLAMAINDEX_QUERY_ENGINE"
        else:
            context[ContextKey.AGENT_TYPE] = f"LLAMAINDEX_{class_name.upper()}"

        # Merge extra context
        context.update(extra_context)

        return IntrospectionResult(
            framework="llamaindex",
            baseline_model=baseline,
            capabilities=capabilities,
            constraints=constraints,
            context=context,
        )
