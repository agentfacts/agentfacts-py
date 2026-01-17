"""
LangChain introspection for AgentFacts SDK.

Extracts metadata from LangChain agents, chains, tools,
LCEL runnables, and LangGraph graphs using reflection.
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


def _is_langgraph_graph(obj: Any) -> bool:
    """Check if object is a LangGraph compiled graph."""
    class_name = type(obj).__name__
    return (
        class_name in ("CompiledGraph", "CompiledStateGraph", "StateGraph", "Graph")
        or "CompiledGraph" in class_name
        or "StateGraph" in class_name
    )


# Mapping of known model name patterns to providers
PROVIDER_PATTERNS: dict[str, ModelProvider] = {
    "gpt-": ModelProvider.OPENAI,
    "text-davinci": ModelProvider.OPENAI,
    "claude": ModelProvider.ANTHROPIC,
    "gemini": ModelProvider.GOOGLE,
    "palm": ModelProvider.GOOGLE,
    "command": ModelProvider.COHERE,
    "mistral": ModelProvider.MISTRAL,
    "mixtral": ModelProvider.MISTRAL,
    "llama": ModelProvider.META,
    "codellama": ModelProvider.META,
}

# Mapping of LangChain LLM class names to providers
LLM_CLASS_PROVIDERS: dict[str, ModelProvider] = {
    "ChatOpenAI": ModelProvider.OPENAI,
    "OpenAI": ModelProvider.OPENAI,
    "AzureChatOpenAI": ModelProvider.OPENAI,
    "ChatAnthropic": ModelProvider.ANTHROPIC,
    "Anthropic": ModelProvider.ANTHROPIC,
    "ChatGoogleGenerativeAI": ModelProvider.GOOGLE,
    "GoogleGenerativeAI": ModelProvider.GOOGLE,
    "ChatVertexAI": ModelProvider.GOOGLE,
    "Cohere": ModelProvider.COHERE,
    "ChatCohere": ModelProvider.COHERE,
    "MistralAI": ModelProvider.MISTRAL,
    "ChatMistralAI": ModelProvider.MISTRAL,
    "HuggingFaceHub": ModelProvider.HUGGINGFACE,
    "HuggingFacePipeline": ModelProvider.LOCAL,
    "Ollama": ModelProvider.LOCAL,
    "LlamaCpp": ModelProvider.LOCAL,
}


def _detect_provider(llm: Any) -> ModelProvider:
    """Detect the model provider from an LLM object."""
    class_name = type(llm).__name__
    if class_name in LLM_CLASS_PROVIDERS:
        return LLM_CLASS_PROVIDERS[class_name]

    model_name = _get_model_name(llm).lower()
    for pattern, provider in PROVIDER_PATTERNS.items():
        if pattern in model_name:
            return provider

    return ModelProvider.UNKNOWN


def _get_model_name(llm: Any) -> str:
    """Extract model name from an LLM object."""
    for attr in ["model_name", "model", "model_id", "repo_id"]:
        if hasattr(llm, attr):
            value = getattr(llm, attr)
            if value and isinstance(value, str):
                return str(value)

    model_kwargs = getattr(llm, "model_kwargs", None)
    if isinstance(model_kwargs, dict) and "model" in model_kwargs:
        return str(model_kwargs["model"])

    return type(llm).__name__


def _get_temperature(llm: Any) -> float | None:
    """Extract temperature setting from an LLM object."""
    if hasattr(llm, "temperature"):
        temp = llm.temperature
        if isinstance(temp, (int, float)):
            return float(temp)
    return None


def _get_max_tokens(llm: Any) -> int | None:
    """Extract max tokens setting from an LLM object."""
    for attr in ["max_tokens", "max_output_tokens", "max_new_tokens"]:
        if hasattr(llm, attr):
            value = getattr(llm, attr)
            if isinstance(value, int) and value > 0:
                return value
    return None


def introspect_llm(llm: Any) -> BaselineModel:
    """
    Introspect a LangChain LLM/ChatModel to extract model metadata.

    Args:
        llm: A LangChain LLM or ChatModel instance

    Returns:
        BaselineModel with extracted metadata
    """
    model_name = _get_model_name(llm)
    provider = _detect_provider(llm)
    temperature = _get_temperature(llm)
    max_tokens = _get_max_tokens(llm)

    extra_params: dict[str, Any] = {}
    param_attrs = [
        "top_p",
        "top_k",
        "frequency_penalty",
        "presence_penalty",
        "stop",
        "streaming",
        "n",
        "seed",
    ]
    for attr in param_attrs:
        if hasattr(llm, attr):
            value = getattr(llm, attr)
            if value is not None:
                extra_params[attr] = value

    return BaselineModel(
        name=model_name,
        provider=provider,
        temperature=temperature,
        max_tokens=max_tokens,
        extra_params=extra_params,
    )


def introspect_tools(tools: list[Any]) -> list[Capability]:
    """
    Introspect a list of LangChain tools to extract capability metadata.

    Args:
        tools: List of LangChain Tool instances

    Returns:
        List of Capability objects
    """
    capabilities = []

    for tool in tools:
        name = getattr(tool, "name", type(tool).__name__)
        description = getattr(tool, "description", "")

        parameters: dict[str, Any] = {}
        if hasattr(tool, "args_schema") and tool.args_schema is not None:
            try:
                if hasattr(tool.args_schema, "model_json_schema"):
                    parameters = tool.args_schema.model_json_schema()
                elif hasattr(tool.args_schema, "schema"):
                    parameters = tool.args_schema.schema()
            except Exception:
                pass

        risk_level = RiskAssessor.assess_tool_risk(name, description)

        capabilities.append(
            Capability(
                name=name,
                description=description,
                parameters=parameters,
                risk_level=risk_level,
            )
        )

    return capabilities


def introspect_chain(chain: Any) -> tuple[BaselineModel | None, list[Capability]]:
    """
    Introspect a LangChain Chain to extract metadata.

    Args:
        chain: A LangChain Chain instance

    Returns:
        Tuple of (BaselineModel or None, list of Capabilities)
    """
    baseline_model = None
    capabilities: list[Capability] = []

    if hasattr(chain, "llm"):
        baseline_model = introspect_llm(chain.llm)
    elif hasattr(chain, "llm_chain") and hasattr(chain.llm_chain, "llm"):
        baseline_model = introspect_llm(chain.llm_chain.llm)

    if hasattr(chain, "tools"):
        capabilities = introspect_tools(chain.tools)

    return baseline_model, capabilities


def _is_llm_like(obj: Any) -> bool:
    """Check if an object looks like an LLM."""
    llm_indicators = ["invoke", "generate", "model_name", "temperature"]
    matches = sum(1 for attr in llm_indicators if hasattr(obj, attr))
    return matches >= 2


def _is_runnable(obj: Any) -> bool:
    """Check if object is an LCEL Runnable."""
    return hasattr(obj, "invoke") and hasattr(obj, "batch")


def _extract_llms_from_runnable(runnable: Any, found_llms: list[Any]) -> None:
    """Recursively extract LLMs from LCEL runnable chains."""
    if _is_llm_like(runnable):
        found_llms.append(runnable)
        return

    if hasattr(runnable, "first"):
        _extract_llms_from_runnable(runnable.first, found_llms)
    if hasattr(runnable, "middle"):
        for item in runnable.middle or []:
            _extract_llms_from_runnable(item, found_llms)
    if hasattr(runnable, "last"):
        _extract_llms_from_runnable(runnable.last, found_llms)

    if hasattr(runnable, "steps") and isinstance(runnable.steps, dict):
        for step in runnable.steps.values():
            _extract_llms_from_runnable(step, found_llms)

    if hasattr(runnable, "branches"):
        for branch in runnable.branches:
            if isinstance(branch, tuple) and len(branch) >= 2:
                _extract_llms_from_runnable(branch[1], found_llms)
        if hasattr(runnable, "default"):
            _extract_llms_from_runnable(runnable.default, found_llms)

    if hasattr(runnable, "bound"):
        _extract_llms_from_runnable(runnable.bound, found_llms)


def introspect_runnable(runnable: Any) -> tuple[BaselineModel | None, list[Capability]]:
    """
    Introspect an LCEL Runnable to extract metadata.

    Args:
        runnable: An LCEL Runnable instance

    Returns:
        Tuple of (BaselineModel or None, list of Capabilities)
    """
    found_llms: list[Any] = []
    _extract_llms_from_runnable(runnable, found_llms)

    baseline_model = None
    if found_llms:
        baseline_model = introspect_llm(found_llms[0])

    capabilities: list[Capability] = []
    if hasattr(runnable, "tools"):
        capabilities = introspect_tools(runnable.tools)

    return baseline_model, capabilities


def introspect_agent(
    agent: Any,
) -> tuple[BaselineModel | None, list[Capability], OperationalConstraints]:
    """
    Introspect a LangChain Agent or AgentExecutor to extract full metadata.

    Args:
        agent: A LangChain Agent or AgentExecutor instance

    Returns:
        Tuple of (BaselineModel or None, list of Capabilities, OperationalConstraints)
    """
    baseline_model = None
    capabilities: list[Capability] = []
    constraints = OperationalConstraints()

    if hasattr(agent, "agent") and hasattr(agent, "tools"):
        inner_agent = agent.agent

        if hasattr(inner_agent, "llm_chain") and hasattr(inner_agent.llm_chain, "llm"):
            baseline_model = introspect_llm(inner_agent.llm_chain.llm)
        elif hasattr(inner_agent, "llm"):
            baseline_model = introspect_llm(inner_agent.llm)

        if hasattr(agent, "tools"):
            capabilities = introspect_tools(agent.tools)

        if hasattr(agent, "max_iterations"):
            constraints.max_iterations = agent.max_iterations
        if hasattr(agent, "max_execution_time"):
            constraints.timeout_seconds = (
                int(agent.max_execution_time) if agent.max_execution_time else None
            )
        if hasattr(agent, "return_intermediate_steps"):
            constraints.requires_human_approval = not getattr(
                agent, "return_intermediate_steps", True
            )

    elif hasattr(agent, "llm"):
        baseline_model = introspect_llm(agent.llm)

    elif hasattr(agent, "first") or hasattr(agent, "middle") or hasattr(agent, "last"):
        for attr in ["first", "middle", "last"]:
            if hasattr(agent, attr):
                component = getattr(agent, attr)
                if hasattr(component, "llm"):
                    baseline_model = introspect_llm(component.llm)
                    break
                elif _is_llm_like(component):
                    baseline_model = introspect_llm(component)
                    break

    return baseline_model, capabilities, constraints


def introspect_langgraph(
    graph: Any,
) -> tuple[BaselineModel | None, list[Capability], OperationalConstraints]:
    """
    Introspect a LangGraph compiled graph to extract metadata.

    Args:
        graph: A LangGraph CompiledGraph or CompiledStateGraph

    Returns:
        Tuple of (BaselineModel or None, list of Capabilities, OperationalConstraints)
    """
    baseline_model = None
    capabilities: list[Capability] = []
    constraints = OperationalConstraints()

    nodes = {}
    if hasattr(graph, "nodes"):
        nodes = graph.nodes if isinstance(graph.nodes, dict) else {}

    found_llms: list[Any] = []
    for node_name, node_value in nodes.items():
        if node_name in ("__start__", "__end__", "START", "END"):
            continue

        if _is_llm_like(node_value):
            found_llms.append(node_value)
        elif _is_runnable(node_value):
            _extract_llms_from_runnable(node_value, found_llms)
        elif hasattr(node_value, "llm"):
            found_llms.append(node_value.llm)

        capabilities.append(
            Capability(
                name=f"node:{node_name}",
                description=f"LangGraph node: {node_name}",
                parameters={},
                risk_level=RiskAssessor.assess_tool_risk(node_name, ""),
            )
        )

    if found_llms:
        baseline_model = introspect_llm(found_llms[0])

    if hasattr(graph, "tools"):
        capabilities.extend(introspect_tools(graph.tools))

    edges = []
    if hasattr(graph, "edges"):
        edges = list(graph.edges) if hasattr(graph.edges, "__iter__") else []

    if edges:
        constraints.allowed_domains = [f"graph_edges:{len(edges)}"]

    return baseline_model, capabilities, constraints


def introspect_any(
    obj: Any,
) -> tuple[BaselineModel | None, list[Capability], OperationalConstraints]:
    """
    Introspect any LangChain/LangGraph object to extract metadata.

    Auto-detects the object type and calls the appropriate introspection function.

    Args:
        obj: A LangChain agent, chain, runnable, or LangGraph graph

    Returns:
        Tuple of (BaselineModel or None, list of Capabilities, OperationalConstraints)
    """
    if _is_langgraph_graph(obj):
        return introspect_langgraph(obj)

    if hasattr(obj, "agent") and hasattr(obj, "tools"):
        return introspect_agent(obj)

    if _is_runnable(obj):
        model, caps = introspect_runnable(obj)
        return model, caps, OperationalConstraints()

    if hasattr(obj, "llm") or hasattr(obj, "llm_chain"):
        model, caps = introspect_chain(obj)
        return model, caps, OperationalConstraints()

    return introspect_agent(obj)


@dataclass
class LangChainIntegration(BaseIntrospector):
    """
    LangChain integration for AgentFacts.

    Provides introspection for LangChain agents, chains, runnables,
    and LangGraph graphs.
    """

    @property
    def framework_name(self) -> str:
        return "langchain"

    def can_introspect(self, obj: Any) -> bool:
        """Check if this integration can handle the object."""
        # Check module name
        module = getattr(type(obj), "__module__", "")
        if any(x in module for x in ["langchain", "langgraph"]):
            return True

        # Check for LangGraph
        if _is_langgraph_graph(obj):
            return True

        # Check for LCEL Runnable
        if _is_runnable(obj):
            return True

        # Check for agent patterns
        return hasattr(obj, "agent") and hasattr(obj, "tools")

    # Alias for Integration protocol

    def introspect(self, obj: Any) -> IntrospectionResult:
        """Introspect a LangChain object."""
        baseline, capabilities, constraints = introspect_any(obj)

        context: dict[ContextKey | str, Any] = {
            ContextKey.FRAMEWORK: "langchain",
            ContextKey.TOOL_COUNT: len(capabilities),
        }

        if _is_langgraph_graph(obj):
            context[ContextKey.AGENT_TYPE] = "LangGraph"
        elif hasattr(obj, "agent") and hasattr(obj, "tools"):
            context[ContextKey.AGENT_TYPE] = "AgentExecutor"
        elif _is_runnable(obj):
            context[ContextKey.AGENT_TYPE] = "Runnable"
        else:
            context[ContextKey.AGENT_TYPE] = type(obj).__name__

        return IntrospectionResult(
            framework="langchain",
            baseline_model=baseline,
            capabilities=capabilities,
            constraints=constraints,
            context=context,
        )
