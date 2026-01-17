"""
CrewAI introspection for AgentFacts SDK.

Extracts metadata from CrewAI Agent and Crew objects.
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
    AgentRole,
    BaselineModel,
    Capability,
    DelegationPolicy,
    ModelProvider,
    OperationalConstraints,
    ProcessType,
)

# Mapping of CrewAI LLM class names to providers
CREWAI_LLM_PROVIDERS: dict[str, ModelProvider] = {
    "ChatOpenAI": ModelProvider.OPENAI,
    "OpenAI": ModelProvider.OPENAI,
    "AzureChatOpenAI": ModelProvider.OPENAI,
    "ChatAnthropic": ModelProvider.ANTHROPIC,
    "Anthropic": ModelProvider.ANTHROPIC,
    "ChatGoogleGenerativeAI": ModelProvider.GOOGLE,
    "Gemini": ModelProvider.GOOGLE,
    "ChatGroq": ModelProvider.UNKNOWN,  # Groq not in standard providers
    "Ollama": ModelProvider.LOCAL,
    "ChatOllama": ModelProvider.LOCAL,
    "LlamaCpp": ModelProvider.LOCAL,
}


def introspect_llm(llm: Any) -> BaselineModel:
    """Extract LLM metadata from a CrewAI agent's LLM."""
    class_name = type(llm).__name__

    # Detect provider
    provider = CREWAI_LLM_PROVIDERS.get(class_name, ModelProvider.UNKNOWN)

    # Get model name
    model_name = "unknown"
    for attr in ["model_name", "model", "model_id"]:
        if hasattr(llm, attr):
            value = getattr(llm, attr)
            if value and isinstance(value, str):
                model_name = value
                break

    # Get temperature
    temperature = None
    if hasattr(llm, "temperature"):
        temp = llm.temperature
        if isinstance(temp, (int, float)):
            temperature = float(temp)

    # Get max tokens
    max_tokens = None
    for attr in ["max_tokens", "max_output_tokens"]:
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
    """Extract tool capabilities from a list of tools."""
    capabilities = []

    for tool in tools:
        name = getattr(tool, "name", type(tool).__name__)
        description = getattr(tool, "description", "")

        # Extract parameter schema if available
        parameters: dict[str, Any] = {}
        if hasattr(tool, "args_schema") and tool.args_schema:
            try:
                if hasattr(tool.args_schema, "model_json_schema"):
                    parameters = tool.args_schema.model_json_schema()
                elif hasattr(tool.args_schema, "schema"):
                    parameters = tool.args_schema.schema()
            except Exception:
                pass

        # Assess risk level
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


def introspect_agent(
    agent: Any,
) -> tuple[
    BaselineModel | None, list[Capability], OperationalConstraints, dict[str, Any]
]:
    """Introspect a CrewAI Agent."""
    baseline_model = None
    capabilities: list[Capability] = []
    constraints = OperationalConstraints()
    context: dict[str, Any] = {}

    # Extract role information
    role = getattr(agent, "role", None)
    goal = getattr(agent, "goal", None)
    backstory = getattr(agent, "backstory", None)

    if role:
        context["role"] = AgentRole(
            role_name=str(role),
            goal=str(goal) if goal else "",
            backstory=str(backstory) if backstory else "",
        ).model_dump()

    # Extract LLM
    llm = getattr(agent, "llm", None)
    if llm is not None:
        baseline_model = introspect_llm(llm)

    # Extract tools
    tools = getattr(agent, "tools", None)
    if tools:
        capabilities = introspect_tools(tools)

    # Extract delegation settings
    allow_delegation = getattr(agent, "allow_delegation", False)
    context["delegation"] = DelegationPolicy(
        can_delegate=bool(allow_delegation),
        requires_approval_to_delegate=True,
    ).model_dump()

    # Extract other settings
    verbose = getattr(agent, "verbose", False)
    context["verbose"] = verbose

    memory = getattr(agent, "memory", None)
    if memory is not None:
        context["has_memory"] = True

    max_iter = getattr(agent, "max_iter", None)
    if max_iter:
        constraints.max_iterations = int(max_iter)

    max_rpm = getattr(agent, "max_rpm", None)
    if max_rpm:
        context["max_rpm"] = max_rpm

    # Check for function calling LLM
    function_calling_llm = getattr(agent, "function_calling_llm", None)
    if function_calling_llm:
        context["has_function_calling_llm"] = True

    return baseline_model, capabilities, constraints, context


def introspect_crew(
    crew: Any,
) -> tuple[
    BaselineModel | None, list[Capability], OperationalConstraints, dict[str, Any]
]:
    """Introspect a CrewAI Crew."""
    baseline_model = None
    capabilities: list[Capability] = []
    constraints = OperationalConstraints()
    context: dict[str, Any] = {}

    # Extract agents
    agents = getattr(crew, "agents", [])
    context["agent_count"] = len(agents)
    context["agent_roles"] = []

    for agent in agents:
        role = getattr(agent, "role", None)
        if role:
            context["agent_roles"].append(str(role))

        # Use first agent's LLM as baseline
        if baseline_model is None:
            llm = getattr(agent, "llm", None)
            if llm:
                baseline_model = introspect_llm(llm)

        # Collect all tools from all agents
        tools = getattr(agent, "tools", None)
        if tools:
            for cap in introspect_tools(tools):
                if cap.name not in [c.name for c in capabilities]:
                    capabilities.append(cap)

    # Extract tasks
    tasks = getattr(crew, "tasks", [])
    context["task_count"] = len(tasks)
    context["task_descriptions"] = []
    for task in tasks:
        desc = getattr(task, "description", None)
        if desc:
            context["task_descriptions"].append(str(desc)[:100])

    # Extract process type
    process = getattr(crew, "process", None)
    if process:
        process_str = str(process).lower()
        if "sequential" in process_str:
            context["process_type"] = ProcessType.SEQUENTIAL.value
        elif "hierarchical" in process_str:
            context["process_type"] = ProcessType.HIERARCHICAL.value
        else:
            context["process_type"] = process_str

    # Extract verbose and other settings
    verbose = getattr(crew, "verbose", False)
    context["verbose"] = verbose

    # Memory settings
    memory = getattr(crew, "memory", None)
    if memory is not None:
        context["shared_memory"] = True

    # Max RPM for rate limiting
    max_rpm = getattr(crew, "max_rpm", None)
    if max_rpm:
        context["max_rpm"] = max_rpm

    # Manager LLM for hierarchical process
    manager_llm = getattr(crew, "manager_llm", None)
    if manager_llm:
        context["has_manager_llm"] = True

    return baseline_model, capabilities, constraints, context


def introspect_any(
    obj: Any,
) -> tuple[
    BaselineModel | None, list[Capability], OperationalConstraints, dict[str, Any]
]:
    """
    Introspect any CrewAI object to extract metadata.

    Auto-detects whether it's an Agent or Crew.
    """
    class_name = type(obj).__name__

    if class_name == "Crew" or (hasattr(obj, "agents") and hasattr(obj, "tasks")):
        return introspect_crew(obj)
    else:
        return introspect_agent(obj)


@dataclass
class CrewAIIntegration(BaseIntrospector):
    """
    CrewAI integration for AgentFacts.

    Provides introspection for CrewAI Agent and Crew objects.
    """

    @property
    def framework_name(self) -> str:
        return "crewai"

    def can_introspect(self, obj: Any) -> bool:
        """Check if object is a CrewAI Agent or Crew."""
        module = getattr(type(obj), "__module__", "")

        # Check module name
        if "crewai" in module.lower():
            return True

        # Check class name patterns
        class_name = type(obj).__name__
        crewai_classes = {"Agent", "Crew", "Task"}
        if class_name in crewai_classes and "crewai" in str(type(obj)):
            return True

        # Check for CrewAI-specific attributes
        if hasattr(obj, "role") and hasattr(obj, "goal") and hasattr(obj, "backstory"):
            return True
        return (
            hasattr(obj, "agents") and hasattr(obj, "tasks") and hasattr(obj, "process")
        )

    # Alias for Integration protocol

    def introspect(self, obj: Any) -> IntrospectionResult:
        """Introspect a CrewAI object."""
        baseline, capabilities, constraints, extra_context = introspect_any(obj)

        context: dict[ContextKey | str, Any] = {
            ContextKey.FRAMEWORK: "crewai",
            ContextKey.TOOL_COUNT: len(capabilities),
        }

        class_name = type(obj).__name__
        if class_name == "Crew" or (hasattr(obj, "agents") and hasattr(obj, "tasks")):
            process = getattr(obj, "process", None)
            if process:
                context[ContextKey.AGENT_TYPE] = f"CREW_{str(process).upper()}"
            else:
                context[ContextKey.AGENT_TYPE] = "CREW"
        elif hasattr(obj, "allow_delegation") and getattr(
            obj, "allow_delegation", False
        ):
            context[ContextKey.AGENT_TYPE] = "DELEGATING_AGENT"
        else:
            context[ContextKey.AGENT_TYPE] = "CREWAI_AGENT"

        # Merge extra context
        context.update(extra_context)

        return IntrospectionResult(
            framework="crewai",
            baseline_model=baseline,
            capabilities=capabilities,
            constraints=constraints,
            context=context,
        )
