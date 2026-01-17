"""
AutoGen introspection for AgentFacts SDK.

Extracts metadata from Microsoft AutoGen agents and group chats.
"""

from dataclasses import dataclass
from typing import Any

from agentfacts.integrations.base import (
    BaseIntrospector,
    ContextKey,
    IntrospectionResult,
)
from agentfacts.models import (
    BaselineModel,
    Capability,
    ModelProvider,
    OperationalConstraints,
    ProcessType,
)


def _detect_provider(model_name: str, api_base: str) -> ModelProvider:
    """Detect provider from model name and API base."""
    model_lower = model_name.lower()
    api_lower = api_base.lower()

    # Check API base first
    if "openai" in api_lower or "azure" in api_lower:
        return ModelProvider.OPENAI
    if "anthropic" in api_lower:
        return ModelProvider.ANTHROPIC
    if "google" in api_lower or "vertex" in api_lower:
        return ModelProvider.GOOGLE

    # Check model name
    if model_lower.startswith("gpt-") or "davinci" in model_lower:
        return ModelProvider.OPENAI
    if "claude" in model_lower:
        return ModelProvider.ANTHROPIC
    if "gemini" in model_lower or "palm" in model_lower:
        return ModelProvider.GOOGLE
    if "mistral" in model_lower or "mixtral" in model_lower:
        return ModelProvider.MISTRAL
    if "llama" in model_lower:
        return ModelProvider.META

    return ModelProvider.UNKNOWN


def _assess_function_risk(name: str, description: str) -> str:
    """Assess function risk level."""
    combined = (name + " " + description).lower()

    high_risk = [
        "execute",
        "shell",
        "bash",
        "command",
        "delete",
        "remove",
        "admin",
        "sudo",
        "system",
        "write",
        "modify",
        "payment",
        "database",
        "sql",
        "credential",
        "password",
        "code",
    ]
    for keyword in high_risk:
        if keyword in combined:
            return "high"

    medium_risk = [
        "file",
        "network",
        "http",
        "request",
        "email",
        "send",
        "create",
        "update",
        "api",
        "browser",
        "web",
    ]
    for keyword in medium_risk:
        if keyword in combined:
            return "medium"

    return "low"


def introspect_llm_config(llm_config: dict[str, Any]) -> BaselineModel:
    """Extract model info from AutoGen llm_config."""
    model_name = "unknown"
    provider = ModelProvider.UNKNOWN
    temperature = None
    max_tokens = None

    # AutoGen supports config_list for multiple models
    config_list = llm_config.get("config_list", [])
    if config_list and len(config_list) > 0:
        first_config = config_list[0]
        model_name = first_config.get("model", "unknown")

        # Detect provider from model name or API base
        api_base = first_config.get("api_base", "")
        provider = _detect_provider(model_name, api_base)

    # Direct model specification
    if model_name == "unknown":
        model_name = llm_config.get("model", "unknown")
        provider = _detect_provider(model_name, "")

    # Extract temperature
    temperature = llm_config.get("temperature")
    if temperature is not None:
        temperature = float(temperature)

    # Extract max tokens
    max_tokens = llm_config.get("max_tokens")

    return BaselineModel(
        name=model_name,
        provider=provider,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def _extract_registered_functions(
    agent: Any, capabilities: list[Capability], context: dict[str, Any]
) -> None:
    """Extract registered functions/tools from an agent."""
    # AutoGen agents can have registered functions
    function_map = getattr(agent, "_function_map", None)
    if function_map and isinstance(function_map, dict):
        for func_name, func in function_map.items():
            # Avoid duplicates
            if any(c.name == func_name for c in capabilities):
                continue

            description = ""
            if hasattr(func, "__doc__") and func.__doc__:
                description = func.__doc__[:200]

            risk_level = _assess_function_risk(func_name, description)

            capabilities.append(
                Capability(
                    name=func_name,
                    description=description,
                    risk_level=risk_level,
                )
            )

    # Check for reply functions
    reply_func_list = getattr(agent, "_reply_func_list", None)
    if reply_func_list:
        context["registered_reply_functions"] = len(reply_func_list)


def introspect_agent(
    agent: Any,
) -> tuple[
    BaselineModel | None, list[Capability], OperationalConstraints, dict[str, Any]
]:
    """Introspect an AutoGen agent."""
    baseline_model = None
    capabilities: list[Capability] = []
    constraints = OperationalConstraints()
    context: dict[str, Any] = {}

    # Extract agent name
    name = getattr(agent, "name", None)
    if name:
        context["agent_name"] = name

    # Extract description (used for agent discovery in AutoGen)
    description = getattr(agent, "description", None)
    if description:
        context["description"] = description

    # Extract system message
    system_message = getattr(agent, "system_message", None)
    if system_message:
        context["system_message"] = str(system_message)[:500]

    # Extract LLM config
    llm_config = getattr(agent, "llm_config", None)
    if llm_config and isinstance(llm_config, dict):
        baseline_model = introspect_llm_config(llm_config)
        context["llm_config_keys"] = list(llm_config.keys())

    # Extract code execution config
    code_execution_config = getattr(agent, "code_execution_config", None)
    if code_execution_config:
        context["code_execution_enabled"] = True
        if isinstance(code_execution_config, dict):
            work_dir = code_execution_config.get("work_dir")
            if work_dir:
                context["code_work_dir"] = str(work_dir)
            use_docker = code_execution_config.get("use_docker", False)
            context["code_uses_docker"] = use_docker

        # Code execution is a high-risk capability
        capabilities.append(
            Capability(
                name="code_execution",
                description="Execute Python code in local environment",
                risk_level="high",
                requires_approval=True,
            )
        )

    # Extract human input mode
    human_input_mode = getattr(agent, "human_input_mode", None)
    if human_input_mode:
        context["human_input_mode"] = str(human_input_mode)
        if str(human_input_mode).upper() == "ALWAYS":
            constraints.requires_human_approval = True

    # Extract registered functions/tools
    _extract_registered_functions(agent, capabilities, context)

    # Extract max consecutive auto reply
    max_consecutive = getattr(agent, "max_consecutive_auto_reply", None)
    if max_consecutive:
        constraints.max_iterations = int(max_consecutive)

    # Check agent type
    class_name = type(agent).__name__
    context["agent_class"] = class_name

    if class_name == "UserProxyAgent":
        context["is_user_proxy"] = True
    elif class_name == "AssistantAgent":
        context["is_assistant"] = True

    return baseline_model, capabilities, constraints, context


def introspect_group_chat(
    group_chat: Any,
) -> tuple[
    BaselineModel | None, list[Capability], OperationalConstraints, dict[str, Any]
]:
    """Introspect an AutoGen GroupChat."""
    baseline_model = None
    capabilities: list[Capability] = []
    constraints = OperationalConstraints()
    context: dict[str, Any] = {}

    # Extract agents
    agents = getattr(group_chat, "agents", [])
    context["agent_count"] = len(agents)
    context["agent_names"] = []
    context["agent_descriptions"] = []

    for agent in agents:
        name = getattr(agent, "name", None)
        if name:
            context["agent_names"].append(name)

        description = getattr(agent, "description", None)
        if description:
            context["agent_descriptions"].append(str(description)[:100])

        # Use first agent with llm_config as baseline
        if baseline_model is None:
            llm_config = getattr(agent, "llm_config", None)
            if llm_config and isinstance(llm_config, dict):
                baseline_model = introspect_llm_config(llm_config)

        # Collect capabilities from all agents
        _extract_registered_functions(agent, capabilities, {})

        # Check for code execution
        code_config = getattr(agent, "code_execution_config", None)
        if code_config:
            cap_exists = any(c.name == "code_execution" for c in capabilities)
            if not cap_exists:
                capabilities.append(
                    Capability(
                        name="code_execution",
                        description="Execute code (available to one or more agents)",
                        risk_level="high",
                        requires_approval=True,
                    )
                )

    # Extract max rounds
    max_round = getattr(group_chat, "max_round", None)
    if max_round:
        context["max_round"] = max_round
        constraints.max_iterations = int(max_round)

    # Extract speaker selection method
    speaker_selection_method = getattr(group_chat, "speaker_selection_method", None)
    if speaker_selection_method:
        context["speaker_selection_method"] = str(speaker_selection_method)

    # Extract admin name
    admin_name = getattr(group_chat, "admin_name", None)
    if admin_name:
        context["admin_name"] = admin_name

    # Process type is event-driven for GroupChat
    context["process_type"] = ProcessType.EVENT_DRIVEN.value

    # Extract allow repeat speaker
    allow_repeat = getattr(group_chat, "allow_repeat_speaker", True)
    context["allow_repeat_speaker"] = allow_repeat

    return baseline_model, capabilities, constraints, context


def introspect_group_chat_manager(
    manager: Any,
) -> tuple[
    BaselineModel | None, list[Capability], OperationalConstraints, dict[str, Any]
]:
    """Introspect a GroupChatManager."""
    baseline_model = None
    capabilities: list[Capability] = []
    constraints = OperationalConstraints()
    context: dict[str, Any] = {}

    context["is_manager"] = True

    # Extract the underlying GroupChat
    groupchat = getattr(manager, "groupchat", None)
    if groupchat:
        baseline_model, capabilities, constraints, gc_context = introspect_group_chat(
            groupchat
        )
        context.update(gc_context)

    # Extract manager's own LLM config
    llm_config = getattr(manager, "llm_config", None)
    if llm_config and isinstance(llm_config, dict) and baseline_model is None:
        baseline_model = introspect_llm_config(llm_config)

    # Manager capability
    capabilities.append(
        Capability(
            name="group_chat_management",
            description="Coordinates multi-agent group conversations",
            risk_level="medium",
        )
    )

    return baseline_model, capabilities, constraints, context


def introspect_any(
    obj: Any,
) -> tuple[
    BaselineModel | None, list[Capability], OperationalConstraints, dict[str, Any]
]:
    """
    Introspect any AutoGen object to extract metadata.

    Auto-detects the object type.
    """
    class_name = type(obj).__name__

    if class_name == "GroupChat" or (
        hasattr(obj, "agents")
        and hasattr(obj, "max_round")
        and hasattr(obj, "messages")
    ):
        return introspect_group_chat(obj)
    elif class_name == "GroupChatManager":
        return introspect_group_chat_manager(obj)
    else:
        return introspect_agent(obj)


@dataclass
class AutoGenIntegration(BaseIntrospector):
    """
    AutoGen integration for AgentFacts.

    Supports:
    - AssistantAgent
    - UserProxyAgent
    - GroupChat
    - GroupChatManager
    - ConversableAgent (base class)
    """

    @property
    def framework_name(self) -> str:
        return "autogen"

    def can_introspect(self, obj: Any) -> bool:
        """Check if object is an AutoGen agent or GroupChat."""
        module = getattr(type(obj), "__module__", "")

        # Check module name
        if "autogen" in module.lower():
            return True

        # Check class name patterns
        class_name = type(obj).__name__
        autogen_classes = {
            "AssistantAgent",
            "UserProxyAgent",
            "ConversableAgent",
            "GroupChat",
            "GroupChatManager",
        }
        if class_name in autogen_classes:
            return True

        # Check for AutoGen-specific attributes
        # AssistantAgent has llm_config and system_message
        if hasattr(obj, "llm_config") and hasattr(obj, "system_message"):
            return True
        # GroupChat has agents and messages
        return (
            hasattr(obj, "agents")
            and hasattr(obj, "messages")
            and hasattr(obj, "max_round")
        )

    # Alias for Integration protocol

    def introspect(self, obj: Any) -> IntrospectionResult:
        """Introspect an AutoGen object."""
        baseline, capabilities, constraints, extra_context = introspect_any(obj)

        context: dict[ContextKey | str, Any] = {
            ContextKey.FRAMEWORK: "autogen",
            ContextKey.TOOL_COUNT: len(capabilities),
        }

        class_name = type(obj).__name__
        if class_name == "GroupChat":
            context[ContextKey.AGENT_TYPE] = "AUTOGEN_GROUP_CHAT"
        elif class_name == "GroupChatManager":
            context[ContextKey.AGENT_TYPE] = "AUTOGEN_GROUP_MANAGER"
        elif class_name == "UserProxyAgent":
            context[ContextKey.AGENT_TYPE] = "AUTOGEN_USER_PROXY"
        elif class_name == "AssistantAgent":
            context[ContextKey.AGENT_TYPE] = "AUTOGEN_ASSISTANT"
        else:
            context[ContextKey.AGENT_TYPE] = f"AUTOGEN_{class_name.upper()}"

        # Merge extra context
        context.update(extra_context)

        return IntrospectionResult(
            framework="autogen",
            baseline_model=baseline,
            capabilities=capabilities,
            constraints=constraints,
            context=context,
        )
