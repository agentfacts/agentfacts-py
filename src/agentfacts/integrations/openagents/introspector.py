"""
OpenAgents introspection for AgentFacts SDK.

Extracts metadata from OpenAgents agents, networks, and configurations.
"""

import contextlib
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
    ProcessType,
)

# Mapping of OpenAgents provider names to our ModelProvider enum
OPENAGENTS_PROVIDERS: dict[str, ModelProvider] = {
    "openai": ModelProvider.OPENAI,
    "anthropic": ModelProvider.ANTHROPIC,
    "claude": ModelProvider.ANTHROPIC,
    "google": ModelProvider.GOOGLE,
    "gemini": ModelProvider.GOOGLE,
    "mistral": ModelProvider.MISTRAL,
    "deepseek": ModelProvider.UNKNOWN,
    "ollama": ModelProvider.LOCAL,
    "local": ModelProvider.LOCAL,
}


def _detect_provider_from_model(model_name: str) -> ModelProvider:
    """Detect provider from model name."""
    model_lower = model_name.lower()

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


def introspect_tools(tools: Any) -> list[Capability]:
    """Extract tool capabilities."""
    capabilities = []

    # Handle various tool formats
    tool_list = tools if hasattr(tools, "__iter__") else [tools]

    for tool in tool_list:
        name = getattr(tool, "name", None) or type(tool).__name__
        description = getattr(tool, "description", "") or ""

        # Extract parameters if available
        parameters: dict[str, Any] = {}
        schema = getattr(tool, "schema", None) or getattr(tool, "parameters", None)
        if schema:
            if hasattr(schema, "model_json_schema"):
                with contextlib.suppress(Exception):
                    parameters = schema.model_json_schema()
            elif isinstance(schema, dict):
                parameters = schema

        risk_level = RiskAssessor.assess_tool_risk(str(name), str(description))

        capabilities.append(
            Capability(
                name=str(name),
                description=str(description)[:200],
                parameters=parameters,
                risk_level=risk_level,
            )
        )

    return capabilities


def introspect_agent_config(
    config: Any,
) -> tuple[
    BaselineModel | None, list[Capability], OperationalConstraints, dict[str, Any]
]:
    """Introspect an AgentConfig."""
    baseline_model = None
    capabilities: list[Capability] = []
    constraints = OperationalConstraints()
    context: dict[str, Any] = {}

    context["config_type"] = type(config).__name__

    # Extract model configuration
    model_name = getattr(config, "model_name", None)
    provider = getattr(config, "provider", None)
    instruction = getattr(config, "instruction", None)

    if model_name:
        context["model_name"] = model_name

        # Detect provider
        model_provider = ModelProvider.UNKNOWN
        if provider:
            context["provider"] = str(provider)
            provider_lower = str(provider).lower()
            model_provider = OPENAGENTS_PROVIDERS.get(
                provider_lower, ModelProvider.UNKNOWN
            )

            if model_provider == ModelProvider.UNKNOWN:
                model_provider = _detect_provider_from_model(model_name)

        # Extract temperature if available
        temperature = getattr(config, "temperature", None)
        max_tokens = getattr(config, "max_tokens", None)

        baseline_model = BaselineModel(
            name=model_name,
            provider=model_provider,
            temperature=float(temperature) if temperature else None,
            max_tokens=int(max_tokens) if max_tokens else None,
        )

    # Extract instruction/system prompt
    if instruction:
        context["instruction"] = str(instruction)[:500]

    # Extract max iterations
    max_iterations = getattr(config, "max_iterations", None)
    if max_iterations:
        constraints.max_iterations = int(max_iterations)

    # Extract tools
    tools = getattr(config, "tools", None)
    if tools:
        capabilities.extend(introspect_tools(tools))

    # Extract MCP servers
    mcp_servers = getattr(config, "mcp_servers", None)
    if mcp_servers:
        context["mcp_server_count"] = (
            len(mcp_servers) if hasattr(mcp_servers, "__len__") else 1
        )
        capabilities.append(
            Capability(
                name="mcp_integration",
                description="Model Context Protocol server integration",
                risk_level="medium",
            )
        )

    # LLM reasoning capability
    capabilities.append(
        Capability(
            name="llm_reasoning",
            description="LLM-based reasoning and response generation",
            risk_level="low",
        )
    )

    return baseline_model, capabilities, constraints, context


def introspect_worker_agent(
    agent: Any,
) -> tuple[
    BaselineModel | None, list[Capability], OperationalConstraints, dict[str, Any]
]:
    """Introspect a WorkerAgent implementation."""
    baseline_model = None
    capabilities: list[Capability] = []
    constraints = OperationalConstraints()
    context: dict[str, Any] = {}

    class_name = type(agent).__name__
    context["agent_class"] = class_name

    # Extract agent ID
    agent_id = getattr(agent, "default_agent_id", None) or getattr(
        agent, "agent_id", None
    )
    if agent_id:
        context["agent_id"] = str(agent_id)

    # Check for AgentConfig
    config = getattr(agent, "config", None) or getattr(agent, "agent_config", None)
    if config:
        baseline_model, config_caps, config_constraints, config_context = (
            introspect_agent_config(config)
        )
        capabilities.extend(config_caps)
        constraints = config_constraints
        context.update(config_context)

    # Fallback to direct model name if config isn't present
    if baseline_model is None:
        model_name = getattr(agent, "model", None) or getattr(agent, "model_name", None)
        if isinstance(model_name, str) and model_name:
            baseline_model = BaselineModel(
                name=model_name,
                provider=_detect_provider_from_model(model_name),
            )
            context["model_name"] = model_name

    # Check for run_agent method (indicates LLM capability)
    if hasattr(agent, "run_agent") and not any(
        c.name == "llm_reasoning" for c in capabilities
    ):
        capabilities.append(
            Capability(
                name="llm_reasoning",
                description="LLM-based agent execution",
                risk_level="low",
            )
        )

    # Check for on_channel_post (message handling)
    if hasattr(agent, "on_channel_post"):
        capabilities.append(
            Capability(
                name="channel_messaging",
                description="Handle channel messages in OpenAgents network",
                risk_level="low",
            )
        )

    # Check for network capabilities
    if hasattr(agent, "network") or hasattr(agent, "_network"):
        context["network_connected"] = True

    return baseline_model, capabilities, constraints, context


def introspect_network_config(
    config: Any,
) -> tuple[
    BaselineModel | None, list[Capability], OperationalConstraints, dict[str, Any]
]:
    """Introspect a NetworkConfig."""
    capabilities: list[Capability] = []
    constraints = OperationalConstraints()
    context: dict[str, Any] = {}

    context["config_type"] = "NetworkConfig"

    # Extract network settings
    name = getattr(config, "name", None)
    if name:
        context["network_name"] = str(name)

    mode = getattr(config, "mode", None)
    if mode:
        context["network_mode"] = str(mode)
        if str(mode).lower() == "centralized":
            context["process_type"] = ProcessType.HIERARCHICAL.value
        else:
            context["process_type"] = ProcessType.EVENT_DRIVEN.value

    transport = getattr(config, "transport", None)
    if transport:
        context["transport"] = str(transport)

    # Network capability
    capabilities.append(
        Capability(
            name="network_coordination",
            description=f"OpenAgents network coordination ({transport or 'default'} transport)",
            risk_level="medium",
        )
    )

    return None, capabilities, constraints, context


def introspect_network(
    network: Any,
) -> tuple[
    BaselineModel | None, list[Capability], OperationalConstraints, dict[str, Any]
]:
    """Introspect a Network instance."""
    baseline_model = None
    capabilities: list[Capability] = []
    constraints = OperationalConstraints()
    context: dict[str, Any] = {}

    context["object_type"] = "Network"

    # Extract config if available
    config = getattr(network, "config", None)
    if config:
        _, config_caps, _, config_context = introspect_network_config(config)
        capabilities.extend(config_caps)
        context.update(config_context)

    # Extract agents if available
    agents = getattr(network, "agents", None) or getattr(network, "_agents", None)
    if agents:
        if isinstance(agents, dict):
            context["agent_count"] = len(agents)
            context["agent_ids"] = list(agents.keys())[:10]
        elif hasattr(agents, "__len__"):
            context["agent_count"] = len(agents)

    # Messaging capability
    if hasattr(network, "send_message"):
        capabilities.append(
            Capability(
                name="message_routing",
                description="Route messages between agents",
                risk_level="low",
            )
        )

    # Registration capability
    if hasattr(network, "register_agent"):
        capabilities.append(
            Capability(
                name="agent_registration",
                description="Register new agents to the network",
                risk_level="medium",
            )
        )

    context["process_type"] = ProcessType.EVENT_DRIVEN.value

    return baseline_model, capabilities, constraints, context


def introspect_any(
    obj: Any,
) -> tuple[
    BaselineModel | None, list[Capability], OperationalConstraints, dict[str, Any]
]:
    """Introspect any OpenAgents object to extract metadata."""
    class_name = type(obj).__name__

    if class_name == "AgentConfig" or (
        hasattr(obj, "model_name") and hasattr(obj, "instruction")
    ):
        return introspect_agent_config(obj)
    elif class_name == "NetworkConfig" or (
        hasattr(obj, "mode") and hasattr(obj, "transport")
    ):
        return introspect_network_config(obj)
    elif class_name == "Network" or (
        hasattr(obj, "register_agent") and hasattr(obj, "send_message")
    ):
        return introspect_network(obj)
    elif hasattr(obj, "default_agent_id") or hasattr(obj, "run_agent"):
        return introspect_worker_agent(obj)
    else:
        return introspect_agent_config(obj)


@dataclass
class OpenAgentsIntegration(BaseIntrospector):
    """
    OpenAgents integration for AgentFacts.

    Supports:
    - AgentConfig: LLM configuration for agents
    - WorkerAgent: Base agent implementation
    - NetworkConfig: Network configuration
    - Network: Agent network instances
    """

    @property
    def framework_name(self) -> str:
        return "openagents"

    def can_introspect(self, obj: Any) -> bool:
        """Check if object is an OpenAgents component."""
        module = getattr(type(obj), "__module__", "")

        # Check module name
        if "openagents" in module.lower():
            return True

        # Check class name patterns
        class_name = type(obj).__name__
        openagents_classes = {
            "AgentConfig",
            "WorkerAgent",
            "NetworkConfig",
            "AgentTool",
            "MCPServerConfig",
            "LLMAssistantAgent",
            "ChannelMessageContext",
            "Network",
        }
        if class_name in openagents_classes:
            return True

        # Check for OpenAgents-specific attributes
        if (
            hasattr(obj, "model_name")
            and hasattr(obj, "instruction")
            and hasattr(obj, "provider")
        ):
            return True
        if hasattr(obj, "default_agent_id") and hasattr(obj, "run_agent"):
            return True
        if hasattr(obj, "mode") and hasattr(obj, "transport") and hasattr(obj, "host"):
            return True
        return hasattr(obj, "register_agent") and hasattr(obj, "send_message")

    # Alias for Integration protocol

    def introspect(self, obj: Any) -> IntrospectionResult:
        """Introspect an OpenAgents object."""
        baseline, capabilities, constraints, extra_context = introspect_any(obj)

        context: dict[ContextKey | str, Any] = {
            ContextKey.FRAMEWORK: "openagents",
            ContextKey.TOOL_COUNT: len(capabilities),
        }

        class_name = type(obj).__name__
        if class_name == "AgentConfig":
            context[ContextKey.AGENT_TYPE] = "OPENAGENTS_CONFIG"
        elif class_name == "NetworkConfig":
            context[ContextKey.AGENT_TYPE] = "OPENAGENTS_NETWORK_CONFIG"
        elif class_name == "Network":
            context[ContextKey.AGENT_TYPE] = "OPENAGENTS_NETWORK"
        elif hasattr(obj, "default_agent_id") or hasattr(obj, "run_agent"):
            context[ContextKey.AGENT_TYPE] = "OPENAGENTS_WORKER_AGENT"
        else:
            context[ContextKey.AGENT_TYPE] = f"OPENAGENTS_{class_name.upper()}"

        # Merge extra context
        context.update(extra_context)

        return IntrospectionResult(
            framework="openagents",
            baseline_model=baseline,
            capabilities=capabilities,
            constraints=constraints,
            context=context,
        )
