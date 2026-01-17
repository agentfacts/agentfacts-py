"""
Hugging Face introspection for AgentFacts SDK.

Extracts metadata from Hugging Face agent frameworks:
- smolagents: CodeAgent, ToolCallingAgent, MultiStepAgent
- tiny-agents: agent.json configuration files
- Hub agents: agents loaded from/pushed to HF Hub
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

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

# Mapping of model prefixes/providers to our ModelProvider enum
HUGGINGFACE_PROVIDERS: dict[str, ModelProvider] = {
    "nebius": ModelProvider.UNKNOWN,
    "together": ModelProvider.UNKNOWN,
    "fireworks": ModelProvider.UNKNOWN,
    "sambanova": ModelProvider.UNKNOWN,
    "cerebras": ModelProvider.UNKNOWN,
    "hyperbolic": ModelProvider.UNKNOWN,
    "novita": ModelProvider.UNKNOWN,
    "replicate": ModelProvider.UNKNOWN,
    "openai": ModelProvider.OPENAI,
    "anthropic": ModelProvider.ANTHROPIC,
    "google": ModelProvider.GOOGLE,
    "mistral": ModelProvider.MISTRAL,
    "cohere": ModelProvider.UNKNOWN,
    "ollama": ModelProvider.LOCAL,
    "transformers": ModelProvider.LOCAL,
    "local": ModelProvider.LOCAL,
}


def _detect_provider_from_model(model_id: Any) -> ModelProvider:
    """Detect provider from model ID."""
    if not isinstance(model_id, str):
        return ModelProvider.UNKNOWN
    model_lower = model_id.lower()

    if (
        model_lower.startswith("gpt-")
        or "davinci" in model_lower
        or model_lower.startswith("o1")
    ):
        return ModelProvider.OPENAI
    if "claude" in model_lower:
        return ModelProvider.ANTHROPIC
    if "gemini" in model_lower or "palm" in model_lower:
        return ModelProvider.GOOGLE
    if "mistral" in model_lower or "mixtral" in model_lower:
        return ModelProvider.MISTRAL
    if "llama" in model_lower:
        return ModelProvider.META
    if "qwen" in model_lower:
        return ModelProvider.UNKNOWN

    return ModelProvider.UNKNOWN


def _extract_server_name(command: str, args: list[str] | None) -> str:
    """Extract a friendly name from MCP server command."""
    if not command:
        return "unknown_server"

    if "@" in command:
        return command.split("/")[-1].replace("@", "").replace("-", "_")

    if not args:
        return command.replace("-", "_")

    for arg in args:
        if "/" in arg:
            parts = arg.split("/")
            scope = parts[0].lstrip("@") if parts[0].startswith("@") else None
            after_slash = parts[-1]
            pkg_name = after_slash.split("@")[0] if "@" in after_slash else after_slash

            if scope and pkg_name in ("mcp", "mcp-server", "server"):
                return scope.replace("-", "_")
            return pkg_name.replace("-", "_")
        elif arg.startswith("mcp-") or arg.startswith("@"):
            name = arg.split("@")[0]
            return name.replace("-", "_")

    return command.replace("-", "_")


def _extract_model_info(model: Any) -> BaselineModel:
    """Extract BaselineModel from a smolagents model object."""
    model_id: str | None = None

    if isinstance(model, str):
        model_id = model
        return BaselineModel(
            name=model_id,
            provider=_detect_provider_from_model(model_id),
            temperature=None,
            max_tokens=None,
        )

    if isinstance(model, dict):
        model_id = (
            model.get("model_id") or model.get("model") or model.get("model_name")
        )
        if model_id and isinstance(model_id, str):
            return BaselineModel(
                name=model_id,
                provider=_detect_provider_from_model(model_id),
                temperature=None,
                max_tokens=None,
            )

    class_name = type(model).__name__
    model_id = "unknown"
    provider = ModelProvider.UNKNOWN

    _sentinel = object()
    for attr in ["model_id", "model", "model_name", "_model_id"]:
        value = getattr(model, attr, _sentinel)
        if value is not _sentinel and value and isinstance(value, str):
            model_id = value
            break

    if "InferenceClientModel" in class_name:
        provider = ModelProvider.UNKNOWN
    elif "LiteLLMModel" in class_name:
        provider = _detect_provider_from_model(model_id)
    elif "TransformersModel" in class_name:
        provider = ModelProvider.LOCAL

    if provider == ModelProvider.UNKNOWN:
        provider = _detect_provider_from_model(model_id)

    temperature = None
    temp = getattr(model, "temperature", None)
    if isinstance(temp, (int, float)):
        temperature = float(temp)

    max_tokens = None
    for attr in ["max_tokens", "max_new_tokens"]:
        value = getattr(model, attr, _sentinel)
        if value is not _sentinel and isinstance(value, int) and value > 0:
            max_tokens = value
            break

    return BaselineModel(
        name=model_id,
        provider=provider,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def introspect_tools(tools: Any) -> list[Capability]:
    """Extract tool capabilities."""
    capabilities = []

    tool_list = tools
    if isinstance(tools, dict):
        tool_list = list(tools.values())
    else:
        inner_tools = getattr(tools, "tools", None)
        if inner_tools is not None:
            tool_list = inner_tools
        elif not hasattr(tools, "__iter__"):
            tool_list = [tools]

    for tool in tool_list:
        name = getattr(tool, "name", None) or type(tool).__name__
        description = getattr(tool, "description", "") or ""

        parameters: dict[str, Any] = {}
        inputs = getattr(tool, "inputs", None)
        if inputs and isinstance(inputs, dict):
            parameters = inputs

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


def introspect_smolagent(
    agent: Any,
) -> tuple[
    BaselineModel | None, list[Capability], OperationalConstraints, dict[str, Any]
]:
    """Introspect a smolagents agent (CodeAgent, ToolCallingAgent, etc.)."""
    baseline_model = None
    capabilities: list[Capability] = []
    constraints = OperationalConstraints()
    context: dict[str, Any] = {}

    class_name = type(agent).__name__
    context["agent_type"] = class_name

    if "CodeAgent" in class_name:
        context["execution_mode"] = "code"
        capabilities.append(
            Capability(
                name="code_execution",
                description="Executes Python code to perform actions",
                risk_level="high",
                requires_approval=True,
            )
        )
    elif "ToolCallingAgent" in class_name:
        context["execution_mode"] = "json_tool_calling"

    # Extract model
    model = getattr(agent, "model", None)
    if model is not None:
        baseline_model = _extract_model_info(model)
        context["model_class"] = type(model).__name__

    # Extract tools
    tools = getattr(agent, "tools", None)
    if tools:
        tool_caps = introspect_tools(tools)
        capabilities.extend(tool_caps)
        try:
            context["tool_count"] = len(tools)
        except TypeError:
            context["tool_count"] = 0

    # Extract max_steps
    max_steps = getattr(agent, "max_steps", None)
    if max_steps:
        constraints.max_iterations = int(max_steps)
        context["max_steps"] = max_steps

    # Extract instructions
    instructions = getattr(agent, "instructions", None)
    if instructions:
        context["instructions"] = str(instructions)[:500]

    # Check for managed agents (multi-agent)
    managed_agents = getattr(agent, "managed_agents", None)
    if managed_agents:
        try:
            context["managed_agent_count"] = len(managed_agents)
        except TypeError:
            context["managed_agent_count"] = 0
        context["has_managed_agents"] = True

    # Check executor type for CodeAgent
    executor = getattr(agent, "executor", None)
    if executor:
        executor_type = type(executor).__name__
        context["executor_type"] = executor_type
        if (
            "Docker" in executor_type
            or "E2B" in executor_type
            or "Modal" in executor_type
        ):
            context["sandboxed_execution"] = True
        elif "Local" in executor_type:
            context["sandboxed_execution"] = False

    # Memory
    memory = getattr(agent, "memory", None)
    if memory:
        context["has_memory"] = True

    return baseline_model, capabilities, constraints, context


def introspect_tiny_agents_config(
    config: dict[str, Any],
) -> tuple[
    BaselineModel | None, list[Capability], OperationalConstraints, dict[str, Any]
]:
    """Introspect a tiny-agents agent.json configuration dict."""
    baseline_model = None
    capabilities: list[Capability] = []
    constraints = OperationalConstraints()
    context: dict[str, Any] = {}

    context["config_type"] = "tiny-agents"

    model_id = config.get("model")
    provider = config.get("provider")
    endpoint_url = config.get("endpointUrl")

    if model_id is not None and not isinstance(model_id, str):
        model_id = str(model_id)

    if model_id:
        context["model_id"] = model_id

        model_provider = ModelProvider.UNKNOWN
        if provider:
            context["provider"] = provider
            model_provider = HUGGINGFACE_PROVIDERS.get(
                provider.lower(), ModelProvider.UNKNOWN
            )

        if model_provider == ModelProvider.UNKNOWN:
            model_provider = _detect_provider_from_model(model_id)

        if endpoint_url:
            context["endpoint_url"] = endpoint_url
            if "localhost" in endpoint_url or "127.0.0.1" in endpoint_url:
                model_provider = ModelProvider.LOCAL
                context["local_inference"] = True

        baseline_model = BaselineModel(
            name=model_id,
            provider=model_provider,
        )

    # Extract MCP servers
    servers = config.get("servers", [])
    if servers and isinstance(servers, list):
        context["mcp_server_count"] = len(servers)
        context["mcp_servers"] = []

        for server in servers:
            if not isinstance(server, dict):
                continue

            server_type = server.get("type", "unknown")
            command = server.get("command", "")
            args = server.get("args", [])

            server_info = {
                "type": server_type,
                "command": command,
            }
            context["mcp_servers"].append(server_info)

            server_name = _extract_server_name(command, args)
            risk_level = RiskAssessor.assess_mcp_server_risk(command, args)

            args_preview = " ".join(args[:3]) if args else ""
            capabilities.append(
                Capability(
                    name=f"mcp_{server_name}",
                    description=f"MCP server: {command} {args_preview}".strip(),
                    risk_level=risk_level,
                )
            )

        capabilities.append(
            Capability(
                name="mcp_integration",
                description="Model Context Protocol integration for tool access",
                risk_level="medium",
            )
        )

    return baseline_model, capabilities, constraints, context


def introspect_agent_json_path(
    path: Any,
) -> tuple[
    BaselineModel | None, list[Capability], OperationalConstraints, dict[str, Any]
]:
    """Introspect an agent.json file from a filesystem path."""
    path = Path(path)

    try:
        resolved_path = path.resolve(strict=False)
    except (OSError, ValueError) as e:
        return None, [], OperationalConstraints(), {"error": f"Invalid path: {e}"}

    path_str = str(resolved_path)
    if any(
        suspicious in path_str
        for suspicious in ["/etc/", "/proc/", "/sys/", "\\windows\\", "\\system32\\"]
    ):
        return (
            None,
            [],
            OperationalConstraints(),
            {"error": "Access to system directories not allowed"},
        )

    agent_json_path = (
        resolved_path / "agent.json" if resolved_path.is_dir() else resolved_path
    )

    if agent_json_path.suffix.lower() != ".json":
        return (
            None,
            [],
            OperationalConstraints(),
            {"error": "Path must point to a .json file or directory"},
        )

    if not agent_json_path.exists():
        return None, [], OperationalConstraints(), {"error": "agent.json not found"}

    try:
        with open(agent_json_path) as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        return (
            None,
            [],
            OperationalConstraints(),
            {"error": f"Invalid JSON in agent.json: {e}"},
        )
    except PermissionError:
        return (
            None,
            [],
            OperationalConstraints(),
            {"error": "Permission denied reading agent.json"},
        )
    except OSError as e:
        return (
            None,
            [],
            OperationalConstraints(),
            {"error": f"Error reading agent.json: {e}"},
        )

    baseline_model, capabilities, constraints, context = introspect_tiny_agents_config(
        config
    )
    context["config_path"] = str(agent_json_path)

    prompt_md = agent_json_path.parent / "PROMPT.md"
    if prompt_md.exists():
        context["has_custom_prompt"] = True

    agents_md = agent_json_path.parent / "AGENTS.md"
    if agents_md.exists():
        context["has_agents_md"] = True

    return baseline_model, capabilities, constraints, context


def introspect_any(
    obj: Any,
) -> tuple[
    BaselineModel | None, list[Capability], OperationalConstraints, dict[str, Any]
]:
    """Introspect any Hugging Face agent object."""
    if isinstance(obj, dict):
        return introspect_tiny_agents_config(obj)

    if isinstance(obj, (str, Path)):
        return introspect_agent_json_path(obj)

    class_name = type(obj).__name__

    if (
        "CodeAgent" in class_name
        or "ToolCallingAgent" in class_name
        or "MultiStepAgent" in class_name
        or hasattr(obj, "tools")
        and hasattr(obj, "model")
    ):
        return introspect_smolagent(obj)
    else:
        return introspect_smolagent(obj)


@dataclass
class HuggingFaceIntegration(BaseIntrospector):
    """
    Hugging Face integration for AgentFacts.

    Supports:
    - smolagents: CodeAgent, ToolCallingAgent, MultiStepAgent
    - tiny-agents: agent.json config files
    - Model classes: InferenceClientModel, LiteLLMModel, TransformersModel
    """

    @property
    def framework_name(self) -> str:
        return "huggingface"

    def can_introspect(self, obj: Any) -> bool:
        """Check if object is a Hugging Face agent component."""
        if isinstance(obj, dict):
            return "model" in obj and (
                "servers" in obj or "provider" in obj or "endpointUrl" in obj
            )

        if isinstance(obj, (str, Path)):
            path = Path(obj)
            return (path.name == "agent.json" and path.exists()) or (
                path.is_dir() and (path / "agent.json").exists()
            )

        module = getattr(type(obj), "__module__", "")

        if "smolagents" in module.lower():
            return True
        if "huggingface_hub" in module.lower():
            return True

        class_name = type(obj).__name__
        smolagents_classes = {
            "CodeAgent",
            "ToolCallingAgent",
            "MultiStepAgent",
            "InferenceClientModel",
            "LiteLLMModel",
            "TransformersModel",
            "Tool",
            "ToolCollection",
            "AgentMemory",
            "GradioUI",
            "Agent",
            "MCPClient",
        }
        if class_name in smolagents_classes:
            return True

        if hasattr(obj, "tools") and hasattr(obj, "model") and hasattr(obj, "run"):
            return True
        return hasattr(obj, "model_id") and (hasattr(obj, "generate") or callable(obj))

    # Alias for Integration protocol

    def introspect(self, obj: Any) -> IntrospectionResult:
        """Introspect a Hugging Face agent object."""
        baseline, capabilities, constraints, extra_context = introspect_any(obj)

        context: dict[ContextKey | str, Any] = {
            ContextKey.FRAMEWORK: "huggingface",
            ContextKey.TOOL_COUNT: len(capabilities),
        }

        if isinstance(obj, dict):
            context[ContextKey.AGENT_TYPE] = "HUGGINGFACE_TINY_AGENTS_CONFIG"
        elif isinstance(obj, (str, Path)):
            context[ContextKey.AGENT_TYPE] = "HUGGINGFACE_AGENT_JSON"
        else:
            class_name = type(obj).__name__
            if class_name == "CodeAgent":
                context[ContextKey.AGENT_TYPE] = "HUGGINGFACE_CODE_AGENT"
            elif class_name == "ToolCallingAgent":
                context[ContextKey.AGENT_TYPE] = "HUGGINGFACE_TOOL_CALLING_AGENT"
            elif class_name == "MultiStepAgent":
                context[ContextKey.AGENT_TYPE] = "HUGGINGFACE_MULTI_STEP_AGENT"
            else:
                context[ContextKey.AGENT_TYPE] = f"HUGGINGFACE_{class_name.upper()}"

        # Merge extra context
        context.update(extra_context)

        return IntrospectionResult(
            framework="huggingface",
            baseline_model=baseline,
            capabilities=capabilities,
            constraints=constraints,
            context=context,
        )


# Convenience function for loading tiny-agents config
def load_tiny_agents_config(path: str | Path) -> dict[str, Any]:
    """
    Load a tiny-agents agent.json configuration file.

    Args:
        path: Path to agent.json file or directory containing it.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If agent.json doesn't exist.
        json.JSONDecodeError: If the file contains invalid JSON.
    """
    path = Path(path)
    if path.is_dir():
        path = path / "agent.json"

    if not path.exists():
        raise FileNotFoundError(f"agent.json not found at {path}")

    with open(path) as f:
        return cast(dict[str, Any], json.load(f))
