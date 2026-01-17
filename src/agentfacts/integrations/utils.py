"""
Shared utilities for introspection across frameworks.

Provides centralized, reusable utilities for:
- Risk assessment (tools, MCP servers)
- Provider detection (from model names, endpoints, class names)
- Safe attribute access (single and multi-path)
- Tool/capability extraction helpers

All introspectors should use these utilities instead of implementing
their own versions to ensure consistency.

Example:
    ```python
    from agentfacts.integrations.utils import RiskAssessor, ProviderDetector

    # Assess tool risk
    risk = RiskAssessor.assess_tool_risk("execute_shell", "Run shell commands")
    # Returns: "high"

    # Detect provider from model name
    provider = ProviderDetector.detect(model_name="gpt-4-turbo")
    # Returns: "openai"
    ```
"""

from collections.abc import Callable, Sequence
from typing import Any, Literal

from agentfacts.models import Capability, ModelProvider

RiskLevel = Literal["high", "medium", "low"]


class RiskAssessor:
    """
    Extensible tool/capability risk assessment.

    Provides consistent risk evaluation across all framework introspectors.
    Risk levels are determined by keyword matching in tool names and descriptions.

    Risk Levels:
        - "high": Tools that can execute code, modify system state, access credentials
        - "medium": Tools that access network, files, or external services
        - "low": Tools with no significant security implications

    Extensibility:
        Framework developers can register additional keywords:
        ```python
        RiskAssessor.register_high_risk("my_dangerous_tool")
        RiskAssessor.register_medium_risk("my_network_tool")
        ```

    Example:
        ```python
        # Single tool assessment
        risk = RiskAssessor.assess_tool_risk("shell_execute", "Run shell commands")
        assert risk == "high"

        # MCP server assessment
        risk = RiskAssessor.assess_mcp_server_risk("docker", ["run", "-it"])
        assert risk == "high"

        # Extend with custom keywords
        RiskAssessor.register_high_risk("my_custom_dangerous_action")
        ```
    """

    # Default keywords (used for reset)
    _DEFAULT_HIGH_RISK: frozenset[str] = frozenset(
        [
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
            "secret",
            "code_interpreter",
            "file_write",
            "eval",
            "exec",
            "rm",
            "drop",
            "truncate",
            "grant",
            "revoke",
            "injection",
        ]
    )

    _DEFAULT_MEDIUM_RISK: frozenset[str] = frozenset(
        [
            "file",
            "network",
            "http",
            "request",
            "email",
            "send",
            "create",
            "update",
            "api",
            "scrape",
            "browser",
            "web",
            "download",
            "upload",
            "fetch",
            "post",
            "put",
            "patch",
            "webhook",
            "socket",
            "connection",
            "smtp",
            "ftp",
        ]
    )

    _DEFAULT_HIGH_RISK_MCP_COMMANDS: frozenset[str] = frozenset(
        [
            "docker",
            "kubectl",
            "aws",
            "gcloud",
            "az",
            "terraform",
            "ansible",
            "ssh",
            "scp",
            "rsync",
            "bash",
            "sh",
            "zsh",
        ]
    )

    _DEFAULT_HIGH_RISK_MCP_SERVERS: frozenset[str] = frozenset(
        [
            "filesystem",
            "file",
            "shell",
            "terminal",
            "docker",
            "kubernetes",
            "aws",
            "database",
            "sql",
            "postgres",
            "mysql",
        ]
    )

    _DEFAULT_MEDIUM_RISK_MCP_SERVERS: frozenset[str] = frozenset(
        [
            "playwright",
            "puppeteer",
            "selenium",
            "browser",
            "github",
            "gitlab",
            "http",
            "fetch",
            "api",
        ]
    )

    # Mutable sets for runtime extension
    _high_risk: set[str] = set(_DEFAULT_HIGH_RISK)
    _medium_risk: set[str] = set(_DEFAULT_MEDIUM_RISK)
    _high_risk_mcp_commands: set[str] = set(_DEFAULT_HIGH_RISK_MCP_COMMANDS)
    _high_risk_mcp_servers: set[str] = set(_DEFAULT_HIGH_RISK_MCP_SERVERS)
    _medium_risk_mcp_servers: set[str] = set(_DEFAULT_MEDIUM_RISK_MCP_SERVERS)

    # Registration methods for extensibility
    @classmethod
    def register_high_risk(cls, *keywords: str) -> None:
        """
        Register additional high-risk keywords.

        Args:
            *keywords: Keywords to add to high-risk set

        Example:
            ```python
            RiskAssessor.register_high_risk("my_dangerous_tool", "unsafe_action")
            ```
        """
        cls._high_risk.update(k.lower() for k in keywords)

    @classmethod
    def register_medium_risk(cls, *keywords: str) -> None:
        """
        Register additional medium-risk keywords.

        Args:
            *keywords: Keywords to add to medium-risk set
        """
        cls._medium_risk.update(k.lower() for k in keywords)

    @classmethod
    def register_high_risk_mcp_command(cls, *commands: str) -> None:
        """Register additional high-risk MCP commands."""
        cls._high_risk_mcp_commands.update(c.lower() for c in commands)

    @classmethod
    def register_high_risk_mcp_server(cls, *servers: str) -> None:
        """Register additional high-risk MCP server names."""
        cls._high_risk_mcp_servers.update(s.lower() for s in servers)

    @classmethod
    def register_medium_risk_mcp_server(cls, *servers: str) -> None:
        """Register additional medium-risk MCP server names."""
        cls._medium_risk_mcp_servers.update(s.lower() for s in servers)

    @classmethod
    def reset(cls) -> None:
        """
        Reset all keyword sets to defaults.

        Useful for testing or when you need a clean slate.
        """
        cls._high_risk = set(cls._DEFAULT_HIGH_RISK)
        cls._medium_risk = set(cls._DEFAULT_MEDIUM_RISK)
        cls._high_risk_mcp_commands = set(cls._DEFAULT_HIGH_RISK_MCP_COMMANDS)
        cls._high_risk_mcp_servers = set(cls._DEFAULT_HIGH_RISK_MCP_SERVERS)
        cls._medium_risk_mcp_servers = set(cls._DEFAULT_MEDIUM_RISK_MCP_SERVERS)

    @classmethod
    def assess_tool_risk(cls, name: str, description: str = "") -> RiskLevel:
        """
        Assess risk level for a tool based on name and description.

        Args:
            name: Tool name
            description: Tool description (optional)

        Returns:
            Risk level: "high", "medium", or "low"
        """
        combined = (name + " " + description).lower()

        # Use mutable sets for extensibility
        for keyword in cls._high_risk:
            if keyword in combined:
                return "high"

        for keyword in cls._medium_risk:
            if keyword in combined:
                return "medium"

        return "low"

    @classmethod
    def assess_function_risk(cls, name: str, description: str = "") -> RiskLevel:
        """Alias for assess_tool_risk for function-style capabilities."""
        return cls.assess_tool_risk(name, description)

    @classmethod
    def assess_mcp_server_risk(
        cls, command: str, args: list[str] | None = None
    ) -> RiskLevel:
        """
        Assess risk level for an MCP server.

        Args:
            command: The command or executable name
            args: Command arguments (optional)

        Returns:
            Risk level: "high", "medium", or "low"
        """
        command_lower = command.lower()
        args_str = " ".join(args or []).lower()

        # Check if command is a high-risk tool (use mutable set)
        for risky_cmd in cls._high_risk_mcp_commands:
            if risky_cmd in command_lower:
                return "high"

        # Check server names in args (e.g., npx -y @playwright/mcp)
        for risky_server in cls._high_risk_mcp_servers:
            if risky_server in args_str:
                return "high"

        # Check medium-risk servers
        for medium_server in cls._medium_risk_mcp_servers:
            if medium_server in args_str:
                return "medium"

        return "medium"  # Default MCP servers to medium (they provide tool access)

    @classmethod
    def assess_capabilities(
        cls, capabilities: list[Capability]
    ) -> dict[str, list[Capability]]:
        """
        Group capabilities by risk level.

        Args:
            capabilities: List of capabilities to assess

        Returns:
            Dict with keys "high", "medium", "low" mapping to capability lists
        """
        result: dict[str, list[Capability]] = {
            "high": [],
            "medium": [],
            "low": [],
        }
        for cap in capabilities:
            level = cap.risk_level or cls.assess_tool_risk(cap.name, cap.description)
            result[level].append(cap)
        return result


class ProviderDetector:
    """
    Centralized model provider detection.

    Provides consistent provider identification across all framework introspectors.
    Detection uses multiple strategies:
    1. Explicit provider (highest priority)
    2. Model name patterns
    3. API endpoint patterns
    4. LLM class name (framework-specific)

    Example:
        ```python
        # From model name
        provider = ProviderDetector.detect(model_name="claude-3-opus")
        assert provider == "anthropic"

        # From endpoint
        provider = ProviderDetector.detect(endpoint="https://api.openai.com/v1")
        assert provider == "openai"

        # Combined detection
        provider = ProviderDetector.detect(
            model_name="my-model",
            endpoint="https://api.together.xyz/v1",
        )
        assert provider == "together"
        ```
    """

    # Model name patterns to provider mapping
    PROVIDER_PATTERNS: dict[str, str] = {
        # OpenAI
        "gpt-": "openai",
        "gpt4": "openai",
        "o1-": "openai",
        "o3-": "openai",
        "davinci": "openai",
        "curie": "openai",
        "babbage": "openai",
        "ada": "openai",
        "text-embedding": "openai",
        # Anthropic
        "claude": "anthropic",
        # Google
        "gemini": "google",
        "palm": "google",
        "bison": "google",
        "gecko": "google",
        # Mistral
        "mistral": "mistral",
        "mixtral": "mistral",
        "codestral": "mistral",
        # Cohere
        "command": "cohere",
        # Meta
        "llama": "meta",
        "meta-llama": "meta",
        "codellama": "meta",
        "opt-": "meta",
        # Others
        "qwen": "alibaba",
        "deepseek": "deepseek",
        "yi-": "01ai",
        "phi-": "microsoft",
        "orca": "microsoft",
        "wizardlm": "microsoft",
        "falcon": "tii",
        "starcoder": "huggingface",
        "bloom": "huggingface",
        "vicuna": "lmsys",
        "zephyr": "huggingface",
        "nous": "nous",
        "openhermes": "teknium",
        "solar": "upstage",
        "dbrx": "databricks",
    }

    # Endpoint patterns to provider mapping
    ENDPOINT_PATTERNS: dict[str, str] = {
        "openai.com": "openai",
        "azure.com": "azure",
        "anthropic.com": "anthropic",
        "googleapis.com": "google",
        "aiplatform.googleapis": "google",
        "generativelanguage.googleapis": "google",
        "mistral.ai": "mistral",
        "cohere.ai": "cohere",
        "cohere.com": "cohere",
        "huggingface.co": "huggingface",
        "replicate.com": "replicate",
        "together.xyz": "together",
        "anyscale.com": "anyscale",
        "deepinfra.com": "deepinfra",
        "fireworks.ai": "fireworks",
        "groq.com": "groq",
        "perplexity.ai": "perplexity",
        "localhost": "local",
        "127.0.0.1": "local",
        "ollama": "ollama",
        "lmstudio": "lmstudio",
    }

    # LLM class name to provider mapping (common across frameworks)
    CLASS_NAME_PATTERNS: dict[str, str] = {
        # OpenAI
        "ChatOpenAI": "openai",
        "OpenAI": "openai",
        "AzureChatOpenAI": "azure",
        "AzureOpenAI": "azure",
        # Anthropic
        "ChatAnthropic": "anthropic",
        "Anthropic": "anthropic",
        "Claude": "anthropic",
        # Google
        "ChatGoogleGenerativeAI": "google",
        "GoogleGenerativeAI": "google",
        "Gemini": "google",
        "ChatVertexAI": "google",
        "VertexAI": "google",
        # Mistral
        "ChatMistralAI": "mistral",
        "MistralAI": "mistral",
        # Cohere
        "Cohere": "cohere",
        "ChatCohere": "cohere",
        # HuggingFace
        "HuggingFaceHub": "huggingface",
        "HuggingFacePipeline": "huggingface",
        "HuggingFaceEndpoint": "huggingface",
        "InferenceClient": "huggingface",
        "InferenceClientModel": "huggingface",
        "TransformersModel": "huggingface",
        # Local
        "Ollama": "local",
        "ChatOllama": "local",
        "LlamaCpp": "local",
        "GPT4All": "local",
        # LiteLLM (proxy - detect from model name)
        "LiteLLM": "unknown",
        "LiteLLMModel": "unknown",
    }

    @classmethod
    def detect_from_model_name(cls, model_name: str) -> str:
        """
        Detect provider from model name.

        Args:
            model_name: The model identifier

        Returns:
            Provider name or "unknown"
        """
        model_lower = model_name.lower()

        for pattern, provider in cls.PROVIDER_PATTERNS.items():
            if pattern in model_lower:
                return provider

        return "unknown"

    @classmethod
    def detect_from_endpoint(cls, endpoint: str) -> str:
        """
        Detect provider from API endpoint URL.

        Args:
            endpoint: The API endpoint URL

        Returns:
            Provider name or "unknown"
        """
        endpoint_lower = endpoint.lower()

        for pattern, provider in cls.ENDPOINT_PATTERNS.items():
            if pattern in endpoint_lower:
                return provider

        return "unknown"

    @classmethod
    def detect_from_class_name(cls, class_name: str) -> str:
        """
        Detect provider from LLM class name.

        Args:
            class_name: The LLM class name

        Returns:
            Provider name or "unknown"
        """
        return cls.CLASS_NAME_PATTERNS.get(class_name, "unknown")

    @classmethod
    def detect(
        cls,
        model_name: str | None = None,
        endpoint: str | None = None,
        explicit_provider: str | None = None,
        class_name: str | None = None,
    ) -> str:
        """
        Detect provider using all available information.

        Priority: explicit > class_name > model_name > endpoint > unknown

        Args:
            model_name: Model identifier (optional)
            endpoint: API endpoint URL (optional)
            explicit_provider: Explicitly specified provider (optional)
            class_name: LLM class name (optional)

        Returns:
            Provider name or "unknown"
        """
        # Explicit provider takes precedence
        if explicit_provider:
            return explicit_provider.lower()

        # Try class name (framework-specific)
        if class_name:
            provider = cls.detect_from_class_name(class_name)
            if provider != "unknown":
                return provider

        # Try model name
        if model_name:
            provider = cls.detect_from_model_name(model_name)
            if provider != "unknown":
                return provider

        # Try endpoint
        if endpoint:
            provider = cls.detect_from_endpoint(endpoint)
            if provider != "unknown":
                return provider

        return "unknown"

    @classmethod
    def to_model_provider(cls, provider_str: str) -> ModelProvider:
        """
        Convert provider string to ModelProvider enum.

        Args:
            provider_str: Provider string from detection

        Returns:
            ModelProvider enum value
        """
        provider_mapping = {
            "openai": ModelProvider.OPENAI,
            "azure": ModelProvider.OPENAI,  # Azure OpenAI
            "anthropic": ModelProvider.ANTHROPIC,
            "google": ModelProvider.GOOGLE,
            "cohere": ModelProvider.COHERE,
            "mistral": ModelProvider.MISTRAL,
            "meta": ModelProvider.META,
            "huggingface": ModelProvider.HUGGINGFACE,
            "local": ModelProvider.LOCAL,
            "ollama": ModelProvider.LOCAL,
            "lmstudio": ModelProvider.LOCAL,
        }
        return provider_mapping.get(provider_str.lower(), ModelProvider.UNKNOWN)


def safe_getattr(obj: object, name: str, default: Any = None) -> Any:
    """
    Safely get an attribute with exception handling.

    More robust than getattr() alone - catches any exception that
    might be raised by property getters or __getattr__.

    Args:
        obj: Object to get attribute from
        name: Attribute name
        default: Default value if attribute doesn't exist or raises

    Returns:
        Attribute value or default
    """
    try:
        return getattr(obj, name, default)
    except Exception:
        return default


def safe_getattr_multi(
    obj: object,
    paths: Sequence[Sequence[str]],
    default: Any = None,
) -> Any:
    """
    Try multiple attribute paths to find a value.

    Useful for frameworks with different attribute names across versions
    or for nested attribute access.

    Example:
        ```python
        # Try llm, then _llm, then agent_worker.llm
        llm = safe_getattr_multi(obj, [
            ["llm"],
            ["_llm"],
            ["agent_worker", "llm"],
            ["agent_worker", "_llm"],
        ])
        ```

    Args:
        obj: Object to get attribute from
        paths: List of attribute paths (each path is a list of attr names)
        default: Default value if no path succeeds

    Returns:
        First found value or default
    """
    for path in paths:
        current = obj
        try:
            for attr_name in path:
                current = getattr(current, attr_name)
            if current is not None:
                return current
        except (AttributeError, TypeError):
            continue
    return default


def extract_tool_info(
    tool: Any,
    name_attrs: Sequence[str] = ("name", "tool_name", "__name__"),
    desc_attrs: Sequence[str] = ("description", "desc", "__doc__"),
    param_attrs: Sequence[str] = ("parameters", "args_schema", "input_schema"),
) -> tuple[str, str, dict[str, Any]]:
    """
    Extract name, description, and parameters from a tool object.

    Works with various tool formats across frameworks.

    Args:
        tool: Tool object to extract info from
        name_attrs: Attribute names to try for tool name
        desc_attrs: Attribute names to try for description
        param_attrs: Attribute names to try for parameters

    Returns:
        Tuple of (name, description, parameters)
    """
    # Extract name
    name = "unknown_tool"
    for attr in name_attrs:
        value = safe_getattr(tool, attr)
        if value and isinstance(value, str):
            name = value
            break
    if name == "unknown_tool":
        name = type(tool).__name__

    # Extract description
    description = ""
    for attr in desc_attrs:
        value = safe_getattr(tool, attr)
        if value and isinstance(value, str):
            description = value
            break

    # Extract parameters
    parameters: dict[str, Any] = {}
    for attr in param_attrs:
        value = safe_getattr(tool, attr)
        if value:
            if isinstance(value, dict):
                parameters = value
                break
            # Handle Pydantic models
            if hasattr(value, "model_json_schema"):
                try:
                    parameters = value.model_json_schema()
                    break
                except Exception:
                    pass
            elif hasattr(value, "schema"):
                try:
                    parameters = value.schema()
                    break
                except Exception:
                    pass

    return name, description, parameters


def normalize_tools(
    tools: Any,
    filter_fn: Callable[[Any], bool] | None = None,
) -> list[Any]:
    """
    Normalize various tool collection formats to a list.

    Handles lists, tuples, generators, dicts, and single tools.

    Args:
        tools: Tools in various formats
        filter_fn: Optional filter function

    Returns:
        List of tool objects
    """
    if tools is None:
        return []

    # Handle different collection types
    if isinstance(tools, dict):
        result = list(tools.values())
    elif hasattr(tools, "__iter__") and not isinstance(tools, (str, bytes)):
        try:
            result = list(tools)
        except TypeError:
            result = [tools]
    else:
        result = [tools]

    # Apply filter if provided
    if filter_fn:
        result = [t for t in result if filter_fn(t)]

    return result
