"""
Base introspection abstractions.

Defines the interface for framework-specific introspectors with template
methods for common operations like LLM extraction, tool introspection,
and risk assessment.

To add a new framework:
    1. Subclass BaseIntrospector
    2. Implement required abstract methods: framework_name, can_introspect, introspect
    3. Override hook methods as needed for framework-specific behavior
    4. Register with the IntegrationRegistry

Example:
    ```python
    class MyFrameworkIntrospector(BaseIntrospector):
        @property
        def framework_name(self) -> str:
            return "myframework"

        def can_introspect(self, obj: Any) -> bool:
            return getattr(type(obj), "__module__", "").startswith("myframework")

        def introspect(self, obj: Any) -> IntrospectionTuple:
            baseline = self._extract_baseline_model(obj)
            capabilities = self._extract_capabilities(obj)
            constraints = self._extract_constraints(obj)
            context = self._build_context(obj)
            return baseline, capabilities, constraints, context
    ```
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from enum import Enum
from typing import Any

from agentfacts.models import (
    BaselineModel,
    Capability,
    ModelProvider,
    OperationalConstraints,
)


class ContextKey(str, Enum):
    """
    Standard context dictionary keys for introspection results.

    Using these constants ensures consistency across all introspectors
    and enables IDE autocomplete.

    Key Categories:
    ---------------

    **REQUIRED** (must be present in all introspection results):
        - AGENT_TYPE: Human-readable agent type (e.g., "AgentExecutor")
        - FRAMEWORK: Framework identifier (e.g., "langchain", "crewai")
        - TOOL_COUNT: Number of capabilities extracted (int)

    **RECOMMENDED** (should be present if applicable):
        - MODEL_CLASS: LLM class name (e.g., "ChatOpenAI")
        - HAS_MEMORY: Whether agent has memory (bool)
        - EXECUTION_MODE: "code", "tool_calling", "chat", or "rag"

    **OPTIONAL** (include when relevant):
        - All others

    Example:
        ```python
        context = {
            # Required
            ContextKey.AGENT_TYPE: "AgentExecutor",
            ContextKey.FRAMEWORK: "langchain",
            ContextKey.TOOL_COUNT: 5,
            # Recommended
            ContextKey.MODEL_CLASS: "ChatOpenAI",
            ContextKey.HAS_MEMORY: True,
        }
        ```
    """

    # === REQUIRED KEYS ===
    # These MUST be present in all introspection results

    AGENT_TYPE = "agent_type"
    """Human-readable agent type. REQUIRED."""

    FRAMEWORK = "framework"
    """Framework identifier (langchain, crewai, etc.). REQUIRED."""

    TOOL_COUNT = "tool_count"
    """Number of capabilities extracted. REQUIRED."""

    # === RECOMMENDED KEYS ===
    # Should be present if the information is available

    MODEL_CLASS = "model_class"
    """LLM class name for provider detection. RECOMMENDED."""

    HAS_MEMORY = "has_memory"
    """Whether agent has memory/state. RECOMMENDED."""

    EXECUTION_MODE = "execution_mode"
    """Execution pattern: 'code', 'tool_calling', 'chat', 'rag'. RECOMMENDED."""

    # === OPTIONAL KEYS ===
    # Include when relevant to the specific framework/agent

    # Identity (optional)
    AGENT_CLASS = "agent_class"

    # Model info (optional)
    MODEL_TYPE = "model_type"

    # Tools/capabilities (optional beyond TOOL_COUNT)
    HAS_TOOLS = "has_tools"

    # Execution (optional beyond EXECUTION_MODE)
    SANDBOXED_EXECUTION = "sandboxed_execution"
    CODE_EXECUTION_ENABLED = "code_execution_enabled"

    # Memory/state (optional beyond HAS_MEMORY)
    SHARED_MEMORY = "shared_memory"

    # Multi-agent
    ROLE = "role"
    DELEGATION = "delegation"
    HIERARCHY_LEVEL = "hierarchy_level"

    # Constraints
    MAX_ITERATIONS = "max_iterations"
    MAX_ROUNDS = "max_rounds"
    TIMEOUT = "timeout"

    # Framework-specific (use sparingly)
    SYSTEM_MESSAGE = "system_message"
    VERBOSE = "verbose"
    STREAMING = "streaming"

    # MCP/servers
    MCP_SERVER_COUNT = "mcp_server_count"
    MCP_SERVERS = "mcp_servers"

    # Errors (deprecated - use IntrospectionResult.errors/warnings instead)
    ERROR = "error"
    WARNING = "warning"


# Required context keys that MUST be present
REQUIRED_CONTEXT_KEYS: frozenset[str] = frozenset(
    {
        ContextKey.AGENT_TYPE.value,
        ContextKey.FRAMEWORK.value,
        ContextKey.TOOL_COUNT.value,
    }
)

# Recommended context keys that SHOULD be present
RECOMMENDED_CONTEXT_KEYS: frozenset[str] = frozenset(
    {
        ContextKey.MODEL_CLASS.value,
        ContextKey.HAS_MEMORY.value,
        ContextKey.EXECUTION_MODE.value,
    }
)

# Type alias for the introspection return tuple
IntrospectionTuple = tuple[
    BaselineModel | None,
    list[Capability],
    OperationalConstraints,
    dict[ContextKey | str, Any],
]


def _normalize_context_key(key: Any) -> str:
    """Normalize context keys to string values for comparison."""
    if isinstance(key, ContextKey):
        return key.value
    return str(key)


class BaseIntrospector(ABC):
    """
    Abstract base class for framework introspectors.

    Each framework (LangChain, CrewAI, AutoGen, etc.) should implement
    this interface to enable automatic metadata extraction.

    Required Methods (must override):
        - framework_name: Return framework identifier
        - can_introspect: Check if object can be handled
        - introspect: Extract metadata from object

    Hook Methods (can override for customization):
        - get_agent_type: Return human-readable agent type
        - _extract_baseline_model: Extract LLM info
        - _extract_capabilities: Extract tools/capabilities
        - _extract_constraints: Extract operational constraints
        - _build_context: Build context dictionary

    Utility Methods (inherit and use):
        - _assess_risk: Centralized risk assessment
        - _detect_provider: Centralized provider detection
        - _safe_getattr: Safe attribute access
        - _safe_getattr_multi: Multi-path attribute access
        - _create_capability: Standardized capability creation
        - _create_baseline_model: Standardized model creation
    """

    # -------------------------------------------------------------------------
    # Abstract Methods (MUST implement)
    # -------------------------------------------------------------------------

    @property
    @abstractmethod
    def framework_name(self) -> str:
        """
        Return the framework name identifier.

        This should be a lowercase string like 'langchain', 'crewai', 'autogen'.
        Used for registration and framework detection.

        Returns:
            Framework identifier string
        """
        pass

    @abstractmethod
    def can_introspect(self, obj: Any) -> bool:
        """
        Check if this introspector can handle the given object.

        Implementation tips:
            - Check module name: getattr(type(obj), "__module__", "").startswith("framework")
            - Use duck typing: hasattr(obj, "framework_specific_attr")
            - Avoid isinstance() to prevent import requirements

        Args:
            obj: The object to check

        Returns:
            True if this introspector can handle the object
        """
        pass

    @abstractmethod
    def introspect(self, obj: Any) -> IntrospectionReturn:
        """
        Introspect the object to extract metadata.

        This is the main entry point. Implementations should:
            1. Extract baseline model info
            2. Extract capabilities/tools
            3. Extract operational constraints
            4. Build context dictionary

        Args:
            obj: The agent/crew/chain object to introspect

        Returns:
            IntrospectionResult or tuple of
            (BaselineModel | None, list[Capability], OperationalConstraints, context_dict)
        """
        pass

    # -------------------------------------------------------------------------
    # Hook Methods (CAN override for customization)
    # -------------------------------------------------------------------------

    def get_agent_type(self, obj: Any) -> str:
        """
        Get the agent type string for context.

        Override to provide more meaningful type names.

        Args:
            obj: The object to get type for

        Returns:
            Human-readable agent type string
        """
        return type(obj).__name__

    def _extract_baseline_model(self, obj: Any) -> BaselineModel | None:
        """
        Extract baseline model information from the object.

        Override to implement framework-specific LLM extraction.
        Use _safe_getattr_multi for multi-path extraction.

        Args:
            obj: Object to extract model from

        Returns:
            BaselineModel if found, None otherwise
        """
        return None

    def _extract_capabilities(self, obj: Any) -> list[Capability]:
        """
        Extract capabilities/tools from the object.

        Override to implement framework-specific tool extraction.
        Use _create_capability for standardized capability creation.

        Args:
            obj: Object to extract capabilities from

        Returns:
            List of Capability objects
        """
        return []

    def _extract_constraints(self, obj: Any) -> OperationalConstraints:
        """
        Extract operational constraints from the object.

        Override to implement framework-specific constraint extraction.

        Args:
            obj: Object to extract constraints from

        Returns:
            OperationalConstraints instance
        """
        return OperationalConstraints()

    def _build_context(self, obj: Any) -> dict[str, Any]:
        """
        Build the context dictionary for the object.

        Override to add framework-specific context.
        Use ContextKey enum for standard keys.

        Args:
            obj: Object to build context for

        Returns:
            Context dictionary
        """
        return {
            ContextKey.AGENT_TYPE.value: self.get_agent_type(obj),
            ContextKey.FRAMEWORK.value: self.framework_name,
        }

    # -------------------------------------------------------------------------
    # Utility Methods (INHERIT and use)
    # -------------------------------------------------------------------------

    def _assess_risk(self, name: str, description: str = "") -> str:
        """
        Assess risk level for a tool/capability.

        Uses centralized RiskAssessor for consistent risk evaluation.

        Args:
            name: Tool/capability name
            description: Tool description (optional)

        Returns:
            Risk level: "high", "medium", or "low"
        """
        from agentfacts.integrations.utils import RiskAssessor

        return RiskAssessor.assess_tool_risk(name, description)

    def _assess_mcp_risk(self, command: str, args: list[str] | None = None) -> str:
        """
        Assess risk level for an MCP server.

        Uses centralized RiskAssessor for consistent risk evaluation.

        Args:
            command: Server command
            args: Command arguments

        Returns:
            Risk level: "high", "medium", or "low"
        """
        from agentfacts.integrations.utils import RiskAssessor

        return RiskAssessor.assess_mcp_server_risk(command, args)

    def _detect_provider(
        self,
        model_name: str | None = None,
        endpoint: str | None = None,
        llm_class_name: str | None = None,
    ) -> ModelProvider:
        """
        Detect model provider using centralized detection.

        Args:
            model_name: Model identifier (e.g., "gpt-4")
            endpoint: API endpoint URL
            llm_class_name: LLM class name for framework-specific detection

        Returns:
            ModelProvider enum value
        """
        from agentfacts.integrations.utils import ProviderDetector

        # Check class name mappings first
        if llm_class_name:
            provider = self._get_provider_from_class_name(llm_class_name)
            if provider != ModelProvider.UNKNOWN:
                return provider

        # Use centralized detector
        provider_str = ProviderDetector.detect(
            model_name=model_name,
            endpoint=endpoint,
        )

        # Convert string to enum
        try:
            return ModelProvider(provider_str)
        except ValueError:
            return ModelProvider.UNKNOWN

    def _get_provider_from_class_name(self, class_name: str) -> ModelProvider:
        """
        Get provider from LLM class name.

        Override to add framework-specific class name mappings.

        Args:
            class_name: LLM class name

        Returns:
            ModelProvider enum value
        """
        # Common mappings across frameworks
        common_mappings = {
            "ChatOpenAI": ModelProvider.OPENAI,
            "OpenAI": ModelProvider.OPENAI,
            "AzureChatOpenAI": ModelProvider.OPENAI,
            "ChatAnthropic": ModelProvider.ANTHROPIC,
            "Anthropic": ModelProvider.ANTHROPIC,
            "Claude": ModelProvider.ANTHROPIC,
            "ChatGoogleGenerativeAI": ModelProvider.GOOGLE,
            "GoogleGenerativeAI": ModelProvider.GOOGLE,
            "Gemini": ModelProvider.GOOGLE,
            "ChatMistralAI": ModelProvider.MISTRAL,
            "MistralAI": ModelProvider.MISTRAL,
            "Cohere": ModelProvider.COHERE,
            "ChatCohere": ModelProvider.COHERE,
            "HuggingFaceHub": ModelProvider.HUGGINGFACE,
            "HuggingFacePipeline": ModelProvider.HUGGINGFACE,
            "Ollama": ModelProvider.LOCAL,
            "ChatOllama": ModelProvider.LOCAL,
            "LlamaCpp": ModelProvider.LOCAL,
        }

        return common_mappings.get(class_name, ModelProvider.UNKNOWN)

    def _safe_getattr(self, obj: object, name: str, default: Any = None) -> Any:
        """
        Safely get an attribute from an object.

        Args:
            obj: Object to get attribute from
            name: Attribute name
            default: Default value if not found

        Returns:
            Attribute value or default
        """
        try:
            return getattr(obj, name, default)
        except Exception:
            return default

    def _safe_getattr_multi(
        self,
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
            llm = self._safe_getattr_multi(obj, [
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

    def _create_capability(
        self,
        name: str,
        description: str = "",
        parameters: dict[str, Any] | None = None,
        risk_level: str | None = None,
        requires_approval: bool = False,
        delegatable: bool = False,
    ) -> Capability:
        """
        Create a Capability with standardized risk assessment.

        Args:
            name: Capability/tool name
            description: Human-readable description
            parameters: Parameter schema
            risk_level: Override risk level (auto-assessed if None)
            requires_approval: Whether approval is required
            delegatable: Whether capability can be delegated

        Returns:
            Capability instance
        """
        if risk_level is None:
            risk_level = self._assess_risk(name, description)

        return Capability(
            name=name,
            description=description,
            parameters=parameters or {},
            risk_level=risk_level,
            requires_approval=requires_approval,
            delegatable=delegatable,
        )

    def _create_baseline_model(
        self,
        name: str,
        provider: ModelProvider | str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **extra_params: Any,
    ) -> BaselineModel:
        """
        Create a BaselineModel with standardized provider detection.

        Args:
            name: Model name/identifier
            provider: Provider (auto-detected if None)
            temperature: Sampling temperature
            max_tokens: Maximum output tokens
            **extra_params: Additional model parameters

        Returns:
            BaselineModel instance
        """
        # Detect provider if not provided
        if provider is None:
            provider = self._detect_provider(model_name=name)
        elif isinstance(provider, str):
            try:
                provider = ModelProvider(provider.lower())
            except ValueError:
                provider = ModelProvider.UNKNOWN

        return BaselineModel(
            name=name,
            provider=provider,
            temperature=temperature,
            max_tokens=max_tokens,
            extra_params=extra_params if extra_params else {},
        )

    def _build_result(
        self,
        baseline_model: BaselineModel | None,
        capabilities: list[Capability],
        constraints: OperationalConstraints,
        context: dict[ContextKey | str, Any],
        errors: list[str] | None = None,
        warnings: list[str] | None = None,
    ) -> IntrospectionResult:
        """
        Build an IntrospectionResult with this introspector's framework.

        This is a convenience method for introspector implementations
        that ensures consistent result construction.

        Args:
            baseline_model: Extracted model info
            capabilities: Extracted capabilities
            constraints: Extracted constraints
            context: Context dictionary
            errors: List of critical errors
            warnings: List of non-critical warnings

        Returns:
            IntrospectionResult instance

        Example:
            ```python
            def introspect(self, obj: Any) -> IntrospectionTuple:
                baseline = self._extract_baseline_model(obj)
                capabilities = self._extract_capabilities(obj)
                constraints = self._extract_constraints(obj)
                context = self._build_context(obj)

                # Can also return IntrospectionResult directly
                # return self._build_result(baseline, capabilities, constraints, context)

                return baseline, capabilities, constraints, context
            ```
        """
        # Ensure required context keys
        context.setdefault(ContextKey.FRAMEWORK.value, self.framework_name)
        context.setdefault(ContextKey.TOOL_COUNT.value, len(capabilities))

        return IntrospectionResult(
            framework=self.framework_name,
            baseline_model=baseline_model,
            capabilities=capabilities,
            constraints=constraints,
            context=context,
            errors=errors,
            warnings=warnings,
        )


class IntrospectionResult:
    """
    Result of introspecting an agent.

    Provides a structured container for introspection results with
    easy access to all metadata components, plus error/warning tracking.

    Attributes:
        framework: Framework identifier
        baseline_model: Model metadata (may be None)
        capabilities: List of extracted capabilities
        constraints: Operational constraints
        context: Additional context dictionary
        errors: List of critical errors during introspection
        warnings: List of non-critical warnings during introspection

    Example:
        ```python
        result = registry.introspect(my_agent)

        # Access result properties
        print(f"Model: {result.model_name}")
        print(f"Capabilities: {result.capability_count}")

        # Check for issues
        if not result.success:
            print(f"Errors: {result.errors}")
        if result.warnings:
            print(f"Warnings: {result.warnings}")

        # Check completeness
        if result.complete:
            print("Full introspection succeeded")
        ```
    """

    def __init__(
        self,
        framework: str,
        baseline_model: BaselineModel | None,
        capabilities: list[Capability],
        constraints: OperationalConstraints,
        context: dict[str, Any],
        errors: list[str] | None = None,
        warnings: list[str] | None = None,
        validate_context: bool = False,
    ):
        self.framework = framework
        self.baseline_model = baseline_model
        self.capabilities = capabilities
        self.constraints = constraints
        self.context = context
        self.errors = errors or []
        self.warnings = warnings or []

        # Validate required context keys if requested
        if validate_context:
            self._validate_context()

    def _validate_context(self) -> None:
        """Validate that required context keys are present."""
        # Convert context keys to strings for comparison
        context_keys = {_normalize_context_key(k) for k in self.context}
        required_str = set(REQUIRED_CONTEXT_KEYS)

        missing = required_str - context_keys
        if missing:
            self.warnings.append(
                f"Missing recommended context keys: {missing}. "
                "See ContextKey documentation for required keys."
            )

    @property
    def model_name(self) -> str | None:
        """Get model name if available."""
        return self.baseline_model.name if self.baseline_model else None

    @property
    def provider(self) -> ModelProvider | None:
        """Get model provider if available."""
        return self.baseline_model.provider if self.baseline_model else None

    @property
    def capability_count(self) -> int:
        """Get number of capabilities."""
        return len(self.capabilities)

    @property
    def high_risk_capabilities(self) -> list[Capability]:
        """Get list of high-risk capabilities."""
        return [c for c in self.capabilities if c.risk_level == "high"]

    @property
    def success(self) -> bool:
        """
        True if no critical errors occurred during introspection.

        Note: A successful introspection may still have warnings
        or missing optional data.
        """
        return len(self.errors) == 0

    @property
    def complete(self) -> bool:
        """
        True if all expected data was extracted.

        A complete result has:
        - No errors
        - A baseline model
        - At least the required context keys
        """
        if not self.success:
            return False
        if self.baseline_model is None:
            return False
        # Check for required context keys
        context_keys = {_normalize_context_key(k) for k in self.context}
        required_str = set(REQUIRED_CONTEXT_KEYS)
        return required_str.issubset(context_keys)

    def add_error(self, error: str) -> None:
        """Add a critical error message."""
        self.errors.append(error)

    def add_warning(self, warning: str) -> None:
        """Add a non-critical warning message."""
        self.warnings.append(warning)

    def __repr__(self) -> str:
        status = "OK" if self.success else f"ERRORS({len(self.errors)})"
        return (
            f"IntrospectionResult(framework={self.framework!r}, "
            f"model={self.model_name}, "
            f"capabilities={self.capability_count}, "
            f"status={status})"
        )


IntrospectionReturn = IntrospectionTuple | IntrospectionResult
