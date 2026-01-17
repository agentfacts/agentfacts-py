"""
Framework integrations for AgentFacts SDK.

This module provides a unified interface for integrating AgentFacts with
various AI agent frameworks. Each integration provides:

1. **Introspection**: Extract metadata from framework-specific agent objects
2. **Callbacks**: Runtime event capture for transparency logs (where applicable)
3. **Convenience functions**: Simplified API for framework users

Supported frameworks:
- LangChain / LangGraph
- CrewAI
- AutoGen
- LlamaIndex
- OpenAgents
- HuggingFace (smolagents, tiny-agents)

Example:
    ```python
    from agentfacts.integrations import get_registry

    registry = get_registry()
    integration = registry.get_for_object(my_agent)
    if integration:
        result = integration.introspect(my_agent)

    # Framework-specific import
    from agentfacts.integrations.langchain import from_agent
    facts = from_agent(my_agent, "Research Agent", sign=True)
    ```

Adding a new framework:
    Subclass BaseIntrospector and implement framework_name, can_introspect,
    and introspect methods. See existing integrations for examples.
"""

import threading
from typing import Any, Protocol, runtime_checkable

# Re-export base classes and utilities
from agentfacts.integrations.base import (
    RECOMMENDED_CONTEXT_KEYS,
    REQUIRED_CONTEXT_KEYS,
    BaseIntrospector,
    ContextKey,
    IntrospectionResult,
    IntrospectionTuple,
)
from agentfacts.integrations.utils import (
    ProviderDetector,
    RiskAssessor,
    RiskLevel,
    extract_tool_info,
    normalize_tools,
    safe_getattr,
    safe_getattr_multi,
)


@runtime_checkable
class Integration(Protocol):
    """
    Protocol for framework integrations.

    Each integration provides introspection capabilities for a specific
    AI agent framework. Integrations are discovered and registered
    automatically when their module is imported.
    """

    @property
    def framework_name(self) -> str:
        """The name of the framework (e.g., 'langchain', 'crewai')."""
        ...

    def can_introspect(self, obj: Any) -> bool:
        """Check if this integration can introspect the given object."""
        ...

    def introspect(self, obj: Any) -> IntrospectionResult:
        """Introspect an object to extract metadata."""
        ...


class IntegrationRegistry:
    """
    Registry for framework integrations.

    Thread-safe registry that discovers and manages framework integrations.
    Integrations are auto-registered when imported.

    Example:
        ```python
        from agentfacts.integrations import get_registry

        registry = get_registry()

        # Get integration for an object
        integration = registry.get_for_object(my_agent)

        # List available integrations
        print(registry.available_frameworks)
        ```
    """

    def __init__(self) -> None:
        self._integrations: dict[str, Integration] = {}
        self._lock = threading.RLock()

    def register(self, integration: Integration) -> None:
        """
        Register an integration.

        Args:
            integration: Integration instance to register
        """
        with self._lock:
            self._integrations[integration.framework_name] = integration

    def unregister(self, framework_name: str) -> bool:
        """
        Unregister an integration by framework name.

        Returns:
            True if found and removed
        """
        with self._lock:
            if framework_name in self._integrations:
                del self._integrations[framework_name]
                return True
            return False

    def get(self, framework_name: str) -> Integration | None:
        """Get an integration by framework name."""
        with self._lock:
            return self._integrations.get(framework_name)

    def get_for_object(self, obj: Any) -> Integration | None:
        """
        Find an integration that can handle the given object.

        Args:
            obj: Object to find integration for

        Returns:
            First matching integration or None
        """
        with self._lock:
            for integration in self._integrations.values():
                if integration.can_introspect(obj):
                    return integration
        return None

    @property
    def available_frameworks(self) -> list[str]:
        """Get list of available framework names."""
        with self._lock:
            return list(self._integrations.keys())

    def has_framework(self, framework: str) -> bool:
        """Check if a framework is registered."""
        with self._lock:
            return framework in self._integrations

    def detect_framework(self, obj: Any) -> str | None:
        """
        Detect which framework the object belongs to.

        Args:
            obj: The object to analyze

        Returns:
            Framework name or None if unrecognized
        """
        # Check registered integrations first
        with self._lock:
            integrations = list(self._integrations.values())

        for integration in integrations:
            if integration.can_introspect(obj):
                return integration.framework_name

        # Fallback to module-based detection
        module = getattr(type(obj), "__module__", "")

        framework_hints = {
            "crewai": "crewai",
            "autogen": "autogen",
            "langchain": "langchain",
            "langgraph": "langchain",
            "llama_index": "llamaindex",
            "semantic_kernel": "semantic_kernel",
            "haystack": "haystack",
            "dspy": "dspy",
            "smolagents": "huggingface",
            "huggingface": "huggingface",
            "openagents": "openagents",
        }

        for hint, framework in framework_hints.items():
            if hint in module.lower():
                return framework

        return None

    def introspect(
        self,
        obj: Any,
        framework: str | None = None,
    ) -> IntrospectionResult:
        """
        Introspect an agent object to extract metadata.

        Args:
            obj: The agent/crew/chain to introspect
            framework: Explicit framework name (auto-detected if None)

        Returns:
            IntrospectionResult with extracted metadata

        Raises:
            ValueError: If framework cannot be detected or is unsupported
        """
        from agentfacts.models import OperationalConstraints

        # Detect framework if not specified
        if framework is None:
            framework = self.detect_framework(obj)

        if framework is None:
            raise ValueError(
                f"Cannot detect framework for {type(obj).__name__}. "
                "Please specify the framework explicitly or register an integration."
            )

        # Get the integration (thread-safe lookup)
        with self._lock:
            integration = self._integrations.get(framework)
            available = list(self._integrations.keys())

        if integration is None:
            raise ValueError(
                f"No integration registered for framework '{framework}'. "
                f"Available: {available}"
            )

        # Perform introspection
        result = integration.introspect(obj)

        # If result is already IntrospectionResult, return it
        if isinstance(result, IntrospectionResult):
            return result

        # Fallback - return empty result
        return IntrospectionResult(
            framework=framework,
            baseline_model=None,
            capabilities=[],
            constraints=OperationalConstraints(),
            context={"framework": framework, "agent_type": type(obj).__name__},
        )

    def introspect_safe(
        self,
        obj: Any,
        framework: str | None = None,
    ) -> IntrospectionResult | None:
        """
        Introspect without raising exceptions.

        Args:
            obj: The agent/crew/chain to introspect
            framework: Explicit framework name

        Returns:
            IntrospectionResult or None if introspection fails
        """
        try:
            return self.introspect(obj, framework)
        except Exception:
            return None

    def clear(self) -> None:
        """Clear all registered integrations."""
        with self._lock:
            self._integrations.clear()


# Global registry
_global_registry: IntegrationRegistry | None = None
_global_registry_lock = threading.Lock()


def get_registry() -> IntegrationRegistry:
    """
    Get the global integration registry.

    Thread-safe lazy initialization.

    Returns:
        The global IntegrationRegistry instance
    """
    global _global_registry
    if _global_registry is None:
        with _global_registry_lock:
            if _global_registry is None:
                _global_registry = IntegrationRegistry()
                _discover_integrations(_global_registry)
    return _global_registry


def reset_registry() -> None:
    """Reset the global integration registry (for testing)."""
    global _global_registry
    with _global_registry_lock:
        _global_registry = None


def _discover_integrations(registry: IntegrationRegistry) -> None:
    """
    Auto-discover and register available integrations.

    This is called once when the registry is first accessed.
    It attempts to import each integration module and register
    any integrations found there.
    """
    integration_modules = [
        "agentfacts.integrations.langchain",
        "agentfacts.integrations.crewai",
        "agentfacts.integrations.autogen",
        "agentfacts.integrations.llamaindex",
        "agentfacts.integrations.openagents",
        "agentfacts.integrations.huggingface",
    ]

    for module_name in integration_modules:
        try:
            module = __import__(module_name, fromlist=["*"])
            register = getattr(module, "_register", None)
            if callable(register):
                register()
        except ImportError:
            pass
        except Exception:
            pass


__all__ = [
    # Protocol
    "Integration",
    # Registry
    "IntegrationRegistry",
    "get_registry",
    "reset_registry",
    # Base classes
    "BaseIntrospector",
    "ContextKey",
    "IntrospectionResult",
    "IntrospectionTuple",
    "REQUIRED_CONTEXT_KEYS",
    "RECOMMENDED_CONTEXT_KEYS",
    # Utilities
    "ProviderDetector",
    "RiskAssessor",
    "RiskLevel",
    "extract_tool_info",
    "normalize_tools",
    "safe_getattr",
    "safe_getattr_multi",
]
