"""
Plugin Registry for AgentFacts extensibility.

Provides centralized registration and lookup for plugins.
"""

import threading
from typing import Any

from agentfacts.plugins.attestation import AttestationVerifier
from agentfacts.plugins.context import VerificationContext
from agentfacts.plugins.did import DIDResolver, ResolvedDID
from agentfacts.plugins.governance import GovernanceAdapter
from agentfacts.plugins.status import StatusChecker


class PluginRegistry:
    """
    Registry for AgentFacts plugins.

    Provides centralized registration and lookup for:
    - DID resolvers (by DID method)
    - Attestation verifiers (by format)
    - Status checkers
    - Governance adapters (by name)

    Thread-safe for concurrent access.

    Example:
        ```python
        from agentfacts.plugins import get_plugin_registry, VerificationContext

        registry = get_plugin_registry()

        # Register a custom DID resolver
        registry.register_did_resolver("web", DidWebResolver())

        # Register an attestation verifier
        registry.register_attestation_verifier(SdJwtVcVerifier())

        # Create context from registry
        context = registry.create_context()

        # Use in verification
        facts.verify(context=context)
        ```
    """

    def __init__(self) -> None:
        self._did_resolvers: dict[str, DIDResolver] = {}
        self._attestation_verifiers: list[AttestationVerifier] = []
        self._status_checkers: list[StatusChecker] = []
        self._governance_adapters: dict[str, GovernanceAdapter] = {}
        self._lock = threading.RLock()

    def register_did_resolver(self, method: str, resolver: DIDResolver) -> None:
        """
        Register a DID resolver for a specific method.

        Args:
            method: The DID method (e.g., "web", "ion", "ethr")
            resolver: The resolver instance

        Example:
            ```python
            registry.register_did_resolver("web", DidWebResolver())
            # Now did:web:example.com will be resolved by DidWebResolver
            ```
        """
        with self._lock:
            self._did_resolvers[method] = resolver

    def unregister_did_resolver(self, method: str) -> bool:
        """
        Unregister a DID resolver by method.

        Returns:
            True if the resolver was found and removed
        """
        with self._lock:
            if method in self._did_resolvers:
                del self._did_resolvers[method]
                return True
            return False

    def get_did_resolver(self, did: str) -> DIDResolver | None:
        """
        Get a resolver for the given DID.

        Args:
            did: The DID URI (e.g., "did:web:example.com")

        Returns:
            Matching resolver or None
        """
        # Extract method from DID
        parts = did.split(":")
        if len(parts) < 2:
            return None
        method = parts[1]

        with self._lock:
            resolver = self._did_resolvers.get(method)

        if resolver and resolver.supports(did):
            return resolver
        return None

    def register_attestation_verifier(self, verifier: AttestationVerifier) -> None:
        """
        Register an attestation verifier.

        The verifier will be used for any attestation format it supports.

        Args:
            verifier: The verifier instance
        """
        with self._lock:
            self._attestation_verifiers.append(verifier)

    def unregister_attestation_verifier(self, verifier: AttestationVerifier) -> bool:
        """
        Unregister an attestation verifier.

        Returns:
            True if the verifier was found and removed
        """
        with self._lock:
            if verifier in self._attestation_verifiers:
                self._attestation_verifiers.remove(verifier)
                return True
            return False

    def get_attestation_verifier(self, format: str) -> AttestationVerifier | None:
        """
        Get a verifier for the given format.

        Args:
            format: The attestation format (e.g., "sd-jwt-vc")

        Returns:
            Matching verifier or None
        """
        with self._lock:
            for verifier in self._attestation_verifiers:
                if format in verifier.formats:
                    return verifier
        return None

    def register_status_checker(self, checker: StatusChecker) -> None:
        """
        Register a status checker.

        Args:
            checker: The status checker instance
        """
        with self._lock:
            self._status_checkers.append(checker)

    def unregister_status_checker(self, checker: StatusChecker) -> bool:
        """
        Unregister a status checker.

        Returns:
            True if the checker was found and removed
        """
        with self._lock:
            if checker in self._status_checkers:
                self._status_checkers.remove(checker)
                return True
            return False

    def get_status_checker(self, status_ref: str) -> StatusChecker | None:
        """
        Get a checker that supports the given status reference.

        Args:
            status_ref: The status list reference

        Returns:
            Matching checker or None
        """
        with self._lock:
            for checker in self._status_checkers:
                if checker.supports(status_ref):
                    return checker
        return None

    def register_governance_adapter(
        self, name: str, adapter: GovernanceAdapter
    ) -> None:
        """
        Register a governance adapter.

        Args:
            name: Adapter name (e.g., "degov", "oid4vc")
            adapter: The adapter instance
        """
        with self._lock:
            self._governance_adapters[name] = adapter

    def unregister_governance_adapter(self, name: str) -> bool:
        """
        Unregister a governance adapter by name.

        Returns:
            True if the adapter was found and removed
        """
        with self._lock:
            if name in self._governance_adapters:
                del self._governance_adapters[name]
                return True
            return False

    def get_governance_adapter(self, name: str) -> GovernanceAdapter | None:
        """
        Get a governance adapter by name.

        Args:
            name: Adapter name

        Returns:
            The adapter or None
        """
        with self._lock:
            return self._governance_adapters.get(name)

    def create_context(
        self,
        governance_adapter_name: str | None = None,
        governance_doc: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> VerificationContext:
        """
        Create a VerificationContext from registered plugins.

        This is a convenience method that builds a context with all
        registered resolvers, verifiers, and checkers.

        Args:
            governance_adapter_name: Name of governance adapter to use
            governance_doc: Governance document to evaluate
            **kwargs: Additional VerificationContext parameters

        Returns:
            Configured VerificationContext

        Example:
            ```python
            registry = get_plugin_registry()
            registry.register_did_resolver("web", DidWebResolver())

            context = registry.create_context()
            facts.verify(context=context)
            ```
        """
        with self._lock:
            # Create a composite DID resolver
            did_resolver = _CompositeResolver(dict(self._did_resolvers))

            # Copy lists
            verifiers = list(self._attestation_verifiers)
            checkers = list(self._status_checkers)

            # Get governance adapter
            gov_adapter = None
            if governance_adapter_name:
                gov_adapter = self._governance_adapters.get(governance_adapter_name)

        return VerificationContext(
            did_resolver=did_resolver if self._did_resolvers else None,
            attestation_verifiers=verifiers,
            status_checkers=checkers,
            governance_adapter=gov_adapter,
            governance_doc=governance_doc,
            **kwargs,
        )

    @property
    def registered_did_methods(self) -> list[str]:
        """Get list of registered DID methods."""
        with self._lock:
            return list(self._did_resolvers.keys())

    @property
    def registered_formats(self) -> set[str]:
        """Get set of all supported attestation formats."""
        with self._lock:
            formats: set[str] = set()
            for verifier in self._attestation_verifiers:
                formats.update(verifier.formats)
            return formats

    @property
    def registered_governance_adapters(self) -> list[str]:
        """Get list of registered governance adapter names."""
        with self._lock:
            return list(self._governance_adapters.keys())

    def clear(self) -> None:
        """Clear all registered plugins."""
        with self._lock:
            self._did_resolvers.clear()
            self._attestation_verifiers.clear()
            self._status_checkers.clear()
            self._governance_adapters.clear()


class _CompositeResolver:
    """Internal composite resolver that delegates to method-specific resolvers."""

    def __init__(self, resolvers: dict[str, DIDResolver]) -> None:
        self._resolvers = resolvers

    def supports(self, did: str) -> bool:
        """Check if any resolver supports this DID."""
        parts = did.split(":")
        if len(parts) < 2:
            return False
        method = parts[1]
        resolver = self._resolvers.get(method)
        return resolver is not None and resolver.supports(did)

    def resolve(self, did: str) -> ResolvedDID:
        """Resolve using the appropriate method resolver."""
        parts = did.split(":")
        if len(parts) < 2:
            raise ValueError(f"Invalid DID format: {did}")
        method = parts[1]
        resolver = self._resolvers.get(method)
        if resolver is None:
            raise ValueError(f"No resolver for DID method: {method}")
        return resolver.resolve(did)


# Global registry instance
_global_registry: PluginRegistry | None = None
_global_registry_lock = threading.Lock()


def get_plugin_registry() -> PluginRegistry:
    """
    Get the global plugin registry.

    Thread-safe lazy initialization.

    Returns:
        The global PluginRegistry instance

    Example:
        ```python
        from agentfacts.plugins import get_plugin_registry

        registry = get_plugin_registry()
        registry.register_did_resolver("web", DidWebResolver())
        ```
    """
    global _global_registry
    if _global_registry is None:
        with _global_registry_lock:
            if _global_registry is None:
                _global_registry = PluginRegistry()
    return _global_registry


def reset_plugin_registry() -> None:
    """Reset the global plugin registry (mainly for testing)."""
    global _global_registry
    with _global_registry_lock:
        _global_registry = None
