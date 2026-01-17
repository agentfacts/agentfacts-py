"""
DID Resolution plugin interfaces.

Defines protocols and data classes for DID resolvers.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol, runtime_checkable


@dataclass
class ResolvedDID:
    """
    Result of resolving a DID to its verification material.

    Attributes:
        did: The resolved DID URI
        public_key_base64: Ed25519 public key in base64 format
        metadata: Optional resolution metadata (e.g., controller, service endpoints)
        resolved_at: Timestamp of resolution
    """

    did: str
    public_key_base64: str
    metadata: dict[str, Any] = field(default_factory=dict)
    resolved_at: datetime | None = None


@runtime_checkable
class DIDResolver(Protocol):
    """
    Protocol for resolving DIDs to their verification keys.

    The default did:key resolver is built-in. Implement this protocol
    to support external DID methods like did:web, did:ion, did:ethr.

    Example:
        ```python
        class DidWebResolver:
            def supports(self, did: str) -> bool:
                return did.startswith("did:web:")

            def resolve(self, did: str) -> ResolvedDID:
                # Fetch .well-known/did.json from domain
                ...
        ```
    """

    def supports(self, did: str) -> bool:
        """
        Check if this resolver supports the given DID method.

        Args:
            did: The DID URI to check

        Returns:
            True if this resolver can resolve the DID
        """
        ...

    def resolve(self, did: str) -> ResolvedDID:
        """
        Resolve a DID to its verification material.

        Args:
            did: The DID URI to resolve

        Returns:
            ResolvedDID with public key and metadata

        Raises:
            ValueError: If DID cannot be resolved
        """
        ...
