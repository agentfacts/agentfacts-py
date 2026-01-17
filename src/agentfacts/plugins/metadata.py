"""
Metadata Provider plugin interfaces.

Defines protocols for resolving agent metadata by DID.
"""

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class MetadataProvider(Protocol):
    """
    Protocol for resolving agent metadata by DID.

    Used by middleware to fetch full AgentFacts metadata
    for verification.

    Example:
        ```python
        class DatabaseMetadataProvider:
            def __init__(self, db):
                self.db = db

            def resolve(self, did: str) -> dict | None:
                return self.db.get_agent_metadata(did)
        ```
    """

    def resolve(self, did: str) -> Any:
        """
        Resolve agent metadata by DID.

        Args:
            did: The agent's DID

        Returns:
            AgentFacts instance, dict, JSON string, or None if not found
        """
        ...
