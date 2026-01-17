"""
Status/Revocation Checking plugin interfaces.

Defines protocols and data classes for checking credential status.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from agentfacts.plugins.context import VerificationContext


@dataclass
class StatusCheckResult:
    """
    Result of checking credential status.

    Attributes:
        valid: True if credential is not revoked/suspended
        status: Current status string (e.g., "active", "revoked", "suspended")
        checked_at: Timestamp of the check
        errors: Error messages if check failed
    """

    valid: bool
    status: str = "unknown"
    checked_at: datetime | None = None
    errors: list[str] = field(default_factory=list)


@runtime_checkable
class StatusChecker(Protocol):
    """
    Protocol for checking credential revocation status.

    Implement this to support status list formats like
    StatusList2021, RevocationList2020, or custom registries.

    Example:
        ```python
        class StatusList2021Checker:
            def supports(self, status_ref: str) -> bool:
                return "StatusList2021" in status_ref

            def check(self, status_ref, context) -> StatusCheckResult:
                # Fetch and decode status list bitstring
                ...
        ```
    """

    def supports(self, status_ref: str) -> bool:
        """
        Check if this checker supports the given status reference.

        Args:
            status_ref: The status list reference URL or ID

        Returns:
            True if this checker can handle the reference
        """
        ...

    def check(
        self,
        status_ref: str,
        context: VerificationContext,
    ) -> StatusCheckResult:
        """
        Check the revocation status of a credential.

        Args:
            status_ref: Reference to the status list entry
            context: Verification context

        Returns:
            StatusCheckResult with current status
        """
        ...
