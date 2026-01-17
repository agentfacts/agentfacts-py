"""
Shared utilities for AgentFacts SDK.

This module provides common utility functions used across the SDK
to avoid code duplication.
"""

from datetime import datetime, timezone


def utcnow() -> datetime:
    """Get current UTC time (timezone-aware).

    Returns:
        A timezone-aware datetime object representing the current UTC time.

    Example:
        >>> from agentfacts.utils import utcnow
        >>> now = utcnow()
        >>> now.tzinfo is not None
        True
    """
    return datetime.now(timezone.utc)
