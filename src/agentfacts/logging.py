"""
Structured logging for AgentFacts SDK.

Provides configurable logging with support for JSON output
and contextual information about agent operations.
"""

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any


class AgentFactsLogger:
    """
    Structured logger for AgentFacts operations.

    Supports both human-readable and JSON output formats.
    """

    def __init__(
        self,
        name: str = "agentfacts",
        level: int = logging.INFO,
        json_output: bool = False,
        stream: Any = None,
    ):
        """
        Initialize the logger.

        Args:
            name: Logger name
            level: Logging level (default: INFO)
            json_output: If True, output logs as JSON
            stream: Output stream (default: sys.stderr)
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.json_output = json_output
        self._context: dict[str, Any] = {}

        # Remove existing handlers
        self.logger.handlers.clear()

        # Add handler
        handler = logging.StreamHandler(stream or sys.stderr)
        handler.setLevel(level)

        if json_output:
            handler.setFormatter(JsonFormatter())
        else:
            handler.setFormatter(
                logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
            )

        self.logger.addHandler(handler)

    def set_context(self, **kwargs: Any) -> None:
        """Set contextual fields that will be included in all log entries."""
        self._context.update(kwargs)

    def clear_context(self) -> None:
        """Clear all contextual fields."""
        self._context.clear()

    def _log(
        self,
        level: int,
        message: str,
        event_type: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Internal log method with structured data."""
        extra = {
            "event_type": event_type,
            "structured_data": {**self._context, **kwargs},
        }
        self.logger.log(level, message, extra=extra)

    def debug(self, message: str, event_type: str | None = None, **kwargs: Any) -> None:
        """Log debug message."""
        self._log(logging.DEBUG, message, event_type, **kwargs)

    def info(self, message: str, event_type: str | None = None, **kwargs: Any) -> None:
        """Log info message."""
        self._log(logging.INFO, message, event_type, **kwargs)

    def warning(
        self, message: str, event_type: str | None = None, **kwargs: Any
    ) -> None:
        """Log warning message."""
        self._log(logging.WARNING, message, event_type, **kwargs)

    def error(self, message: str, event_type: str | None = None, **kwargs: Any) -> None:
        """Log error message."""
        self._log(logging.ERROR, message, event_type, **kwargs)

    def critical(
        self, message: str, event_type: str | None = None, **kwargs: Any
    ) -> None:
        """Log critical message."""
        self._log(logging.CRITICAL, message, event_type, **kwargs)

    # Convenience methods for common AgentFacts events

    def log_signature_created(self, agent_did: str, **kwargs: Any) -> None:
        """Log signature creation event."""
        self.info(
            f"Signature created for agent {agent_did}",
            event_type="signature_created",
            agent_did=agent_did,
            **kwargs,
        )

    def log_signature_verified(
        self, agent_did: str, valid: bool, **kwargs: Any
    ) -> None:
        """Log signature verification event."""
        level = logging.INFO if valid else logging.WARNING
        self._log(
            level,
            f"Signature verification {'passed' if valid else 'failed'} for {agent_did}",
            event_type="signature_verified",
            agent_did=agent_did,
            valid=valid,
            **kwargs,
        )

    def log_policy_evaluation(
        self,
        policy_name: str,
        agent_did: str,
        passed: bool,
        violations: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Log policy evaluation event."""
        level = logging.INFO if passed else logging.WARNING
        self._log(
            level,
            f"Policy '{policy_name}' {'passed' if passed else 'failed'} for {agent_did}",
            event_type="policy_evaluated",
            policy_name=policy_name,
            agent_did=agent_did,
            passed=passed,
            violations=violations or [],
            **kwargs,
        )

    def log_handshake(
        self,
        initiator_did: str,
        responder_did: str,
        success: bool,
        **kwargs: Any,
    ) -> None:
        """Log handshake event."""
        level = logging.INFO if success else logging.WARNING
        self._log(
            level,
            f"Handshake {'completed' if success else 'failed'}: {initiator_did} -> {responder_did}",
            event_type="handshake",
            initiator_did=initiator_did,
            responder_did=responder_did,
            success=success,
            **kwargs,
        )

    def log_introspection(
        self, agent_name: str, model_name: str | None, tool_count: int, **kwargs: Any
    ) -> None:
        """Log agent introspection event."""
        self.info(
            f"Introspected agent '{agent_name}': model={model_name}, tools={tool_count}",
            event_type="introspection",
            agent_name=agent_name,
            model_name=model_name,
            tool_count=tool_count,
            **kwargs,
        )


class JsonFormatter(logging.Formatter):
    """JSON log formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add event type if present
        if hasattr(record, "event_type") and record.event_type:
            log_data["event_type"] = record.event_type

        # Add structured data if present
        if hasattr(record, "structured_data") and record.structured_data:
            log_data["data"] = record.structured_data

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data)


# Global logger instance
_logger: AgentFactsLogger | None = None

# Configuration for default warnings
_warn_on_defaults: bool = False


def enable_default_warnings(enabled: bool = True) -> None:
    """
    Enable or disable warnings when silent defaults are used.

    When enabled, the SDK will warn when:
    - No baseline model is detected (uses "unknown")
    - No tools/capabilities are found
    - Constraints use default values
    - Introspection falls back to defaults

    This helps users understand what data was actually extracted
    vs what was silently defaulted.

    Args:
        enabled: Whether to enable warnings (default: True)

    Example:
        ```python
        from agentfacts.logging import enable_default_warnings

        # Enable verbose warnings about defaults
        enable_default_warnings(True)

        # Create AgentFacts - will warn if model/tools not found
        facts = AgentFacts.from_agent(my_agent, "MyAgent")
        ```
    """
    global _warn_on_defaults
    _warn_on_defaults = enabled


def warn_on_defaults() -> bool:
    """Check if default warnings are enabled."""
    return _warn_on_defaults


def warn_default(
    field: str,
    default_value: Any,
    context: str = "",
) -> None:
    """
    Emit a warning when a default value is used.

    Only emits if enable_default_warnings(True) has been called.

    Args:
        field: Name of the field using a default
        default_value: The default value being used
        context: Additional context about where this happened
    """
    if not _warn_on_defaults:
        return

    logger = get_logger()
    msg = f"Using default for '{field}': {default_value!r}"
    if context:
        msg += f" ({context})"
    logger.warning(
        msg,
        event_type="default_used",
        field=field,
        default_value=str(default_value),
        context=context,
    )


def get_logger() -> AgentFactsLogger:
    """Get the global AgentFacts logger instance."""
    global _logger
    if _logger is None:
        _logger = AgentFactsLogger()
    return _logger


def configure_logging(
    level: int = logging.INFO,
    json_output: bool = False,
    stream: Any = None,
) -> AgentFactsLogger:
    """
    Configure the global AgentFacts logger.

    Args:
        level: Logging level
        json_output: If True, output logs as JSON
        stream: Output stream

    Returns:
        Configured logger instance
    """
    global _logger
    _logger = AgentFactsLogger(
        level=level,
        json_output=json_output,
        stream=stream,
    )
    return _logger
