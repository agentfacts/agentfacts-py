"""
LangChain callback handler for AgentFacts SDK.

Captures runtime events and updates agent metadata dynamically.
"""

import contextlib
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any
from uuid import UUID

LANGCHAIN_AVAILABLE = False

if TYPE_CHECKING:
    from langchain_core.agents import AgentAction, AgentFinish
    from langchain_core.outputs import LLMResult

    class BaseCallbackHandler:
        """Type-checking stub for LangChain's BaseCallbackHandler."""

        pass
else:
    try:
        from langchain_core.agents import AgentAction, AgentFinish
        from langchain_core.callbacks import BaseCallbackHandler
        from langchain_core.outputs import LLMResult

        LANGCHAIN_AVAILABLE = True
    except ImportError:

        class BaseCallbackHandler:
            """Fallback base class when LangChain is unavailable."""

            pass


class AgentFactsCallbackHandler(BaseCallbackHandler):
    """
    Callback handler that captures agent execution events for the transparency log.

    Records tool invocations, LLM calls, and errors for audit purposes.
    """

    def __init__(
        self,
        agent_facts: Any | None = None,
        log_llm_calls: bool = True,
        log_tool_calls: bool = True,
        log_errors: bool = True,
    ) -> None:
        """
        Initialize the callback handler.

        Args:
            agent_facts: Optional AgentFacts instance to update
            log_llm_calls: Whether to log LLM invocations
            log_tool_calls: Whether to log tool invocations
            log_errors: Whether to log errors
        """
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "LangChain is required for AgentFactsCallbackHandler. "
                "Install it with: pip install agentfacts[langchain]"
            )

        self.agent_facts = agent_facts
        self.log_llm_calls = log_llm_calls
        self.log_tool_calls = log_tool_calls
        self.log_errors = log_errors

        # Event collection
        self.events: list[dict[str, Any]] = []
        self.run_started_at: datetime | None = None
        self.current_run_id: str | None = None

    def _record_event(self, event_type: str, data: dict[str, Any]) -> None:
        """Record an event to the internal log."""
        timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        event = {
            "type": event_type,
            "timestamp": timestamp,
            "run_id": self.current_run_id,
            **data,
        }
        self.events.append(event)

        # If we have an AgentFacts instance with a transparency log, record there too
        if self.agent_facts is not None and hasattr(self.agent_facts, "log_evidence"):
            with contextlib.suppress(Exception):
                self.agent_facts.log_evidence(event_type, data)

    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when a chain starts running."""
        self.run_started_at = datetime.now(timezone.utc)
        self.current_run_id = str(run_id)

    def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when a chain finishes running."""
        if self.run_started_at:
            duration = (
                datetime.now(timezone.utc) - self.run_started_at
            ).total_seconds()
            self._record_event(
                "chain_completed",
                {
                    "duration_seconds": duration,
                    "success": True,
                },
            )

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when a chain errors."""
        if self.log_errors:
            self._record_event(
                "chain_error",
                {
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                },
            )

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when an LLM starts generating."""
        if self.log_llm_calls:
            self._record_event(
                "llm_invocation",
                {
                    "model": serialized.get("name", "unknown"),
                    "prompt_count": len(prompts),
                },
            )

    def on_llm_end(
        self,
        response: "LLMResult",
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when an LLM finishes generating."""
        if self.log_llm_calls:
            token_usage = {}
            if hasattr(response, "llm_output") and response.llm_output:
                token_usage = response.llm_output.get("token_usage", {})

            self._record_event(
                "llm_completed",
                {
                    "generations": len(response.generations),
                    "token_usage": token_usage,
                },
            )

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when an LLM errors."""
        if self.log_errors:
            self._record_event(
                "llm_error",
                {
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                },
            )

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when a tool starts running."""
        if self.log_tool_calls:
            self._record_event(
                "tool_invocation",
                {
                    "tool_name": serialized.get("name", "unknown"),
                    "input_length": len(input_str),
                },
            )

    def on_tool_end(
        self,
        output: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when a tool finishes running."""
        if self.log_tool_calls:
            self._record_event(
                "tool_completed",
                {
                    "output_length": len(output) if output else 0,
                },
            )

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when a tool errors."""
        if self.log_errors:
            self._record_event(
                "tool_error",
                {
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                },
            )

    def on_agent_action(
        self,
        action: "AgentAction",
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when an agent takes an action."""
        self._record_event(
            "agent_action",
            {
                "tool": action.tool,
                "log": action.log[:200] if action.log else None,  # Truncate for privacy
            },
        )

    def on_agent_finish(
        self,
        finish: "AgentFinish",
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when an agent finishes."""
        self._record_event(
            "agent_finished",
            {
                "has_output": bool(finish.return_values),
            },
        )

    def get_events(self) -> list[dict[str, Any]]:
        """Get all recorded events."""
        return self.events.copy()

    def clear_events(self) -> None:
        """Clear recorded events."""
        self.events.clear()

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of the recorded events."""
        tool_calls = [e for e in self.events if e["type"] == "tool_invocation"]
        llm_calls = [e for e in self.events if e["type"] == "llm_invocation"]
        errors = [e for e in self.events if "error" in e["type"]]

        return {
            "total_events": len(self.events),
            "tool_invocations": len(tool_calls),
            "llm_invocations": len(llm_calls),
            "errors": len(errors),
            "tools_used": list(
                {e.get("tool_name") for e in tool_calls if e.get("tool_name")}
            ),
        }

    # Async callback methods for LangChain async operations

    async def on_chain_start_async(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Async version of on_chain_start."""
        self.on_chain_start(
            serialized,
            inputs,
            run_id=run_id,
            parent_run_id=parent_run_id,
            tags=tags,
            metadata=metadata,
            **kwargs,
        )

    async def on_chain_end_async(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Async version of on_chain_end."""
        self.on_chain_end(outputs, run_id=run_id, parent_run_id=parent_run_id, **kwargs)

    async def on_chain_error_async(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Async version of on_chain_error."""
        self.on_chain_error(error, run_id=run_id, parent_run_id=parent_run_id, **kwargs)

    async def on_llm_start_async(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Async version of on_llm_start."""
        self.on_llm_start(
            serialized,
            prompts,
            run_id=run_id,
            parent_run_id=parent_run_id,
            tags=tags,
            metadata=metadata,
            **kwargs,
        )

    async def on_llm_end_async(
        self,
        response: "LLMResult",
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Async version of on_llm_end."""
        self.on_llm_end(response, run_id=run_id, parent_run_id=parent_run_id, **kwargs)

    async def on_llm_error_async(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Async version of on_llm_error."""
        self.on_llm_error(error, run_id=run_id, parent_run_id=parent_run_id, **kwargs)

    async def on_tool_start_async(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Async version of on_tool_start."""
        self.on_tool_start(
            serialized,
            input_str,
            run_id=run_id,
            parent_run_id=parent_run_id,
            tags=tags,
            metadata=metadata,
            **kwargs,
        )

    async def on_tool_end_async(
        self,
        output: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Async version of on_tool_end."""
        self.on_tool_end(output, run_id=run_id, parent_run_id=parent_run_id, **kwargs)

    async def on_tool_error_async(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Async version of on_tool_error."""
        self.on_tool_error(error, run_id=run_id, parent_run_id=parent_run_id, **kwargs)
