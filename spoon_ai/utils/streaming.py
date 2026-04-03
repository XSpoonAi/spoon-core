from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from spoon_ai.schema import (
    LLMResponse,
    LLMResponseChunk,
    Message,
    ToolCall,
)


@dataclass
class StreamOutcome:
    """Accumulator for streaming output state."""

    content: str = ""
    finish_reason: Optional[str] = None
    usage: Optional[dict] = None
    tool_calls: List[ToolCall] = field(default_factory=list)

    def update_from_chunk(self, chunk: LLMResponseChunk) -> None:
        if chunk.content:
            self.content = chunk.content
        elif chunk.delta:
            self.content += chunk.delta

        if chunk.finish_reason:
            self.finish_reason = chunk.finish_reason
        if chunk.usage:
            self.usage = chunk.usage
        if chunk.tool_calls:
            self.tool_calls = chunk.tool_calls

    def update_from_response(self, response: LLMResponse) -> None:
        if getattr(response, "content", None):
            self.content = response.content
        finish = getattr(response, "finish_reason", None) or getattr(
            response, "native_finish_reason", None
        )
        if finish:
            self.finish_reason = finish
        if getattr(response, "tool_calls", None):
            self.tool_calls = response.tool_calls

    def build_response(self) -> LLMResponse:
        finish = self.finish_reason or "stop"
        return LLMResponse(
            content=self.content,
            finish_reason=finish,
            native_finish_reason=finish,
            tool_calls=self.tool_calls,
        )


def message_to_dict(message: Any) -> Dict[str, Any]:
    if isinstance(message, dict):
        return dict(message)
    if isinstance(message, Message):
        return message.model_dump()
    raise ValueError(f"Unsupported message type: {type(message)}")


def sanitize_stream_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    sanitized = dict(kwargs)
    for key in ("callbacks", "stream"):
        sanitized.pop(key, None)
    return sanitized


def build_output_queue_event(
    *,
    event_type: str,
    delta: str = "",
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a backward-compatible output queue event payload."""
    payload: Dict[str, Any] = {
        "type": event_type,
        "delta": delta,
        "content": delta,
        "metadata": dict(metadata or {}),
    }
    return payload
