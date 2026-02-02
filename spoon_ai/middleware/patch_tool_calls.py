"""PatchToolCalls Middleware - Fix Dangling Tool Calls.

Patches message history to handle dangling tool calls that occur when:
- HITL (Human-in-the-Loop) interrupts tool execution
- Errors cause tool execution to be skipped
- Agent is resumed from a checkpoint mid-execution

Compatible with LangChain DeepAgents PatchToolCallsMiddleware interface.

Usage:
    from spoon_ai.middleware.patch_tool_calls import PatchToolCallsMiddleware

    agent = ToolCallAgent(
        middleware=[PatchToolCallsMiddleware()],
        ...
    )
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Set

from spoon_ai.middleware.base import (
    AgentMiddleware,
    AgentRuntime,
    ModelRequest,
    ModelResponse,
)
from spoon_ai.schema import Message, Role, ToolCall

logger = logging.getLogger(__name__)


# ============================================================================
# Constants
# ============================================================================

CANCELLED_TOOL_MESSAGE = (
    "Tool call {tool_name} with id {tool_call_id} was cancelled - "
    "another message came in before it could be completed."
)


# ============================================================================
# PatchToolCalls Middleware
# ============================================================================

class PatchToolCallsMiddleware(AgentMiddleware):
    """Middleware to patch dangling tool calls in message history.

    A "dangling tool call" occurs when an AI message contains tool_calls
    but there's no corresponding tool response message. This violates
    the OpenAI/Anthropic API requirements and causes errors.

    This middleware:
    1. Scans message history for AI messages with tool_calls
    2. Checks if each tool call has a corresponding tool response
    3. Injects synthetic tool response messages for any missing ones

    Common causes of dangling tool calls:
    - HITL approval flow rejects or edits a tool call
    - Error during tool execution before response is added
    - Agent resumed from checkpoint mid-execution
    - Network timeout during tool execution

    Example:
        ```python
        from spoon_ai.middleware.patch_tool_calls import PatchToolCallsMiddleware

        middleware = PatchToolCallsMiddleware()

        agent = ToolCallAgent(
            middleware=[middleware],
            ...
        )
        ```
    """

    def __init__(
        self,
        cancelled_message_template: Optional[str] = None,
        log_patches: bool = True,
    ):
        """Initialize PatchToolCalls middleware.

        Args:
            cancelled_message_template: Custom message template for cancelled tools.
                Must contain {tool_name} and {tool_call_id} placeholders.
            log_patches: Whether to log when patches are applied (default: True)
        """
        super().__init__()
        self._message_template = cancelled_message_template or CANCELLED_TOOL_MESSAGE
        self._log_patches = log_patches
        self._patch_count = 0

    def _get_tool_call_id(self, tool_call: Any) -> Optional[str]:
        """Extract tool call ID from various formats."""
        if isinstance(tool_call, dict):
            return tool_call.get('id')
        elif hasattr(tool_call, 'id'):
            return tool_call.id
        return None

    def _get_tool_call_name(self, tool_call: Any) -> str:
        """Extract tool name from various formats."""
        if isinstance(tool_call, dict):
            func = tool_call.get('function', {})
            if isinstance(func, dict):
                return func.get('name', 'unknown')
            return 'unknown'
        elif hasattr(tool_call, 'function'):
            if hasattr(tool_call.function, 'name'):
                return tool_call.function.name
        return 'unknown'

    def _find_dangling_tool_calls(
        self,
        messages: List[Message]
    ) -> List[Dict[str, Any]]:
        """Find all dangling tool calls in message history.

        Returns list of dicts with:
        - message_index: Index of AI message with the tool call
        - tool_call_id: ID of the dangling tool call
        - tool_name: Name of the tool
        """
        dangling = []

        # Collect all tool response IDs
        tool_response_ids: Set[str] = set()
        for msg in messages:
            if msg.role == Role.TOOL and hasattr(msg, 'tool_call_id') and msg.tool_call_id:
                tool_response_ids.add(msg.tool_call_id)

        # Find AI messages with tool calls that lack responses
        for i, msg in enumerate(messages):
            if msg.role != Role.ASSISTANT:
                continue

            tool_calls = getattr(msg, 'tool_calls', None)
            if not tool_calls:
                continue

            for tc in tool_calls:
                tc_id = self._get_tool_call_id(tc)
                if tc_id and tc_id not in tool_response_ids:
                    dangling.append({
                        'message_index': i,
                        'tool_call_id': tc_id,
                        'tool_name': self._get_tool_call_name(tc),
                    })

        return dangling

    def _create_patch_message(self, tool_call_id: str, tool_name: str) -> Message:
        """Create a synthetic tool response message for a cancelled tool call."""
        content = self._message_template.format(
            tool_name=tool_name,
            tool_call_id=tool_call_id
        )
        return Message(
            role=Role.TOOL,
            content=content,
            tool_call_id=tool_call_id,
            name=tool_name,
        )

    def _patch_messages(self, messages: List[Message]) -> List[Message]:
        """Patch message list to fix all dangling tool calls.

        Returns new message list with synthetic tool responses inserted
        after their corresponding AI messages.
        """
        dangling = self._find_dangling_tool_calls(messages)

        if not dangling:
            return messages

        # Group patches by message index for efficient insertion
        patches_by_index: Dict[int, List[Message]] = {}
        for d in dangling:
            idx = d['message_index']
            patch_msg = self._create_patch_message(d['tool_call_id'], d['tool_name'])

            if idx not in patches_by_index:
                patches_by_index[idx] = []
            patches_by_index[idx].append(patch_msg)

            if self._log_patches:
                logger.info(
                    f"PatchToolCallsMiddleware: Patching dangling tool call "
                    f"'{d['tool_name']}' (id={d['tool_call_id']})"
                )

        # Build new message list with patches inserted
        patched = []
        for i, msg in enumerate(messages):
            patched.append(msg)

            # Insert patches after this message
            if i in patches_by_index:
                patched.extend(patches_by_index[i])

        self._patch_count += len(dangling)
        return patched

    def before_agent(
        self,
        state: Dict[str, Any],
        runtime: AgentRuntime
    ) -> Optional[Dict[str, Any]]:
        """Patch dangling tool calls when agent starts.

        This handles cases where agent is resumed from a checkpoint
        with incomplete tool execution.
        """
        messages = runtime.messages

        if not messages:
            return None

        patched = self._patch_messages(messages)

        if len(patched) != len(messages):
            # Update runtime messages
            runtime.messages = patched

            # Update agent memory if accessible
            if hasattr(runtime, '_agent_instance'):
                agent = runtime._agent_instance
                if hasattr(agent, 'memory') and hasattr(agent.memory, 'messages'):
                    agent.memory.messages = patched
                    logger.debug("Updated agent memory with patched messages")

            return {"messages_patched": True}

        return None

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable
    ) -> ModelResponse:
        """Patch dangling tool calls before model call."""
        messages = list(request.messages) if request.messages else []

        if not messages:
            return await handler(request)

        patched = self._patch_messages(messages)

        if len(patched) != len(messages):
            # Create new request with patched messages
            request = ModelRequest(
                messages=patched,
                system_prompt=request.system_prompt,
                tools=request.tools,
                phase=request.phase,
                runtime=request.runtime,
            )

            # Update agent memory if accessible
            if request.runtime and hasattr(request.runtime, '_agent_instance'):
                agent = request.runtime._agent_instance
                if hasattr(agent, 'memory') and hasattr(agent.memory, 'messages'):
                    agent.memory.messages = patched

        return await handler(request)

    def get_stats(self) -> Dict[str, int]:
        """Get patching statistics."""
        return {
            "patches_applied": self._patch_count,
        }


# ============================================================================
# Factory Function
# ============================================================================

def create_patch_tool_calls_middleware(
    log_patches: bool = True,
) -> PatchToolCallsMiddleware:
    """Create a PatchToolCalls middleware.

    Args:
        log_patches: Whether to log when patches are applied

    Returns:
        Configured PatchToolCallsMiddleware
    """
    return PatchToolCallsMiddleware(log_patches=log_patches)
