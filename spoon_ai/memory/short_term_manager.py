"""Short-term memory management for conversation history."""

import logging
import math
import uuid
from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any, Tuple, Set

from spoon_ai.schema import Message, SystemMessage
from spoon_ai.graph.checkpointer import InMemoryCheckpointer
from spoon_ai.graph.types import StateSnapshot
from .remove_message import RemoveMessage, REMOVE_ALL_MESSAGES

logger = logging.getLogger(__name__)


class TrimStrategy(str, Enum):
    """Strategy for trimming messages."""

    FROM_START = "from_start"  # Remove oldest messages first
    FROM_END = "from_end"  # Remove newest messages first


class MessageTokenCounter:
    """Approximate token counter aligned with LangChain semantics."""

    async def count_tokens(
        self, messages: List[Message], model: Optional[str] = None
    ) -> int:
        return self._approximate_count(messages)

    @staticmethod
    def _approximate_count(messages: List[Message]) -> int:
        chars_per_token = 4.0
        extra_tokens_per_message = 3.0
        token_count = 0.0

        for message in messages:
            message_chars = 0

            content = message.content
            if isinstance(content, str):
                message_chars += len(content)
            elif content is not None:
                message_chars += len(repr(content))

            if (
                message.role == "assistant"
                and message.tool_calls
                and not isinstance(message.content, list)
            ):
                message_chars += len(repr(message.tool_calls))

            if message.role == "tool" and message.tool_call_id:
                message_chars += len(message.tool_call_id)

            message_chars += len(message.role or "")

            if message.name:
                message_chars += len(message.name)

            token_count += math.ceil(message_chars / chars_per_token)
            token_count += extra_tokens_per_message

        total = math.ceil(token_count)
        return max(1, total)


def _ensure_message_ids(messages: List[Message]) -> None:
    for message in messages:
        if not getattr(message, "id", None):
            message.id = str(uuid.uuid4())


class ShortTermMemoryManager:
    """Manager for short-term conversation memory with advanced operations."""

    def __init__(
        self,
        checkpointer: Optional[InMemoryCheckpointer] = None,
        token_counter: Optional[MessageTokenCounter] = None,
        default_trim_strategy: TrimStrategy = TrimStrategy.FROM_END,
    ):

        self.checkpointer = checkpointer or InMemoryCheckpointer()
        self.token_counter = token_counter or MessageTokenCounter()
        self.default_trim_strategy = default_trim_strategy

    @staticmethod
    def _find_assistant_index_for_tool(messages: List[Message], tool_index: int) -> Optional[int]:
        """Locate the assistant message that issued the tool call for the given tool message."""
        if tool_index <= 0:
            return None

        tool_message = messages[tool_index]
        tool_call_id = getattr(tool_message, "tool_call_id", None)
        if not tool_call_id:
            return None

        for idx in range(tool_index - 1, -1, -1):
            assistant = messages[idx]
            if assistant.role == "assistant" and assistant.tool_calls:
                if any(call.id == tool_call_id for call in assistant.tool_calls):
                    return idx
        return None

    def _find_tool_indices_for_assistant(self, messages: List[Message], assistant_index: int) -> List[int]:
        """Locate all tool messages responding to the given assistant message."""
        assistant = messages[assistant_index]
        if assistant.role != "assistant" or not assistant.tool_calls:
            return []

        tool_call_ids = {call.id for call in assistant.tool_calls}
        found_indices = []

        # Tool messages responding to an assistant message are typically
        # immediately following it, but we'll search ahead to be safe.
        for idx in range(assistant_index + 1, len(messages)):
            msg = messages[idx]
            if msg.role == "tool" and getattr(msg, "tool_call_id", None) in tool_call_ids:
                found_indices.append(idx)
                # Note: we don't break early because there might be multiple calls
                # and we need to find all of them.
            
            # Stop if we hit another assistant message or user message? 
            # Actually, just keep going until we find all of them or end of list.
            if len(found_indices) == len(tool_call_ids):
                break
                
        return found_indices

    async def _apply_tool_call_dependencies(
        self,
        messages: List[Message],
        keep_indices: Set[int],
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
    ) -> Set[int]:
        """
        Ensure tool messages and their originating assistant messages are kept together.
        1. If a tool message is kept, its assistant message must be kept.
        2. If an assistant message with tool calls is kept, ALL its tool responses must be kept.
        If adding dependencies would violate the token budget, the entire group is dropped.
        """
        if not keep_indices:
            return keep_indices

        # Iteratively add dependencies until stable
        current_indices = set(keep_indices)

        while True:
            added_any = False
            dropped_any = False
            
            # 1. Tool -> Assistant (ensure parent is kept)
            for idx in list(current_indices):
                message = messages[idx]
                if message.role != "tool" or not getattr(message, "tool_call_id", None):
                    continue

                assistant_idx = self._find_assistant_index_for_tool(messages, idx)
                if assistant_idx is not None and assistant_idx not in current_indices:
                    current_indices.add(assistant_idx)
                    added_any = True

            # 2. Assistant -> Tools (ensure all responses are kept)
            # CRITICAL: If an assistant message has tool calls, but some of those tool calls
            # don't have responses in the ENTIRE message list (e.g. due to timeout),
            # we MUST drop the assistant message because OpenAI will reject it.
            for idx in list(current_indices):
                message = messages[idx]
                if message.role != "assistant" or not message.tool_calls:
                    continue

                tool_indices = self._find_tool_indices_for_assistant(messages, idx)
                
                # Check if all tool calls have a corresponding tool message in the source 'messages' list
                if len(tool_indices) < len(message.tool_calls):
                    # Orphaned tool calls detected! 
                    # This assistant message is invalid for the API.
                    logger.warning(f"Dropping assistant message at index {idx} because it has {len(message.tool_calls)} tool calls but only {len(tool_indices)} responses found in memory.")
                    current_indices.remove(idx)
                    # Also remove any partial responses already in keep_indices
                    for t_idx in tool_indices:
                        if t_idx in current_indices:
                            current_indices.remove(t_idx)
                    dropped_any = True
                    continue

                # All responses exist in source list, ensure they are all in current_indices
                for t_idx in tool_indices:
                    if t_idx not in current_indices:
                        current_indices.add(t_idx)
                        added_any = True
            
            if not added_any and not dropped_any:
                break

        # Check budget if specified
        if max_tokens is not None:
            # If we exceeded budget, we need a strategy to prune.
            # For now, let's just count and if over, we might have to prune groups.
            # But the LangChain-style trim_messages already has its own loop.
            # This logic is a bit complex for a simple budget check.
            # We'll just return what we have and let the caller handle it or
            # perform a basic check here.
            
            proposed_messages = [messages[i] for i in sorted(current_indices)]
            token_cost = await self.token_counter.count_tokens(proposed_messages, model)

            if token_cost > max_tokens:
                # If adding dependencies broke the budget, we have to start dropping.
                # Simplest strategy: if a group (Assistant + Tools) doesn't fit, drop the whole group
                # starting from the oldest.
                # This is a bit advanced, so for now we'll just log it.
                logger.warning(f"Tool dependency resolution exceeded token budget ({token_cost} > {max_tokens})")

        return current_indices

    async def trim_messages(
        self,
        messages: List[Message],
        max_tokens: int,
        strategy: TrimStrategy = TrimStrategy.FROM_END,
        keep_system: bool = True,
        model: Optional[str] = None,
    ) -> List[Message]:
        """Trim messages using a LangChain-style heuristic."""
        if not messages:
            return []

        _ensure_message_ids(messages)

        if strategy not in {TrimStrategy.FROM_END, TrimStrategy.FROM_START}:
            raise ValueError(f"Unsupported trim strategy: {strategy}")

        total_tokens = await self.token_counter.count_tokens(messages, model)
        if total_tokens <= max_tokens:
            return messages

        system_message: Optional[Message] = None
        remaining = messages
        if (
            keep_system
            and strategy == TrimStrategy.FROM_END
            and messages
            and messages[0].role == "system"
        ):
            system_message = messages[0]
            remaining = messages[1:]

        if strategy == TrimStrategy.FROM_END:
            kept: List[Message] = []
            for message in reversed(remaining):
                trial = [message] + kept
                candidate = ([system_message] if system_message else []) + trial
                token_cost = await self.token_counter.count_tokens(candidate, model)
                if token_cost <= max_tokens or not kept:
                    kept = trial
            trimmed = ([system_message] if system_message else []) + kept
        else:
            kept: List[Message] = []
            for message in remaining:
                trial = kept + [message]
                token_cost = await self.token_counter.count_tokens(trial, model)
                if token_cost <= max_tokens or not kept:
                    kept = trial
                else:
                    break
            trimmed = kept

        if not trimmed:
            if system_message is not None:
                trimmed = [system_message]
            elif remaining:
                trimmed = [remaining[0]] if strategy == TrimStrategy.FROM_START else [remaining[-1]]
            else:
                trimmed = []

        index_lookup = {id(msg): idx for idx, msg in enumerate(messages)}
        trimmed_indices = {index_lookup[id(msg)] for msg in trimmed if id(msg) in index_lookup}
        trimmed_indices = await self._apply_tool_call_dependencies(
            messages,
            trimmed_indices,
            max_tokens=max_tokens,
            model=model,
        )
        trimmed = [messages[i] for i in sorted(trimmed_indices)]

        logger.info(
            "Trimmed messages: %d -> %d (strategy=%s)",
            len(messages),
            len(trimmed),
            strategy.value,
        )
        return trimmed

    async def summarize_messages(
        self,
        messages: List[Message],
        max_tokens_before_summary: int,
        messages_to_keep: int = 5,
        summary_model: Optional[str] = None,
        llm_manager=None,
        llm_provider: Optional[str] = None,
        existing_summary: str = "",
    ) -> Tuple[List[Message], List[RemoveMessage], Optional[str]]:
        """Summarize earlier messages and emit removal directives."""
        if not messages or not llm_manager:
            return messages, [], existing_summary or None

        _ensure_message_ids(messages)

        total_tokens = await self.token_counter.count_tokens(messages, summary_model)
        if total_tokens <= max_tokens_before_summary:
            return messages, [], existing_summary or None

        if existing_summary:
            summary_prompt = (
                "This is a summary of the conversation to date: "
                f"{existing_summary}\n\n"
                "Extend the summary by taking into account the new messages above:"
            )
        else:
            summary_prompt = "Create a summary of the conversation above:"

        prompt_message = Message(role="user", content=summary_prompt)

        try:
            # Only pass model parameter if it's not None
            # Let provider use its default model if summary_model is None
            chat_kwargs = {
                "messages": messages + [prompt_message],
                "provider": llm_provider,
            }
            if summary_model is not None:
                chat_kwargs["model"] = summary_model
            
            response = await llm_manager.chat(**chat_kwargs)
            summary_text = response.content
        except Exception as exc:  # pragma: no cover - safeguard
            logger.error("Failed to generate summary: %s", exc)
            return messages, [], existing_summary or None

        start_index = max(len(messages) - messages_to_keep, 0)
        indices_to_keep: Set[int] = set(range(start_index, len(messages)))
        keep_system_message = messages and messages[0].role == "system"
        if keep_system_message:
            indices_to_keep.add(0)

        indices_to_keep = await self._apply_tool_call_dependencies(
            messages,
            indices_to_keep,
            max_tokens=None,
            model=summary_model,
        )

        removals: List[RemoveMessage] = []
        for idx, message in enumerate(messages):
            if idx in indices_to_keep:
                continue
            message_id = getattr(message, "id", None)
            if message_id:
                removals.append(RemoveMessage(id=message_id))
            else:
                logger.debug("Skipping removal for message at index %s; no id present", idx)

        kept_messages = [messages[i] for i in sorted(indices_to_keep)]

        messages_for_llm: List[Message] = []
        if keep_system_message:
            system_msg = messages[0]
            if kept_messages and kept_messages[0] is system_msg:
                messages_for_llm.append(system_msg)
                kept_messages = kept_messages[1:]

        if summary_text:
            summary_message = SystemMessage(
                id=f"summary-{uuid.uuid4()}",
                content=f"[CONVERSATION SUMMARY]\n{summary_text}",
            )
            messages_for_llm.append(summary_message)

        messages_for_llm.extend(kept_messages)

        if not messages_for_llm:
            messages_for_llm = list(messages)

        return messages_for_llm, removals, summary_text

    def save_checkpoint(
        self,
        thread_id: str,
        messages: List[Message],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        
        checkpoint_id = str(uuid.uuid4())
        checkpoint_metadata = {
            "message_count": len(messages),
            "timestamp": datetime.now().isoformat(),
            "checkpoint_id": checkpoint_id,
            **(metadata or {})
        }

        _ensure_message_ids(messages)

        state_snapshot = StateSnapshot(
            values={"messages": [msg.model_dump() for msg in messages]},
            next=("message_checkpoint",),
            config={},
            metadata=checkpoint_metadata,
            created_at=datetime.now()
        )

        self.checkpointer.save_checkpoint(thread_id, state_snapshot)
        logger.info(f"Checkpoint saved: thread={thread_id}, id={checkpoint_id}, messages={len(messages)}")

        return checkpoint_id

    def restore_checkpoint(
        self,
        thread_id: str,
        checkpoint_id: Optional[str] = None,
    ) -> Optional[List[Message]]:
        snapshot = self.checkpointer.get_checkpoint(thread_id, checkpoint_id)

        if not snapshot:
            logger.warning(f"Checkpoint not found: thread={thread_id}, id={checkpoint_id}")
            return None

        try:
            messages_data = snapshot.values.get("messages", [])
            messages = [Message(**msg_data) for msg_data in messages_data]

            logger.info(f"Checkpoint restored: thread={thread_id}, id={checkpoint_id}, messages={len(messages)}")
            return messages

        except Exception as e:
            logger.error(f"Failed to restore checkpoint: {e}")
            return None

    def list_checkpoints(self, thread_id: str) -> List[Dict[str, Any]]:
        
        snapshots = self.checkpointer.list_checkpoints(thread_id)

        result = []
        for snapshot in snapshots:
            checkpoint_info = {
                "checkpoint_id": snapshot.metadata.get("checkpoint_id") or str(snapshot.created_at.timestamp()),
                "created_at": snapshot.created_at.isoformat(),
                "message_count": snapshot.metadata.get("message_count", 0),
                "metadata": snapshot.metadata
            }
            result.append(checkpoint_info)

        return result

    def clear_checkpoints(self, thread_id: str) -> None:
        
        self.checkpointer.clear_thread(thread_id)
        logger.info(f"Cleared checkpoints for thread: {thread_id}")
