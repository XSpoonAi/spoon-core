"""Shared utilities for sanitising Message sequences before provider conversion.

Every provider that sends tool-role messages to an API should call
``drop_orphaned_tool_messages`` **before** provider-specific conversion so
that malformed / orphaned tool messages never reach the remote API.
"""

from __future__ import annotations

import logging
from typing import List

from spoon_ai.schema import Message

logger = logging.getLogger(__name__)


def drop_orphaned_tool_messages(messages: List[Message]) -> List[Message]:
    """Return *messages* with orphaned tool messages removed.

    A tool message is considered **orphaned** (and dropped) when any of the
    following is true:

    1. It has no ``tool_call_id`` at all.
    2. There is no preceding assistant message that contains ``tool_calls``.
    3. Its ``tool_call_id`` does not match any ``tool_calls[].id`` in the
       nearest preceding assistant message that carries tool calls.

    The function preserves the original ordering of all non-orphaned messages.
    """
    if not messages:
        return messages

    all_preceding_tool_call_ids: set[str] = set()

    cleaned: List[Message] = []
    for msg in messages:
        if msg.role == "assistant" and msg.tool_calls:
            for tc in msg.tool_calls:
                if tc.id:
                    all_preceding_tool_call_ids.add(tc.id)
            cleaned.append(msg)
            continue

        if msg.role != "tool":
            cleaned.append(msg)
            continue

        tool_call_id = getattr(msg, "tool_call_id", None)
        if not tool_call_id:
            logger.warning("Dropping tool message with missing tool_call_id")
            continue

        if not all_preceding_tool_call_ids:
            logger.warning(
                "Dropping orphaned tool message (tool_call_id=%s): "
                "no preceding assistant message with tool_calls",
                tool_call_id,
            )
            continue

        if tool_call_id not in all_preceding_tool_call_ids:
            logger.warning(
                "Dropping tool message with unmatched tool_call_id=%s "
                "(known ids: %s)",
                tool_call_id,
                all_preceding_tool_call_ids,
            )
            continue

        cleaned.append(msg)

    dropped = len(messages) - len(cleaned)
    if dropped:
        logger.info("Dropped %d orphaned tool message(s)", dropped)

    return cleaned
