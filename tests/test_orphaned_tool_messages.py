"""Tests for orphaned tool message handling across ALL providers.

Validates that the shared ``drop_orphaned_tool_messages`` utility and each
provider's message conversion pipeline correctly drops tool-role messages
that have no valid pairing with a preceding assistant tool_calls entry.
"""

from __future__ import annotations

import pytest

from spoon_ai.schema import Message, ToolCall, Function
from spoon_ai.llm.message_utils import drop_orphaned_tool_messages

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _assistant_with_tool_calls(*call_ids: str) -> Message:
    """Build an assistant message that requested *call_ids*."""
    return Message(
        role="assistant",
        content="",
        tool_calls=[
            ToolCall(id=cid, type="function", function=Function(name=f"fn_{i}", arguments="{}"))
            for i, cid in enumerate(call_ids)
        ],
    )


def _tool_result(call_id: str, content: str = "result") -> Message:
    return Message(role="tool", tool_call_id=call_id, content=content)


def _tool_result_no_id(content: str = "stale") -> Message:
    return Message(role="tool", content=content)


def _user(content: str = "Hello") -> Message:
    return Message(role="user", content=content)


def _system(content: str = "You are helpful.") -> Message:
    return Message(role="system", content=content)


def _assistant(content: str = "Sure.") -> Message:
    return Message(role="assistant", content=content)


# ===================================================================
# 1. Shared utility – drop_orphaned_tool_messages
# ===================================================================

class TestDropOrphanedToolMessages:
    """Unit tests for the shared utility function."""

    def test_empty_list(self):
        assert drop_orphaned_tool_messages([]) == []

    def test_no_tool_messages(self):
        msgs = [_system(), _user(), _assistant()]
        assert drop_orphaned_tool_messages(msgs) == msgs

    def test_keeps_valid_tool_messages(self):
        msgs = [
            _user(),
            _assistant_with_tool_calls("call_1", "call_2"),
            _tool_result("call_1"),
            _tool_result("call_2"),
            _assistant("Done."),
        ]
        result = drop_orphaned_tool_messages(msgs)
        assert len(result) == 5
        tool_msgs = [m for m in result if m.role == "tool"]
        assert len(tool_msgs) == 2

    def test_drops_tool_without_tool_call_id(self):
        msgs = [
            _user(),
            _tool_result_no_id("stale"),
            _assistant(),
        ]
        result = drop_orphaned_tool_messages(msgs)
        assert len(result) == 2
        assert all(m.role != "tool" for m in result)

    def test_drops_tool_without_preceding_assistant(self):
        msgs = [
            _system(),
            _tool_result("call_ghost"),
            _user(),
        ]
        result = drop_orphaned_tool_messages(msgs)
        assert len(result) == 2
        assert all(m.role != "tool" for m in result)

    def test_drops_tool_with_unmatched_id(self):
        msgs = [
            _user(),
            _assistant_with_tool_calls("call_A"),
            _tool_result("call_A"),
            _tool_result("call_NONEXISTENT"),
        ]
        result = drop_orphaned_tool_messages(msgs)
        tool_msgs = [m for m in result if m.role == "tool"]
        assert len(tool_msgs) == 1
        assert tool_msgs[0].tool_call_id == "call_A"

    def test_multiple_orphans_all_dropped(self):
        msgs = [
            _system(),
            _tool_result_no_id("orphan1"),
            _tool_result("gone_call"),
            _user(),
            _assistant(),
        ]
        result = drop_orphaned_tool_messages(msgs)
        assert all(m.role != "tool" for m in result)
        assert len(result) == 3

    def test_multi_turn_tool_calls_preserved(self):
        """Two separate assistant+tool turns should both be preserved."""
        msgs = [
            _user("Turn 1"),
            _assistant_with_tool_calls("c1"),
            _tool_result("c1", "r1"),
            _assistant("Middle"),
            _user("Turn 2"),
            _assistant_with_tool_calls("c2", "c3"),
            _tool_result("c2", "r2"),
            _tool_result("c3", "r3"),
            _assistant("Final"),
        ]
        result = drop_orphaned_tool_messages(msgs)
        assert len(result) == len(msgs)

    def test_tool_from_earlier_turn_still_valid(self):
        """A tool_call_id from an earlier assistant turn should still be valid."""
        msgs = [
            _user(),
            _assistant_with_tool_calls("early_call"),
            _user("injected user msg"),
            _tool_result("early_call"),
        ]
        result = drop_orphaned_tool_messages(msgs)
        tool_msgs = [m for m in result if m.role == "tool"]
        assert len(tool_msgs) == 1


# ===================================================================
# 2. OpenAI-compatible provider (dict-level validation)
# ===================================================================

class TestOpenAICompatibleProviderOrphanedTools:
    """Validates the existing dict-level validation in OpenAICompatibleProvider."""

    @staticmethod
    def _make_provider():
        from spoon_ai.llm.providers.openai_compatible_provider import OpenAICompatibleProvider
        return OpenAICompatibleProvider()

    def test_drops_tool_message_without_tool_call_id(self):
        provider = self._make_provider()
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "tool", "content": "stale result"},
            {"role": "assistant", "content": "Hi there"},
        ]
        fixed = provider._validate_and_fix_message_sequence(messages)
        assert "tool" not in [m["role"] for m in fixed]

    def test_drops_tool_without_preceding_assistant(self):
        provider = self._make_provider()
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "tool", "tool_call_id": "call_123", "content": "orphaned"},
            {"role": "user", "content": "Hello"},
        ]
        fixed = provider._validate_and_fix_message_sequence(messages)
        assert "tool" not in [m["role"] for m in fixed]

    def test_drops_tool_with_unmatched_id(self):
        provider = self._make_provider()
        messages = [
            {"role": "user", "content": "Run"},
            {
                "role": "assistant", "content": "",
                "tool_calls": [{"id": "call_A", "type": "function",
                                "function": {"name": "shell", "arguments": '{}'}}],
            },
            {"role": "tool", "tool_call_id": "call_A", "content": "ok"},
            {"role": "tool", "tool_call_id": "call_MISSING", "content": "bad"},
        ]
        fixed = provider._validate_and_fix_message_sequence(messages)
        tool_msgs = [m for m in fixed if m["role"] == "tool"]
        assert len(tool_msgs) == 1
        assert tool_msgs[0]["tool_call_id"] == "call_A"

    def test_keeps_valid_tool_messages(self):
        provider = self._make_provider()
        messages = [
            {"role": "user", "content": "Run commands"},
            {
                "role": "assistant", "content": "",
                "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "a", "arguments": "{}"}},
                    {"id": "c2", "type": "function", "function": {"name": "b", "arguments": "{}"}},
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "r1"},
            {"role": "tool", "tool_call_id": "c2", "content": "r2"},
            {"role": "assistant", "content": "Done."},
        ]
        fixed = provider._validate_and_fix_message_sequence(messages)
        tool_msgs = [m for m in fixed if m["role"] == "tool"]
        assert len(tool_msgs) == 2


# ===================================================================
# 3. Anthropic provider – end-to-end through _convert_messages
# ===================================================================

class TestAnthropicProviderOrphanedTools:
    """Verifies AnthropicProvider._convert_messages drops orphans."""

    @staticmethod
    def _make_provider():
        from spoon_ai.llm.providers.anthropic_provider import AnthropicProvider
        p = AnthropicProvider()
        p.enable_prompt_cache = False
        return p

    def test_drops_orphan_tool(self):
        provider = self._make_provider()
        msgs = [
            _system(),
            _tool_result("orphan_id"),
            _user(),
        ]
        _, converted = provider._convert_messages(msgs)
        roles = [m["role"] for m in converted]
        assert "tool" not in roles
        content_types = []
        for m in converted:
            if isinstance(m.get("content"), list):
                for block in m["content"]:
                    content_types.append(block.get("type"))
        assert "tool_result" not in content_types

    def test_keeps_valid_tool(self):
        provider = self._make_provider()
        msgs = [
            _user("Do stuff"),
            _assistant_with_tool_calls("call_ok"),
            _tool_result("call_ok", "great"),
        ]
        _, converted = provider._convert_messages(msgs)
        tool_result_blocks = []
        for m in converted:
            if isinstance(m.get("content"), list):
                for block in m["content"]:
                    if block.get("type") == "tool_result":
                        tool_result_blocks.append(block)
        assert len(tool_result_blocks) == 1
        assert tool_result_blocks[0]["tool_use_id"] == "call_ok"


# ===================================================================
# 4. Gemini provider – _convert_messages_for_tools
# ===================================================================

class TestGeminiProviderOrphanedTools:
    """Verifies GeminiProvider._convert_messages_for_tools drops orphans."""

    @staticmethod
    def _make_provider():
        pytest.importorskip("google.genai")
        from spoon_ai.llm.providers.gemini_provider import GeminiProvider
        return GeminiProvider()

    def test_drops_orphan_tool(self):
        provider = self._make_provider()
        msgs = [
            _system(),
            _tool_result("orphan_id"),
            _user(),
        ]
        _, converted = provider._convert_messages_for_tools(msgs)
        for content in converted:
            for part in content.parts:
                assert not hasattr(part, "function_response") or part.function_response is None

    def test_keeps_valid_tool(self):
        provider = self._make_provider()
        msgs = [
            _user("Do stuff"),
            _assistant_with_tool_calls("call_ok"),
            _tool_result("call_ok", "result"),
        ]
        _, converted = provider._convert_messages_for_tools(msgs)
        fn_response_parts = []
        for content in converted:
            for part in content.parts:
                if hasattr(part, "function_response") and part.function_response is not None:
                    fn_response_parts.append(part)
        assert len(fn_response_parts) == 1


# ===================================================================
# 5. Ollama provider – _convert_messages
# ===================================================================

class TestOllamaProviderOrphanedTools:
    """Verifies OllamaProvider._convert_messages drops orphans."""

    @staticmethod
    def _make_provider():
        from spoon_ai.llm.providers.ollama_provider import OllamaProvider
        return OllamaProvider()

    def test_drops_orphan_tool(self):
        provider = self._make_provider()
        msgs = [
            _system(),
            _tool_result("orphan_id"),
            _user(),
        ]
        converted = provider._convert_messages(msgs)
        roles = [m["role"] for m in converted]
        assert "tool" not in roles

    def test_keeps_valid_tool(self):
        provider = self._make_provider()
        msgs = [
            _user("Do stuff"),
            _assistant_with_tool_calls("call_ok"),
            _tool_result("call_ok", "result"),
        ]
        converted = provider._convert_messages(msgs)
        tool_msgs = [m for m in converted if m["role"] == "tool"]
        assert len(tool_msgs) == 1
