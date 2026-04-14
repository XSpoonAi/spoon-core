"""Tests for orphaned tool message handling in _validate_and_fix_message_sequence.

Covers the fix where tool-role messages without matching tool_calls were
previously kept (causing OpenAI 400 errors) and are now properly dropped.
"""

from spoon_ai.llm.providers.openai_compatible_provider import OpenAICompatibleProvider


def _make_provider() -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider()


def test_drops_tool_message_without_tool_call_id():
    """Tool message with no tool_call_id should be dropped entirely."""
    provider = _make_provider()
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "tool", "content": "stale result"},  # no tool_call_id
        {"role": "assistant", "content": "Hi there"},
    ]
    fixed = provider._validate_and_fix_message_sequence(messages)

    roles = [m["role"] for m in fixed]
    assert "tool" not in roles
    assert roles == ["user", "assistant"]


def test_drops_tool_message_without_preceding_assistant_tool_calls():
    """Tool message at start of conversation (no assistant with tool_calls) should be dropped."""
    provider = _make_provider()
    messages = [
        {"role": "system", "content": "You are helpful."},
        {
            "role": "tool",
            "tool_call_id": "call_123",
            "content": "orphaned result",
        },
        {"role": "user", "content": "Hello"},
    ]
    fixed = provider._validate_and_fix_message_sequence(messages)

    roles = [m["role"] for m in fixed]
    assert "tool" not in roles
    assert roles == ["system", "user"]


def test_drops_tool_message_with_unmatched_tool_call_id():
    """Tool message whose tool_call_id doesn't match any preceding tool_calls is dropped."""
    provider = _make_provider()
    messages = [
        {"role": "user", "content": "Run a command"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_A",
                    "type": "function",
                    "function": {"name": "shell", "arguments": '{"cmd":"ls"}'},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_A",
            "content": "file1.txt",
        },
        {
            "role": "tool",
            "tool_call_id": "call_NONEXISTENT",
            "content": "this should be dropped",
        },
    ]
    fixed = provider._validate_and_fix_message_sequence(messages)

    tool_msgs = [m for m in fixed if m["role"] == "tool"]
    assert len(tool_msgs) == 1
    assert tool_msgs[0]["tool_call_id"] == "call_A"


def test_keeps_valid_tool_messages():
    """Properly paired tool messages should be kept."""
    provider = _make_provider()
    messages = [
        {"role": "user", "content": "Run commands"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "shell", "arguments": '{"cmd":"pwd"}'},
                },
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {"name": "shell", "arguments": '{"cmd":"ls"}'},
                },
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "/workspace"},
        {"role": "tool", "tool_call_id": "call_2", "content": "file.txt"},
        {"role": "assistant", "content": "Done."},
    ]
    fixed = provider._validate_and_fix_message_sequence(messages)

    tool_msgs = [m for m in fixed if m["role"] == "tool"]
    assert len(tool_msgs) == 2
    assert [m["tool_call_id"] for m in tool_msgs] == ["call_1", "call_2"]


def test_reorder_then_validate_handles_interleaved_user_messages():
    """Tool results separated from assistant by user messages should be reordered then validated."""
    provider = _make_provider()
    messages = [
        {"role": "user", "content": "Do something"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_X",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": '{"path":"a.txt"}'},
                }
            ],
        },
        {"role": "user", "content": "Focus on the task."},
        {"role": "tool", "tool_call_id": "call_X", "content": "contents of a.txt"},
    ]
    fixed = provider._validate_and_fix_message_sequence(messages)

    roles = [m["role"] for m in fixed]
    assert roles == ["user", "assistant", "tool", "user"]
    assert fixed[2]["tool_call_id"] == "call_X"


def test_multiple_orphaned_tool_messages_all_dropped():
    """Multiple orphaned tool messages from session history injection should all be dropped."""
    provider = _make_provider()
    messages = [
        {"role": "system", "content": "System prompt"},
        {"role": "tool", "content": "orphan 1"},
        {"role": "tool", "tool_call_id": "gone_call", "content": "orphan 2"},
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
    ]
    fixed = provider._validate_and_fix_message_sequence(messages)

    roles = [m["role"] for m in fixed]
    assert "tool" not in roles
    assert roles == ["system", "user", "assistant"]
