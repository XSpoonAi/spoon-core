"""Provider-level regressions for tool-call message ordering."""

from spoon_ai.llm.providers.gemini_provider import GeminiProvider
from spoon_ai.llm.providers.openai_compatible_provider import OpenAICompatibleProvider
from spoon_ai.schema import Function, Message, ToolCall


def _tool_call(tool_call_id: str, name: str, arguments: str) -> ToolCall:
    return ToolCall(
        id=tool_call_id,
        type="function",
        function=Function(name=name, arguments=arguments),
    )


def test_openai_validator_moves_tool_result_immediately_after_assistant_tool_call():
    provider = OpenAICompatibleProvider()

    messages = [
        {"role": "user", "content": "Inspect the workspace."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_shell",
                    "type": "function",
                    "function": {
                        "name": "shell",
                        "arguments": "{\"command\":\"pwd\"}",
                    },
                }
            ],
        },
        {"role": "user", "content": "Focus on the current request."},
        {
            "role": "tool",
            "tool_call_id": "call_shell",
            "name": "shell",
            "content": "/workspace",
        },
    ]

    fixed = provider._validate_and_fix_message_sequence(messages)

    assert [message["role"] for message in fixed] == ["user", "assistant", "tool", "user"]
    assert fixed[2]["tool_call_id"] == "call_shell"
    assert fixed[3]["content"] == "Focus on the current request."


def test_gemini_converter_merges_adjacent_user_turns_and_groups_tool_results():
    provider = GeminiProvider()

    messages = [
        Message(role="user", content="Inspect the workspace."),
        Message(role="user", content="Focus on the active request."),
        Message(
            role="assistant",
            content="",
            tool_calls=[
                _tool_call("call_read", "read_file", '{"path":"README.md"}'),
                _tool_call("call_shell", "shell", '{"command":"pwd"}'),
            ],
        ),
        Message(role="tool", content="/workspace", tool_call_id="call_shell", name="shell"),
        Message(role="tool", content="[file output]", tool_call_id="call_read", name="read_file"),
    ]

    system_content, gemini_messages = provider._convert_messages_for_tools(messages)

    assert system_content == ""
    assert [message.role for message in gemini_messages] == ["user", "model", "user"]

    first_user_parts = gemini_messages[0].parts
    assert [part.text for part in first_user_parts] == [
        "Inspect the workspace.",
        "Focus on the active request.",
    ]

    model_parts = gemini_messages[1].parts
    assert [part.function_call.name for part in model_parts] == ["read_file", "shell"]

    tool_parts = gemini_messages[2].parts
    assert [part.function_response.name for part in tool_parts] == ["read_file", "shell"]
