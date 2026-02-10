from spoon_ai.llm.providers.gemini_provider import GeminiProvider
from spoon_ai.schema import Message, ToolCall, Function


def test_convert_messages_for_tools_splits_text_and_function_call_turns():
    provider = GeminiProvider()

    messages = [
        Message(role="user", content="记住代号 蓝鲸-729"),
        Message(
            role="assistant",
            content="收到，我来记住。",
            tool_calls=[
                ToolCall(
                    id="call_1",
                    function=Function(name="memory", arguments='{"action":"remember","content":"蓝鲸-729"}'),
                )
            ],
        ),
        Message(role="tool", name="memory", tool_call_id="call_1", content="Remembered: 蓝鲸-729"),
        Message(role="assistant", content="已记住。"),
    ]

    _sys, gemini_messages = provider._convert_messages_for_tools(messages)

    # Expect model text turn split from model function_call turn
    roles_and_kinds = []
    for msg in gemini_messages:
        role = msg.role
        kinds = []
        for part in msg.parts:
            if getattr(part, "text", None):
                kinds.append("text")
            elif getattr(part, "function_call", None):
                kinds.append("function_call")
            elif getattr(part, "function_response", None):
                kinds.append("function_response")
        roles_and_kinds.append((role, kinds))

    assert roles_and_kinds == [
        ("user", ["text"]),
        ("model", ["text"]),
        ("model", ["function_call"]),
        ("user", ["function_response"]),
        ("model", ["text"]),
    ]
