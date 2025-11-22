"""spoon_ai short-term memory demos."""
import asyncio
from spoon_ai.graph.types import StateSnapshot
import uuid
from typing import Any, Dict, List, Optional, Tuple

from spoon_ai.chat import ChatBot
from spoon_ai.schema import Message
from spoon_ai.memory.short_term_manager import (
    ShortTermMemoryManager,
    TrimStrategy,
)
from spoon_ai.graph.engine import StateGraph, SummarizationNode, END
from spoon_ai.graph.checkpointer import InMemoryCheckpointer
from spoon_ai.graph.reducers import add_messages


class ShortTermMemoryDemoAgent:
    """Utility agent that seeds a reusable conversation transcript."""

    def __init__(self, system_prompt: str, prompts: List[str]) -> None:
        self.system_prompt = system_prompt
        self.prompts = prompts
        self.chatbot: Optional[ChatBot] = None
        self._history: Optional[List[Message]] = None

    async def get_history(self) -> List[Message]:
        if self._history is None:
            await self._build_history()
        history = self._history or []
        return [msg.model_copy(deep=True) for msg in history]

    async def _build_history(self) -> None:
        # Configure a chatbot with short-term memory enabled so the demo reflects reality.
        self.chatbot = ChatBot(
            enable_short_term_memory=True,
            llm_provider="openrouter",
            model_name="anthropic/claude-3.5-sonnet",
        )

        history: List[Message] = []
        history.append(Message(id=str(uuid.uuid4()), role="system", content=self.system_prompt))

        for prompt in self.prompts:
            user_msg = Message(id=str(uuid.uuid4()), role="user", content=prompt)
            history.append(user_msg)
            response_text = await self.chatbot.ask(list[dict | Message](history))
            assistant_msg = Message(
                id=str(uuid.uuid4()),
                role="assistant",
                content=response_text,
            )
            history.append(assistant_msg)

        self._history = history


DEMO_AGENT = ShortTermMemoryDemoAgent(
    system_prompt="You are a patient blockchain mentor who explains decentralized technologies in plain language.",
    prompts=[
        "Hello, what is a blockchain and why does it matter?",
        "Can you describe how a smart contract runs on a public chain in one sentence?",
        "How does proof-of-stake secure a blockchain network?",
        "Mention one major risk when using decentralized finance protocols.",
    ],
)


async def example_trim_messages() -> None:
    print("Example 1: Trim Messages")
    manager = ShortTermMemoryManager()
    messages = await DEMO_AGENT.get_history()
    print("Original messages")
    for idx, msg in enumerate(messages):
        print(f"  {idx}: id={getattr(msg, 'id', None)} role={msg.role} -> {msg.content}")

    total_tokens = await manager.token_counter.count_tokens(messages)
    print(f"Total tokens: {total_tokens}")

    max_tokens = 500
    print(f"Max tokens allowed: {max_tokens}")

    trimmed = await manager.trim_messages(
        messages=messages,
        max_tokens=max_tokens,
        strategy=TrimStrategy.FROM_END,
        keep_system=True,
    )
    print("Messages after trim")
    for idx, msg in enumerate(trimmed):
        print(f"  {idx}: id={getattr(msg, 'id', None)} role={msg.role} -> {msg.content}")


async def example_remove_messages() -> None:
    print("Example 2: RemoveMessage Directives")
    history = await DEMO_AGENT.get_history()
    chatbot = ChatBot(enable_short_term_memory=True)
    thread_id = "remove-demo"
    cp_id = chatbot.save_checkpoint(thread_id, history)
    restored = chatbot.restore_checkpoint(thread_id, cp_id) or []
    assistant_ids = [msg.id for msg in restored if msg.role == "assistant" and getattr(msg, "id", None)]
    removals = [
        chatbot.remove_message(assistant_ids[0]),
        chatbot.remove_message(assistant_ids[-1]),
        # chatbot.remove_all_messages(),
    ]
    print("Removal directives:")
    for rm in removals:
        print(f"  -> {rm.type} {rm.target_id}")

    updated_history = add_messages(restored, removals)
    remaining = [getattr(msg, "id", None) for msg in updated_history]
    print(f"Remaining message ids: {remaining}")
    chatbot.clear_checkpoints(thread_id)


async def example_summarise_messages() -> None:
    print("Example 3: Summarise Messages")
    history = await DEMO_AGENT.get_history()

    manager = DEMO_AGENT.chatbot.short_term_memory_manager
    llm_manager = DEMO_AGENT.chatbot.llm_manager
    summary_model = DEMO_AGENT.chatbot.model_name

    if not manager or not llm_manager:
        print("  Short-term memory manager or LLM manager missing.")
        return

    print("Scenario A: Prompt-requested summary")
    summary_prompt = Message(
        id=str(uuid.uuid4()),
        role="user",
        content="Please summarize the recent conversation we just had about blockchain topics.",
    )
    history_with_request = history + [summary_prompt]
    history_with_requesty_ids = [msg.id for msg in history_with_request if getattr(msg, "id", None)]
    print(f"History with request ids: {history_with_requesty_ids}")
    tokens_after_request = await manager.token_counter.count_tokens(history_with_request, summary_model)
    print(f"  Tokens after adding summary request: {tokens_after_request}")

    prompt_threshold = max(1, tokens_after_request // 2)
    print(f"  Max tokens before summary (prompt scenario, forced): {prompt_threshold}")

    _, prompt_removals, prompt_summary = await manager.summarize_messages(
        messages=history_with_request,
        max_tokens_before_summary=prompt_threshold,
        messages_to_keep=2,
        summary_model=summary_model,
        llm_manager=llm_manager,
        llm_provider=DEMO_AGENT.chatbot.llm_provider,
        existing_summary="",
    )

    if prompt_summary:
        print(f"  Prompt summary text: {prompt_summary}...")
    else:
        print("  Prompt summary unavailable.")

    print("  Prompt scenario removals:")
    for rm in prompt_removals:
        print(f"    -> remove {rm.target_id}")
    prompt_history = add_messages(history_with_request, prompt_removals)
    if prompt_summary:
        prompt_history.append(
            Message(
                id=str(uuid.uuid4()),
                role="assistant",
                content=f"Here is the summary so far:\n{prompt_summary}",
            )
        )
    prompt_history_ids = [msg.id for msg in prompt_history if getattr(msg, "id", None)]
    print(f"  Prompt scenario history ids: {prompt_history_ids}")

    print("Scenario B: Token-limit summary")
    baseline_tokens = await manager.token_counter.count_tokens(history, summary_model)
    limit_threshold = max(1, baseline_tokens - 100)
    print(f"  Max tokens before summary (limit scenario, constrained): {limit_threshold}")

    _, limit_removals, limit_summary = await manager.summarize_messages(
        messages=history,
        max_tokens_before_summary=limit_threshold,
        messages_to_keep=2,
        summary_model=summary_model,
        llm_manager=llm_manager,
        llm_provider=DEMO_AGENT.chatbot.llm_provider,
        existing_summary="",
    )

    if limit_summary:
        print(f"  Limit summary text: {limit_summary}...")
    else:
        print("  Limit summary unavailable.")

    print("  Limit scenario removals:")
    for rm in limit_removals:
        print(f"    -> remove {rm.target_id}")
    limit_history = add_messages(history, limit_removals)
    if limit_summary:
        limit_history.append(
            Message(
                id=str(uuid.uuid4()),
                role="assistant",
                content=f"Here is the summary so far:\n{limit_summary}",
            )
        )
    limit_history_ids = [msg.id for msg in limit_history if getattr(msg, "id", None)]
    print(f"  Limit scenario history ids: {limit_history_ids}")


async def example_graph_summarization_node() -> Tuple[StateGraph, Dict[str, Dict[str, str]]]:
    print("Example 4: Summarise via StateGraph node")
    history = await DEMO_AGENT.get_history()
    chatbot = DEMO_AGENT.chatbot

    graph = StateGraph(dict, checkpointer=InMemoryCheckpointer())

    summarization_node = SummarizationNode(
        name="summarize_history",
        llm_manager=chatbot.llm_manager,
        max_tokens=500,
        messages_to_keep=2,
        summary_model=chatbot.model_name,
        summary_key="summary_context",
        output_messages_key="llm_input_messages",
        manager=chatbot.short_term_memory_manager,
    )

    async def call_llm_node(state: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        llm_messages = state.get("llm_input_messages") or state.get("messages", [])
        assistant_text = ""
        try:
            response = await chatbot.llm_manager.chat(
                messages=llm_messages,
                provider=chatbot.llm_provider,
                model=chatbot.model_name,
            )
            assistant_text = response.content or ""
        except Exception as exc:  # pragma: no cover - demo fallback
            assistant_text = f"[LLM call failed: {exc}]"
            print(f"  call_model warning: {assistant_text}")

        assistant_message = Message(
            id=str(uuid.uuid4()),
            role="assistant",
            content=assistant_text,
        )
        return {
            "messages": [assistant_message],
            "latest_reply": assistant_text,
        }

    graph.add_node("summarize_history", summarization_node)
    graph.add_node("call_model", call_llm_node)
    graph.set_entry_point("summarize_history")
    graph.add_edge("summarize_history", "call_model")
    graph.add_edge("call_model", END)

    compiled = graph.compile()
    run_config: Dict[str, Dict[str, str]] = {"configurable": {"thread_id": "memory_demo_thread"}}
    result = await compiled.invoke(
        initial_state={
            "messages": history,
            "summary_context": {},
        },
        config=run_config,
    )

    latest_reply = result.get("latest_reply")
    summary_context = result.get("summary_context", {})
    llm_input = result.get("llm_input_messages", [])

    print(f"  Summarization node produced {len(llm_input)} messages for the LLM.")
    print(f"  Latest assistant reply: {latest_reply!r}")
    if summary_context:
        summary_text = summary_context.get("summary", "")
        print(f"  Stored summary: {summary_text}")
    else:
        print("  No summary stored in context.")
    print(f"  Thread id recorded in checkpoints: {run_config['configurable']['thread_id']}")

    return graph, run_config


async def example_view_graph_state(graph: StateGraph, config: Dict[str, Dict[str, str]]) -> None:
    print("Example 5: Inspect Graph State & History")

    snapshot = graph.get_state(config)
    if snapshot:
        checkpoint_id = snapshot.metadata.get("checkpoint_id")
        message_count = len(snapshot.values.get("messages", []))
        print(f"  Latest checkpoint: {checkpoint_id} (messages={message_count})")
    else:
        print("  No checkpoint found for thread.")

    history_snapshots = list[StateSnapshot](graph.get_state_history(config))
    print(f"  Total checkpoints for thread: {len(history_snapshots)}")
    for snap in history_snapshots:
        cid = snap.metadata.get("checkpoint_id")
        created = snap.created_at.isoformat()
        next_node = snap.next[0] if snap.next else END
        print(f"    -> id={cid} created={created} next={next_node}")

    checkpointer_tuple = graph.checkpointer.get_checkpoint_tuple(config)
    if checkpointer_tuple:
        print("  Checkpointer tuple:")
        print(f"    config={checkpointer_tuple.config}")
        print(f"    checkpoint={checkpointer_tuple.checkpoint}")
        print(f"    metadata={checkpointer_tuple.metadata}")
        print(f"    parent_config={checkpointer_tuple.parent_config}")
        print(f"    pending_writes={checkpointer_tuple.pending_writes}")
    else:
        print("  Checkpointer tuple not available.")

    tuple_history = list(graph.checkpointer.iter_checkpoint_history(config))
    print(f"  Checkpointer history count: {len(tuple_history)}")
    for entry in tuple_history:
        print("    -> tuple entry:")
        print(f"       config={entry.config}")
        print(f"       checkpoint={entry.checkpoint}")
        print(f"       metadata={entry.metadata}")
        print(f"       parent_config={entry.parent_config}")


async def example_checkpoint_management() -> None:
    print("Example 6: Checkpoint Management")

    chatbot = ChatBot(enable_short_term_memory=True)

    thread_id = "demo-thread"

    messages_v1 = [
        Message(id="m1", role="user", content="Hello"),
        Message(id="m2", role="assistant", content="Hi there!"),
    ]
    cp1 = chatbot.save_checkpoint(
        thread_id, messages_v1, metadata={"stage": "initial"}
    )
    print(f"Saved checkpoint: {cp1}")

    messages_v2 = messages_v1 + [
        Message(id="m3", role="user", content="How are you?"),
        Message(id="m4", role="assistant", content="Doing great!"),
    ]
    cp2 = chatbot.save_checkpoint(
        thread_id, messages_v2, metadata={"stage": "follow_up"}
    )
    print(f"Saved checkpoint: {cp2}")

    print("Checkpoint history:")
    for entry in chatbot.list_checkpoints(thread_id):
        print(f"  -> id={entry['checkpoint_id']} created={entry['created_at']} count={entry['message_count']}")

    # 恢复到指定的检查点（第一个检查点）
    restored = chatbot.restore_checkpoint(thread_id, cp1) or []
    print("Messages restored from first checkpoint (specified by checkpoint_id):")
    for idx, msg in enumerate(restored):
        print(f"  {idx}: id={getattr(msg, 'id', None)} role={msg.role} -> {msg.content}")

    # 恢复到指定的检查点（第二个检查点）
    restored2 = chatbot.restore_checkpoint(thread_id, cp2) or []
    print("\nMessages restored from second checkpoint (specified by checkpoint_id):")
    for idx, msg in enumerate(restored2):
        print(f"  {idx}: id={getattr(msg, 'id', None)} role={msg.role} -> {msg.content}")

    # 恢复到最新的检查点（不指定 checkpoint_id，恢复最新的）
    restored_latest = chatbot.restore_checkpoint(thread_id) or []
    print("\nMessages restored from latest checkpoint (no checkpoint_id specified):")
    for idx, msg in enumerate(restored_latest):
        print(f"  {idx}: id={getattr(msg, 'id', None)} role={msg.role} -> {msg.content}")

    chatbot.clear_checkpoints(thread_id)
    print("\nAll checkpoints cleared.")


async def main() -> None:
    # await example_trim_messages()
    # await example_remove_messages()
    # await example_summarise_messages()
    # graph, graph_config = await example_graph_summarization_node()
    # await example_view_graph_state(graph, graph_config)
    await example_checkpoint_management()


if __name__ == "__main__":
    asyncio.run(main())
