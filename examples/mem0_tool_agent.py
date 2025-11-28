"""
Mem0 toolkit demo using SpoonReactAI.
This demo requires spoon-toolkit to be installed
"""

import asyncio
from typing import Any, Dict, List

from pydantic import Field

from spoon_ai.agents.spoon_react import SpoonReactAI
from spoon_ai.chat import ChatBot
from spoon_ai.tools.tool_manager import ToolManager
from spoon_ai.tools.base import ToolResult
from spoon_toolkits.memory import AddMemoryTool, SearchMemoryTool, GetAllMemoryTool


USER_ID = "defi_user_002"


class DeFiMemoryAgent(SpoonReactAI):
    """Brain from spoon-core + memory tools from spoon-toolkit."""

    mem0_config: Dict[str, Any] = Field(default_factory=dict)
    available_tools: ToolManager = Field(default_factory=lambda: ToolManager([]))

    def model_post_init(self, __context: Any = None) -> None:
        super().model_post_init(__context)
        # Rebuild tools with the injected mem0_config for this agent
        memory_tools = [
            AddMemoryTool(mem0_config=self.mem0_config),
            SearchMemoryTool(mem0_config=self.mem0_config),
            GetAllMemoryTool(mem0_config=self.mem0_config),
        ]
        self.available_tools = ToolManager(memory_tools)
        # Refresh prompts so SpoonReactAI lists the newly provided tools
        if hasattr(self, "_refresh_prompts"):
            self._refresh_prompts()


def build_agent(mem0_cfg: Dict[str, Any]) -> DeFiMemoryAgent:
    return DeFiMemoryAgent(
        llm=ChatBot(
            llm_provider="openrouter",
            base_url="https://openrouter.ai/api/v1",
            model_name="anthropic/claude-3.5-sonnet",
            enable_long_term_memory=False,  # memory comes from toolkit tools instead
        ),
        mem0_config=mem0_cfg,
        system_prompt=(
            "You are a DeFi investment advisor. Use the provided Mem0 tools to recall "
            "and update user preferences before answering."
        ),
    )


def print_memories(result: ToolResult, label: str) -> None:
    if not isinstance(result, ToolResult):
        print(f"[Mem0] {label}: error -> {result}")
        return
    memories: List[str] = result.output.get("memories", []) if result and result.output else []
    print(f"[Mem0] {label}:")
    for m in memories:
        print(f"  - {m}")


async def phase_capture(agent: DeFiMemoryAgent) -> None:
    print("\n=== Phase 1: Capture high-risk Solana preferences ===")
    await agent.available_tools.execute(
        name="add_memory",
        tool_input={
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "I am a high-risk degen trader. I exclusively trade meme coins on Solana "
                        "and dislike Ethereum gas fees."
                    ),
                }
            ]
        },
    )
    memories = await agent.available_tools.execute(
        name="search_memory",
        tool_input={"query": "Solana meme coins high risk"},
    )
    print_memories(memories, "After Phase 1 store")


async def phase_recall(mem0_cfg: Dict[str, Any]) -> None:
    print("\n=== Phase 2: Recall with a fresh agent instance ===")
    agent = build_agent(mem0_cfg)
    memories = await agent.available_tools.execute(
        name="search_memory",
        tool_input={"query": "trading strategy solana meme"},
    )
    print_memories(memories, "Retrieved for Phase 2")


async def phase_update(agent: DeFiMemoryAgent) -> None:
    print("\n=== Phase 3: Update preferences to safer Arbitrum yield ===")
    await agent.available_tools.execute(
        name="add_memory",
        tool_input={
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "I lost too much money. I want to pivot to safe stablecoin yield farming on Arbitrum now."
                    ),
                }
            ]
        },
    )
    memories = await agent.available_tools.execute(
        name="search_memory",
        tool_input={"query": "stablecoin yield chain choice"},
    )
    print_memories(memories, "Retrieved after update (Phase 3)")


async def main() -> None:
    mem0_cfg = {
        "user_id": USER_ID,
        "metadata": {"project": "defi-investment-advisor"},
        "async_mode": False,  # synchronous writes so the next search sees new data
    }
    agent = build_agent(mem0_cfg)
    await phase_capture(agent)
    await phase_recall(mem0_cfg)
    await phase_update(agent)
    


if __name__ == "__main__":
    asyncio.run(main())
