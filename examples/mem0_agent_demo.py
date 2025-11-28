"""Mem0 long-term memory demo: The Intelligent Web3 Portfolio Assistant"""

import asyncio
import os
from typing import List

from spoon_ai.chat import ChatBot

USER_ID = "crypto_whale_001"
SYSTEM_PROMPT = (
    "You are the Intelligent Web3 Portfolio Assistant. "
    "Remember user risk appetite, preferred chains, and asset types. "
    "Recommend actionable strategies without re-asking for already stored preferences."
)


def new_llm(mem0_config: dict) -> ChatBot:
    """Create a new ChatBot configured for long-term memory with Mem0."""
    return ChatBot(
        llm_provider="openrouter",
        model_name="openai/gpt-5.1", 
        enable_long_term_memory=True,
        mem0_config=mem0_config,
    )


def print_memories(memories: List[str], label: str) -> None:
    print(f"[Mem0] {label}:")
    for m in memories:
        print(f"  - {m}")


async def main() -> None:
    mem0_config = {
        "user_id": USER_ID,
        "metadata": {"project": "web3-portfolio-assistant"},
        "async_mode": False,  # synchronous writes so retrieval in the next turn works immediately
    }

    # Step 1: Introduction / preference capture
    print(" Session 1: Capturing preferences")
    llm = new_llm(mem0_config)
    first_reply = await llm.ask(
        [{"role": "user", "content": (
        "I am a high-risk degen trader. I exclusively trade meme coins on the Solana blockchain. "
        "I hate Ethereum gas fees.")}],
        system_msg=SYSTEM_PROMPT,
    )
    print("First reply:", first_reply)

    memories = llm.mem0_client.search_memory("Solana meme coins high risk")
    print_memories(memories, "After Session 1")

    # Step 2: Recall after re-initialization
    print(" Session 2: Recall with a brand new agent instance")
    llm_reloaded = new_llm(mem0_config)
    second_reply = await llm_reloaded.ask(
        [{"role": "user", "content": "Recommend a trading strategy for me today."}],
        system_msg=SYSTEM_PROMPT,
    )
    print("Second reply:", second_reply)

    memories = llm_reloaded.mem0_client.search_memory("trading strategy solana meme")
    print_memories(memories, "Retrieved for Session 2")


    # Step 3: Update preferences and verify recency/relevance
    print(" Session 3: Updating preferences to safer Arbitrum yield")
    third_reply = await llm_reloaded.ask(
        [{"role": "user", "content": "I lost too much money. I want to pivot to safe stablecoin yield farming on Arbitrum now."}],
        system_msg=SYSTEM_PROMPT,
    )
    print("Third reply:", third_reply)
    fourth_reply = await llm_reloaded.ask(
        [{"role": "user", "content": "What chain should I use?"}],
        system_msg=SYSTEM_PROMPT,
    )
    print("Fourth reply:", fourth_reply)
    memories = llm_reloaded.mem0_client.search_memory("stablecoin yield chain choice")
    print_memories(memories, "Retrieved after update (Session 3)")

if __name__ == "__main__":
    asyncio.run(main())
