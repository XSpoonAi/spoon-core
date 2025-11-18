"""
Streaming ChatBot demo that prints token deltas in real time.
"""

import asyncio
from typing import Optional

from dotenv import load_dotenv

from spoon_ai.chat import ChatBot

load_dotenv(override=True)

# Simple conversation starters
PROMPTS = [
    "Give me a concise overview of the Neo blockchain in two paragraphs.",
    "List three practical agentic OS use cases for Web3 builders.",
    "Suggest a short social post promoting SpoonOS and its streaming agent demo.",
]


def create_chatbot() -> ChatBot:
    return ChatBot(
        llm_provider="openrouter",
        model_name="anthropic/claude-3.5-sonnet",
    )


async def stream_chatbot_response(question: str, timeout: float = 60.0) -> None:
    chatbot = create_chatbot()
    messages = [{"role": "user", "content": question}]

    print("🔁 Streaming ChatBot response...\n")

    async for chunk in chatbot.astream(messages, timeout=timeout):
        print(chunk.delta, end="", flush=True)
        
async def main():
    print("🔷 Streaming ChatBot Demo")
    print("=" * 50)

    for prompt in PROMPTS:
        print("\n" + "=" * 50)
        print(f"Q: {prompt}")
        print("=" * 50)
        await stream_chatbot_response(prompt)
        
    print("\n✅ Demo Complete!")

if __name__ == "__main__":
    asyncio.run(main())

