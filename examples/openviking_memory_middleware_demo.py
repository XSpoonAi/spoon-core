"""Run a Spoon agent with optional OpenViking long-term memory.

Start an OpenViking server and set OPENVIKING_URL (plus OPENVIKING_API_KEY
for a remote server) before running this example.
"""

import asyncio

from spoon_ai.agents import SpoonReactAI
from spoon_ai.middleware import OpenVikingMemoryMiddleware


async def main() -> None:
    memory = OpenVikingMemoryMiddleware(
        # A stable session keeps memories connected across separate agent runs.
        session_id="spoon-openviking-demo",
        recall_limit=5,
    )
    agent = SpoonReactAI(
        name="openviking-demo",
        system_prompt="You are a concise assistant.",
        middleware=[memory],
        x402_enabled=False,
    )
    try:
        await agent.initialize()
        print(await agent.run("Remember that I prefer short answers."))
    finally:
        memory.close()


if __name__ == "__main__":
    asyncio.run(main())
