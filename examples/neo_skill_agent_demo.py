#!/usr/bin/env python3
"""
Neo N3 Research Skill Agent Demo

This example demonstrates how to use the SpoonReactSkill agent with the
Neo Query skill for comprehensive blockchain data analysis.

Features:
- Skill-based agent for Neo N3
- Script-based Neo RPC integration (via neo-query skill)
- Advanced analysis of blocks, addresses, and governance
"""

import os
import sys
import asyncio
import logging
from pathlib import Path
from dotenv import load_dotenv

from spoon_ai.agents import SpoonReactSkill
from spoon_ai.chat import ChatBot

# Load environment variables
load_dotenv(override=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Path to example skills
EXAMPLES_SKILLS_PATH = str(Path(__file__).parent / "skills")


class NeoResearchSkillAgent(SpoonReactSkill):
    """
    A Neo-focused research agent that uses the neo-query skill
    for deep blockchain data analysis.
    """

    def __init__(self, **kwargs):
        # Set default values before super().__init__
        kwargs.setdefault('name', 'neo_research_skill_agent')
        kwargs.setdefault('description', 'AI agent specialized in Neo N3 blockchain research')
        kwargs.setdefault('system_prompt', self._get_system_prompt())
        kwargs.setdefault('max_steps', 10)
        kwargs.setdefault('_default_timeout', 120.0)

        # Configure skill paths to include examples/skills
        kwargs.setdefault('skill_paths', [EXAMPLES_SKILLS_PATH])

        # Enable scripts for data query
        kwargs.setdefault('scripts_enabled', True)

        super().__init__(**kwargs)

    @staticmethod
    def _get_system_prompt() -> str:
        return """You are a top-tier Neo N3 Blockchain Analyst.

Your mission is to provide deep, accurate, and professional analysis of the Neo ecosystem.
You have access to the `neo-query` skill, which provides direct RPC access to Neo data.

When analyzing Neo topics:
1. Use `run_script_neo-query_neo_rpc_query` to fetch real-time data.
2. For addresses: Always check balances (NEO/GAS) and recent transfer history to understand the user's profile.
3. For governance: Use committee and candidate tools to explain the voting landscape.
4. For contracts: Look for verification status and analyze recent invocation logs if needed.
5. Context: Default to Neo Testnet unless Mainnet is specified.

Structure your responses professionally, using tables for data comparison where appropriate.
Always explain technical terms (like NEP-17, GAS, UInt160) in a user-friendly way.
"""

    async def initialize(self, __context=None):
        """Initialize the agent and activate Neo skills."""
        await super().initialize(__context)

        skills = self.list_skills()
        logger.info(f"Available skills: {skills}")

    async def analyze(self, query: str) -> str:
        logger.info(f"Starting Neo analysis: {query}")
        response = await self.run(query)
        return response


async def demo_neo_analysis():
    """Run a demo of Neo blockchain analysis."""
    print("\n" + "=" * 60)
    print("Neo N3 Research Skill Agent Demo")
    print("(Powered by neo-query skill scripts)")
    print("=" * 60)

    # Create agent
    agent = NeoResearchSkillAgent(
        llm=ChatBot(),
        auto_trigger_skills=True
    )

    # Initialize
    await agent.initialize()

    # Test cases
    test_queries = [
        "What is the current status of the Neo Testnet? Show me the latest block height and committee members.",
        "Check the balance and recent NEP-17 activity for Neo address NUTtedVrz5RgKAdCvtKiq3sRkb9pizcewe.",
        "Search for a contract named 'Flamingo' and tell me its hash and verification status."
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n[Test {i}] Query: {query}")
        print("-" * 60)
        response = await agent.analyze(query)
        print(f"\nAnalysis Result:\n{response}")
        print("-" * 60)
        await asyncio.sleep(2)


async def demo_interactive():
    """Interactive mode for Neo research."""
    print("\n" + "=" * 60)
    print("Neo N3 Research Agent - Interactive Mode")
    print("Type your Neo-related questions (e.g., 'Check balance of N...', 'Who is in the council?')")
    print("Type 'exit' to quit.")
    print("=" * 60)

    agent = NeoResearchSkillAgent(llm=ChatBot())
    await agent.initialize()

    while True:
        try:
            user_input = input("\nYou: ").strip()
            if not user_input or user_input.lower() in ['exit', 'quit', 'q']:
                break
            
            response = await agent.analyze(user_input)
            print(f"\nAgent: {response}")
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

async def main():
    # Intelligent LLM Provider selection
    from spoon_ai.llm.manager import get_llm_manager
    if not (os.getenv("OPENAI_API_KEY") or os.getenv("GEMINI_API_KEY") or os.getenv("OPENROUTER_API_KEY")):
        print("Error: No LLM API key found.")
        sys.exit(1)

    print("\nSelect demo mode:")
    print("1. Automatic Demo (3 scenarios)")
    print("2. Interactive mode")
    
    choice = input("\nEnter choice (1-2, default=1): ").strip() or "1"

    if choice == "1":
        await demo_neo_analysis()
    else:
        await demo_interactive()


if __name__ == "__main__":
    asyncio.run(main())

