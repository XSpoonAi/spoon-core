#!/usr/bin/env python3
"""
Web3 Research Skill Agent Demo

This example demonstrates how to use the SpoonReactSkill agent with the
Web3 research skill for cryptocurrency and blockchain analysis.

This demo uses the SKILL-BASED approach with script execution.

Features:
- Skill-based agent with auto-activation
- Script-based Tavily search integration (via skill scripts)
- Web3/crypto specialized research capabilities

Prerequisites:
- Set TAVILY_API_KEY environment variable
- Set OPENAI_API_KEY (or other LLM provider key)
- Install tavily-python: pip install tavily-python

Usage:
    python examples/web3_research_skill_agent_demo.py
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


class Web3ResearchSkillAgent(SpoonReactSkill):
    """
    A Web3-focused research agent that combines skill-based prompting
    with script-based Tavily search for comprehensive crypto analysis.

    This agent:
    1. Loads skills from examples/skills directory
    2. Uses script-based tavily_search tool from web3-research skill
    3. Automatically activates web3-research skill for crypto queries
    """

    def __init__(self, **kwargs):
        # Set default values before super().__init__
        kwargs.setdefault('name', 'web3_research_skill_agent')
        kwargs.setdefault('description', 'AI agent specialized in Web3 and cryptocurrency research (skill-based)')
        kwargs.setdefault('system_prompt', self._get_system_prompt())
        kwargs.setdefault('max_steps', 10)
        kwargs.setdefault('_default_timeout', 120.0)

        # Configure skill paths to include examples/skills
        kwargs.setdefault('skill_paths', [EXAMPLES_SKILLS_PATH])

        # Enable scripts for Tavily search
        kwargs.setdefault('scripts_enabled', True)

        super().__init__(**kwargs)

    @staticmethod
    def _get_system_prompt() -> str:
        return """You are an expert Web3 and cryptocurrency research analyst.

Your capabilities include:
1. Real-time market research using the tavily_search script
2. Fundamental analysis of crypto projects
3. Technical analysis and market trends
4. DeFi protocol evaluation
5. NFT and tokenomics analysis

When analyzing crypto assets or Web3 topics:
- Use the run_script_web3-research_tavily_search tool to gather current information
- Cross-reference multiple sources for accuracy
- Provide balanced analysis with both opportunities and risks
- Include relevant on-chain metrics when available
- Cite your sources clearly

Always structure your analysis professionally and acknowledge uncertainty
where appropriate. Cryptocurrency markets are highly volatile and speculative.
"""

    async def initialize(self, __context=None):
        """
        Initialize the agent with skill system.
        """
        # Check for Tavily API key
        tavily_key = os.getenv("TAVILY_API_KEY", "")
        if not tavily_key:
            logger.warning(
                "TAVILY_API_KEY not set. Tavily search will fail. "
                "Get your API key from https://tavily.com"
            )

        # Initialize parent (includes skill system with script support)
        await super().initialize(__context)

        # Log available skills
        skills = self.list_skills()
        logger.info(f"Available skills: {skills}")

        # # Activate web3-research skill to get access to tavily_search script
        # if "web3-research" in skills:
        #     await self.activate_skill("web3-research")
        #     logger.info("Activated web3-research skill with tavily_search script")

    async def research(self, query: str) -> str:
        """
        Perform Web3 research on a given query.

        Args:
            query: The research query (e.g., "Analyze Ethereum staking yields")

        Returns:
            Comprehensive research analysis
        """
        logger.info(f"Starting research: {query}")

        # Run the agent with the query
        response = await self.run(query)

        # Log active skills
        active = self.list_active_skills()
        if active:
            logger.info(f"Skills used: {active}")

        return response


async def demo_basic_research():
    """Basic research demo with a single query."""
    print("\n" + "=" * 60)
    print("Web3 Research Skill Agent Demo - Basic Research")
    print("(Using skill-based tavily_search script)")
    print("=" * 60)

    # Create agent with OpenAI
    agent = Web3ResearchSkillAgent(
        llm=ChatBot(
        ),
        auto_trigger_skills=True,
        max_auto_skills=2
    )

    # Initialize
    await agent.initialize()

    # Show available tools
    tools = agent.skill_manager.get_active_tools() if agent.skill_manager else []
    print(f"\nAvailable tools: {[t.name for t in tools]}")

    # Research query
    query = "What are the latest developments in Ethereum Layer 2 solutions? Compare Arbitrum and Optimism."

    print(f"\nQuery: {query}\n")
    print("-" * 60)

    response = await agent.research(query)

    print("\nResearch Results:")
    print("-" * 60)
    print(response)


async def demo_with_skill_info():
    """Demo showing skill system details."""
    print("\n" + "=" * 60)
    print("Web3 Research Skill Agent Demo - With Skill Info")
    print("(Using skill-based tavily_search script)")
    print("=" * 60)

    agent = Web3ResearchSkillAgent(
        llm=ChatBot(
        ),
        auto_trigger_skills=True
    )

    await agent.initialize()

    # Show skill statistics
    stats = agent.get_skill_stats()
    print(f"\nSkill System Stats:")
    print(f"  - Total skills: {stats['total_skills']}")
    print(f"  - Active skills: {stats['active_skills']}")
    print(f"  - Scripts enabled: {stats.get('scripts_enabled', False)}")

    # Show skill info
    info = agent.skill_manager.get_skill_info("web3-research") if agent.skill_manager else None
    if info:
        print(f"\nWeb3-Research Skill Info:")
        print(f"  - Scripts: {info.get('script_names', [])}")
        print(f"  - Has scripts: {info.get('has_scripts', False)}")

    # Show available tools (including script tools)
    if agent.skill_manager:
        tools = agent.skill_manager.get_active_tools()
        print(f"\nActive Tools:")
        for tool in tools:
            desc = tool.description[:60] + "..." if len(tool.description) > 60 else tool.description
            print(f"  - {tool.name}: {desc}")

    # Run a query
    query = "What are the key risks of investing in Ethereum right now?"
    print(f"\nQuery: {query}")
    print("-" * 60)

    response = await agent.research(query)
    print(f"\nResponse:\n{response}")

    # Deactivate skill
    await agent.deactivate_skill("web3-research")
    print(f"\nAfter deactivation, active skills: {agent.list_active_skills()}")


async def demo_interactive():
    """Interactive demo for user queries."""
    print("\n" + "=" * 60)
    print("Web3 Research Skill Agent - Interactive Mode")
    print("(Using skill-based tavily_search script)")
    print("=" * 60)
    print("Type your crypto/Web3 research questions.")
    print("Type 'quit' or 'exit' to end the session.")
    print("Type 'skills' to see available and active skills.")
    print("Type 'tools' to see available tools.")
    print("=" * 60)

    agent = Web3ResearchSkillAgent(
        llm=ChatBot(
        ),
        auto_trigger_skills=True
    )

    await agent.initialize()

    while True:
        try:
            user_input = input("\nYou: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break

            if user_input.lower() == 'skills':
                print(f"Available skills: {agent.list_skills()}")
                print(f"Active skills: {agent.list_active_skills()}")
                continue

            if user_input.lower() == 'tools':
                if agent.skill_manager:
                    tools = agent.skill_manager.get_active_tools()
                    print(f"Active tools: {[t.name for t in tools]}")
                continue

            if user_input.lower() == 'clear':
                agent.clear()
                await agent.deactivate_all_skills()
                print("Conversation and skills cleared.")
                continue

            response = await agent.research(user_input)
            print(f"\nAgent: {response}")

        except KeyboardInterrupt:
            print("\nInterrupted. Goodbye!")
            break
        except Exception as e:
            logger.error(f"Error: {e}")
            print(f"Error occurred: {e}")


async def main():
    """Main entry point."""
    # Check environment
    if not os.getenv("TAVILY_API_KEY"):
        print("Warning: TAVILY_API_KEY environment variable is not set.")
        print("Tavily search will not work. Get your API key from https://tavily.com")
        print()

    if not (os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")or os.getenv("GEMINI_API_KEY") or os.getenv("OPENROUTER_API_KEY")):
        print("Error: No LLM API key found. Set OPENAI_API_KEY or ANTHROPIC_API_KEY,GEMINI_API_KEY or OPENROUTER_API_KEY.")
        sys.exit(1)

    # Run demos
    print("\nSelect demo mode:")
    print("1. Basic research (single query)")
    print("2. With skill info")
    print("3. Interactive mode")

    choice = input("\nEnter choice (1-3, default=1): ").strip() or "1"

    if choice == "1":
        await demo_basic_research()
    elif choice == "2":
        await demo_with_skill_info()
    elif choice == "3":
        await demo_interactive()
    else:
        print("Invalid choice, running basic demo...")
        await demo_basic_research()


if __name__ == "__main__":
    asyncio.run(main())
