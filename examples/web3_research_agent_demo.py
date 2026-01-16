#!/usr/bin/env python3
"""
Web3 Research Agent Demo (MCP-based)

This example demonstrates how to use the SpoonReactMCP agent with MCPTool
for Web3 and cryptocurrency research using Tavily MCP server.

This demo uses the MCP-BASED approach with Tavily MCP server.

Features:
- MCP-based agent with Tavily web search
- Web3/crypto specialized research capabilities
- Direct MCP server integration

For the SKILL-BASED approach, see: web3_research_skill_agent_demo.py

Prerequisites:
- Set TAVILY_API_KEY environment variable
- Set OPENAI_API_KEY (or other LLM provider key)
- npx available for running tavily-mcp server

Usage:
    python examples/web3_research_agent_demo.py
"""

import os
import sys
import asyncio
import logging
from dotenv import load_dotenv

from spoon_ai.agents import SpoonReactMCP
from spoon_ai.chat import ChatBot
from spoon_ai.tools import ToolManager
from spoon_ai.tools.mcp_tool import MCPTool

# Load environment variables
load_dotenv(override=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Web3ResearchAgent(SpoonReactMCP):
    """
    A Web3-focused research agent that uses MCPTool with Tavily MCP server
    for comprehensive cryptocurrency and blockchain analysis.

    This agent:
    1. Uses MCPTool to connect to Tavily MCP server for web search
    2. Provides specialized crypto analysis capabilities
    3. Handles real-time market research queries
    """

    def __init__(self, **kwargs):
        # Set default values before super().__init__
        kwargs.setdefault('name', 'web3_research_agent')
        kwargs.setdefault('description', 'AI agent specialized in Web3 and cryptocurrency research (MCP-based)')
        kwargs.setdefault('system_prompt', self._get_system_prompt())
        kwargs.setdefault('max_steps', 10)

        super().__init__(**kwargs)

    @staticmethod
    def _get_system_prompt() -> str:
        return """You are an expert Web3 and cryptocurrency research analyst.

Your capabilities include:
1. Real-time market research using the tavily-search tool
2. Fundamental analysis of crypto projects
3. Technical analysis and market trends
4. DeFi protocol evaluation
5. NFT and tokenomics analysis

When analyzing crypto assets or Web3 topics:
- Use the tavily-search tool to gather current information
- Cross-reference multiple sources for accuracy
- Provide balanced analysis with both opportunities and risks
- Include relevant on-chain metrics when available
- Cite your sources clearly

Always structure your analysis professionally and acknowledge uncertainty
where appropriate. Cryptocurrency markets are highly volatile and speculative.
"""

    async def initialize(self, __context=None):
        """
        Initialize the agent with MCP tools.
        """
        # Check for Tavily API key
        tavily_key = os.getenv("TAVILY_API_KEY", "")
        if not tavily_key:
            logger.warning(
                "TAVILY_API_KEY not set. Tavily search will fail. "
                "Get your API key from https://tavily.com"
            )

        # Create Tavily MCP tool
        tavily_tool = MCPTool(
            name="tavily-search",
            description="Search the web for current cryptocurrency, blockchain, and Web3 information using Tavily API.",
            mcp_config={
                "command": "npx",
                "args": ["--yes", "tavily-mcp"],
                "env": {"TAVILY_API_KEY": tavily_key},
                "timeout": 30,
            }
        )

        # Set up tool manager
        self.available_tools = ToolManager([tavily_tool])
        logger.info(f"Available tools: {list(self.available_tools.tool_map.keys())}")

        # Initialize parent (will pre-load MCP tool parameters)
        await super().initialize(__context)

        logger.info("Web3 Research Agent initialized with Tavily MCP tool")

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

        return response


async def demo_basic_research():
    """Basic research demo with a single query."""
    print("\n" + "=" * 60)
    print("Web3 Research Agent Demo - Basic Research")
    print("(Using MCP-based Tavily search)")
    print("=" * 60)

    # Create agent with OpenAI
    agent = Web3ResearchAgent(
        llm=ChatBot(
            llm_provider="openai",
            model_name="gpt-4o-mini"
        )
    )

    # Initialize
    await agent.initialize()

    # Show available tools
    tools = list(agent.available_tools.tool_map.keys())
    print(f"\nAvailable tools: {tools}")

    # Research query
    query = "What are the latest developments in Ethereum Layer 2 solutions? Compare Arbitrum and Optimism."

    print(f"\nQuery: {query}\n")
    print("-" * 60)

    response = await agent.research(query)

    print("\nResearch Results:")
    print("-" * 60)
    print(response)


async def demo_multi_query():
    """Demo with multiple research queries."""
    print("\n" + "=" * 60)
    print("Web3 Research Agent Demo - Multi-Query Research")
    print("(Using MCP-based Tavily search)")
    print("=" * 60)

    agent = Web3ResearchAgent(
        llm=ChatBot(
            llm_provider="openai",
            model_name="gpt-4o-mini"
        )
    )

    await agent.initialize()

    queries = [
        "What is the current state of Solana DeFi ecosystem?",
        "Analyze the tokenomics of the UNI token",
    ]

    for query in queries:
        print(f"\n{'=' * 60}")
        print(f"Query: {query}")
        print("=" * 60)

        # Clear previous conversation
        agent.clear()

        response = await agent.research(query)
        # Truncate for demo
        truncated = response[:1000] + "..." if len(response) > 1000 else response
        print(f"\nResponse:\n{truncated}")


async def demo_interactive():
    """Interactive demo for user queries."""
    print("\n" + "=" * 60)
    print("Web3 Research Agent - Interactive Mode")
    print("(Using MCP-based Tavily search)")
    print("=" * 60)
    print("Type your crypto/Web3 research questions.")
    print("Type 'quit' or 'exit' to end the session.")
    print("Type 'tools' to see available tools.")
    print("Type 'clear' to clear conversation history.")
    print("=" * 60)

    agent = Web3ResearchAgent(
        llm=ChatBot(
            llm_provider="openai",
            model_name="gpt-4o-mini"
        )
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

            if user_input.lower() == 'tools':
                tools = list(agent.available_tools.tool_map.keys())
                print(f"Available tools: {tools}")
                continue

            if user_input.lower() == 'clear':
                agent.clear()
                print("Conversation history cleared.")
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

    if not (os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")):
        print("Error: No LLM API key found. Set OPENAI_API_KEY or ANTHROPIC_API_KEY.")
        sys.exit(1)

    # Run demos
    print("\nSelect demo mode:")
    print("1. Basic research (single query)")
    print("2. Multi-query research")
    print("3. Interactive mode")

    choice = input("\nEnter choice (1-3, default=1): ").strip() or "1"

    if choice == "1":
        await demo_basic_research()
    elif choice == "2":
        await demo_multi_query()
    elif choice == "3":
        await demo_interactive()
    else:
        print("Invalid choice, running basic demo...")
        await demo_basic_research()


if __name__ == "__main__":
    asyncio.run(main())
