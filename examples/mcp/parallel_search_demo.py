"""Search and fetch through Spoon's MCPTool without API keys or an LLM."""

import argparse
import asyncio
import uuid

from spoon_ai import __version__
from spoon_ai.tools.mcp_tool import MCPTool


def create_parallel_tool() -> MCPTool:
    """Create an opt-in anonymous Streamable HTTP connection."""
    return MCPTool(
        name="parallel_search",
        description="Search the web and fetch pages with Parallel Search MCP",
        mcp_config={
            "url": "https://search.parallel.ai/mcp",
            "transport": "http",
            "timeout": 30,
            "max_retries": 1,
            "headers": {
                "User-Agent": f"spoon-core/{__version__} (Parallel Search MCP example)",
            },
        },
    )


async def run_demo(query: str, fetch_url: str | None = None) -> dict[str, str]:
    """Discover real server tools, search, and optionally fetch a known URL."""
    server = create_parallel_tool()
    expanded = []
    session_id = str(uuid.uuid4())
    try:
        async with asyncio.timeout(60):
            expanded = await server.expand_server_tools()
            tools = {tool.name: tool for tool in expanded}
            required = {"web_search", "web_fetch"} if fetch_url else {"web_search"}
            if not required.issubset(tools):
                raise RuntimeError("Parallel Search MCP did not expose the required tools")

            results = {
                "web_search": await tools["web_search"].execute(
                    objective=query,
                    search_queries=[query],
                    session_id=session_id,
                )
            }
            if fetch_url:
                results["web_fetch"] = await tools["web_fetch"].execute(
                    urls=[fetch_url], objective=query[:200], session_id=session_id,
                )
            return results
    finally:
        for tool in [*expanded, server]:
            await tool.cleanup()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", nargs="?", default="SpoonOS MCP framework documentation")
    parser.add_argument("--fetch", metavar="URL", help="Also fetch a specific page")
    args = parser.parse_args()
    for name, result in asyncio.run(run_demo(args.query, args.fetch)).items():
        print(f"\n{name}:\n{result}")


if __name__ == "__main__":
    main()
