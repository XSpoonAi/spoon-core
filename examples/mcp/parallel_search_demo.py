"""Search and fetch through Spoon's MCPTool without API keys or an LLM."""

import argparse
import asyncio
import json

from spoon_ai import __version__
from spoon_ai.tools.mcp_tool import MCPTool

PARALLEL_MCP_URL = "https://mcp.parallel.ai/mcp"


def create_parallel_tool() -> MCPTool:
    """Create an opt-in anonymous Streamable HTTP connection."""
    return MCPTool(
        name="parallel_mcp",
        description="Search the web and fetch pages with the Parallel MCP server",
        mcp_config={
            "url": PARALLEL_MCP_URL,
            "transport": "http",
            "timeout": 30,
            "max_retries": 1,
            "headers": {
                "User-Agent": f"spoon-core/{__version__} (Parallel MCP example)",
            },
        },
    )


def extract_session_id(result) -> str | None:
    """Return the session identifier the server attached to a tool result."""
    if isinstance(result, (str, bytes, bytearray)):
        try:
            result = json.loads(result)
        except ValueError:
            return None
    if isinstance(result, dict):
        session_id = result.get("session_id")
        if isinstance(session_id, str) and session_id:
            return session_id
    return None


async def run_demo(query: str, fetch_url: str | None = None) -> dict[str, str]:
    """Discover real server tools, search, and optionally fetch a known URL."""
    server = create_parallel_tool()
    expanded = []
    try:
        async with asyncio.timeout(60):
            expanded = await server.expand_server_tools()
            tools = {tool.name: tool for tool in expanded}
            required = {"web_search", "web_fetch"} if fetch_url else {"web_search"}
            missing = sorted(required - tools.keys())
            if missing:
                raise RuntimeError(
                    "Parallel MCP server did not expose the required tools: "
                    + ", ".join(missing)
                )

            search_result = await tools["web_search"].execute(
                objective=query,
                search_queries=[query],
            )
            results = {"web_search": search_result}
            if fetch_url:
                fetch_args: dict[str, object] = {
                    "urls": [fetch_url],
                    "objective": query[:200],
                }
                # The server issues the session identifier on the first call.
                session_id = extract_session_id(search_result)
                if session_id:
                    fetch_args["session_id"] = session_id
                results["web_fetch"] = await tools["web_fetch"].execute(**fetch_args)
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
