"""Find a code entity in a local checkout using Spoon's MCPTool."""
import argparse
import asyncio
from pathlib import Path

from spoon_ai.tools.mcp_tool import MCPTool


async def run_demo(repo: Path, query: str):
    server = MCPTool(
        name="sem", description="Inspect entities in a local Git checkout",
        mcp_config={"command": "sem", "args": ["mcp"],
                    "env": {"SEM_REPO": str(repo.resolve(strict=True)), "SEM_CLOUD": "0"}},
    )
    expanded = []
    try:
        async with asyncio.timeout(60):
            expanded = await server.expand_server_tools()
            tools = {tool.name: tool for tool in expanded}
            if "sem_find" not in tools:
                raise RuntimeError("The server did not expose sem_find")
            return await tools["sem_find"].execute(query=query)
    finally:
        for tool in [*expanded, server]:
            await tool.cleanup()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", type=Path)
    parser.add_argument("query", help="Entity name to find")
    args = parser.parse_args()
    print(asyncio.run(run_demo(args.repo.expanduser(), args.query)))


if __name__ == "__main__":
    main()
