# Parallel Search MCP example

Search the web and fetch page excerpts through Spoon's existing `MCPTool`
Streamable HTTP transport. This example calls the discovered tools directly;
it does not run an LLM or require an LLM key. It is opt-in and does not change
any agent's default tools or configuration.

From the repository root, use Python 3.12+ and a fresh virtual environment:

```bash
python -m venv .venv-parallel
source .venv-parallel/bin/activate
pip install -e . -r examples/mcp/requirements-parallel.txt
python -m examples.mcp.parallel_search_demo "SpoonOS MCP framework documentation"
python -m examples.mcp.parallel_search_demo "Python asyncio documentation" --fetch https://docs.python.org/3/library/asyncio.html
```

The example requirements select FastMCP 2.12.5, which provides the transport
classes imported by `MCPTool`, and a compatible Pydantic version. Install them
together with the SDK in a fresh environment.

The script discovers `web_search` and `web_fetch` at
`https://search.parallel.ai/mcp`, prints the search response, and optionally
prints page excerpts for `--fetch`. Both calls share a random session identifier.
It sends a Spoon project User-Agent and no Authorization header, and does not
read Parallel credentials from environment variables or saved configuration.
Calls have a 30-second timeout, and discovery plus execution is bounded to
60 seconds. Connection failures, timeouts, and server errors exit with an error.

[Parallel Search MCP](https://docs.parallel.ai/integrations/mcp/search-mcp)
is free for anonymous exploration and light use, with lower rate limits.
Anonymous searches use server-managed fast mode. Internet access is required;
rate limits still apply. The server returns source URLs and excerpts rather
than a synthesized answer.

To reuse the connection in your own code, import `create_parallel_tool()` from
the example and call `expand_server_tools()` to obtain one `MCPTool` per
discovered tool with its server-provided input schema. Execute `web_search` or
`web_fetch` by name and call `cleanup()` on each tool when finished.
