# Local code inspection with Sem

Install [Sem](https://github.com/Ataraxy-Labs/sem) on PATH and install Spoon with
its MCP dependencies. This example uses the same MCPTool stdio transport as
other local servers; no hosted endpoint or model key is needed.

Use an MCP client that supports Sem's protocol version. This example was tested
with Sem 0.29.0, FastMCP 2.14.7 and MCP Python SDK 1.30.0. Older environments
(for example FastMCP 2.12.5 with MCP 1.16.0) reject the server's 2025-11-25
protocol version. Upgrade the MCP dependencies in your environment if needed:

```bash
python -m pip install --upgrade 'fastmcp>=2.14.7,<3' 'mcp>=1.30.0,<2'
```

```bash
python -m examples.mcp.sem_code_demo /absolute/path/to/git/repo parse_config
```

The example discovers tool schemas and calls `sem_find` for the supplied name.
It closes all MCP connections on success or failure and bounds the request to
60 seconds. Sem may create its local index; cloud uploads are disabled.

The expanded tools can also be given to a Spoon agent through its normal tool
manager. In that case, source excerpts enter the configured model's context.
Choose only repositories the agent is allowed to inspect.
