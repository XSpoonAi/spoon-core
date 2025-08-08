# MCP Mode Usage: Stdio & SSE Tool Integration in SpoonOS

This guide explains **how to configure and use MCP tools in SpoonOS**, focusing on the two main integration methods:
**Stdio-based MCP tools** (recommended for most modern tools) and **SSE-based MCP tools** (for legacy/custom servers).

---

## 1. What is MCP?

**MCP (Model Context Protocol)** is a unified protocol for integrating external tools and services (like Tavily, Brave, GitHub, custom Web3 APIs) into SpoonOS agents.
MCP tools can be provided by:
- **Stdio tools**: Tools that communicate via standard input/output (Stdio), typically started as a subprocess by SpoonOS.
- **SSE tools**: Tools running as local/remote SSE servers, communicating over HTTP.

---

## 2. Recommended: Stdio MCP Tools

### What are Stdio MCP Tools?

- **Stdio tools** are MCP tools that run as a subprocess and communicate with SpoonOS via stdin/stdout.
- No need to run or configure any server manually.
- Just declare the tool and its command in your agent's config, and SpoonOS will auto-start and manage the tool process.

### How to Use Stdio MCP Tools

**Step 1: Add the tool to your agent config**

```json
{
  "agents": {
    "my_agent": {
      "class": "SpoonReactMCP",
      "tools": [
        {
          "name": "tavily_search",
          "type": "mcp",
          "mcp_server": {
            "command": "npx",
            "args": ["-y", "tavily-mcp"],
            "transport": "stdio"
          }
        },
        {
          "name": "github_tools",
          "type": "mcp",
          "mcp_server": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-github"],
            "transport": "stdio"
          }
        }
      ]
    }
  }
}
```

- `"transport": "stdio"` tells SpoonOS to use Stdio for communication.
- SpoonOS will auto-start the process and manage its lifecycle.

**Step 2: Use the agent as normal**
SpoonOS will handle all Stdio tool connections automatically.

---

## 3. Advanced: SSE MCP Tools (Custom/Legacy)

### What are SSE Tools?

- **SSE tools** are MCP tools you run yourself, typically for custom or in-development integrations.
- You must start the SSE server and specify its endpoint in your config.

### How to Use SSE Tools

**Step 1: Start your SSE MCP server**
(Example: running a custom FastMCP server)

```bash
python my_mcp_server.py
# or
npx -y @modelcontextprotocol/server-github
```

**Step 2: Add the tool to your agent config**

```json
{
  "agents": {
    "my_agent": {
      "class": "SpoonReactMCP",
      "tools": [
        {
          "name": "my_custom_tool",
          "type": "mcp",
          "mcp_server": {
            "endpoint": "http://127.0.0.1:8765/sse",
            "transport": "sse"
          }
        }
      ]
    }
  }
}
```

- `"transport": "sse"` is used for SSE tools.
- `"endpoint"` specifies the SSE server URL.
- You are responsible for starting and managing the SSE server process.

**Step 3: Use the agent as normal**
SpoonOS will connect to your SSE server and expose the tool to the agent.

---

## 4. Configuration Summary Table

| Tool Type   | Config `"type"` | Transport   | How to Use/Connect                | Example Use Case                |
|-------------|-----------------|-------------|-----------------------------------|---------------------------------|
| Stdio MCP   | `"mcp"`         | `"stdio"`   | Auto-managed subprocess           | Tavily, GitHub, Brave, etc.     |
| SSE MCP     | `"mcp"`         | `"sse"`     | Start your own SSE server         | Custom Web3, in-house tools     |
| Built-in    | `"builtin"`     | N/A         | Provided by SpoonOS               | Crypto tools, price data, etc.  |

---

## 5. Example: Mixed Agent Config

```json
{
  "agents": {
    "web3_agent": {
      "class": "SpoonReactMCP",
      "tools": [
        {
          "name": "tavily_search",
          "type": "mcp",
          "mcp_server": {
            "command": "npx",
            "args": ["-y", "tavily-mcp"],
            "transport": "stdio"
          }
        },
        {
          "name": "my_custom_tool",
          "type": "mcp",
          "mcp_server": {
            "endpoint": "http://127.0.0.1:8765/sse",
            "transport": "sse"
          }
        },
        { "name": "crypto_tools", "type": "builtin" }
      ]
    }
  }
}
```

---

## 6. Best Practices

- **Prefer Stdio MCP tools** for most use cases—no server management, always up to date, auto-restart.
- Use `"sse"` tools only for custom or in-development integrations.
- You can mix Stdio, SSE, and built-in tools in a single agent.
- All tool configuration is managed in `config.json`—no need to modify code for tool integration.

---

## 7. Troubleshooting

- If a Stdio tool is not available, check the command and args in your config.
- For SSE tools, ensure the server is running and the endpoint is correct.
- Use the CLI or logs to see which tools are loaded for each agent.

---

## Next Steps

- [Agent Development Guide](./agent.md)
- [Configuration Guide](./configuration.md)
- [MCP Protocol Documentation](https://modelcontextprotocol.io/)