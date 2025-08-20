
# 🤖 Agent Development Guide

This guide provides a comprehensive walkthrough for developing and configuring agents in the SpoonOS Core Developer Framework (SCDF). We will use a practical example, the `SpoonMacroAnalysisAgent`, to illustrate key concepts, including agent definition, tool integration, and execution.

For details on the configuration file system, see [`configuration.md`](./configuration.md).

---

## 1. Core Concepts: Agent and Tools

In SpoonOS, an **Agent** is an autonomous entity that leverages **Tools** to achieve specific goals. `SpoonMacroAnalysisAgent` is our example for demonstrating how to combine on-chain data with real-time news.

### Agent and Tool Architecture

- **`SpoonReactMCP`**: A powerful base class for agents, providing a ReAct (Reasoning + Acting) loop and built-in support for the Model Context Protocol (MCP).
- **Built-in Tools**: Standard Python classes, like `CryptoPowerDataCEXTool`, that are integrated directly into the framework.
- **MCP Tools**: External tools accessed over a network or via a subprocess. `MCPTool` is a generic class for connecting to these tools.

---

## 2. Building an Agent: A Step-by-Step Example

### Step 1: Define the Agent Class

Define the agent by inheriting from `SpoonReactMCP` and setting its core properties:

```python
class SpoonMacroAnalysisAgent(SpoonReactMCP):
    name: str = "SpoonMacroAnalysisAgent"
    system_prompt: str = (
        '''You are a cryptocurrency market analyst. Your task is to provide a comprehensive
        macroeconomic analysis of a given token.

        To do this, you will perform the following steps:
        1. Use the `crypto_power_data_cex` tool to get the latest candlestick data and
           technical indicators.
        2. Use the `tavily-search` tool to find the latest news and market sentiment.
        3. Synthesize the data from both tools to form a holistic analysis.'''
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.avaliable_tools = ToolManager([])
```

### Step 2: Configure and Load Tools

In the `initialize` method, we set up and load all necessary tools.

#### MCP Tool Configuration (Transport-Agnostic)

`MCPTool` is designed to be **transport-agnostic**. You can configure it to connect to any MCP server using **stdio**, **HTTP (SSE)**, or **WebSocket**.

**1. Stdio-based Transport (Recommended)**

**Stdio tools** are the recommended approach for most MCP integrations. They run as subprocesses and communicate via stdin/stdout, with automatic lifecycle management by SpoonOS.

Use the `command` field to execute local command-line tools. The tool's `name` should directly match the MCP tool's name.

```python
tavily_tool = MCPTool(
    name="tavily-search",
    description="Performs a web search using the Tavily API.",
    mcp_config={
        "command": "npx",
        "args": ["--yes", "tavily-mcp"],
        "env": {"TAVILY_API_KEY": os.getenv("TAVILY_API_KEY")},
        "transport": "stdio"
    }
)
```

**Benefits of Stdio Transport:**
- No server management required
- Auto-managed lifecycle by SpoonOS
- Always up to date with latest tool versions
- Automatic restart on failures

**Common Stdio MCP Tools:**

```python
# GitHub integration
github_tool = MCPTool(
    name="github_tools",
    description="GitHub repository and issue management",
    mcp_config={
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-github"],
        "env": {"GITHUB_TOKEN": os.getenv("GITHUB_TOKEN")},
        "transport": "stdio"
    }
)

# Brave Search
brave_tool = MCPTool(
    name="brave_search",
    description="Web search using Brave Search API",
    mcp_config={
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-brave-search"],
        "env": {"BRAVE_API_KEY": os.getenv("BRAVE_API_KEY")},
        "transport": "stdio"
    }
)
```

**2. HTTP Transport (via SSE) - Advanced Use Cases**

For HTTP-based MCP servers, `fastmcp` uses **Server-Sent Events (SSE)**. Use this for custom or in-development integrations.

```python
http_tool = MCPTool(
    name="http_example_tool",
    mcp_config={
        "url": "http://127.0.0.1:8765/sse",
        "transport": "sse"
    }
)
```

**Note:** You must manually start and manage SSE servers:

```bash
# Start your custom SSE server
python my_mcp_server.py
# or
npx -y @modelcontextprotocol/server-github --sse
```

**3. WebSocket Transport**

For WebSocket-based servers, provide the `ws` or `wss` URL.

```python
ws_tool = MCPTool(
    name="websocket_example_tool",
    mcp_config={"url": "ws://127.0.0.1:8766"}
)
```

#### Add Tools to the ToolManager

After configuring your tools, add them to the agent’s `ToolManager`:

```python
# In the initialize method:
self.avaliable_tools = ToolManager([tavily_tool, crypto_tool, http_tool, ws_tool])
```

### Step 3: Execute the Agent

The `main` function orchestrates the agent's run, from initialization to query execution.

---

## 3. Full Code Example (Refactored)

This complete, refactored code for `spoon_search_agent.py` demonstrates the simplified and transport-agnostic approach.

```python
import os
import sys
import asyncio
import logging
from typing import Dict, Any

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../spoon-toolkit')))

from spoon_ai.agents.spoon_react_mcp import SpoonReactMCP
from spoon_ai.tools.mcp_tool import MCPTool
from spoon_ai.tools.tool_manager import ToolManager
from spoon_ai.chat import ChatBot
from spoon_toolkits.crypto.crypto_powerdata.tools import CryptoPowerDataCEXTool

logging.basicConfig(level=logging.INFO)

class SpoonMacroAnalysisAgent(SpoonReactMCP):
    name: str = "SpoonMacroAnalysisAgent"
    system_prompt: str = (
        '''You are a cryptocurrency market analyst. Your task is to provide a comprehensive
        macroeconomic analysis of a given token...'''
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.avaliable_tools = ToolManager([])

    async def initialize(self):
        logging.info("Initializing agent and loading tools...")
        tavily_key = os.getenv("TAVILY_API_KEY", "")
        if not tavily_key or "your-tavily-api-key-here" in tavily_key:
            raise ValueError("TAVILY_API_KEY is not set or is a placeholder.")

        tavily_tool = MCPTool(
            name="tavily-search",
            description="Performs a web search using the Tavily API.",
            mcp_config={
                "command": "npx",
                "args": ["--yes", "tavily-mcp"],
                "env": {"TAVILY_API_KEY": tavily_key}
            }
        )

        crypto_tool = CryptoPowerDataCEXTool()
        self.avaliable_tools = ToolManager([tavily_tool, crypto_tool])
        logging.info(f"Available tools: {list(self.avaliable_tools.tool_map.keys())}")

async def main():
    print("--- SpoonOS Macro Analysis Agent Demo ---")
    agent = SpoonMacroAnalysisAgent(llm=ChatBot(llm_provider="openai"))
    print("Agent instance created.")
    await agent.initialize()
    query = "Perform a macro analysis of the NEO token."
    print(f"\nRunning query: {query}")
    response = await agent.run(query)
    print(f"\n--- Analysis Complete ---\n{response}")

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 4. MCP Tool Best Practices

### Choosing the Right Transport

| Transport | Use Case | Pros | Cons |
|-----------|----------|------|------|
| **Stdio** | Most MCP tools (Tavily, GitHub, Brave) | Auto-managed, reliable, up-to-date | Limited to command-line tools |
| **SSE** | Custom/in-development tools | Flexible, supports custom servers | Manual server management |
| **WebSocket** | Real-time bidirectional communication | Low latency, persistent connection | Complex setup, manual management |

### Configuration Best Practices

1. **Prefer Stdio for standard tools**: Use stdio transport for well-established MCP tools
2. **Environment variable management**: Store API keys in environment variables, not in code
3. **Error handling**: Set appropriate timeouts and retry attempts
4. **Tool naming**: Use descriptive names that match the tool's functionality
5. **Resource management**: Limit concurrent tool usage to avoid rate limits

### Example: Production-Ready Tool Configuration

```python
async def initialize(self):
    """Initialize agent with production-ready tool configuration"""

    # Validate required environment variables
    required_env_vars = ["TAVILY_API_KEY", "GITHUB_TOKEN", "OKX_API_KEY"]
    for var in required_env_vars:
        if not os.getenv(var):
            raise ValueError(f"Required environment variable {var} is not set")

    # Configure tools with proper error handling
    tools = []

    # Stdio-based tools (recommended)
    try:
        tavily_tool = MCPTool(
            name="tavily-search",
            description="Web search using Tavily API",
            mcp_config={
                "command": "npx",
                "args": ["-y", "tavily-mcp"],
                "env": {"TAVILY_API_KEY": os.getenv("TAVILY_API_KEY")},
                "transport": "stdio",
                "timeout": 30,
                "retry_attempts": 3
            }
        )
        tools.append(tavily_tool)
    except Exception as e:
        logging.warning(f"Failed to initialize Tavily tool: {e}")

    # Built-in tools
    try:
        crypto_tool = CryptoPowerDataCEXTool()
        tools.append(crypto_tool)
    except Exception as e:
        logging.warning(f"Failed to initialize crypto tool: {e}")

    self.avaliable_tools = ToolManager(tools)
    logging.info(f"Initialized {len(tools)} tools successfully")
```

### Troubleshooting Common Issues

#### Stdio Tool Issues

**Problem**: Tool command not found
```bash
Error: Command 'npx' not found
```
**Solution**: Ensure Node.js and npm are installed and in PATH

**Problem**: Environment variables not loaded
```bash
Error: TAVILY_API_KEY is required
```
**Solution**: Check environment variable configuration in mcp_config

#### SSE Tool Issues

**Problem**: Connection refused
```bash
Error: Connection refused to http://127.0.0.1:8765/sse
```
**Solution**: Ensure the SSE server is running and accessible

**Problem**: Timeout errors
```bash
Error: Request timeout after 30 seconds
```
**Solution**: Increase timeout in mcp_config or check server performance

#### General Debugging

1. **Enable debug logging**:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

2. **Check tool availability**:
```python
print(f"Available tools: {list(self.avaliable_tools.tool_map.keys())}")
```

3. **Test tool connectivity**:
```python
# Test individual tool
result = await tool.execute("test_function", {})
```

---

## 5. Next Steps

- **Customize**: Modify the `system_prompt` or add new tools to the `ToolManager`
- **Explore**: Investigate other built-in toolkits or connect to different MCP services
- **Advanced Configuration**: For more complex setups, refer to the [Configuration Guide](./configuration.md)
- **MCP Protocol**: Learn more at [MCP Protocol Documentation](https://modelcontextprotocol.io/)
