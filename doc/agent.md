
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

**1. Stdio-based Transport (npx, python, etc.)**

Use the `command` field to execute local command-line tools. The tool's `name` should directly match the MCP tool's name.

```python
tavily_tool = MCPTool(
    name="tavily-search",
    description="Performs a web search using the Tavily API.",
    mcp_config={
        "command": "npx",
        "args": ["--yes", "tavily-mcp"],
        "env": {"TAVILY_API_KEY": os.getenv("TAVILY_API_KEY")}
    }
)
```

**2. HTTP Transport (via SSE)**

For HTTP-based MCP servers, `fastmcp` uses **Server-Sent Events (SSE)**. Simply provide the server's URL.

```python
http_tool = MCPTool(
    name="http_example_tool",
    mcp_config={"url": "http://127.0.0.1:8765/sse"}
)
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

## 4. Next Steps

- **Customize**: Modify the `system_prompt` or add new tools to the `ToolManager`.
- **Explore**: Investigate other built-in toolkits or connect to different MCP services.
- **Advanced Configuration**: For more complex setups, refer to the [Configuration Guide](./configuration.md).
