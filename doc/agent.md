# 🤖 Agent Development Guide

This guide explains how to **develop, extend, and understand agents in SpoonOS Core Developer Framework (SCDF)**.
**For configuration file details, see [`configuration.md`](./configuration.md).**

---

## 1. Agent Architecture & Lifecycle

SpoonOS agents are autonomous reasoning entities that follow the **ReAct (Reasoning + Acting) loop** and can be extended for custom logic and tool integration.

### Core Agent Types

- **BaseAgent**: Abstract base for all agents. Defines the main lifecycle and state management.
- **ReActAgent**: Implements the ReAct loop (`think` → `act` → feedback).
- **ToolCallAgent**: Adds tool selection and calling logic.
- **SpoonReactAI**: Standard agent with built-in crypto tools (no MCP support).
- **SpoonReactMCP**: Inherits from SpoonReactAI, adds MCP protocol support for external tools (including studio tools, MCP tools, and built-in tools).
- **CustomAgent**: User-extendable agent for custom logic and tools.

### Agent Lifecycle (Code Logic)

1. **Initialization**: Agent is created, tools are registered (from config or code).
2. **Observation**: Receives user input or environment context.
3. **Reasoning**: `think()` method plans the next step.
4. **Action**: `act()` method executes a tool or action.
5. **Feedback**: Tool results are processed for further reasoning.
6. **Loop**: Steps 3-5 repeat until goal is reached or `max_steps` is hit.
7. **Finish**: Agent returns the final result.

**Key code locations:**
- [`react.py`](../spoon_ai/agents/react.py): `ReActAgent` base logic, `think`/`act`/`step`.
- [`toolcall.py`](../spoon_ai/agents/toolcall.py): Tool selection, tool call, and memory.
- [`spoon_react.py`](../spoon_ai/agents/spoon_react.py): `SpoonReactAI` implementation.
- [`spoon_react_mcp.py`](../spoon_ai/agents/spoon_react_mcp.py): `SpoonReactMCP` with MCP integration.
- [`custom_agent.py`](../spoon_ai/agents/custom_agent.py): User-extendable agent.

---

## 2. Tool Integration: Built-in, MCP, and Studio Tools

Agents in SpoonOS can use **three types of tools**:
- **Built-in tools**: Provided by the core system (e.g., crypto_tools).
- **MCP tools**: Tools provided by MCP servers (e.g., Tavily, Brave, GitHub, etc).
- **Studio tools**: Tools registered and managed via the Spoon Studio platform (no need to configure SSE or custom transport).

**All tool types can be configured directly in the agent's `tools` array in config.**
You do **not** need to manually configure SSE or transport for studio/MCP tools—just declare them in config and the system will handle the connection.

**Example agent config (see `configuration.md` for full details):**
```json
{
  "agents": {
    "my_agent": {
      "class": "SpoonReactMCP",
      "description": "Agent with built-in, MCP, and studio tools",
      "tools": [
        { "name": "crypto_tools", "type": "builtin" },
        { "name": "tavily_search", "type": "mcp" },
        { "name": "studio_tool_example", "type": "studio" }
      ]
    }
  }
}
```
- **Built-in tools**: `"type": "builtin"`
- **MCP tools**: `"type": "mcp"` (system will auto-connect to the correct MCP server)
- **Studio tools**: `"type": "studio"` (auto-managed by Spoon Studio, no manual transport config needed)

---

## 3. Extending Agents (Code-Level)

### a. Create a Custom Tool

```python
from spoon_ai.tools.base import BaseTool

class MyCustomTool(BaseTool):
    name: str = "my_custom_tool"
    description: str = "Custom tool for specific tasks"
    parameters: dict = {
        "type": "object",
        "properties": {
            "param1": {"type": "string", "description": "First parameter"},
            "param2": {"type": "integer", "description": "Second parameter"}
        },
        "required": ["param1"]
    }

    async def execute(self, param1: str, param2: int = 0) -> str:
        # Tool logic here
        return f"Processing: {param1}, {param2}"
```

### b. Create a Custom Agent

```python
from spoon_ai.agents import ToolCallAgent
from spoon_ai.tools import ToolManager
from pydantic import Field

class MyCustomAgent(ToolCallAgent):
    name: str = "my_custom_agent"
    description: str = "My custom Agent"
    max_steps: int = 5
    avaliable_tools: ToolManager = Field(default_factory=lambda: ToolManager([
        MyCustomTool(),
        # Add more tools...
    ]))
```

### c. Add/Remove Tools Dynamically

```python
agent = MyCustomAgent()
agent.add_tool(MyCustomTool())
agent.remove_tool("my_custom_tool")
```

---

## 4. Agent Execution Example

```python
import asyncio
from spoon_ai.chat import ChatBot

async def main():
    agent = MyCustomAgent(llm=ChatBot(llm_provider="openai", model_name="gpt-4.1"))
    agent.clear()  # Reset state
    response = await agent.run("What is the weather in Hong Kong?")
    print(f"Answer: {response}")

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 5. Advanced: Multi-Provider & Fallback

You can use the LLM Manager for advanced provider fallback and multi-provider logic.

```python
from spoon_ai.llm import LLMManager, ConfigurationManager

config_manager = ConfigurationManager()
llm_manager = LLMManager(config_manager)
llm_manager.set_fallback_chain(["openai", "anthropic", "gemini"])

agent = MyCustomAgent(llm=llm_manager)
```

---

## 6. Debugging & Best Practices

- Use `max_steps` to control loop length.
- Add logging in `execute()` for custom tools.
- Use `agent.clear()` to reset state between runs.
- Handle exceptions in tools to test agent fallback.
- Use `list_tools()` to see all registered tools.

---

## 7. Built-in & Example Agents

- **Built-in**: `react`, `spoon_react`, `spoon_react_mcp` (see codebase for details).
- **Examples**: See [`examples/agent/`](../examples/agent/) for runnable demos.

---

## 8. Configuration

**All agent configuration (JSON, environment variables, CLI) is documented in [`configuration.md`](./configuration.md).**
This includes: agent registration, tool config (including MCP and studio tools), LLM provider config, MCP server config, and schema reference.

---

## Next Steps

- [Configuration Guide](./configuration.md)
- [MCP Mode Usage](./mcp_mode_usage.md)
- [Graph Agent](./graph_agent.md)