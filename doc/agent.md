# 🤖 Agent Development Guide (agent.md)

This guide walks you through how to define, configure, and run intelligent agents within the SpoonOS Core Developer Framework (SCDF). In SCDF, agents are autonomous reasoning entities capable of perceiving, reasoning, and acting via tool calls.

SpoonOS adopts the ReAct (Reasoning + Acting) architecture, enabling the development of composable, modular, and intelligent AI agents tailored for Web3 and beyond.

## 🧠 Agent Architecture Overview

Every agent in SCDF extends from the BaseAgent class and follows the ReAct (Reasoning + Action) loop. Common types include:

- **ToolCallAgent**: Automatically selects and calls tools based on the task.
- **SpoonReactAI**: Standard agent with built-in crypto tools (no MCP support).
- **SpoonReactMCP**: MCP-enabled agent that can connect to external services.
- **CustomAgent**: Your own implementation with custom tools, logic, or control.

### Agent Types Comparison

| Agent Type | MCP Support | Use Case | Configuration |
|------------|-------------|----------|---------------|
| `SpoonReactAI` | ❌ | Standard blockchain operations | Built-in tools only |
| `SpoonReactMCP` | ✅ | External service integration | Requires MCP server config |

### Built-in Agents

The following agents are built into the system:

| Agent Name | Aliases | Type | Description |
|------------|---------|------|-------------|
| `react` | `spoon_react` | SpoonReactAI | Standard blockchain agent |
| `spoon_react_mcp` | - | SpoonReactMCP | MCP-enabled blockchain agent |

**Note**: Additional agents can be configured in `config.json` (see configuration examples below).

## 🔄 ReAct Agent Loop

SpoonOS implements a ReAct-style agent with iterative reasoning + action capability. The lifecycle includes:

1. **Observation**: Understanding user intent and environment.
2. **Reasoning**: Analyzing and planning the next step.
3. **Acting**: Executing the chosen tool.
4. **Feedback**: Using tool results to guide further action.

This loop continues until either the goal is achieved or max_steps is reached.

## 🛠️ Step-by-Step: Build Your Own Agent

### 1. Define a Custom Tool

Creating custom tools is one of SpoonAI's most powerful features. Each tool should inherit from the `BaseTool` class:

```python
from spoon_ai.tools.base import BaseTool

class MyCustomTool(BaseTool):
    name: str = "my_custom_tool"
    description: str = "This is a custom tool for performing specific tasks"
    parameters: dict = {
        "type": "object",
        "properties": {
            "param1": {
                "type": "string",
                "description": "Description of the first parameter"
            },
            "param2": {
                "type": "integer",
                "description": "Description of the second parameter"
            }
        },
        "required": ["param1"]
    }

    async def execute(self, param1: str, param2: int = 0) -> str:
        """Implement the tool's specific logic"""
        # Implement your tool logic here
        result = f"Processing parameters: {param1}, {param2}"
        return result
```

### 2. Create Your Agent Class

```python
from spoon_ai.agents import ToolCallAgent
from spoon_ai.tools import ToolManager
from pydantic import Field

class MyCustomAgent(ToolCallAgent):
    name: str = "my_custom_agent"
    description: str = "This is my custom Agent"

    system_prompt: str = """You are an AI assistant specialized in performing specific tasks.
    You can use the provided tools to complete tasks."""

    next_step_prompt: str = "What should be the next step?"

    max_steps: int = 5

    # Define available tools
    avaliable_tools: ToolManager = Field(default_factory=lambda: ToolManager([
        MyCustomTool(),
        # Add other tools...
    ]))
```

## ⚙️ Agent Configuration via config.json

Instead of creating agents programmatically, you can configure them in `spoon-core/config.json` for use with the CLI.

### Basic Agent Configuration

```json
{
  "default_agent": "my_agent",
  "agents": {
    "my_agent": {
      "class": "SpoonReactAI",
      "aliases": ["my", "custom"],
      "description": "My custom agent",
      "config": {
        "max_steps": 10,
        "tool_choice": "auto"
      },
      "tools": ["crypto_tools"]
    }
  }
}
```

### MCP-Enabled Agent Configuration Example

Here's how to configure a custom agent with external service integration:

```json
{
  "agents": {
    "my_search_agent": {
      "class": "SpoonReactMCP",
      "aliases": ["search", "web"],
      "description": "Custom agent with web search capabilities",
      "config": {
        "max_steps": 20,
        "tool_choice": "auto"
      },
      "mcp_servers": ["tavily-mcp"],
      "tools": ["web_search", "crypto_tools"]
    }
  },
  "mcp_servers": {
    "tavily-mcp": {
      "transport": "npx",
      "command": "npx",
      "args": ["-y", "tavily-mcp"],
      "env": {
        "TAVILY_API_KEY": "your-api-key"
      },
      "disabled": false
    }
  },
  "tool_sets": {
    "web_search": {
      "type": "mcp_server",
      "server": "tavily-mcp",
      "enabled": true
    },
    "crypto_tools": {
      "type": "builtin",
      "enabled": true
    }
  }
}
```

### Configuration Parameters

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `class` | string | Agent type: "SpoonReactAI" or "SpoonReactMCP" | `"SpoonReactMCP"` |
| `aliases` | array | Alternative names for loading the agent | `["search", "web"]` |
| `description` | string | Human-readable description | `"Web search agent"` |
| `config.max_steps` | integer | Maximum reasoning steps (default: 10) | `20` |
| `config.tool_choice` | string | Tool selection mode: "auto", "required", "none" | `"auto"` |
| `mcp_servers` | array | MCP servers to connect to (MCP agents only) | `["tavily-mcp"]` |
| `tools` | array | Tool sets to enable | `["web_search", "crypto_tools"]` |

### MCP Server Configuration

| Transport | Use Case | Required Fields | Example |
|-----------|----------|-----------------|---------|
| `npx` | Node.js packages | `command`, `args` | Tavily search |
| `python` | Python scripts | `script_path` | Custom servers |
| `sse` | HTTP streaming | `url` | Real-time data |
| `websocket` | Real-time | `url` | Live connections |

### Using Configured Agents

Once configured, custom agents will appear in the CLI:

```bash
# List available agents (includes built-in + configured)
> list-agents

# Load built-in agent
> load-agent react

# Load custom configured agent by name
> load-agent my_search_agent

# Load custom agent by alias
> load-agent search

# Start conversation
> action chat
```

## 🧪 Run the Agent

```python
import asyncio
from spoon_ai.chat import ChatBot
async def main():
    # Create an InfoAssistantAgent instance
    info_agent = MyCustomAgent(llm=ChatBot(llm_provider="openai",model_name="anthropic/claude-sonnet-4", base_url="https://openrouter.ai/api/v1"))

    # Reset the Agent state
    info_agent.clear()

    response = await info_agent.run("What is the weather like in hongkong today xxxxx?")
    print(f"Answer: {response}\n")

if __name__ == "__main__":
    asyncio.run(main())
```

## 🛠️ Agent Debugging Tips

- Use max_steps to control reasoning loop length.
- Add print() or logging in execute() for custom tools.
- Raise exceptions on tool misuse to test agent fallback.
- Use agent.clear() to reset memory/state between runs.

#### 📁 Examples Directory

The following directory contains runnable examples:

```yaml
examples/
├── agents/
│   └── my_agent_demo.py
```

## ✅ Next Steps

Now that you understand agents:

- 🧪 [Try running agents via CLI](./cli.md)
- 🌐 [Integrate Web3 tools with MCP](./mcp_mode_usage.md)
