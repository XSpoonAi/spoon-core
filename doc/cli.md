# 🛠️ SpoonOS CLI Usage Guide

SCDF CLI is a powerful command-line tool that provides rich functionality, including interacting with AI agents, managing chat history, processing cryptocurrency transactions, and loading documents.

## 📦 Prerequisite: Start the MCP Server

Before starting the CLI, make sure the MCP (Message Connectivity Protocol) server is running:

```bash
python -m spoon_ai.tools.mcp_tools_collection
```

## 🚀 Start the CLI

Once the MCP server is running, launch the CLI:

```bash
python main.py
```

## 📋 Available Agents

The CLI includes these built-in agents:

| Agent | Aliases | Type | MCP Support | Description |
|-------|---------|------|-------------|-------------|
| `react` | `spoon_react` | SpoonReactAI | ❌ | Standard blockchain analysis agent |
| `spoon_react_mcp` | - | SpoonReactMCP | ✅ | MCP-enabled blockchain agent |

**Note**: Additional agents can be configured in `config.json` (see examples below).

### Loading Agents

```bash
# List all available agents
> list-agents

# Load built-in agent by name
> load-agent react
> load-agent spoon_react_mcp

# Load agent by alias
> load-agent spoon_react

# Check currently loaded agent
> list-agents
```

## Basic Commands

| Command             | Aliases           | Description                                                                                                               |
| ------------------- | ----------------- | ------------------------------------------------------------------------------------------------------------------------- |
| `help`              | `h`, `?`          | Display help information                                                                                                  |
| `exit`              | `quit`, `q`       | Exit the CLI                                                                                                              |
| `load-agent <name>` | `load`            | Load an agent with the specified name                                                                                     |
| `list-agents`       | `agents`          | List all available agents                                                                                                 |
| `config`            | `cfg`, `settings` | Configure settings (such as API keys)                                                                                     |
| `reload-config`     | `reload`          | Reload the current agent's configuration                                                                                  |
| `action <action>`   | `a`               | Perform a specific action using the current agent. For example, `action react` to start a step-by-step reasoning session. |

### Chat Management Commands

| Command          | Aliases | Description                         |
| ---------------- | ------- | ----------------------------------- |
| `new-chat`       | `new`   | Start a new chat (clear history)    |
| `list-chats`     | `chats` | List available chat history records |
| `load-chat <ID>` | -       | Load a specific chat history record |

### Cryptocurrency-Related Commands

| Command                                       | Aliases  | Description                            |
| --------------------------------------------- | -------- | -------------------------------------- |
| `transfer <address> <amount> <token>`         | `send`   | Transfer tokens to a specified address |
| `swap <source_token> <target_token> <amount>` | -        | Exchange tokens using an aggregator    |
| `token-info <address>`                        | `token`  | Get token information by address       |
| `token-by-symbol <symbol>`                    | `symbol` | Get token information by symbol        |

### Document Management Commands

| Command                      | Aliases | Description                                                      |
| ---------------------------- | ------- | ---------------------------------------------------------------- |
| `load-docs <directory_path>` | `docs`  | Load documents from the specified directory to the current agent |

## ⚙️ Configuration Management

### Environment Variables

Set required API keys before starting:

```bash
# Required for web search (Tavily MCP)
export TAVILY_API_KEY="your-tavily-api-key"

# Required for LLM functionality
export OPENAI_API_KEY="your-openai-api-key"

# Optional: Debug mode
export DEBUG=true
export LOG_LEVEL=debug
```

### Agent Configuration

Custom agents can be configured in `spoon-core/config.json`. The basic structure is:

```json
{
  "default_agent": "react",
  "agents": {
    "my_custom_agent": {
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

For MCP-enabled agents, additional configuration is needed (see examples below).

### MCP Server Types

| Transport | Use Case | Configuration Example |
|-----------|----------|----------------------|
| `npx` | Node.js packages | `"command": "npx", "args": ["-y", "package-name"]` |
| `python` | Python scripts | `"script_path": "server.py", "args": ["--port", "8766"]` |
| `sse` | HTTP streaming | `"url": "http://localhost:8765/sse"` |
| `websocket` | Real-time | `"url": "ws://localhost:8765/ws"` |

### Configuration Commands

```bash
# Reload configuration after changes
> reload-config

# Check current configuration
> config

# Update specific settings
> config API_KEY your-new-key
```

## CLI Usage Examples

#### Configure Settings

1. View current configuration:

```
> config
Current configuration:
API_KEY: sk-***********
MODEL: gpt-4
...
```

2. Modify configuration:

```
> config API_KEY sk-your-new-api-key
API_KEY updated
```

#### Basic Interaction

1. Start a new chat:

```
> action chat
New chat session started
```

3. Directly input text to interact with the AI agent:

```
> Hello, please introduce yourself
[AI reply will be displayed here]
```

#### Cryptocurrency Operations

1. View token information:

```
> token-by-symbol SPO
Token information:
Name: SpoonOS not a meme
Symbol:SPO
Address: 0x...
Decimals: 18
...
```

2. Transfer operation:

```
> transfer 0x123... 0.1 SPO
Preparing to transfer 0.1 SPO to 0x123...
[Transfer details will be displayed here]
```

## 🔧 Troubleshooting

### Common Issues

| Issue | Symptoms | Solution |
|-------|----------|----------|
| **Agent not found** | `Agent 'name' not found` | Use `list-agents` to check available agents, verify spelling |
| **MCP connection failed** | `Failed to create transport` | Check API keys, verify MCP server is running |
| **Tool not available** | `Tool 'name' not found` | Check `tool_sets` config, verify MCP server enabled |
| **Duplicate agents** | Same agent appears twice | Fixed in latest version, restart if needed |
| **API key missing** | Authentication errors | Set environment variables: `TAVILY_API_KEY`, `OPENAI_API_KEY` |

### Debug Mode

Enable detailed logging:

```bash
export DEBUG=true
export LOG_LEVEL=debug
python main.py
```

### Configuration Validation

1. **Check JSON syntax**: Use a JSON validator for `config.json`
2. **Verify required fields**: Ensure all required parameters are present
3. **Test MCP servers**: Verify external services are accessible
4. **Check environment variables**: Confirm all API keys are set

### Example: Custom Search Agent Configuration

Here's how to configure a custom search agent with MCP integration:

```json
{
  "default_agent": "my_search_agent",
  "agents": {
    "my_search_agent": {
      "class": "SpoonReactMCP",
      "aliases": ["search", "web"],
      "description": "Custom search agent with web capabilities",
      "mcp_servers": ["tavily-mcp"],
      "tools": ["web_search", "crypto_tools"]
    }
  },
  "mcp_servers": {
    "tavily-mcp": {
      "transport": "npx",
      "command": "npx",
      "args": ["-y", "tavily-mcp"],
      "env": { "TAVILY_API_KEY": "your-key" }
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

After adding this configuration, the agent will appear in `list-agents` and can be loaded with:
```bash
> load-agent my_search_agent
> load-agent search  # using alias
```

## ✅ Next Steps

To extend CLI usage:

- 🤖 [Explore agent capabilities](./agent.md)
- 🌐 [Use Web3 tools via MCP](./mcp_mode_usage.md)
