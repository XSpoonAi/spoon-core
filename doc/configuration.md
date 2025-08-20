# 🔐 SpoonOS Configuration Guide

This guide provides a comprehensive overview of configuring SpoonOS, including unified tool and agent configuration, environment variables, LLM provider setup, MCP server integration, and best practices for secure, maintainable deployments.

---

## Overview

SpoonOS uses a **unified configuration system** that consolidates all agent, tool, and provider settings in a single place. This enables:

- **Single Configuration Point**: All settings per agent, tool, and provider are centralized.
- **Embedded MCP Server Support**: MCP server configuration is included directly in tool/agent definitions.
- **Automatic Lifecycle Management**: MCP servers start/stop automatically based on tool usage.
- **Type Safety & Validation**: Strong validation for all configuration options.
- **Environment Variable Integration**: Tool- and server-specific environment variables can be set directly in config or via `.env`.
- **Priority-based Configuration**: Tool-level env vars override system env vars for flexible management.

---

## 1. 🧾 Recommended: .env File

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Edit and fill in your credentials:

```bash
# LLM APIs
OPENAI_API_KEY=sk-your-openai-key
ANTHROPIC_API_KEY=sk-your-anthropic-key
DEEPSEEK_API_KEY=your-deepseek-key

# Blockchain
PRIVATE_KEY=your-wallet-private-key
RPC_URL=https://mainnet.rpc
CHAIN_ID=12345

# Tool-specific environment variables
TAVILY_API_KEY=your-tavily-api-key
BRAVE_API_KEY=your-brave-search-key
GITHUB_TOKEN=your-github-token

# Built-in tool environment variables
OKX_API_KEY=your_okx_api_key
OKX_SECRET_KEY=your_okx_secret_key
OKX_API_PASSPHRASE=your_okx_api_passphrase
OKX_PROJECT_ID=your_okx_project_id
CHAINBASE_API_KEY=your_chainbase_api_key
THIRDWEB_CLIENT_ID=your_thirdweb_client_id
BITQUERY_API_KEY=your_bitquery_api_key
```

Load it at the top of your Python entry file:

```python
from dotenv import load_dotenv
load_dotenv(override=True)
```

---

## 2. 💻 Shell Environment Variables

**Linux/macOS:**

```bash
export OPENAI_API_KEY="sk-your-openai-api-key"
export ANTHROPIC_API_KEY="sk-ant-your-anthropic-api-key"
export DEEPSEEK_API_KEY="your-deepseek-api-key"
export PRIVATE_KEY="your-wallet-private-key"
```

**Windows (PowerShell):**

```powershell
$env:OPENAI_API_KEY="sk-your-openai-api-key"
$env:ANTHROPIC_API_KEY="sk-ant-your-anthropic-api-key"
$env:DEEPSEEK_API_KEY="your-deepseek-api-key"
$env:PRIVATE_KEY="your-wallet-private-key"
```

---

## 3. 🧪 CLI Configuration Commands

After starting the CLI, use the `config` command:

```bash
python main.py

> config api_key openai sk-your-openai-api-key
> config api_key anthropic sk-ant-your-anthropic-api-key
> config api_key deepseek your-deepseek-api-key
> config PRIVATE_KEY your-wallet-private-key
> config
```

---

## 4. 📁 Unified Configuration File (`config.json`)

All settings can be consolidated in `config.json`. This includes API keys, agents, tools, providers, MCP servers, and global LLM settings.

### Example: Complete Configuration

```json
{
  "api_keys": {
    "openai": "sk-your-openai-api-key",
    "anthropic": "sk-ant-your-anthropic-api-key",
    "deepseek": "your-deepseek-api-key"
  },
  "default_agent": "web_researcher",
  "providers": {
    "openai": {
      "api_key": "sk-your-openai-key",
      "model": "gpt-4.1",
      "max_tokens": 4096,
      "temperature": 0.3,
      "timeout": 30,
      "retry_attempts": 3
    },
    "anthropic": {
      "api_key": "sk-ant-your-key",
      "model": "claude-3-5-sonnet-20241022",
      "max_tokens": 4096,
      "temperature": 0.3,
      "timeout": 30,
      "retry_attempts": 3
    }
  },
  "llm_settings": {
    "default_provider": "openai",
    "fallback_chain": ["openai", "anthropic"],
    "enable_monitoring": true,
    "enable_caching": true,
    "enable_debug_logging": false,
    "max_concurrent_requests": 10
  },
  "agents": {
    "web_researcher": {
      "class": "SpoonReactMCP",
      "description": "Agent with web search and analysis capabilities",
      "aliases": ["researcher", "web"],
      "config": {
        "max_steps": 10,
        "tool_choice": "auto"
      },
      "tools": [
        {
          "name": "web_search",
          "type": "mcp",
          "description": "Web search via Tavily API",
          "enabled": true,
          "mcp_server": {
            "command": "npx",
            "args": ["-y", "@tavily/mcp-server"],
            "env": {
              "TAVILY_API_KEY": "your-tavily-api-key"
            },
            "disabled": false,
            "autoApprove": ["search", "get_content"],
            "timeout": 30,
            "retry_attempts": 3
          },
          "config": {
            "max_results": 10,
            "include_raw_content": true
          }
        },
        {
          "name": "crypto_powerdata_cex",
          "type": "builtin",
          "description": "Crypto PowerData CEX market data tool",
          "enabled": true,
          "env": {
            "OKX_API_KEY": "your_okx_api_key",
            "OKX_SECRET_KEY": "your_okx_secret_key",
            "OKX_API_PASSPHRASE": "your_okx_api_passphrase",
            "OKX_PROJECT_ID": "your_okx_project_id"
          },
          "config": {
            "timeout": 30,
            "max_retries": 3
          }
        }
      ]
    }
  }
}
```

---

## 5. 🧠 Agent Configuration

### Agent Definition

Each agent is defined in the `agents` section. Key fields:

| Field         | Type    | Required | Description                                      |
|---------------|---------|----------|--------------------------------------------------|
| `class`       | string  | Yes      | Agent class name (`SpoonReactAI`, `SpoonReactMCP`)|
| `description` | string  | No       | Agent description                                |
| `aliases`     | array   | No       | List of agent aliases                            |
| `config`      | object  | No       | Agent-specific configuration parameters          |
| `tools`       | array   | No       | List of tool configurations                      |

#### Example

```json
"agents": {
  "custom_react": {
    "class": "SpoonReactAI",
    "aliases": ["custom", "my_react"],
    "description": "Custom configured SpoonReact agent",
    "config": {
      "max_steps": 15,
      "tool_choice": "auto",
      "llm_provider": "openai"
    }
  }
}
```

#### Supported Agent Classes

- **SpoonReactAI**: Basic ReAct agent
- **SpoonReactMCP**: Advanced agent with MCP protocol support

#### Agent Configuration Parameters

- `max_steps`: Maximum execution steps
- `tool_choice`: Tool selection strategy (`auto`, `required`, `none`)
- `llm_provider`: Specific LLM provider for this agent
- `llm_model`, `llm_temperature`, `enable_fallback`, etc.

---

## 6. 🔧 Tool Configuration

### Tool Types

- **builtin**: Built-in tool collections (e.g., crypto_tools)
- **mcp**: Tools provided by MCP servers

### Tool Configuration Schema

#### Built-in Tool

```json
{
  "name": "crypto_powerdata_cex",
  "type": "builtin",
  "description": "Crypto PowerData CEX market data",
  "enabled": true,
  "env": {
    "OKX_API_KEY": "your_okx_api_key",
    "OKX_SECRET_KEY": "your_okx_secret_key",
    "OKX_API_PASSPHRASE": "your_okx_api_passphrase",
    "OKX_PROJECT_ID": "your_okx_project_id"
  },
  "config": {
    "timeout": 30,
    "max_retries": 3
  }
}
```

#### MCP Tool

```json
{
  "name": "web_search",
  "type": "mcp",
  "description": "Web search via Tavily",
  "enabled": true,
  "mcp_server": {
    "command": "npx",
    "args": ["-y", "tavily-mcp"],
    "env": {
      "TAVILY_API_KEY": "your-tavily-api-key"
    },
    "autoApprove": ["search", "get_content"],
    "timeout": 30,
    "retry_attempts": 3
  },
  "config": {
    "max_results": 10,
    "include_raw_content": true
  }
}
```

#### Environment Variable Precedence

- Tool-level `env` overrides server-level `env`, which overrides system environment variables.

---

## 7. 🏗️ LLM Provider Configuration

SpoonOS supports multiple LLM providers with a flexible configuration and smart fallback logic.

### Provider Configuration Schema

```json
"providers": {
  "openai": {
    "api_key": "sk-your-openai-key",
    "model": "gpt-4.1",
    "max_tokens": 4096,
    "temperature": 0.3,
    "timeout": 30,
    "retry_attempts": 3
  }
}
```

| Option           | Type    | Required | Description                        | Default         |
|------------------|---------|----------|------------------------------------|-----------------|
| `api_key`        | string  | Yes      | Provider API key                   | None            |
| `model`          | string  | No       | Model name to use                  | Provider default|
| `max_tokens`     | integer | No       | Maximum tokens per request         | 4096            |
| `temperature`    | float   | No       | Response randomness (0.0-1.0)      | Provider default|
| `timeout`        | integer | No       | Request timeout in seconds         | 30              |
| `retry_attempts` | integer | No       | Number of retry attempts           | 3               |
| `base_url`       | string  | No       | Custom API endpoint                | Provider default|
| `custom_headers` | object  | No       | Additional HTTP headers            | {}              |

### Global LLM Settings

```json
"llm_settings": {
  "default_provider": "openai",
  "fallback_chain": ["openai", "anthropic"],
  "enable_monitoring": true,
  "enable_caching": true,
  "enable_debug_logging": false,
  "max_concurrent_requests": 10
}
```

---

## 8. 🚀 Agent Management & Usage

### Listing Agents

```bash
> list-agents
Available agents:
  react (aliases: spoon_react): A smart ai agent in neo blockchain
  spoon_react (aliases: react): A smart ai agent in neo blockchain
  spoon_react_mcp: SpoonReact agent with MCP support
  custom_react (aliases: custom, my_react): Custom configured SpoonReact agent
  search_agent (aliases: search, tavily): Search agent with Tavily MCP integration

Currently loaded agents:
  spoon_react: A smart ai agent in neo blockchain
```

### Loading Agents

By name or alias:

```bash
> load-agent search_agent
Loaded agent: spoon_react

> load-agent search
Loaded agent: spoon_react
```

### Error Handling

If an agent doesn't exist, the system will display all available agents:

```bash
> load-agent nonexistent
Agent nonexistent not found
Available agents:
  react (aliases: spoon_react): A smart ai agent in neo blockchain
  search_agent (aliases: search, tavily): Search agent with Tavily MCP integration
```

---

## 9. 🛠️ Configuration Management & Troubleshooting

### Viewing Current Configuration

```bash
> config
Current configuration:
API Keys:
  openai: sk-o...xxxx
  anthropic: Not set
  deepseek: Not set
base_url: https://openrouter.ai/api/v1
default_agent: spoon_react
agents: [object Object]
```

### Modifying Configuration

```bash
> config default_agent search_agent
Set default_agent = search_agent
```

### Troubleshooting

#### Common Issues

- **Configuration File Not Found**: Defaults are used. Create `config.json` to customize.
- **Agent Loading Failed**: Check class name, config syntax, required fields, MCP server accessibility.
- **Alias Conflicts**: First matching agent is used.
- **MCP Server Connection Issues**: Check command, env vars, network, dependencies.
- **LLM Provider Issues**: Ensure provider is configured, API key is valid, model name is supported.
- **Fallback Chain Issues**: Verify all providers in chain are configured and healthy.
- **Performance Issues**: Adjust timeout, reduce max_tokens, enable caching.
- **Debugging**: Enable debug logging in `llm_settings`.

#### Example: Debug Logging

```json
"llm_settings": {
  "enable_debug_logging": true
}
```

#### Example: Validate Configuration

```bash
python -c "import json; print('Valid JSON' if json.load(open('config.json')) else 'Invalid JSON')"
> config validate
```

---

## 10. 🛡️ Security Best Practices

1. **Never commit API keys to version control**
   - Add `.env` to `.gitignore`
2. **Use environment variables in production**
   - Avoid hardcoding keys in source code
3. **Wallet private key security**
   - NEVER share your private key
   - Store in secure environment variables only
4. **API key rotation**
   - Rotate keys regularly, monitor usage, use restrictions
5. **Restrictive file permissions**
   - `chmod 600 .env`
6. **Use dedicated wallets for testing**

---

## 11. 📋 Configuration Schema Reference

### Agent Configuration Schema

```json
"agent_name": {
  "class": "SpoonReactAI | SpoonReactMCP",
  "description": "string (optional)",
  "aliases": ["array", "of", "strings"] (optional),
  "config": {
    "max_steps": "integer (default: 10)",
    "tool_choice": "string (default: 'auto')"
  },
  "tools": [
    // Array of tool configurations (see below)
  ]
}
```

### Tool Configuration Schema

#### Built-in Tool

```json
{
  "name": "string (required, unique per agent)",
  "type": "builtin",
  "description": "string (optional)",
  "enabled": "boolean (default: true)",
  "env": {
    "KEY": "value"
  } (optional),
  "config": {
    // Tool-specific configuration options
  }
}
```

#### MCP Tool

```json
{
  "name": "string (required, unique per agent)",
  "type": "mcp",
  "description": "string (optional)",
  "enabled": "boolean (default: true)",
  "env": {
    "KEY": "value"
  } (optional),
  "mcp_server": {
    "command": "string (required)",
    "args": ["array", "of", "strings"] (optional),
    "env": {
      "KEY": "value"
    } (optional),
    "cwd": "string (optional)",
    "disabled": "boolean (default: false)",
    "autoApprove": ["array", "of", "tool", "names"] (optional),
    "timeout": "integer (default: 30)",
    "retry_attempts": "integer (default: 3)"
  },
  "config": {
    // Tool-specific configuration options
  } (optional)
}
```

---

---

## 12. 🔌 MCP Tool Integration Guide

SpoonOS supports two main types of MCP (Model Context Protocol) tools for integrating external services and APIs:

### MCP Tool Types Overview

| Tool Type   | Config `"type"` | Transport   | How to Use/Connect                | Example Use Case                |
|-------------|-----------------|-------------|-----------------------------------|---------------------------------|
| Stdio MCP   | `"mcp"`         | `"stdio"`   | Auto-managed subprocess           | Tavily, GitHub, Brave, etc.     |
| SSE MCP     | `"mcp"`         | `"sse"`     | Start your own SSE server         | Custom Web3, in-house tools     |
| Built-in    | `"builtin"`     | N/A         | Provided by SpoonOS               | Crypto tools, price data, etc.  |

### Recommended: Stdio MCP Tools

**Stdio tools** are MCP tools that run as a subprocess and communicate with SpoonOS via stdin/stdout. They are the recommended approach for most use cases because:

- No need to run or configure any server manually
- Auto-managed lifecycle by SpoonOS
- Always up to date with latest tool versions
- Automatic restart on failures

#### Stdio Tool Configuration

```json
{
  "name": "tavily_search",
  "type": "mcp",
  "mcp_server": {
    "command": "npx",
    "args": ["-y", "tavily-mcp"],
    "transport": "stdio",
    "env": {
      "TAVILY_API_KEY": "your-tavily-api-key"
    },
    "timeout": 30,
    "retry_attempts": 3
  }
}
```

#### Common Stdio MCP Tools

```json
{
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
    },
    {
      "name": "brave_search",
      "type": "mcp",
      "mcp_server": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-brave-search"],
        "transport": "stdio"
      }
    }
  ]
}
```

### Advanced: SSE MCP Tools

**SSE tools** are MCP tools that run as separate servers and communicate via Server-Sent Events (SSE). Use these for:

- Custom or in-development integrations
- Tools that need to run independently
- Legacy MCP servers

#### SSE Tool Configuration

```json
{
  "name": "my_custom_tool",
  "type": "mcp",
  "mcp_server": {
    "endpoint": "http://127.0.0.1:8765/sse",
    "transport": "sse",
    "timeout": 30,
    "retry_attempts": 3
  }
}
```

#### Starting SSE Servers

You must manually start SSE servers before using them:

```bash
# Example: Custom FastMCP server
python my_mcp_server.py

# Example: GitHub MCP server as SSE
npx -y @modelcontextprotocol/server-github --sse
```

### Mixed Tool Configuration Example

You can combine Stdio, SSE, and built-in tools in a single agent:

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
            "transport": "stdio",
            "env": {
              "TAVILY_API_KEY": "your-tavily-api-key"
            }
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
        {
          "name": "crypto_tools",
          "type": "builtin"
        }
      ]
    }
  }
}
```

### MCP Tool Best Practices

1. **Prefer Stdio MCP tools** for most use cases—no server management, always up to date, auto-restart
2. Use `"sse"` tools only for custom or in-development integrations
3. You can mix Stdio, SSE, and built-in tools in a single agent
4. All tool configuration is managed in `config.json`—no need to modify code for tool integration
5. Use environment variables for API keys and sensitive configuration
6. Set appropriate timeouts and retry attempts for reliability

### MCP Tool Troubleshooting

- **Stdio tool not available**: Check the command and args in your config
- **SSE tool connection failed**: Ensure the server is running and the endpoint is correct
- **Environment variables not loaded**: Verify env vars are set in tool config or system environment
- **Tool timeout issues**: Increase timeout values in mcp_server config
- **Permission errors**: Check file permissions for command executables

---

## Next Steps

- [Agent Development Guide](./agent.md)
- [MCP Protocol Documentation](https://modelcontextprotocol.io/)