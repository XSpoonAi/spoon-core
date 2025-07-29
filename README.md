# 🚀 SpoonOS Core Developer Framework(SCDF)

<div align="center">
  <img src="logo/spoon.gif" alt="SpoonAI Logo" width="200"/>
  <p><strong>Core developer framework of SpoonOS ——Agentic OS for the sentient economy. Next-Generation AI Agent Framework | Powerful Interactive CLI | Web3 infrastructure optimized Support</strong></p>
</div>

## 📘 How to Use This README

This README is your guide to getting started with the **SpoonOS Core Developer Framework (SCDF)**. It walks you through everything you need—from understanding core capabilities to actually running your own agents.

Here's how to navigate it:

- [✨ Features](#features): Start here to understand what SpoonOS can do. This section gives you a high-level overview of its agentic, composable, and interoperable architecture.

- [🔧 Installation](#installation): As of **June 2025**, SpoonOS currently supports **Python only**. This section tells you which Python version to use and how to set up a virtual environment.

- [🔐 Environment & API Key Config](#environment-variables-and-api-key-Configuration): Learn how to configure the API keys for various LLMs (e.g., OpenAI, Claude, deepseek). We also provide configuration methods for Web3 infrastructure such as chains, RPC endpoints, databases, and blockchain explorers.

- [🚀 Quick Start](#quick-start): Once your environment is ready, start calling our **MCP server**, which bundles a wide range of tools. Other servers are also available.

- [🛠️ CLI Tools](#cli-tools): This section shows how to use the CLI to run LLM-powered tasks with ease.

- [🧩 Agent Framework](#agent-framework): Learn how to create your own agents, register custom tools, and extend SpoonOS with minimal setup.

- [🔌 API Integration](#api-integration): Plug in external APIs to enhance your agent workflows.

- [🤝 Contributing](#contributing): Want to get involved? Check here for contribution guidelines.

- [📄 License](#license): Standard license information.

By the end of this README, you'll not only understand what SCDF is—but you'll be ready to build and run your own AI agents and will gain ideas on scenarios what SCDF could empower. **Have fun!**

## Features

SpoonOS is a living, evolving agentic operating system. Its SCDF is purpose-built to meet the growing demands of Web3 developers — offering a complete toolkit for building sentient, composable, and interoperable AI agents.

- **🧠 ReAct Intelligent Agent** - Advanced agent architecture combining reasoning and action
- **🔧 Custom Tool Ecosystem** - Modular tool system for easily extending agent capabilities
- **💬 Multi-Model Support** - Compatible with major large language models including OpenAI, Anthropic, DeepSeek, and more Web3 fine-tuned LLM
- **🏗️ Unified LLM Architecture** - Extensible provider system with automatic fallback, load balancing, and comprehensive monitoring
- **⚡ Prompt Caching** - Intelligent caching for Anthropic models to reduce token costs and improve response times
- **🌐 Web3-Native Interoperability** - Enables AI agents to communicate and coordinate across ecosystems via DID and ZKML-powered interoperability protocols.
- **🔌 MCP (Model Context Protocol)** – Dynamic, protocol-driven tool invocation system. Agents can discover and execute tools at runtime over `stdio`, `http`, or `websocket` transports — without hardcoding or restarts.
- **📡 Scalable Data Access** – Combined with MCP, agents gain seamless access to structured/unstructured data, including databases, Web3 RPCs, external APIs, and more.
- **💻 Interactive CLI** - Feature-rich command line interface
- **🔄 State Management** - Comprehensive session history and state persistence
- **🔗Composable Agent Logic** - Create agents that can sense, reason, plan, and execute modularly — enabling use cases across DeFi, creator economy, and more
- **🚀 Easy to Use** - Well-designed API for rapid development and integration

## ⚙️ Quick Installation

### Prerequisites

- Python 3.10+
- pip package manager (or uv as a faster alternative)

```bash
# Clone the repo
$ git clone https://github.com/XSpoonAi/spoon-core.git
$ cd spoon-core

# Create a virtual environment
$ python -m venv spoon-env
$ source spoon-env/bin/activate  # For macOS/Linux

# Install dependencies
$ pip install -r requirements.txt
```

Prefer faster install? See docs/installation.md for uv-based setup.

## 🔐 API Key & Environment Setup

Create a .env file in the root directory:

```bash
cp .env.example .env
```

Fill in your keys:

```bash
OPENAI_API_KEY=sk-your-openai-key
ANTHROPIC_API_KEY=sk-your-claude-key
DEEPSEEK_API_KEY=your-deepseek-key
PRIVATE_KEY=your-wallet-private-key
RPC_URL=https://mainnet.rpc
CHAIN_ID=12345
```

Then in your Python entry file:

```bash
from dotenv import load_dotenv
load_dotenv(override=True)
```

For advanced config methods (CLI setup, config.json, PowerShell), see docs/configuration.md.

### Configuration Model

SpoonOS uses a hybrid configuration system that combines a `.env` file for initial setup with a dynamic `config.json` for runtime settings. This provides flexibility for both static environment setup and on-the-fly adjustments via the CLI.

#### Loading Priority

The configuration is loaded with the following priority:

1.  **`config.json` (Highest Priority)**: This file is the primary source of configuration at runtime. If it exists, its values are used directly, **overriding** any corresponding environment variables set in `.env`. You can modify this file using the `config` command in the CLI.

2.  **Environment Variables (`.env`) (Lowest Priority)**: This file is used for initial setup. On the first run, if `config.json` is not found, the system will read the variables from your `.env` file to generate a new `config.json`. Any changes to `.env` after `config.json` has been created will **not** be reflected unless you delete `config.json` and restart the application.

This model ensures that sensitive keys and environment-specific settings are kept in `.env` (which should not be committed to version control), while `config.json` handles user-level customizations and runtime state.

#### `config.json` Parameters

The `config.json` file manages agent and API settings. Below are the supported parameters:

| Parameter       | Type     | Description                                                                                               | Default                               |
|-----------------|----------|-----------------------------------------------------------------------------------------------------------|---------------------------------------|
| `api_keys`      | `object` | A dictionary containing API keys for different LLM providers (e.g., `openai`, `anthropic`, `deepseek`).      | `{}`                                  |
| `base_url`      | `string` | The base URL for the API endpoint, particularly useful for custom or proxy servers like OpenRouter.           | `""`                                  |
| `default_agent` | `string` | The default agent to use for tasks.                                                                       | `"default"`                           |
| `llm_provider`  | `string` | The name of the LLM provider to use (e.g., `openai`, `anthropic`). Overrides provider detection from model name. | `"openai"`                            |
| `model_name`    | `string` | The specific model to use for the selected provider (e.g., `gpt-4.1`, `claude-sonnet-4-20250514`).           | `null`                                |

Here is an example `config.json` where a user wants to use OpenAI. You only need to provide the key for the service you intend to use.

```json
{
  "api_keys": {
    "openai": "sk-proj-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
  },
  "base_url": "https://api.openai.com/v1",
  "default_agent": "default",
  "llm_provider": "openai",
  "model_name": "gpt-4.1"
}
```

## 🏗️ Unified LLM Architecture

SpoonOS features a unified LLM infrastructure that provides seamless integration with multiple providers, automatic fallback mechanisms, and comprehensive monitoring.

### Key Benefits

- **Provider Agnostic**: Switch between OpenAI, Anthropic, Gemini, and custom providers without code changes
- **Automatic Fallback**: Built-in fallback chains ensure high availability
- **Load Balancing**: Distribute requests across multiple provider instances
- **Comprehensive Monitoring**: Request logging, performance metrics, and error tracking
- **Easy Extension**: Add new providers with minimal code

### Basic Usage

```python
from spoon_ai.llm import LLMManager, ConfigurationManager

# Initialize the LLM manager
config_manager = ConfigurationManager()
llm_manager = LLMManager(config_manager)

# Simple chat request (uses default provider)
response = await llm_manager.chat([
    {"role": "user", "content": "Hello, world!"}
])
print(response.content)

# Use specific provider
response = await llm_manager.chat(
    messages=[{"role": "user", "content": "Hello!"}],
    provider="anthropic"
)

# Chat with tools
tools = [{"name": "get_weather", "description": "Get weather info"}]
response = await llm_manager.chat_with_tools(
    messages=[{"role": "user", "content": "What's the weather?"}],
    tools=tools,
    provider="openai"
)
```

### Provider Configuration

Configure providers in your `config.json`:

```json
{
  "llm_providers": {
    "openai": {
      "api_key": "sk-your-openai-key",
      "model": "gpt-4.1",
      "max_tokens": 4096,
      "temperature": 0.3
    },
    "anthropic": {
      "api_key": "sk-ant-your-key",
      "model": "claude-sonnet-4-20250514",
      "max_tokens": 4096,
      "temperature": 0.3
    },
    "gemini": {
      "api_key": "your-gemini-key",
      "model": "gemini-2.5-pro",
      "max_tokens": 4096
    }
  },
  "llm_settings": {
    "default_provider": "openai",
    "fallback_chain": ["openai", "anthropic", "gemini"],
    "enable_monitoring": true,
    "enable_caching": true
  }
}
```

### Fallback and Load Balancing

```python
# Set up fallback chain
llm_manager.set_fallback_chain(["openai", "anthropic", "gemini"])

# The manager will automatically try providers in order if one fails
response = await llm_manager.chat([
    {"role": "user", "content": "Hello!"}
])
# If OpenAI fails, it will try Anthropic, then Gemini
```

### Custom Provider Integration

```python
from spoon_ai.llm import LLMProviderInterface, register_provider

@register_provider("custom", capabilities=["chat", "completion"])
class CustomProvider(LLMProviderInterface):
    async def initialize(self, config):
        self.api_key = config["api_key"]
        # Initialize your provider
    
    async def chat(self, messages, **kwargs):
        # Implement chat functionality
        return LLMResponse(
            content="Custom response",
            provider="custom",
            model="custom-model",
            finish_reason="stop"
        )
    
    # Implement other required methods...
```

### Monitoring and Debugging

```python
from spoon_ai.llm import get_debug_logger, get_metrics_collector

# Get monitoring instances
debug_logger = get_debug_logger()
metrics = get_metrics_collector()

# View provider statistics
stats = metrics.get_provider_stats("openai")
print(f"Success rate: {stats['success_rate']:.1f}%")
print(f"Average response time: {stats['avg_response_time']:.2f}s")

# Get recent logs
logs = debug_logger.get_recent_logs(limit=10)
for log in logs:
    print(f"{log.timestamp}: {log.provider} - {log.method}")
```

## Using OpenRouter (Multi-LLM Gateway)

```python
from spoon_ai.chat import ChatBot
from spoon_ai.agents import SpoonReactAI

# Using OpenAI's GPT-4
openai_agent = SpoonReactAI(
    llm=ChatBot(model_name="gpt-4.1", llm_provider="openai")
)

# Using Anthropic's Claude
claude_agent = SpoonReactAI(
    llm=ChatBot(model_name="claude-sonnet-4-20250514", llm_provider="anthropic")
)

# Using OpenRouter (OpenAI-compatible API)
# Uses OPENAI_API_KEY environment variable with your OpenRouter API key
openrouter_agent = SpoonReactAI(
    llm=ChatBot(
        model_name="anthropic/claude-sonnet-4",     # Model name from OpenRouter
        llm_provider="openai",                      # MUST be "openai"
        base_url="https://openrouter.ai/api/v1"     # OpenRouter API endpoint
)
)
```

## 🚀 Run the CLI

### Start the MCP Server

```bash
# Start the MCP server with all available tools
python -m spoon_ai.tools.mcp_tools_collection

# The server will start and display:
# MCP Server running on stdio transport
# Available tools: [list of tools]
```

### Start the CLI

```bash
python main.py
```

Try chatting with your agent:

```bash
> action chat
> Hello, Spoon!
```

## 🧩 Build Your Own Agent

### 1. Define Your Own Tool

```python
from spoon_ai.tools.base import BaseTool

class MyCustomTool(BaseTool):
    name: str = "my_tool"
    description: str = "Description of what this tool does"
    parameters: dict = {
        "type": "object",
        "properties": {
            "param1": {"type": "string", "description": "Parameter description"}
        },
        "required": ["param1"]
    }

    async def execute(self, param1: str) -> str:
        # Tool implementation
        return f"Result: {param1}"

```

### 2. Define Your Own Agent

```python
from spoon_ai.agents import ToolCallAgent
from spoon_ai.tools import ToolManager

class MyAgent(ToolCallAgent):
    name: str = "my_agent"
    description: str = "Agent description"
    system_prompt: str = "You are a helpful assistant..."
    max_steps: int = 5

    available_tools: ToolManager = Field(
        default_factory=lambda: ToolManager([MyCustomTool()])
    )
```

#### 3. Run the Agent and Interact via Prompt

```python
import asyncio

async def main():
    agent = MyCustomAgent(llm=ChatBot())
    result = await agent.run("Say hello to Scarlett")
    print("Result:", result)

if __name__ == "__main__":
    asyncio.run(main())
```

Register your own tools, override run(), or extend with MCP integrations. See docs/agent.md or docs/mcp_mode_usage.md

📖 [Full guide](/doc/agent.md)

📁 [Example agent](/examples/agent/my_agent_demo)

## 🔌 Advanced: Use Web3 Tools via MCP

SpoonOS supports runtime pluggable agents using the MCP (Model Context Protocol) — allowing your agent to connect to a live tool server (via SSE/WebSocket/HTTP) and call tools like get_contract_events or get_wallet_activity with no extra code.

Two ways to build MCP-powered agents:

Built-in Agent Mode: Build and run your own MCP server (e.g., mcp_thirdweb_collection.py) and connect to it using an MCPClientMixin agent.

Community Agent Mode: Use mcp-proxy to connect to open-source agents hosted on GitHub.

📁 [Full guide](/doc/mcp_mode_usage.md)

📁 [Example mcp](/examples/mcp/)

## ⚡ Prompt Caching

SpoonOS supports prompt caching for Anthropic models to reduce costs and improve performance. Enable/disable globally:

```python
from spoon_ai.chat import ChatBot

# Enable prompt caching (default: True)
chatbot = ChatBot(
    llm_provider="anthropic",
    enable_prompt_cache=True
)
```

## 🗂️ Project Structure

- [README.md](./README.md)
- [.env.example](./.env.example)
- [requirements.txt](./requirements.txt)
- [main.py](./main.py)
- [examples/](./examples)
  - [agent/](./examples/agent/) – 🧠 Agent demos ( Weather)
  - [mcp/](./examples/mcp/) – 🔌 Tool server examples
- [spoon_ai/](./spoon_ai) – 🍴 Core agent framework
- [doc/](./doc)
  - [installation.md](./doc/installation.md)
  - [configuration.md](./doc/configuration.md)
  - [openrouter.md](./doc/openrouter.md)
  - [cli.md](./do/cli.md)
  - [agent.md](./doc/agent.md)
  - [mcp_mode_usage.md](./doc/mcp_mode_usage.md)
