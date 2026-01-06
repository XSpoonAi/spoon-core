# Noise Document (干扰文档)

> 目的：用于测试 retriever 的 score threshold / “低相关不返回垃圾”策略。
> 该文档刻意包含大量“看起来像技术关键词”的词：port/config/alpha/beta/handshake/cache/retry… 但不包含任何 CANARY_5xx/6xx 或 NOT_IN_CORPUS_CANARY_999。

关键词堆叠（无答案）：
- alpha alpha alpha, beta beta beta
- port 8080, port 9090, config file, config loader, retry policy, cache invalidation
- handshake timeout, exponential backoff, circuit breaker, rate limit, log sampling
- table schema, markdown code block, python snippet, function return, unit test, assertion


- alpha alpha alpha, beta beta beta
- port 8080, port 9090, config file, config loader, retry policy, cache invalidation
- handshake timeout, exponential backoff, circuit breaker, rate limit, log sampling
- table schema, markdown code block, python snippet, function return, unit test, assertion

- alpha alpha alpha, beta beta beta
- port 8080, port 9090, config file, config loader, retry policy, cache invalidation
- handshake timeout, exponential backoff, circuit breaker, rate limit, log sampling
- table schema, markdown code block, python snippet, function return, unit test, assertion

- alpha alpha alpha, beta beta beta
- port 8080, port 9090, config file, config loader, retry policy, cache invalidation
- handshake timeout, exponential backoff, circuit breaker, rate limit, log sampling
- table schema, markdown code block, python snippet, function return, unit test, assertion
- - alpha alpha alpha, beta beta beta
- port 8080, port 9090, config file, config loader, retry policy, cache invalidation
- handshake timeout, exponential backoff, circuit breaker, rate limit, log sampling
- table schema, markdown code block, python snippet, function return, unit test, assertion
- - alpha alpha alpha, beta beta beta
- port 8080, port 9090, config file, config loader, retry policy, cache invalidation
- handshake timeout, exponential backoff, circuit breaker, rate limit, log sampling
- table schema, markdown code block, python snippet, function return, unit test, assertion
- - alpha alpha alpha, beta beta beta
- port 8080, port 9090, config file, config loader, retry policy, cache invalidation
- handshake timeout, exponential backoff, circuit breaker, rate limit, log sampling
- table schema, markdown code block, python snippet, function return, unit test, assertion
- 

伪代码（无 canary）：
```python
def calc(x):
    return x + 1
```

表格（无 canary）：
| key   | value   |
| ----- | ------- |
| alpha | maybe   |
| beta  | maybe   |
| port  | unknown |



关键词堆叠（无答案）：
- alpha alpha alpha, beta beta beta
- port 8080, port 9090, config file, config loader, retry policy, cache invalidation
- handshake timeout, exponential backoff, circuit breaker, rate limit, log sampling
- table schema, markdown code block, python snippet, function return, unit test, assertion


- alpha alpha alpha, beta beta beta
- port 8080, port 9090, config file, config loader, retry policy, cache invalidation
- handshake timeout, exponential backoff, circuit breaker, rate limit, log sampling
- table schema, markdown code block, python snippet, function return, unit test, assertion

- alpha alpha alpha, beta beta beta
- port 8080, port 9090, config file, config loader, retry policy, cache invalidation
- handshake timeout, exponential backoff, circuit breaker, rate limit, log sampling
- table schema, markdown code block, python snippet, function return, unit test, assertion

- alpha alpha alpha, beta beta beta
- port 8080, port 9090, config file, config loader, retry policy, cache invalidation
- handshake timeout, exponential backoff, circuit breaker, rate limit, log sampling
- table schema, markdown code block, python snippet, function return, unit test, assertion
- - alpha alpha alpha, beta beta beta
- port 8080, port 9090, config file, config loader, retry policy, cache invalidation
- handshake timeout, exponential backoff, circuit breaker, rate limit, log sampling
- table schema, markdown code block, python snippet, function return, unit test, assertion
- - alpha alpha alpha, beta beta beta
- port 8080, port 9090, config file, config loader, retry policy, cache invalidation
- handshake timeout, exponential backoff, circuit breaker, rate limit, log sampling
- table schema, markdown code block, python snippet, function return, unit test, assertion
- - alpha alpha alpha, beta beta beta
- port 8080, port 9090, config file, config loader, retry policy, cache invalidation
- handshake timeout, exponential backoff, circuit breaker, rate limit, log sampling
- table schema, markdown code block, python snippet, function return, unit test, assertion



# 🚀 SpoonOS Core Developer Framework(SCDF)

<div align="center">
  <img src="logo/spoon.gif" alt="SpoonAI Logo" width="200"/>
  <p><strong>Core developer framework of SpoonOS ——Agentic OS for the sentient economy. Next-Generation AI Agent Framework | Powerful Interactive CLI | Web3 infrastructure optimized Support</strong></p>
</div>

<div align="center">
<a href="https://deepwiki.com/XSpoonAi/spoon-core"><img src="https://deepwiki.com/badge.svg" alt="Ask DeepWiki"></a>
<a href="https://zread.ai/XSpoonAi/spoon-core"><img src="https://img.shields.io/badge/Ask_Zread-_.svg?style=flat&color=00b0aa&labelColor=000000&logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB3aWR0aD0iMTYiIGhlaWdodD0iMTYiIHZpZXdCb3g9IjAgMCAxNiAxNiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTQuOTYxNTYgMS42MDAxSDIuMjQxNTZDMS44ODgxIDEuNjAwMSAxLjYwMTU2IDEuODg2NjQgMS42MDE1NiAyLjI0MDFWNC45NjAxQzEuNjAxNTYgNS4zMTM1NiAxLjg4ODEgNS42MDAxIDIuMjQxNTYgNS42MDAxSDQuOTYxNTZDNS4zMTUwMiA1LjYwMDEgNS42MDE1NiA1LjMxMzU2IDUuNjAxNTYgNC45NjAxVjIuMjQwMUM1LjYwMTU2IDEuODg2NjQgNS4zMTUwMiAxLjYwMDEgNC45NjE1NiAxLjYwMDFaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00Ljk2MTU2IDEwLjM5OTlIMi4yNDE1NkMxLjg4ODEgMTAuMzk5OSAxLjYwMTU2IDEwLjY4NjQgMS42MDE1NiAxMS4wMzk5VjEzLjc1OTlDMS42MDE1NiAxNC4xMTM0IDEuODg4MSAxNC4zOTk5IDIuMjQxNTYgMTQuMzk5OUg0Ljk2MTU2QzUuMzE1MDIgMTQuMzk5OSA1LjYwMTU2IDE0LjExMzQgNS42MDE1NiAxMy43NTk5VjExLjAzOTlDNS42MDE1NiAxMC42ODY0IDUuMzE1MDIgMTAuMzk5OSA0Ljk2MTU2IDEwLjM5OTlaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik0xMy43NTg0IDEuNjAwMUgxMS4wMzg0QzEwLjY4NSAxLjYwMDEgMTAuMzk4NCAxLjg4NjY0IDEwLjM5ODQgMi4yNDAxVjQuOTYwMUMxMC4zOTg0IDUuMzEzNTYgMTAuNjg1IDUuNjAwMSAxMS4wMzg0IDUuNjAwMUgxMy43NTg0QzE0LjExMTkgNS42MDAxIDE0LjM5ODQgNS4zMTM1NiAxNC4zOTg0IDQuOTYwMVYyLjI0MDFDMTQuMzk4NCAxLjg4NjY0IDE0LjExMTkgMS42MDAxIDEzLjc1ODQgMS42MDAxWiIgZmlsbD0iI2ZmZiIvPgo8cGF0aCBkPSJNNCAxMkwxMiA0TDQgMTJaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00IDEyTDEyIDQiIHN0cm9rZT0iI2ZmZiIgc3Ryb2tlLXdpZHRoPSIxLjUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPgo8L3N2Zz4K&logoColor=ffffff" alt="Ask Zread"></a>
</div>

<div align="center">
<!-- Keep these links. Translations will automatically update with the README. -->
<a href="https://zdoc.app/de/XSpoonAi/spoon-core">Deutsch</a> |
<a href="https://zdoc.app/en/XSpoonAi/spoon-core">English</a> |
<a href="https://zdoc.app/es/XSpoonAi/spoon-core">Español</a> |
<a href="https://zdoc.app/fr/XSpoonAi/spoon-core">français</a> |
<a href="https://zdoc.app/ja/XSpoonAi/spoon-core">日本語</a> |
<a href="https://zdoc.app/ko/XSpoonAi/spoon-core">한국어</a> |
<a href="https://zdoc.app/pt/XSpoonAi/spoon-core">Português</a> |
<a href="https://zdoc.app/ru/XSpoonAi/spoon-core">Русский</a> |
<a href="https://zdoc.app/zh/XSpoonAi/spoon-core">中文</a>
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

- [📊 Enhanced Graph System](#enhanced-graph-system): Discover the powerful graph-based workflow orchestration system for complex AI agent workflows.

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
- **📊 Enhanced Graph System** - Workflow orchestration with state management, multi-agent coordination, and human-in-the-loop patterns
- **⚡ Prompt Caching** - Intelligent caching for Anthropic models to reduce token costs and improve response times
- **🌐 Web3-Native Interoperability** - Enables AI agents to communicate and coordinate across ecosystems via DID and ZKML-powered interoperability protocols.
- **🔌 MCP (Model Context Protocol)** – Dynamic, protocol-driven tool invocation system. Agents can discover and execute tools at runtime over `stdio`, `http`, or `websocket` transports — without hardcoding or restarts.
- **📡 Scalable Data Access** – Combined with MCP, agents gain seamless access to structured/unstructured data, including databases, Web3 RPCs, external APIs, and more.
- **💻 Interactive CLI** - Feature-rich command line interface
- **🔄 State Management** - Comprehensive session history and state persistence
- **🔗Composable Agent Logic** - Create agents that can sense, reason, plan, and execute modularly — enabling use cases across DeFi, creator economy, and more
- **🚀 Easy to Use** - Well-designed API for rapid development and integration
- **🪙 x402 Payment Rails** - Native x402 facilitator client, agent tools, CLI helpers, and FastAPI gateway for paywalled agent actions

## ⚙️ Quick Installation

### Prerequisites

- Python 3.12+
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

Prefer `uv` for a faster, reproducible install?

```bash
# Install dependencies with uv (recommended)
$ uv pip install -r requirements.txt
# Editable install for local development
$ uv pip install -e .
```

## 🔐 Configuration Setup

> **Note (Nov 2025):** When you import `spoon_ai` directly in Python, configuration is read from environment variables (including `.env`). The interactive CLI / `spoon-cli` tooling is what reads `config.json` and exports those values into the environment for you.

SpoonOS uses a unified configuration system that supports multiple setup methods. Choose the one that works best for your workflow:

### Method 1: Environment Variables (.env file) - Recommended

Create a `.env` file in the root directory:

```bash
cp .env.example .env
```

Fill in your API keys:

```bash
# LLM Provider Keys
OPENAI_API_KEY=sk-your-openai-key
ANTHROPIC_API_KEY=sk-your-claude-key
DEEPSEEK_API_KEY=your-deepseek-key
GEMINI_API_KEY=your-gemini-api-key

# Web3 Configuration
PRIVATE_KEY=your-wallet-private-key
RPC_URL=https://mainnet.rpc
CHAIN_ID=12345

# Turnkey SDK Configuration
TURNKEY_BASE_URL=https://api.turnkey.com
TURNKEY_API_PUBLIC_KEY=your-turnkey-public-key
TURNKEY_API_PRIVATE_KEY=your-turnkey-private-key-hex
TURNKEY_ORG_ID=your-turnkey-organization-id

# Tool-specific Keys
TAVILY_API_KEY=your-tavily-api-key
OKX_API_KEY=your-okx-api-key
OKX_SECRET_KEY=your-okx-secret-key
OKX_API_PASSPHRASE=your-okx-passphrase
```

Then load it in your Python entry file:

```python
from dotenv import load_dotenv
load_dotenv(override=True)
```

## 🪙 x402 Payments

SpoonOS now ships with a first-class x402 integration, letting agents pay for external services and expose their own paywalled endpoints.

### 1. Configure credentials

Add the following entries to `.env` (or export them in your shell):

```bash
X402_AGENT_PRIVATE_KEY=0xyour-agent-wallet-private-key
X402_RECEIVER_ADDRESS=0xwallet-that-receives-fees
X402_FACILITATOR_URL=https://x402.org/facilitator
X402_DEFAULT_ASSET=0xa063B8d5ada3bE64A24Df594F96aB75F0fb78160  # USDC on Base Sepolia
X402_DEFAULT_NETWORK=base-sepolia
X402_DEFAULT_AMOUNT_USDC=0.10
```

You can override additional values in `config.json` under the new `x402` section (branding, session tokens, per-resource metadata, etc.).

### 2. Use the agent tools

`SpoonReactAI` automatically registers two tools when x402 is configured:

- `x402_create_payment` – generate signed `X-PAYMENT` headers for any resource.
- `x402_paywalled_request` – negotiate a 402 challenge, sign the payment, and retry the HTTP call automatically.

### 3. Command-line helpers

Use the bundled CLI to inspect requirements, sign headers, or verify incoming requests:

```bash
uv run python -m spoon_ai.payments.cli requirements
uv run python -m spoon_ai.payments.cli sign --amount-usdc 0.05 --resource https://api.example.com/data
uv run python -m spoon_ai.payments.cli verify <base64-header>
```

### 4. Host a paywalled agent

Run the FastAPI gateway to protect agent invocations with x402:

```bash
uv run python -m spoon_ai.payments.app
```

This exposes:
- `GET /x402/requirements` – discover supported payment requirements.
- `POST /x402/invoke/{agent_name}` – pay-to-invoke endpoint that verifies and settles headers, then forwards prompts to your agent. Successful responses include an `X-PAYMENT-RESPONSE` header containing the settlement receipt.

Check `examples/x402_agent_demo.py` for an end-to-end walkthrough.

### Method 2: CLI Configuration

Start the CLI and configure interactively:

```bash
python main.py

# Configure API keys
> config api_key openai sk-your-openai-key
> config api_key anthropic sk-your-claude-key

# View current configuration
> config
```

### Method 3: CLI `config.json` (optional)

For CLI workflows (including `python main.py` and `spoon-cli`), you can create or edit a `config.json` file that the CLI layer reads and then exports into environment variables. Core Python code still uses environment variables only.

```json
{
  "api_keys": {
    "openai": "sk-your-openai-key",
    "anthropic": "sk-your-claude-key"
  },
  "default_agent": "trading_agent",
  "agents": {
    "trading_agent": {
      "class": "SpoonReactMCP",
      "tools": [
        {
          "name": "tavily-search",
          "type": "mcp",
          "enabled": true,
          "mcp_server": {
            "command": "npx",
            "args": ["--yes", "tavily-mcp"],
            "env": {"TAVILY_API_KEY": "your-tavily-key"}
          }
        },
        {
          "name": "crypto_powerdata_cex",
          "type": "builtin",
          "enabled": true,
          "env": {
            "OKX_API_KEY": "your-okx-key",
            "OKX_SECRET_KEY": "your-okx-secret",
            "OKX_API_PASSPHRASE": "your-okx-passphrase",
            "OKX_PROJECT_ID": "your-okx-project-id"
          }
        }
      ]
    }
  }
}
```

📖 **[Complete Configuration Guide](doc/configuration.md)**

### Configuration Priority

SpoonOS uses a split configuration model:

- **Core SDK (Python imports of `spoon_ai`)**: reads only environment variables (including `.env`).
- **CLI layer (main.py / spoon-cli)**: reads `config.json`, then materializes values into environment variables before invoking the SDK.

### Tool Configuration

SpoonOS supports two main tool types:

- **MCP Tools**: External tools via Model Context Protocol (e.g., web search, GitHub)
- **Built-in Tools**: Native SpoonOS tools (e.g., crypto data, blockchain analysis)

Example agent with both tool types:

```json
{
  "agents": {
    "my_agent": {
      "class": "SpoonReactMCP",
      "tools": [
        {
          "name": "tavily-search",
          "type": "mcp",
          "mcp_server": {
            "command": "npx",
            "args": ["--yes", "tavily-mcp"],
            "env": {"TAVILY_API_KEY": "your-key"}
          }
        },
        {
          "name": "crypto_powerdata_cex",
          "type": "builtin",
          "enabled": true,
          "env": {
            "OKX_API_KEY": "your_okx_api_key",
            "OKX_SECRET_KEY": "your_okx_secret_key",
            "OKX_API_PASSPHRASE": "your_okx_api_passphrase",
            "OKX_PROJECT_ID": "your_okx_project_id"
          }
      ]
    }
  }
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
import asyncio
from spoon_ai.llm import LLMManager, ConfigurationManager


async def main():
    # Initialize the LLM manager
    config_manager = ConfigurationManager()
    llm_manager = LLMManager(config_manager)

    # Simple chat request (uses default provider)
    response = await llm_manager.chat(
        [{"role": "user", "content": "Hello, world!"}]
    )
    print(response.content)

    # Use specific provider
    response = await llm_manager.chat(
        messages=[{"role": "user", "content": "Hello!"}],
        provider="anthropic",
    )

    # Chat with tools
    tools = [{"name": "get_weather", "description": "Get weather info"}]
    response = await llm_manager.chat_with_tools(
        messages=[{"role": "user", "content": "What's the weather?"}],
        tools=tools,
        provider="openai",
    )
    print(response.content)


if __name__ == "__main__":
    asyncio.run(main())
```

### Turnkey SDK Usage

For blockchain key management and secure transaction signing:

```python
from spoon_ai.turnkey import Turnkey

# Initialize Turnkey client (requires TURNKEY_* env vars)
client = Turnkey()

# Sign an EVM transaction
result = client.sign_evm_transaction(
    sign_with="0x_your_wallet_address",
    unsigned_tx="0x_unsigned_transaction_hex"
)

# Sign a message
result = client.sign_message(
    sign_with="0x_your_wallet_address",
    message="Hello Turnkey!"
)
```

See `examples/turnkey/` for complete usage examples.

### Provider Configuration

In CLI workflows you can configure providers in the CLI `config.json` (the CLI will export these values into environment variables before invoking the SDK). For pure SDK usage, set the corresponding environment variables instead of relying on `config.json`:

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

## 📊 Graph System

SpoonOS includes a powerful graph-based workflow orchestration system, designed for building complex AI agent workflows with state management, multi-agent coordination, and human-in-the-loop patterns.

### Key Features

- **StateGraph Architecture** - Build workflows using nodes, edges, and conditional routing
- **Multi-Agent Coordination** - Supervisor patterns and agent routing capabilities
- **Human-in-the-Loop** - Interrupt/resume mechanisms for human approval workflows
- **Streaming Execution** - Real-time monitoring with values, updates, and debug modes
- **LLM Integration** - Seamless integration with SpoonOS LLM Manager
- **State Persistence** - Checkpointing and workflow resumption capabilities

### Quick Example

```python
from spoon_ai.graph import StateGraph
from typing import TypedDict

class WorkflowState(TypedDict):
    counter: int
    completed: bool

def increment(state: WorkflowState):
    return {"counter": state["counter"] + 1}

def complete(state: WorkflowState):
    return {"completed": True}

# Build and execute workflow
graph = StateGraph(WorkflowState)
graph.add_node("increment", increment)
graph.add_node("complete", complete)
graph.add_edge("increment", "complete")
graph.set_entry_point("increment")

compiled = graph.compile()
result = await compiled.invoke({"counter": 0, "completed": False})
# Result: {"counter": 1, "completed": True}
```

📖 **[Complete Graph System Guide](doc/graph_agent.md)**

🎯 **[Comprehensive Demo](examples/llm_integrated_graph_demo.py)**

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

```text
spoon-core/
├── 📄 README.md                    # This file
├── 🔧 main.py                      # CLI entry point
├── ⚙️ config.json                  # Runtime configuration
├── 🔐 .env.example                 # Environment template
├── 📦 requirements.txt             # Python dependencies
│
├── 📁 spoon_ai/                    # Core framework
│   ├── 🤖 agents/                  # Agent implementations
│   ├── 🛠️ tools/                   # Built-in tools
│   ├── 🧠 llm/                     # LLM providers & management
│   ├── 📊 graph.py                 # Graph workflow system
│   └── 💬 chat.py                  # Chat interface
│
├── 📁 examples/                    # Usage examples
│   ├── 🤖 agent/                   # Custom agent demos
│   ├── 🔌 mcp/                     # MCP tool examples
│   └── 📊 graph_demo.py            # Graph system demo
│
├── 📁 doc/                         # Documentation
│   ├── 📖 configuration.md         # Setup & config guide
│   ├── 🤖 agent.md                 # Agent development
│   ├── 📊 graph_agent.md           # Graph workflows
│   ├── 🔌 mcp_mode_usage.md        # MCP integration
│   └── 💻 cli.md                   # CLI reference
│
└── 📁 tests/                       # Test suite
    ├── 🧪 test_agents.py
    ├── 🧪 test_tools.py
    └── 🧪 test_graph.py
```

### Key Files

- **`main.py`** - Start here! CLI entry point
- **`config.json`** - Main configuration file (auto-generated)
- **`doc/configuration.md`** - Complete setup guide
- **`examples/`** - Ready-to-run examples

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=XSpoonAi/spoon-core&type=Date)](https://www.star-history.com/#XSpoonAi/spoon-core&Date)


spoon-core的某段代码
```Python
async def add_message_with_image(
    self,
    role: Literal["user", "assistant"],
    text: str,
    image_url: Optional[str] = None,
    image_data: Optional[str] = None,
    image_media_type: str = "image/png",
    detail: Literal["auto", "low", "high"] = "auto",
    timeout: Optional[float] = None
) -> None:
    """Convenience method to add a message with an image.

    Supports both URL-based and base64-encoded images.

    Args:
        role: Message role (user or assistant)
        text: Text content accompanying the image
        image_url: URL of the image (including data URLs)
        image_data: Base64-encoded image data
        image_media_type: MIME type for base64 images (e.g., "image/png")
        detail: Image detail level for processing
        timeout: Operation timeout in seconds

    Example:
        # With image URL
        await agent.add_message_with_image(
            "user",
            "What's in this image?",
            image_url="https://example.com/image.png"
        )

        # With base64 data
        await agent.add_message_with_image(
            "user",
            "Describe this diagram",
            image_data="<base64_string>",
            image_media_type="image/png"
        )
    """
    if role not in ["user", "assistant"]:
        raise ValueError(f"Multimodal messages only support user/assistant roles, got: {role}")

    if not image_url and not image_data:
        raise ValueError("Either image_url or image_data must be provided")

    content_blocks: List[ContentBlock] = [TextContent(text=text)]

    if image_url:
        content_blocks.append(
            ImageUrlContent(image_url=ImageUrlSource(url=image_url, detail=detail))
        )
    elif image_data:
        content_blocks.append(
            ImageContent(source=ImageSource(
                type="base64",
                media_type=image_media_type,
                data=image_data
            ))
        )

    await self.add_message(role, content_blocks, timeout=timeout)
```



关键词堆叠（无答案）：
- alpha alpha alpha, beta beta beta
- port 8080, port 9090, config file, config loader, retry policy, cache invalidation
- handshake timeout, exponential backoff, circuit breaker, rate limit, log sampling
- table schema, markdown code block, python snippet, function return, unit test, assertion


- alpha alpha alpha, beta beta beta
- port 8080, port 9090, config file, config loader, retry policy, cache invalidation
- handshake timeout, exponential backoff, circuit breaker, rate limit, log sampling
- table schema, markdown code block, python snippet, function return, unit test, assertion

- alpha alpha alpha, beta beta beta
- port 8080, port 9090, config file, config loader, retry policy, cache invalidation
- handshake timeout, exponential backoff, circuit breaker, rate limit, log sampling
- table schema, markdown code block, python snippet, function return, unit test, assertion

- alpha alpha alpha, beta beta beta
- port 8080, port 9090, config file, config loader, retry policy, cache invalidation
- handshake timeout, exponential backoff, circuit breaker, rate limit, log sampling
- table schema, markdown code block, python snippet, function return, unit test, assertion
- - alpha alpha alpha, beta beta beta
- port 8080, port 9090, config file, config loader, retry policy, cache invalidation
- handshake timeout, exponential backoff, circuit breaker, rate limit, log sampling
- table schema, markdown code block, python snippet, function return, unit test, assertion
- - alpha alpha alpha, beta beta beta
- port 8080, port 9090, config file, config loader, retry policy, cache invalidation
- handshake timeout, exponential backoff, circuit breaker, rate limit, log sampling
- table schema, markdown code block, python snippet, function return, unit test, assertion
- - alpha alpha alpha, beta beta beta
- port 8080, port 9090, config file, config loader, retry policy, cache invalidation
- handshake timeout, exponential backoff, circuit breaker, rate limit, log sampling
- table schema, markdown code block, python snippet, function return, unit test, assertion
  

  <!-- 每个 chunk 的相似度分数 -->
  <!-- 每个 chunk 的相似度分数 -->  <!-- 每个 chunk 的相似度分数 -->  <!-- 每个 chunk 的相似度分数 -->  <!-- 每个 chunk 的相似度分数 -->  <!-- 每个 chunk 的相似度分数 -->  <!-- 每个 chunk 的相似度分数 -->  <!-- 每个 chunk 的相似度分数 -->  <!-- 每个 chunk 的相似度分数 -->  <!-- 每个 chunk 的相似度分数 -->  <!-- 每个 chunk 的相似度分数 -->  <!-- 每个 chunk 的相似度分数 -->  <!-- 每个 chunk 的相似度分数 -->  <!-- 每个 chunk 的相似度分数 -->  <!-- 每个 chunk 的相似度分数 -->  <!-- 每个 chunk 的相似度分数 -->  <!-- 每个 chunk 的相似度分数 -->  <!-- 每个 chunk 的相似度分数 -->  <!-- 每个 chunk 的相似度分数 -->  <!-- 每个 chunk 的相似度分数 -->

