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
- **🌐 Web3-Native Interoperability** - Enables AI agents to communicate and coordinate across ecosystems via DID and ZKML-powered interoperability protocols.
- **📡 Scalable Data Access** - Supports structured and unstructured data via MCP integration
- **💻 Interactive CLI** - Feature-rich command line interface
- **🔄 State Management** - Comprehensive session history and state persistence
- **🔗Composable Agent Logic** - Create agents that can sense, reason, plan, and execute modularly — enabling use cases across DeFi, creator economy, and more
- **🚀 Easy to Use** - Well-designed API for rapid development and integration

## Installation

### Prerequisites

- Python 3.10+
- pip package manager (or uv as a faster alternative)

### Create a Virtual Environment

It is recommended to install and use SpoonOS in a virtual environment to avoid dependency conflicts.

```bash
# Create a virtual environment
python -m venv spoon-env

# Activate the virtual environment on Linux/macOS
source spoon-env/bin/activate

# Activate the virtual environment on Windows
# spoon-env\Scripts\activate
```

### Install from Source

#### Option 1: Using pip (standard)

```bash
# Clone the repository
git clone git@github.com:XSpoonAi/spoon-core.git
cd spoon-core

# Install dependencies
pip install -r requirements.txt

# Install in development mode (optional)
pip install -e .
```

#### Option 2: Using uv (faster alternative)

```bash
# Clone the repository
git clone git@github.com:XSpoonAi/spoon-core.git
cd spoon-core

# Install dependencies with uv
uv pip install -r requirements.txt

# Install in development mode (optional)
uv pip install -e .
```

### Install via pip

```bash
pip install spoon-ai-sdk
```

## Environment Variables and API Key Configuration (.env Recommended)

SCDF supports various API services and requires proper configuration of environment variables and API keys. This section provides comprehensive guidance on setting up your environment.

### 🔧 Configuration Methods

#### Method 1: .env File (Recommended for Development)

SpoonOS automatically loads environment variables using the python-dotenv package. This allows you to configure all your API keys and network settings in a simple .env file.

1、Create a `.env` file in the project root directory. You can use the provided template:
Copy the Example File

```bash
# Copy the example file and edit it
cp .env.example .env
```

2、Fill in Your API Keys and Config Values
Open .env and set your API keys, private keys, RPC URLs, etc.

```bash
# LLM API Keys
OPENAI_API_KEY=sk-your-openai-api-key-here
ANTHROPIC_API_KEY=sk-your-anthropic-api-key-here
DEEPSEEK_API_KEY=your-deepseek-api-key-here

# Blockchain
PRIVATE_KEY=your-wallet-private-key
RPC_URL=https://mainnet-1.rpc.banelabs.org
SCAN_URL=https://xt4scan.ngd.network/
CHAIN_ID=47763
```

3、Load the .env File in Your Python Code
At the very top of your entry script (e.g., main.py, SpoonAgent.py, etc), add:

```python
from dotenv import load_dotenv
load_dotenv(override=True)
```

This ensures that SpoonOS will load your .env variables at runtime, even if your system shell has conflicting environment variables.

#### Method2 : Environment Variables

**Linux/macOS:**

```bash
# Set environment variables in your shell
export OPENAI_API_KEY="sk-your-openai-api-key-here"
export ANTHROPIC_API_KEY="sk-ant-your-anthropic-api-key-here"
export DEEPSEEK_API_KEY="your-deepseek-api-key-here"
export PRIVATE_KEY="your-wallet-private-key-here"

# Make them persistent by adding to your shell profile
echo 'export OPENAI_API_KEY="sk-your-openai-api-key-here"' >> ~/.bashrc
echo 'export ANTHROPIC_API_KEY="sk-ant-your-anthropic-api-key-here"' >> ~/.bashrc
source ~/.bashrc
```

**Windows (PowerShell):**

```powershell
# Set environment variables
$env:OPENAI_API_KEY="sk-your-openai-api-key-here"
$env:ANTHROPIC_API_KEY="sk-ant-your-anthropic-api-key-here"
$env:DEEPSEEK_API_KEY="your-deepseek-api-key-here"
$env:PRIVATE_KEY="your-wallet-private-key-here"

# Make them persistent
[Environment]::SetEnvironmentVariable("OPENAI_API_KEY", "sk-your-openai-api-key-here", "User")
[Environment]::SetEnvironmentVariable("ANTHROPIC_API_KEY", "sk-ant-your-anthropic-api-key-here", "User")
```

#### Method 3: CLI Configuration Commands

After starting the CLI, use the `config` command:

```bash
# Start the CLI
python main.py

# Configure API keys using the CLI
> config api_key openai sk-your-openai-api-key-here
✅ OpenAI API key configured successfully

> config api_key anthropic sk-ant-your-anthropic-api-key-here
✅ Anthropic API key configured successfully

> config api_key deepseek your-deepseek-api-key-here
✅ DeepSeek API key configured successfully



# Configure wallet private key
> config PRIVATE_KEY your-wallet-private-key-here
✅ Private key configured successfully

# View current configuration (keys are masked for security)
> config
Current configuration:
API Keys:
  openai: sk-12...ab34
  anthropic: sk-an...xy89
  deepseek: ****...****
PRIVATE_KEY: 0x12...ab34
```

#### Method 4: Configuration File

The CLI creates a configuration file at `config.json` in the project root directory:

```json
{
  "api_keys": {
    "openai": "sk-your-openai-api-key-here",
    "anthropic": "sk-ant-your-anthropic-api-key-here",
    "deepseek": "your-deepseek-api-key-here"
  },
  "base_url": "your_base_url_here",
  "default_agent": "default"
}
```

### 🔍 Verification & Testing

#### Check Environment Variables

```bash
# Verify environment variables are set
echo $OPENAI_API_KEY
echo $ANTHROPIC_API_KEY
echo $DEEPSEEK_API_KEY

# Test with a simple Python script
python -c "import os; print('OpenAI:', 'SET' if os.getenv('OPENAI_API_KEY') else 'NOT SET')"
```

#### Test API Connectivity

```bash
# Start CLI and test
python main.py

# start chat and test
> action chat
> Hello, can you respond to test the API connection?
```

### 🔒 Security Best Practices

#### 🚨 Critical Security Guidelines

1. **Never commit API keys to version control**

   ```bash
   # Ensure .env is in .gitignore
   echo ".env" >> .gitignore
   ```

2. **Use environment variables in production**

   - Avoid hardcoding keys in source code
   - Use secure environment variable management in deployment

3. **Wallet private key security**

   - **NEVER share your private key with anyone**
   - Store in secure environment variables only
   - Consider using hardware wallets for production

4. **API key rotation**
   - Regularly rotate API keys (monthly recommended)
   - Monitor API usage for unusual activity
   - Use API key restrictions when available

#### 🛡️ Additional Security Measures

```bash
# Set restrictive file permissions for .env
chmod 600 .env

# Use a dedicated wallet for testing with minimal funds
# Never use your main wallet's private key

# Monitor API usage regularly
# Set up billing alerts on API provider dashboards
```

### 🌐 OpenRouter Configuration Guide

OpenRouter provides an OpenAI-compatible API interface that allows you to access multiple AI models through a single API key. To use OpenRouter:

1. **Get your OpenRouter API key:**

   - Visit [OpenRouter Platform](https://openrouter.ai/keys)
   - Register an account and create an API key

2. **Set environment variables:**

   ```bash
   # Use OPENAI_API_KEY environment variable to store OpenRouter API key
   export OPENAI_API_KEY="sk-or-your-openrouter-api-key-here"
   ```

3. **Use OpenRouter in your code:**

   ```python
   from spoon_ai.chat import ChatBot
   from spoon_ai.agents import SpoonReactAI

   # Configure to use OpenRouter
   openrouter_agent = SpoonReactAI(
       llm=ChatBot(
           model_name="anthropic/claude-sonnet-4",  # Model name supported by OpenRouter，value:GPT-4o ...
           llm_provider="openai",                    # Use openai provider
           base_url="https://openrouter.ai/api/v1"  # OpenRouter API endpoint
           # Automatically uses OpenRouter API key from OPENAI_API_KEY environment variable
       )
   )
   ```

4. **Supported model examples:**
   - `openai/gpt-4` - GPT-4 model
   - `openai/gpt-3.5-turbo` - GPT-3.5 Turbo
   - `anthropic/claude-sonnet-4` - Claude 3.5 Sonnet
   - `anthropic/claude-3-opus` - Claude 3 Opus
   - `meta-llama/llama-3.1-8b-instruct` - Llama 3.1 8B
   - For more models, see [OpenRouter Models List](https://openrouter.ai/models)

## Quick Start

##**Start the MCP Server**

Before using the MCP-enabled agent, you must start the MCP server with your tools:

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

After entering the interactive command line interface, you can start using the various features of SpoonAI.

### Create a ReAct Agent

```python
from spoon_ai.agents import SpoonReactAI
from spoon_ai.chat import ChatBot
import asyncio

async def main():
    # Create a ReAct agent
    react_agent = SpoonReactAI(llm=ChatBot())

    # Run the ReAct agent and get a response
    response = await react_agent.run("Analyze the transaction history of this wallet address: 0x123...")
    print(response)

asyncio.run(main())
```

## CLI Tools

SCDF CLI is a powerful command-line tool that provides rich functionality, including interacting with AI agents, managing chat history, processing cryptocurrency transactions, and loading documents.

### Basic Commands

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

### CLI Usage Examples

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
> action react
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

## Model Context Protocol Integration

<div align="center">
  <p><strong>Enhanced MCP integration for SpoonOS</strong></p>
</div>

SpoonOS integrates with the Model Context Protocol (MCP) to provide enhanced data availability and tool access. This allows developers to access external data sources and invoke tools more easily and efficiently.

### ✨ Key Features

- **🫎 Unified Data Access Layer** - Abstracts diverse data sources into a standardized interface for AI agents
- **⚡️ Streaming Responses** - Real-time streaming output from language models
- **📈 Modular Integration** - Enables dynamic loading of external APIs, on-chain data, or local resources
- **📡 Access Control & Permissioning** - Supports granular permissions and scoped data/task access

## Agent Framework

SDCF provides a powerful Agent framework for creating custom agents with your own tools and logic.

### ReAct Intelligent Agent

SDCF implements an intelligent agent based on the ReAct (Reasoning + Acting) paradigm, which is an advanced AI agent architecture that combines reasoning and action capabilities. The ReAct agent can think, plan, and execute in complex tasks, solving problems through an iterative reasoning-action loop.

#### ReAct Workflow

The ReAct agent workflow includes the following key steps:

1. **Observation**: Collecting environment and task-related information
2. **Reasoning**: Analyzing information and reasoning
3. **Acting**: Executing specific operations
4. **Feedback**: Obtaining action results and updating cognition

This cycle repeats continuously until the task is completed or the preset goal is achieved.

#### Agent Termination

SpoonOS agents now use an intelligent termination mechanism based on LLM finish reasons rather than explicit termination tools. This provides more natural and efficient conversation completion:

**Finish Reason Termination:**
- Agents automatically detect when the LLM indicates task completion through `finish_reason: "stop"` and `native_finish_reason: "stop"`
- This allows for immediate termination of straightforward queries without unnecessary tool execution steps
- The agent gracefully transitions to `FINISHED` state and returns the response content

**Benefits:**
- **Improved Efficiency**: Prevents unnecessary step execution for simple questions
- **Natural Flow**: Allows LLM to signal completion organically
- **Provider Compatibility**: Works with both OpenAI and Anthropic LLM providers
- **Backward Compatibility**: Maintains existing agent functionality for complex multi-step tasks</search>
</search_and_replace>

### Custom Tools

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

### Custom Agents

There are two ways to create custom Agents:

**Method 1: Inheriting from an existing Agent class**

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

    max_steps: int = 8

    # Define available tools
    avaliable_tools: ToolManager = Field(default_factory=lambda: ToolManager([
        MyCustomTool(),
        # Add other tools...
    ]))
```

**Method 2: Directly using ToolCallAgent and configuring tools**

```python
from spoon_ai.agents import ToolCallAgent
from spoon_ai.tools import ToolManager
from spoon_ai.chat import ChatBot

# Create a tool manager
tool_manager = ToolManager([
    MyCustomTool(),
    # Add more tools...
])

# Create an Agent
my_agent = ToolCallAgent(
    name="my_agent",
    description="Custom configured Agent",
    llm=ChatBot(model="gpt-4"),
    avaliable_tools=tool_manager,
    system_prompt="Custom system prompt",
    max_steps=12
)
```

### Tool Combination and Indexing

SpoonAI supports dynamic tool combination and semantic indexing, allowing Agents to more intelligently select appropriate tools:

```python
from spoon_ai.tools import ToolManager

# Create multiple tools
tools = [
    MyCustomTool(),
    # More tools...
]

# Create a tool manager
tool_manager = ToolManager(tools)

# Create a semantic index for tools (requires OpenAI API key)
tool_manager.index_tools()

# Find the most relevant tools based on a query
relevant_tools = tool_manager.query_tools(
    query="I need to analyze this data",
    top_k=3  # Return the top 3 most relevant tools
)
```

## Advanced Usage

## API Integration

SpoonAI supports multiple AI service providers, including:

- **OpenAI** - GPT-3.5/GPT-4 series models
- **Anthropic** - Claude series models
- **DeepSeek** - DeepSeek series models
- **OpenRouter** - Access to multiple AI models through OpenAI-compatible API
- **More...** - Easily extendable to support other AI providers

### Integration Examples

```python
from spoon_ai.chat import ChatBot
from spoon_ai.agents import SpoonReactAI

# Using OpenAI's GPT-4
openai_agent = SpoonReactAI(
    llm=ChatBot(model_name="gpt-4", llm_provider="openai")
)

# Using Anthropic's Claude
claude_agent = SpoonReactAI(
    llm=ChatBot(model_name="claude-3-7-sonnet-20250219", llm_provider="anthropic")
)


# Using OpenRouter (OpenAI-compatible API)
openrouter_agent = SpoonReactAI(
    llm=ChatBot(
        model_name="anthropic/claude-3.5-sonnet",  # or any model available on OpenRouter
        llm_provider="openai",  # Use openai provider for compatibility
        base_url="https://openrouter.ai/api/v1"  # OpenRouter API endpoint
        # Uses OPENAI_API_KEY environment variable with your OpenRouter API key
    )
)
```

## Tool Integration Modes

### Mode 1: Built-in Agent Mode

In this mode, you encapsulate your custom tools into the MCP tool collection (such as creating a new mcp_thirdweb_collection, or directly changing the mcp_tools_collection.py file), and then call it through an Agent that inherits from SpoonReactAI and MCPClientMixin (such as SpoonThirdWebMCP). This mode is maintained by the platform Agent configuration and can be used directly by users.

#### Structural diagram

[User Prompt]
↓
[SpoonThirdWebMCP Agent] 🧠
↓ calls
[FastMCP over SSE]
↓
[GetBlocksFromThirdwebInsight / GetWalletTransactionsTool / etc.]
↓
[Thirdweb Insight API]

#### Step-by-Step

```yaml
spoon-core/
│   ├── agents/
│   │   └── spoon_thirdweb_mcp.py
│   ├── tool_collection
│   │   └── mcp_thirdweb_collection.py
spoon_toolkits/                ← This is a standalone tool library (installable as a module)
```

##### 0. Install Dependencies

You have two ways to install the spoon-toolkits package:

Option 1: Install from GitHub source (for development or latest changes):

```bash
git clone https://github.com/XSpoonAi/spoon-toolkit.git
cd spoon-toolkit
pip install -e .
```

Option 2: Install from PyPI (recommended for general use):

```bash
pip install spoon-toolkits

```

👉 Tip:

Use Option 1 if you want to modify the toolkit source code or track the latest updates.

##### 1. Register the tool to the MCP service（Take the third_web tool as an example）

```python
from fastmcp import FastMCP
import asyncio
# from typing import Any, Dict, List, Optional

# Import base tool classes and tool manager
from spoon_ai.tools.base import BaseTool, ToolResult
from spoon_ai.tools.tool_manager import ToolManager

# Import all available tools
from spoon_toolkits import (
    GetContractEventsFromThirdwebInsight,
    GetMultichainTransfersFromThirdwebInsight,
    GetTransactionsTool,
    GetContractTransactionsTool,
    GetContractTransactionsBySignatureTool,
    GetBlocksFromThirdwebInsight,
    GetWalletTransactionsFromThirdwebInsight
)

mcp = FastMCP("SpoonAI MCP Tools")

class MCPToolsCollection:
    """Collection class that wraps existing tools as MCP tools"""

    def __init__(self):
        """Initialize MCP tools collection

        Args:
            name: Name of the MCP server
        """
        self.mcp = mcp
        self._setup_tools()

    def _setup_tools(self):
        """Set up all available tools as MCP tools"""
        # Create all tool instances
        tools = [
            GetContractEventsFromThirdwebInsight(),
            GetMultichainTransfersFromThirdwebInsight(),
            GetTransactionsTool(),
            GetContractTransactionsTool(),
            GetContractTransactionsBySignatureTool(),
            GetBlocksFromThirdwebInsight(),
            GetWalletTransactionsFromThirdwebInsight()
        ]

        # Create tool manager
        self.tool_manager = ToolManager(tools)

        # Create MCP wrapper for each tool
        for tool in tools:
            self.mcp.add_tool(tool.execute, name=tool.name, description=tool.description)

    async def run(self, **kwargs):
        """Start the MCP server

        Args:
            **kwargs: Parameters passed to FastMCP.run()
        """
        await self.mcp.run_async(transport="sse", port=8765, **kwargs)

# Create default instance that can be imported directly
mcp_tools = MCPToolsCollection()

if __name__ == "__main__":
    # Start MCP server when this script is run directly
    asyncio.run(mcp_tools.run())
```

Before calling the agent, make sure the MCP service is running:

```bash
python spoon_toolkits/mcp_thirdweb_collection.py
# or if you renamed it:
python your_project/tools/mcp_tools_collection.py
```

##### 2

###### 2.1 Define Agent and connect to MCP

```python
from spoon_ai.agents.spoon_react import SpoonReactAI
from spoon_ai.agents.mcp_client_mixin import MCPClientMixin
from fastmcp.client.transports import SSETransport
from spoon_ai.tools.tool_manager import ToolManager

from pydantic import Field
from spoon_ai.chat import ChatBot
import os
import asyncio


class SpoonThirdWebMCP(SpoonReactAI, MCPClientMixin):
    name: str = "SpoonThirdWebMCP"
    description: str = (
        "An AI assistant specialized in querying EVM blockchain data using the Thirdweb Insight API. "
        "Supports retrieving smart contract events (e.g. Transfer), function call transactions, wallet activity, "
        "recent cross-chain token transfers (especially USDT), block metadata, and contract-specific transaction logs. "
        "Use this agent when the user asks about on-chain behavior, such as token transfers, contract usage, wallet history, or recent block/transaction activity."
    )
    system_prompt: str = """
        You are ThirdwebInsightAgent, a blockchain data analyst assistant powered by Thirdweb Insight API.
        You can fetch EVM contract events, transactions, token transfers, blocks, and wallet activity across multiple chains.

        Use the appropriate tool when the user asks about:
        - contract logs or Transfer events → use `get_contract_events_from_thirdweb_insight`
        - USDT transfers across chains → use `get_multichain_transfers_from_thirdweb_insight`
        - recent cross-chain transactions → use `get_transactions`
        - a specific contract's transaction history → use `get_contract_transactions`
        - contract function call history (e.g., swap, approve) → use `get_contract_transactions_by_signature`
        - recent block info by chain → use `get_blocks_from_thirdweb_insight`
        - wallet activity across chains → use `get_wallet_transactions_from_thirdweb_insight`

        Always extract necessary parameters like:
        - `contract_address` (if user mentions a token, e.g. USDT, WETH, use its address)
        - `chain_id` (Ethereum = 1, Polygon = 137, etc.)
        - `event_signature` (e.g., 'Transfer(address,address,uint256)')
        - `limit` (default to 10 if unspecified)
        - `client_id` can be pulled from environment variable or injected context

        If something is unclear, ask for clarification. Otherwise, call the appropriate tool.
    """

    avaliable_tools: ToolManager = Field(default_factory=lambda: ToolManager([]))
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        MCPClientMixin.__init__(self, mcp_transport=kwargs.get('mcp_transport', SSETransport("http://127.0.0.1:8765/sse")))
```

###### 2.2 User operation mode

Get client_id from https://thirdweb.com/login

```python
async def main():
    # Ensure necessary API keys are set
    # Create an InfoAssistantAgent
    info_agent = SpoonThirdWebMCP(llm=ChatBot())

    # Query standard ERC20 transfer events (Transfer)
    info_agent.clear()
    result = await info_agent.run("Get the last 10 Transfer events from the USDT contract on Ethereum using client ID xxxx.")
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
```

###### Run Everything End-to-End

1. Start the MCP Server:

`````bash
python spoon_toolkits/mcp_thirdweb_collection.py```

2.Run the Agent:

```bash
python spoon_toolkits/mcp_thirdweb_collection.py

3.Sample Query:
````python
await agent.run("Show me the latest 10 USDT transfers on Ethereum.")
`````

Expected Result:
[
{
"block_number": "19202222",
"from": "0x...",
"to": "0x...",
"amount": "1000 USDT"
},
...
]

### Mode 2: Community Agent Mode

In this mode, you can reuse agents published by others in the community, without writing your own tool code. These agents are registered via GitHub using the MCP protocol, and called via mcp-proxy.

This is useful when:

You want to quickly try a public Agent from GitHub

You don't want to define Tool, ToolManager, or custom logic

You want to orchestrate many agents from different repos

Register the tool to the MCP service

#### Step-by-Step: Community Agent Mode

Use Community Agent Mode to connect with agents hosted on GitHub via the MCP protocol — without writing custom tool or agent code.

##### 1. Install mcp-proxy via UV

```bash
uv tool install mcp-proxy
```

This will install the proxy server that bridges your CLI or client agent to remote GitHub agents.

##### 2. Start the Community Agent via MCP Proxy,Example using @modelcontextprotocol/server-github

```bash
mcp.proxy --sse-port 8123 -- npx -y @modelcontextprotocol/server-github
```

This command will:

Start an SSE server on http://localhost:8123/sse

Load an agent from the @modelcontextprotocol/server-github package

Allow your local agent to communicate with this GitHub-based agent over MCP

##### 3. Connect Your Local Python Agent to the Proxy

```python
from spoon_ai.agents.spoon_react import SpoonReactAI
from spoon_ai.agents.mcp_client_mixin import MCPClientMixin
from fastmcp.client.transports import SSETransport
from spoon_ai.tools.tool_manager import ToolManager

from pydantic import Field
class SpoonReactMCP(SpoonReactAI, MCPClientMixin):
    description: str = ()
    system_prompt: str = """ """
    name: str = "spoon_react_mcp"
    description: str = "A smart ai agent in neo blockchain with mcp"
    avaliable_tools: ToolManager = Field(default_factory=lambda: ToolManager([]))
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        MCPClientMixin.__init__(self, mcp_transport=kwargs.get('mcp_transport', SSETransport("http://127.0.0.1:8123/sse")))

```

##### ✅ Summary

| Feature                | Description                                   |
| ---------------------- | --------------------------------------------- |
| 🧠 No code required    | Just connect to an agent hosted on GitHub     |
| 🔗 Plug-and-play setup | Proxy auto-loads GitHub-hosted agents         |
| 🔧 Extensible          | You can still override agent behavior locally |

### Mode 3: Custom Agent Mode

In this mode, you define your own agent from scratch. You have full control over its behavior, prompt, toolset, and integration logic. This is ideal for building advanced or business-specific agents that operate independently of platform configuration.

🧩 Use Cases
You want to build a fully custom agent for your domain (e.g., GitHub analytics, Fluence pricing, database QA)

You need to tightly integrate tools with your backend/business logic

You prefer to operate the agent fully through code (no UI or config dependency)

#### 🛠️ Step-by-Step Guide

##### 1. Define Your Own Tool

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

#### 2. Define Your Own Agent

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
    print("🤖 Result:", result)

if __name__ == "__main__":
    asyncio.run(main())
```

#### 📌 Key Benefits

Feature Description
🎯 Fully Customizable You control the prompt, logic, and available tools
🛠️ Tool Management Easily add or remove tools, supports tool chaining
🔗 Optional MCP You can add MCPClientMixin to integrate with remote tools via MCP

✅ Advanced Extensions (Optional)
Chain tools for multi-step workflows (e.g., scrape → analyze → summarize)

#### 4. (Optional) Register Your Custom Tool to an MCP Tool Collection

If you want to expose your custom tool to remote agents via the MCP protocol (e.g., allow other agents to call it via SSE or WebSocket), you need to register it into a tool collection and run a local MCP server.

##### 4.1 Define the MCP Tool Collection

```python
from fastmcp import FastMCP
import asyncio
from typing import Any, Dict, List, Optional

# Import base tool classes and tool manager
from spoon_ai.tools import BaseTool, ToolManager

from tools import (
MyCustomTool,
...
)

mcp = FastMCP("SpoonAI MCP Tools")

class MCPToolsCollection:
    """Collection class that wraps existing tools as MCP tools"""

    def __init__(self):
        """Initialize MCP tools collection

        Args:
            name: Name of the MCP server
        """
        self.mcp = mcp
        self._setup_tools()

    def _setup_tools(self):
        """Set up all available tools as MCP tools"""
        # Create all tool instances
        tools = [
          MyCustomTool,
          ...
        ]

        # Create tool manager
        self.tool_manager = ToolManager(tools)

        # Create MCP wrapper for each tool
        for tool in tools:
            self.mcp.add_tool(tool.execute, name=tool.name, description=tool.description)

    async def run(self, **kwargs):
        """Start the MCP server

        Args:
            **kwargs: Parameters passed to FastMCP.run()
        """
        await self.mcp.run_async(transport="sse", port=8765, **kwargs)

    async def add_tool(self, tool: BaseTool):
        """Add a tool to the MCP server"""
        self.mcp.add_tool(tool.execute, name=tool.name, description=tool.description)

# Create default instance that can be imported directly
mcp_tools = MCPToolsCollection()

if __name__ == "__main__":
    # Start MCP server when this script is run directly
    asyncio.run(mcp_tools.run())
```

##### 4.2 Start Your MCP Server

```bash
python mcp_tool_collection.py
```

This will start an SSE server on http://localhost:8765/sse and allow other MCP-compatible agents to call your tool remotely.

## 💼 Enterprise Application Scenarios

Add MCPClientMixin to enable remote tool invocation
SpoonAI can be applied to various enterprise scenarios:

- **Financial Analysis** - Cryptocurrency market analysis, investment advice, risk assessment
- **Customer Service** - Intelligent customer service, problem-solving, ticket processing
- **Document Processing** - Contract analysis, report generation, content summarization
- **Business Automation** - Process automation, task coordination, intelligent decision support
- **Research Assistant** - Information retrieval, data analysis, research report generation

## 🔍 Advanced Features

### Available Tools

SDCF comes with a comprehensive set of built-in tools for various use cases:

#### Cryptocurrency Tools

- **GetTokenPriceTool** - Get real-time token prices
- **Get24hStatsTool** - Get 24-hour trading statistics
- **GetKlineDataTool** - Get candlestick chart data
- **PriceThresholdAlertTool** - Set price alerts
- **TokenTransfer** - Transfer tokens between addresses
- **WalletAnalysis** - Analyze wallet transactions and holdings
- **UniswapLiquidity** - Monitor Uniswap liquidity pools
- **LstArbitrageTool** - Liquid staking token arbitrage opportunities

#### Monitoring Tools

- **PredictPrice** - Price prediction using ML models
- **TokenHolders** - Analyze token holder distribution
- **TradingHistory** - Track trading history and patterns
- **LendingRateMonitorTool** - Monitor DeFi lending rates

## 🎯 Project Roadmap

- [ ] **Web Interface** - Develop a web-based user interface
- [ ] **Agent Marketplace** - Create a sharing platform for agents and tools
- [ ] **Agent Interoperability** - Implement collaboration capabilities between multiple agents
- [ ] **Local Model Support** - Add support for locally running open-source models
- [ ] **Plugin System** - Build an extensible plugin architecture
- [ ] **Advanced Monitoring** - Enhance agent execution monitoring and analysis capabilities
- [ ] **Multi-Language Support** - Extend support for more languages
- [ ] **Cloud Deployment** - Simplify cloud environment deployment process

## Contributing

We welcome contributions of all forms!

1. Fork this repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Create a Pull Request

Please ensure you follow our code style and contribution guidelines.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgements

- Thanks to all developers who have contributed to this project
- Special thanks to the major AI model providers for their support
- Thanks to the open-source community for their valuable feedback

---

<div align="center">
  <p>Made with ❤️ | Developed by the SpoonOS Team</p>
  <p>
    <a href="https://github.com/XSpoonAi">GitHub</a> •
    <a href="hhttps://x.com/Spoonai_OS">Twitter</a> •
    <a href="https://discord.gg/G6y3ZCFK4h">Discord</a>
  </p>
</div>
```
