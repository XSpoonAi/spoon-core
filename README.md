# 🚀 SpoonOS Core Developer Framework(SCDF)

<div align="center">
  <img src="logo/spoon.gif" alt="SpoonAI Logo" width="200"/>
  <p><strong>Core developer framework of SpoonOS ——Agentic OS for the sentient economy. Next-Generation AI Agent Framework | Powerful Interactive CLI | Web3 infrastructure optimized Support</strong></p>
</div>

<div align="center">
  <a href="#✨-features">Features</a> •
  <a href="#🔧-installation">Installation</a> •
  <a href="#🚀-quick-start">Quick Start</a> •
  <a href="#💡-usage-examples">Usage Examples</a> •
  <a href="#🛠️-cli-tools">CLI Tools</a> •
  <a href="#🧩-agent-framework">Agent Framework</a> •
  <a href="#🔌-api-integration">API Integration</a> •
  <a href="#🤝-contributing">Contributing</a> •
  <a href="#📄-license">License</a>
</div>

## ✨ Features

SpoonOS is a living, evolving agentic operating system. Its SCDF is purpose-built to meet the growing demands of Web3 developers — offering a complete toolkit for building sentient, composable, and interoperable AI agents.

- **🧠 ReAct Intelligent Agent** - Advanced agent architecture combining reasoning and action
- **🔧 Custom Tool Ecosystem** - Modular tool system for easily extending agent capabilities
- **💬 Multi-Model Support** - Compatible with major large language models including OpenAI, Anthropic, DeepSeek, and more Web3 fine-tuned LLM
- **🌐 Web3-Native Interoperability** - Enables AI agents to communicate and coordinate across ecosystems via DID and ZKML-powered interoperability protocols.
- **📡 Scalable Data Access** - Supports structured and unstructured data via MCP+
- **💻 Interactive CLI** - Feature-rich command line interface
- **🔄 State Management** - Comprehensive session history and state persistence
- **🔗Composable Agent Logic** - Create agents that can sense, reason, plan, and execute modularly — enabling use cases across DeFi, creator economy, and more
- **🚀 Easy to Use** - Well-designed API for rapid development and integration

## 🔧 Installation

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

### Install via pip (Coming Soon)

```bash
pip install spoon-ai
```

## 🔑 API Key Configuration

SDCF supports various API services that require different API keys. Here are the configuration methods for the main API keys:

### Configuration Methods

1. **Via CLI Command**:
```bash
> config <key_name> <key_value>
```

2. **Via Environment Variables**:
Set the corresponding environment variables in your system

3. **Via Configuration File**:
Edit the `~/.config/spoonai/config.json` file

### Common API Keys

| Key Name | Description | How to Obtain |
|------------|------|---------|
| `OPENAI_API_KEY` | OpenAI API key for GPT models | [OpenAI Website](https://platform.openai.com/api-keys) |
| `ANTHROPIC_API_KEY` | Anthropic API key for Claude models | [Anthropic Website](https://console.anthropic.com/keys) |
| `DEEPSEEK_API_KEY` | DeepSeek API key | [DeepSeek Website](https://platform.deepseek.com/) |
| `PRIVATE_KEY` | Blockchain wallet private key for cryptocurrency transactions | Export from your wallet |

### Configuration Examples

```bash
# Configure OpenAI API key
> config OPENAI_API_KEY sk-your-openai-api-key
OPENAI_API_KEY updated

# Configure Anthropic API key
> config ANTHROPIC_API_KEY sk-ant-your-anthropic-api-key
ANTHROPIC_API_KEY updated

# Configure wallet private key
> config PRIVATE_KEY your-private-key-here
PRIVATE_KEY updated
```

### Key Security Considerations

1. API keys are sensitive; never share them with others or expose them in public
2. Wallet private keys are especially important; leakage may result in asset loss
3. It is recommended to store keys using environment variables or configuration files rather than entering them directly in the command line
4. Regularly change API keys to improve security

## 🚀 Quick Start

### Start the CLI

```bash
python main.py
```

After entering the interactive command line interface, you can start using the various features of SpoonAI.

### Basic Example

```python
from spoon_ai.agents import SpoonChatAI
from spoon_ai.chat import ChatBot

# Create a chat agent
chat_agent = SpoonChatAI(llm=ChatBot())

# Run the agent and get a response
response = await chat_agent.run("Hello, please introduce yourself")
print(response)
```

### Create a ReAct Agent

```python
from spoon_ai.agents import SpoonReactAI
from spoon_ai.chat import ChatBot

# Create a ReAct agent
react_agent = SpoonReactAI(llm=ChatBot())

# Run the ReAct agent and get a response
response = await react_agent.run("Analyze the transaction history of this wallet address: 0x123...")
print(response)
```

## 💡 Usage Examples

### Chat Assistant

```python
from spoon_ai.agents import SpoonChatAI
from spoon_ai.chat import ChatBot

# Create an advanced chat agent
chat_agent = SpoonChatAI(
    llm=ChatBot(model="gpt-4"),  # Use specified model
    system_prompt="You are an AI assistant focused on cryptocurrency, proficient in blockchain technology, DeFi, and NFTs."
)

# Run the agent
response = await chat_agent.run("What are the main technical improvements in Ethereum 2.0?")
print(response)
```

### Cryptocurrency Trading Assistant

```python
from spoon_ai.agents import SpoonReactAI
from spoon_ai.chat import ChatBot
from spoon_ai.tools import ToolManager
from spoon_ai.tools.crypto import TokenInfoTool, SwapTool, TransferTool

# Create a tool manager and add cryptocurrency-related tools
tool_manager = ToolManager([
    TokenInfoTool(),
    SwapTool(),
    TransferTool()
])

# Create a cryptocurrency trading agent
crypto_agent = SpoonReactAI(
    llm=ChatBot(model="gpt-4"),
    avaliable_tools=tool_manager,
    system_prompt="You are a cryptocurrency trading assistant that can help users get token information, exchange tokens, and make transfers."
)

# Run the agent
response = await crypto_agent.run("Help me check the current price of ETH and analyze if it's a good time to buy")
print(response)
```

### Document Analysis Assistant

```python
from spoon_ai.agents import SpoonReactAI
from spoon_ai.chat import ChatBot
from spoon_ai.tools.docs import LoadDocsTool, QueryDocsTool

# Create a document analysis agent
docs_agent = SpoonReactAI(
    llm=ChatBot(),
    avaliable_tools=ToolManager([
        LoadDocsTool(),
        QueryDocsTool()
    ]),
    system_prompt="You are a document analysis assistant who can help users load and analyze various documents."
)

# Run the agent
response = await docs_agent.run("Load all PDF files in the './docs' directory, then summarize their main content")
print(response)
```

## 🛠️ CLI Tools

SCDF CLI is a powerful command-line tool that provides rich functionality, including interacting with AI agents, managing chat history, processing cryptocurrency transactions, and loading documents.

### Basic Commands

| Command | Aliases | Description |
|------|------|------|
| `help` | `h`, `?` | Display help information |
| `exit` | `quit`, `q` | Exit the CLI |
| `load-agent <name>` | `load` | Load an agent with the specified name |
| `list-agents` | `agents` | List all available agents |
| `config` | `cfg`, `settings` | Configure settings (such as API keys) |
| `reload-config` | `reload` | Reload the current agent's configuration |
| `action <action>` | `a` | Perform a specific action using the current agent |

### Chat Management Commands

| Command | Aliases | Description |
|------|------|------|
| `new-chat` | `new` | Start a new chat (clear history) |
| `list-chats` | `chats` | List available chat history records |
| `load-chat <ID>` | - | Load a specific chat history record |

### Cryptocurrency-Related Commands

| Command | Aliases | Description |
|------|------|------|
| `transfer <address> <amount> <token>` | `send` | Transfer tokens to a specified address |
| `swap <source_token> <target_token> <amount>` | - | Exchange tokens using an aggregator |
| `token-info <address>` | `token` | Get token information by address |
| `token-by-symbol <symbol>` | `symbol` | Get token information by symbol |

### Document Management Commands

| Command | Aliases | Description |
|------|------|------|
| `load-docs <directory_path>` | `docs` | Load documents from the specified directory to the current agent |

### CLI Usage Examples

#### Basic Interaction

1. Start the CLI and load an agent:
```
> load-agent chat
chat agent loaded
```

2. Start a new chat:
```
> new-chat
New chat session started
```

3. Directly input text to interact with the AI agent:
```
> Hello, please introduce yourself
[AI reply will be displayed here]
```

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

## 📡 MCP+ (Model Context Protocol plus)

<div align="center">
  <h3>🌐 Connect • Orchestrate • Scale 🌐</h3>
  <p><strong>Capture data availability and scalability of SpoonOS</strong></p>
</div>

MCP+ is SpoonOS’s capability extension of the original MCP. It places a stronger focus on integrating MCP servers with enhanced data availability and scalability, and is deeply integrated with existing components like NeoFS and BeVec (SpoonOS’s vector database solution). This allows developers to access data and invoke tools more easily and efficiently.

### ✨ Key Features

- **🫎 Unified Data Access Layer** - Abstracts diverse data sources into a standardized interface for AI agents
- **⚡️ Streaming Responses** - Real-time streaming output from language models
- **📈 Modular Integration** - Enables dynamic loading of external APIs, on-chain data, or local resources
- **📡 Access Control & Permissioning** - Supports granular permissions and scoped data/task access

<div align="center">
  <pre>
  User → [Reasoning] → [Planning] → [Reflecting] → Final Response
             ↓               ↑
        [Data Analyst] ←→ [MCP Servers]
  </pre>
</div>

### 🚀 Quick Example

```python
from spoon_ai.mcp import MCPConfig, MCPAgentAdapter
from spoon_ai.agents.custom_agent import CustomAgent

async def main():
    # Initialize the adapter
    adapter = MCPAgentAdapter(config=MCPConfig(server_url="ws://localhost:8765"))
    await adapter.connect()
    
    # Create and start an agent
    agent_id = await adapter.create_custom_agent(
        name="test_agent",
        description="Test agent",
        system_prompt="You are a helpful assistant."
    )
    await adapter.agent_subscribe(agent_id, "test_topic")
    await adapter.start_agent(agent_id)
    
    # Send a message to the agent
    await adapter.send_message_to_agent(
        agent_id=agent_id,
        message="Hello, can you help me?",
        sender_id="user",
        topic="test_topic"
    )
```

### 🔌 Running MCP Server and Client

To use MCP with all available tools (optional), you need to run both the MCP server and client:

#### Starting the MCP Server

```bash
# Start the MCP server with all available tools
python spoon_ai/tools/mcp_tools_collcetion.py
```

#### Starting the MCP Client in CLI

```bash
# Launch the CLI
python main.py

# Load the MCP-enabled agent
> load-agent spoon_react_mcp
spoon_react_mcp agent loaded
```

For comprehensive documentation and examples, see the [MCP README](spoon_ai/mcp/README.md).

## 🧩 Agent Framework

SDCF provides a powerful Agent framework that supports two ways of use:
1. Using predefined Agents - Simple declaration and execution
2. Custom Agents - Creating your own tools and logic

### ReAct Intelligent Agent

SDCF implements an intelligent agent based on the ReAct (Reasoning + Acting) paradigm, which is an advanced AI agent architecture that combines reasoning and action capabilities. The ReAct agent can think, plan, and execute in complex tasks, solving problems through an iterative reasoning-action loop.

#### ReAct Workflow

The ReAct agent workflow includes the following key steps:

1. **Observation**: Collecting environment and task-related information
2. **Reasoning**: Analyzing information and reasoning
3. **Acting**: Executing specific operations
4. **Feedback**: Obtaining action results and updating cognition

This cycle repeats continuously until the task is completed or the preset goal is achieved.

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
    AnotherTool(),
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
    AnotherTool(),
    ThirdTool(),
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

## 🔌 API Integration

SpoonAI supports multiple AI service providers, including:

- **OpenAI** - GPT-3.5/GPT-4 series models
- **Anthropic** - Claude series models
- **DeepSeek** - DeepSeek series models
- **More...** - Easily extendable to support other AI providers

### Integration Examples

```python
from spoon_ai.chat import ChatBot
from spoon_ai.agents import SpoonChatAI

# Using OpenAI's GPT-4
openai_agent = SpoonChatAI(
    llm=ChatBot(model="gpt-4", provider="openai")
)

# Using Anthropic's Claude
claude_agent = SpoonChatAI(
    llm=ChatBot(model="claude-3-opus-20240229", provider="anthropic")
)

# Using DeepSeek
deepseek_agent = SpoonChatAI(
    llm=ChatBot(model="deepseek-llm", provider="deepseek")
)
```

## 💼 Enterprise Application Scenarios

SpoonAI can be applied to various enterprise scenarios:

- **Financial Analysis** - Cryptocurrency market analysis, investment advice, risk assessment
- **Customer Service** - Intelligent customer service, problem-solving, ticket processing
- **Document Processing** - Contract analysis, report generation, content summarization
- **Business Automation** - Process automation, task coordination, intelligent decision support
- **Research Assistant** - Information retrieval, data analysis, research report generation

## 🔍 Advanced Features

### Tool Chain Orchestration

SDCF supports complex tool chain orchestration, allowing the creation of multi-step, multi-tool execution flows:

```python
from spoon_ai.tools import ToolChain
from spoon_ai.tools.crypto import TokenInfoTool, PriceAnalysisTool

# Create a tool chain
tool_chain = ToolChain([
    (TokenInfoTool(), "Get token information"),
    (PriceAnalysisTool(), "Analyze price trends")
])

# Execute the tool chain
result = await tool_chain.execute("ETH")
```

### Event Listening and Callbacks

SDCF provides a powerful event system that supports registering callbacks at different stages of agent execution:

```python
from spoon_ai.callbacks import register_callback

# Register before execution callback
@register_callback("before_execution")
async def before_execution_callback(agent, query):
    print(f"Agent {agent.name} is about to execute query: {query}")

# Register after execution callback
@register_callback("after_execution")
async def after_execution_callback(agent, query, result):
    print(f"Agent {agent.name} completed execution with result: {result}")
```

## 🎯 Project Roadmap

- [ ] **Web Interface** - Develop a web-based user interface
- [ ] **Agent Marketplace** - Create a sharing platform for agents and tools
- [ ] **Agent Interoperability** - Implement collaboration capabilities between multiple agents
- [ ] **Local Model Support** - Add support for locally running open-source models
- [ ] **Plugin System** - Build an extensible plugin architecture
- [ ] **Advanced Monitoring** - Enhance agent execution monitoring and analysis capabilities
- [ ] **Multi-Language Support** - Extend support for more languages
- [ ] **Cloud Deployment** - Simplify cloud environment deployment process

## 🤝 Contributing

We welcome contributions of all forms!

1. Fork this repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Create a Pull Request

Please ensure you follow our code style and contribution guidelines.

## 📄 License

This project is licensed under the [MIT License](LICENSE).

## 🌟 Acknowledgements

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
