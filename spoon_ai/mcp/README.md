# 🌐 SpoonOS MCP+ (Model Context Protocol plus)

<div align="center">
  <h3>Unified • Scalable •Context-Aware</h3>
  <p><strong>The data availability and context backbone of SpoonOS — empowering agents with scalable, real-time knowledge.</strong></p>
  
  [![Python 3.12+](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/downloads/)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
  [![Async Support](https://img.shields.io/badge/Async-Supported-green.svg)](https://docs.python.org/3/library/asyncio.html)
</div>

<hr>

## What is MCP+?

MCP+ (Model Context Protocol Plus) is SpoonOS’s unified data access and action interface for AI agents. It provides a structured, scalable protocol for connecting large language models to real-time, permissioned, and decentralized data sources — across both Web2 and Web3.

Unlike traditional context injection methods, MCP+ allows agents to access external knowledge and trigger tool-based actions in a modular, verifiable way — whether through direct API calls, on-chain endpoints, or vectorized semantic memory via BeVec.
---

## Architecture

MCP+ consists of three core layers, each representing a distinct functional capability within SpoonOS’s data interaction stack. SpoonOS maintains and hosts a default MCP Client, designed to interoperate with multiple MCP-compliant Servers across diverse data environments.

### 1. Data Adapters
Provide a unified interface to ingest structured and unstructured data from heterogeneous sources — including Web2 APIs, Web3 RPC endpoints, decentralized storage protocols like NeoFS, and on-chain smart contract data.  
These adapters normalize and abstract data retrieval for use by downstream agents.

### 2. Access Controller
Manages authentication, request validation, and permission gating.  
It can incorporate identity-bound controls using agent DID profiles or token-based access, ensuring that only authorized agents can query or act on sensitive data streams.

### 3. Query Engine
Powers semantic and symbolic reasoning. Supports both:
- **Vector-based retrieval** via the native BeVec memory layer (for context-rich grounding),
- **Structured querying** for fetching deterministic facts or invoking external data-computation tools.

Together, this architecture enables AI agents to reason over, react to, and act on real-world data in a verifiable and composable way — forming the foundation of intelligent behavior in SpoonOS.

### 4. MCP+ Architecture Diagram (Text Version)

```text
            +----------------------+
            |    SpoonOS Agent      |
            +----------------------+
                      |
                      v
            +----------------------+
            |    MCP+ Client        |
            +----------------------+
                      |
        +----------------------------+
        |       Query Engine          |
        |  (Vector Retrieval & Tools) |
        +----------------------------+
                      |
        +----------------------------+
        |      Access Controller      |
        | (Permissions / Auth / DID)   |
        +----------------------------+
                      |
        +----------------------------+
        |         Data Adapters       |
        | (Web2 APIs / Web3 RPC / NeoFS)|
        +----------------------------+
                      |
        +----------------------------+
        | External Data & Resources   |
        +----------------------------+
```
---

## Core Components

MCP+ is built around modular, flexible components that give AI agents scalable and verifiable access to real-world knowledge and tools:

- **MCP+ Client**  
  Acts as the gateway between SpoonOS agents and external data sources, handling queries, tool calls, and embedding generation.

- **Query Engine**  
  Processes incoming agent requests, performing either structured data fetching or semantic retrieval from BeVec vector memory.

- **Access Controller**  
  Manages authorization and identity verification for agent queries. Supports DID-based permissioning and fine-grained access control.

- **Data Adapters**  
  Unified wrappers for Web2 APIs, Web3 RPC endpoints, decentralized storage systems (like NeoFS), and other data services.

- **BeVec Vector Database**  
  SpoonOS’s self-developed high-performance vector database, optimized for RAG (Retrieval-Augmented Generation) use cases.

---

## Example Workflows

Here are some simple examples of how agents can interact through MCP+:

**1. Real-Time Web3 Data Query**

An agent wants to fetch the latest on-chain transaction history for a user.

```flow
Agent → MCP+ Client → Query Engine → Web3 RPC Adapter → Blockchain Data
```
**2. Accessing Decentralized Storage**

An agent retrieves a document from NeoFS based on user intent.

```flow
Agent → MCP+ Client → Data Adapter (NeoFS) → File Retrieval
```
**3. RAG-based Knowledge Augmentation**

An agent performs semantic search across indexed research papers to answer a user’s technical question.

```flow
Agent → MCP+ Client → Query Engine (Vector Search) → BeVec → Contextualized Answer
```
**4.  Tool Invocation Example**

An agent triggers a third-party API (like a real-time price feed) to make a trading decision.

```flow
Agent → MCP+ Client → Data Adapter (Web2 API) → Price Information
```

## 🚀 Quick Start with MCP+

### 1. Install MCP+ Client

```bash
git clone https://github.com/spoonos-ai/mcp-plus.git
cd mcp-plus/python-client
pip install -r requirements.txt
```
MCP+ Python Client allows your agents to connect to decentralized and permissioned data sources easily.

### 2. Connect an Agent
In your SpoonOS Agent setup file:
```python
from mcp_client import MCPClient

mcp = MCPClient(
    api_key='your-api-key',
    bevec_endpoint='https://bevec.spoonos.ai',
    adapters=['web3', 'neofs', 'openweather']
)

# Execute a semantic query
context = mcp.query(
    query_type='semantic',
    input_text='latest ETH staking APR'
)

print(context)
```

### 3. Add a Custom Adapter
You can easily extend MCP+ by writing your own adapter:
```python
def coin_gecko_adapter(query):
    import requests
    response = requests.get('https://api.coingecko.com/api/v3/simple/price?ids=ethereum&vs_currencies=usd')
    return response.json()

mcp.register_adapter('coingecko', coin_gecko_adapter)
```

### 4. Run the Local MCP+ Client
You can easily extend MCP+ by writing your own adapter:
```bash
python app.py
```
Your MCP+ client is now running and ready to serve your AI agents with real-time, verifiable external data.

For detailed implementation examples, refer to the `agent_integration.py` file in the MCP module.
