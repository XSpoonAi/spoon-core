# Configuration Examples

This directory contains example configuration files demonstrating various SpoonAI agent setups.

## Files

### `config_with_mcp.json`

A comprehensive configuration example showing:

- **Multiple Agent Types**: Search, crypto, development, and general-purpose agents
- **MCP Server Integration**: Tavily for web search and GitHub for code management
- **Tool Set Configuration**: Built-in and MCP-based tool sets
- **Agent Specialization**: Different agents optimized for specific tasks

## Usage

1. Copy the desired configuration file to your working directory as `config.json`
2. Update API keys and environment variables
3. Install required MCP servers:
   ```bash
   # For Tavily search
   npm install -g tavily-mcp
   
   # For GitHub integration
   npm install -g @modelcontextprotocol/server-github
   ```
4. Set environment variables:
   ```bash
   export TAVILY_API_KEY="your-tavily-api-key"
   export GITHUB_PERSONAL_ACCESS_TOKEN="your-github-token"
   ```
5. Start SpoonAI and load your desired agent

## Agent Types

### Search Agent (`search_agent`)
- **Purpose**: Web search and information retrieval
- **Tools**: Tavily web search + crypto tools
- **Best for**: Research, fact-checking, current events

### Crypto Agent (`crypto_agent`)
- **Purpose**: Cryptocurrency and blockchain operations
- **Tools**: Built-in crypto tools only
- **Best for**: Trading analysis, DeFi operations, token research

### Development Agent (`dev_agent`)
- **Purpose**: Software development and code management
- **Tools**: GitHub integration + crypto tools
- **Best for**: Code review, repository management, development tasks

### General Agent (`general_agent`)
- **Purpose**: Multi-purpose agent with all capabilities
- **Tools**: All available tool sets
- **Best for**: Complex tasks requiring multiple tool types

### `x402_agent_demo.py`
- **Purpose**: Showcase x402 payment setup, header signing, and agent tooling
- **Tools**: `X402PaymentService`, `x402_create_payment`, `x402_paywalled_request`
- **Usage**: `uv run python examples/x402_agent_demo.py`
- **Best for**: Understanding how agents initiate payments and how paywall negotiations work

## MCP Server Configuration

### Tavily Search
```json
{
  "tavily-mcp": {
    "command": "npx",
    "args": ["-y", "tavily-mcp"],
    "env": {
      "TAVILY_API_KEY": "your-api-key"
    }
  }
}
```

### GitHub Integration
```json
{
  "github-mcp": {
    "command": "npx",
    "args": ["-y", "@modelcontextprotocol/server-github"],
    "env": {
      "GITHUB_PERSONAL_ACCESS_TOKEN": "your-token"
    }
  }
}
```

## Tool Set Types

### Built-in Tools
- **crypto_tools**: Cryptocurrency analysis and trading tools
- **Type**: `builtin`
- **Configuration**: Enable/disable only

### MCP Server Tools
- **web_search**: Web search via Tavily
- **github_tools**: GitHub repository management
- **Type**: `mcp_server`
- **Configuration**: Requires MCP server setup

## Best Practices

1. **Start Simple**: Begin with basic agents and add complexity gradually
2. **Test MCP Servers**: Verify MCP server connectivity before deployment
3. **Secure API Keys**: Use environment variables for sensitive data
4. **Agent Specialization**: Create specialized agents for specific workflows
5. **Tool Optimization**: Only enable tools needed for each agent's purpose

## Troubleshooting

### MCP Server Issues
- Verify Node.js and npm are installed
- Check network connectivity
- Ensure API keys are valid
- Review server logs for errors

### Agent Loading Problems
- Validate JSON syntax
- Check agent class names
- Verify tool set references
- Review configuration file path

## Advanced Configuration

For more advanced configurations, see the main documentation at `doc/agent_configuration.md`.

---

## Turnkey Examples (`examples/turnkey`)

Turnkey integration demos for secure key management and signing via Turnkey API.

- Run from project root, ensure Python path includes the repo root.
- Copy env template and fill required values:
  ```bash
  cp examples/turnkey/env.example .env
  ```

### Scripts
- `examples.turnkey.build_unsigned_eip1559_tx`: Build a minimal unsigned EIP-1559 tx and print `TURNKEY_UNSIGNED_TX_HEX`.
  ```bash
  python -m examples.turnkey.build_unsigned_eip1559_tx
  ```
- `examples.turnkey.turnkey_trading_use_case`: Guided demo for tx signing, broadcasting (optional), message signing, EIP-712, and audit.
  ```bash
  python -m examples.turnkey.turnkey_trading_use_case
  ```
- `examples.turnkey.multi_account_use_case`: Enumerate wallets/accounts, per-account tx signing, optional broadcast, per-account message signing, and audit.
  ```bash
  python -m examples.turnkey.multi_account_use_case
  ```

### Requirements
- Uses repo-level `requirements.txt`. If broadcasting or tx building:
  - `web3`, `eth-utils`, `rlp` are required (already pinned in repo requirements).
- Ensure `.env` includes:
  - `TURNKEY_BASE_URL`, `TURNKEY_API_PUBLIC_KEY`, `TURNKEY_API_PRIVATE_KEY`, `TURNKEY_ORG_ID`
  - `TURNKEY_SIGN_WITH` (address or private key ID)
  - Optional: `TURNKEY_UNSIGNED_TX_HEX`, `WEB3_RPC_URL`, and tx params.
