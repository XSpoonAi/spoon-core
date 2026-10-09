# Examples Overview 

Runnable SpoonAI examples and configs. 
- Top-level demos ：`graph_crypto_analysis.py`, `intent_graph_demo.py`, `memory_suite_demo.py`, `my_agent_demo.py`, `neofs-agent-demo.py`, `x402_agent_demo.py`
- MCP demos: `examples/mcp/` — tool calling, Thirdweb Insight, Tavily search. 
- Turnkey demos: `examples/turnkey/` — secure signing, wallets, audit.

---

## Top-level demos 

- `graph_crypto_analysis.py`  
  Declarative crypto analysis graph using real Binance 24h ticker data, PowerData indicators, optional Tavily MCP news. 
  Prereqs: Python deps; `TAVILY_API_KEY` for news (optional); internet access to Binance API.
  Run: `python examples/graph_crypto_analysis.py`

- `intent_graph_demo.py`  
  Intent-based graph with PowerData, optional Tavily MCP, and EVM swap tool. 
  Prereqs: Python deps; `TAVILY_API_KEY` (optional); internet for data/tool calls.  
  Run: `python examples/intent_graph_demo.py`

- `memory_suite_demo.py`  
  Streams responses while showcasing short-term trimming/summarization and Mem0 long-term recall. 
  Modes: `--mode short-term | mem0 | all` (default all).  
  Run: `python examples/memory_suite_demo.py --mode all`

- `my_agent_demo.py`  
  Weather + outfit suggestions using OpenStreetMap geocoding and open-meteo API. 
  Prereqs: internet access to OSM/open-meteo.  
  Run: `python examples/my_agent_demo.py`

- `neofs-agent-demo.py`  
  NeoFS storage agent: containers, eACL, upload/download, bearer tokens.  
  Prereqs: `.env` with NeoFS credentials/endpoint (see script), internet to NeoFS REST.  
  Run: `python examples/neofs-agent-demo.py`

- `x402_agent_demo.py`  
  x402 payment demo: payment creation, paywalled requests, header signing. 
  Prereqs: x402-related keys/config per script; internet for API calls.  
  Run: `python examples/x402_agent_demo.py`

---

## MCP Examples (`examples/mcp`) 

Model Context Protocol demos for calling external tools. 
- [Parallel Search MCP](mcp/parallel_search.md): opt-in, keyless web search and page fetch through `MCPTool`, without an LLM. Run `python -m examples.mcp.parallel_search_demo` after following its setup instructions.
- Scripts: `deepwiki_demo.py`, `mcp_thirdweb_collection.py`, `SpoonThirdWebagent.py`, `spoon_search_agent.py`
- Typical prereqs: Python deps; LLM key; Thirdweb `client_id`; `TAVILY_API_KEY`; port 8765 free for FastMCP.
- Run from repo root, e.g.:
  ```bash
  python -m examples.mcp.deepwiki_demo
  python -m examples.mcp.mcp_thirdweb_collection   # starts FastMCP server
  python -m examples.mcp.SpoonThirdWebagent        # MCP client, start server first
  python -m examples.mcp.spoon_search_agent
  ```

---

## Turnkey Examples (`examples/turnkey`) 

Secure signing and wallet management via Turnkey. 
- Env template 
  ```bash
  cp examples/turnkey/env.example .env
  ```
- Scripts ：
  ```bash
  python examples/turnkey/build_unsigned_eip1559_tx.py    
  python examples/turnkey/turnkey_trading_use_case.py    
  python examples/turnkey/multi_account_use_case.py               
  ```
- Requirements 
  - Repo deps incl. `web3`, `eth-utils`, `rlp`
  - `.env`: `TURNKEY_API_PUBLIC_KEY`, `TURNKEY_API_PRIVATE_KEY`, `TURNKEY_ORG_ID`, `TURNKEY_SIGN_WITH`
  - Optional : `WEB3_RPC_URL` + `TX_*` for building/broadcasting
