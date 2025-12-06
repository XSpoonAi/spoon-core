# ERC-8004 DID Demo (Server + Client)

End-to-end walkthrough for ERC-8004 on NeoX with NeoFS storage:
- **Server**: ReAct agent (`ERC8004SearchAgent`) using spoon_ai core + tavily-search; exposes `/ask`.
- **Client**: Reads on-chain metadata from ERC-8004 IdentityRegistry and asks the server.
- **Storage**: DID and Agent Card stored on NeoFS (T5 testnet).

Everything runs locally on `127.0.0.1`.

## Pre-baked resources
- **NeoX Testnet** RPC: `https://testnet.rpc.banelabs.org` (chainId `12227332`, PoA → `ExtraDataToPOAMiddleware`)
- **ERC-8004 registries** (same as `core/.env`):
  - IdentityRegistry: `0x25fC6fF0D5d64a8e12b1f37c476A49637035Db46`
  - ReputationRegistry: `0x2f39DB667cc8d7460677E98790b911C3c2FdC323`
  - ValidationRegistry: `0x12ce445C862AB2d635cCdD966a41B6128e9278d3`
- **NeoFS artifacts** (container `CK44Vxzo21RED5LAZcBjQAGVQsPW5cLCxnr36vEBXvLG`):
  - DID doc: `neofs://CK44Vxzo21RED5LAZcBjQAGVQsPW5cLCxnr36vEBXvLG/5r1VFgZi5Vnh4woLyYyTpSv67sNJ2E1MC75H21bpKwQE`
  - Agent Card: `neofs://CK44Vxzo21RED5LAZcBjQAGVQsPW5cLCxnr36vEBXvLG/QXxz1MnhzuU7sxDiBfUdbWUGtLNpgFBtQrA2mqowzKb`
- **Pre-registered agent**: `agentId=3` with the above DID/Card already set on-chain.

## Environment variables
```
OPENAI_API_KEY=<openai key>             # required
TAVILY_API_KEY=<valid tavily key>       # required for live search
NEOX_RPC_URL=https://testnet.rpc.banelabs.org
NEOX_CHAIN_ID=12227332
NEOX_IDENTITY_REGISTRY=0x25fC6fF0D5d64a8e12b1f37c476A49637035Db46
NEOX_REPUTATION_REGISTRY=0x2f39DB667cc8d7460677E98790b911C3c2FdC323
NEOX_VALIDATION_REGISTRY=0x12ce445C862AB2d635cCdD966a41B6128e9278d3
NEOX_AGENT_REGISTRY=<optional, if you use SpoonAgentRegistry>
PRIVATE_KEY=<funded NeoX key>           # only needed when --register-agent
ERC8004_AGENT_ID=3                      # default: pre-registered agent
ERC8004_SERVER_URL=http://127.0.0.1:8004
WEB3_ENABLE_CKZG=0                      # avoid ckzg import issues on Windows
```

## One-shot demo
```bash
python -m core.examples.erc8004_did.run_demo
```
What happens:
1) Server starts on `http://127.0.0.1:8004/ask`, loads tavily-search, and uses spoon_ai’s ReAct stack.
2) Client reads `tokenURI` / `did_uri` / `card_uri` for `agentId` from IdentityRegistry, then posts a sample question to `/ask`.

## Optional: register a new agent
```bash
python -m core.examples.erc8004_did.run_demo \
  --register-agent \
  --token-uri neofs://<your-card-oid>
  --registry <identity_registry> \
  --reputation <reputation_registry> \
  --validation <validation_registry> \
  --agent-registry <agent_registry_if_any>
```
Requires `PRIVATE_KEY` with NeoX gas. After registration, you can set on-chain metadata (`did_uri` / `card_uri`) via `setMetadata`.

## Customize
- Change contract addresses via `NEOX_*_REGISTRY`.
- Change NeoFS URIs by writing new `did_uri` / `card_uri` on-chain; the client picks them up automatically.
- To use another agentId, set `ERC8004_AGENT_ID=<id>`.

## Expected output
- Client prints on-chain metadata:
```json
{
  "agent_id": 3,
  "token_uri": "neofs://.../QXxz1Mnhz...",
  "did_uri": "neofs://.../5r1VFgZi...",
  "card_uri": "neofs://.../QXxz1Mnhz..."
}
```
- `/ask` returns a concise ERC-8004 answer enriched with live tavily search results.
