"""
ERC-8004 DID Search Agent (Server)
----------------------------------

- ReAct-style agent powered by spoon_ai.agents.spoon_react_mcp
- Uses Tavily MCP tool for live web search (requires TAVILY_API_KEY)
- Focuses on answering questions about ERC-8004 agents/standard
- Exposes a very small HTTP JSON API on /ask for demo purposes

Run:
    python -m core.examples.erc8004_did.server_agent --port 8004

Env (or CLI flags):
    TAVILY_API_KEY          Tavily API key (required)
    ERC8004_AGENT_NAME      Optional display name returned with answers
    ERC8004_AGENT_DID_URI   Optional DID/metadata URI to echo in responses
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
os.environ.setdefault("WEB3_ENABLE_CKZG", "0")
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Optional

from spoon_ai.agents.spoon_react_mcp import SpoonReactMCP
from spoon_ai.chat import ChatBot
from spoon_ai.tools.mcp_tool import MCPTool
from spoon_ai.tools.tool_manager import ToolManager
from spoon_ai.identity.erc8004_client import ERC8004Client


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("erc8004.server")


class ERC8004SearchAgent(SpoonReactMCP):
    """ReAct agent specialized for ERC-8004 Q&A with live web search."""

    name: str = "ERC8004SearchAgent"
    system_prompt: str = (
        "You are an on-chain ERC-8004 agent.\n"
        "- Use tavily-search to fetch recent info about ERC-8004, agent registries, and ecosystem projects.\n"
        "- Explain concisely, cite key projects (Identity, Reputation, Validation registries), "
        "and highlight how to register/resolve agents.\n"
        "- Prefer actionable steps. Keep answers under 8 sentences."
    )

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.available_tools = ToolManager([])

    async def initialize(self) -> None:
        tavily_key = os.getenv("TAVILY_API_KEY", "").strip()
        if not tavily_key or "your-tavily-api-key" in tavily_key:
            raise ValueError("TAVILY_API_KEY is missing; tavily-search cannot run.")

        tavily_tool = MCPTool(
            name="tavily-search",
            description="Web search via Tavily MCP server",
            mcp_config={
                "command": "npx",
                "args": ["--yes", "tavily-mcp"],
                "env": {"TAVILY_API_KEY": tavily_key},
            },
        )
        self.available_tools = ToolManager([tavily_tool])
        logger.info("Loaded tools: %s", list(self.available_tools.tool_map.keys()))


class _RequestHandler(BaseHTTPRequestHandler):
    agent: Optional[ERC8004SearchAgent] = None
    loop: Optional[asyncio.AbstractEventLoop] = None
    agent_name: str = "ERC8004 Search Agent"
    agent_did_uri: Optional[str] = None

    def _send_json(self, code: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # type: ignore
        if self.path != "/ask":
            self._send_json(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            data = json.loads(self.rfile.read(length) or b"{}")
            question = str(data.get("question", "")).strip()
        except Exception as exc:  # pragma: no cover - demo handler
            logger.exception("Malformed request: %s", exc)
            self._send_json(400, {"error": "bad request"})
            return

        if not question:
            self._send_json(400, {"error": "question is required"})
            return

        assert self.agent and self.loop
        fut = asyncio.run_coroutine_threadsafe(self.agent.run(question), self.loop)
        try:
            answer = fut.result(timeout=120)
        except Exception as exc:  # pragma: no cover - demo handler
            logger.exception("Agent failed: %s", exc)
            self._send_json(500, {"error": "agent_error", "detail": str(exc)})
            return

        self._send_json(
            200,
            {
                "agent": self.agent_name,
                "did_uri": self.agent_did_uri,
                "question": question,
                "answer": answer,
            },
        )

    def log_message(self, fmt: str, *args: Any) -> None:  # type: ignore
        logger.info("%s - %s", self.address_string(), fmt % args)


async def _prepare_agent(agent_name: str, agent_did_uri: Optional[str]) -> ERC8004SearchAgent:
    agent = ERC8004SearchAgent(llm=ChatBot(llm_provider="openai"))
    await agent.initialize()
    logger.info("Agent ready: %s (DID=%s)", agent_name, agent_did_uri)
    return agent


def _register_on_chain_if_requested(
    rpc_url: Optional[str],
    chain_id: int,
    private_key: Optional[str],
    agent_card_uri: Optional[str],
    did_uri: Optional[str],
    did_doc_uri: Optional[str],
    identity_registry: Optional[str],
    reputation_registry: Optional[str],
    validation_registry: Optional[str],
    agent_registry: Optional[str],
) -> Optional[int]:
    if not all(
        [
            rpc_url,
            private_key,
            agent_card_uri,
            did_uri,
            identity_registry,
            reputation_registry,
            validation_registry,
            agent_registry,
        ]
    ):
        logger.info("Skip on-chain registration (missing config).")
        return None
    client = ERC8004Client(
        rpc_url=rpc_url,
        agent_registry_address=agent_registry,
        identity_registry_address=identity_registry,
        reputation_registry_address=reputation_registry,
        validation_registry_address=validation_registry,
        private_key=private_key,
    )
    try:
        metadata = []
        if did_uri:
            metadata.append(("did_uri", did_uri.encode()))
        if did_doc_uri:
            metadata.append(("did_doc_uri", did_doc_uri.encode()))
        agent_id = client.register_agent(agent_card_uri, metadata=metadata or None)
        logger.info("Registered agentId=%s", agent_id)
        return agent_id
    except Exception as exc:
        logger.error("Register failed: %s", exc)
        return None


def run_server(port: int, agent_name: str, agent_did_uri: Optional[str]) -> None:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    agent = loop.run_until_complete(_prepare_agent(agent_name, agent_did_uri))

    _RequestHandler.agent = agent
    _RequestHandler.loop = loop
    _RequestHandler.agent_name = agent_name
    _RequestHandler.agent_did_uri = agent_did_uri

    server = HTTPServer(("0.0.0.0", port), _RequestHandler)
    logger.info("HTTP server listening on http://127.0.0.1:%d/ask", port)

    try:
        loop.run_in_executor(None, server.serve_forever)
        loop.run_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        server.shutdown()
        loop.stop()
        loop.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="ERC-8004 DID Search Agent server")
    parser.add_argument("--port", type=int, default=int(os.getenv("ERC8004_AGENT_PORT", "8004")))
    parser.add_argument("--agent-name", default=os.getenv("ERC8004_AGENT_NAME", "ERC8004 Search Agent"))
    parser.add_argument("--did-uri", default=os.getenv("ERC8004_AGENT_DID_URI"))
    parser.add_argument("--did-doc-uri", default=os.getenv("AGENT_DID_DOC_URI") or os.getenv("DID_DOC_URI"))
    parser.add_argument("--register-agent", action="store_true", help="Register agent on-chain before serving")
    parser.add_argument("--registry", default=os.getenv("NEOX_IDENTITY_REGISTRY"))
    parser.add_argument("--reputation", default=os.getenv("NEOX_REPUTATION_REGISTRY"))
    parser.add_argument("--validation", default=os.getenv("NEOX_VALIDATION_REGISTRY"))
    parser.add_argument("--agent-registry", default=os.getenv("NEOX_AGENT_REGISTRY") or os.getenv("AGENT_REGISTRY"))
    parser.add_argument("--rpc", default=os.getenv("NEOX_RPC_URL", "https://testnet.rpc.banelabs.org"))
    parser.add_argument("--chain-id", type=int, default=int(os.getenv("NEOX_CHAIN_ID", "12227332")))
    parser.add_argument(
        "--private-key",
        default=os.getenv("PRIVATE_KEY") or os.getenv("NEOX_PRIVATE_KEY") or os.getenv("REACT_PRIVATE_KEY"),
    )
    parser.add_argument(
        "--token-uri",
        default=os.getenv("AGENT_CARD_URI") or os.getenv("AGENT_DID_URI"),
        help="Agent Card URI used as tokenURI in register()",
    )
    args = parser.parse_args()

    if args.register_agent:
        _register_on_chain_if_requested(
            rpc_url=args.rpc,
            chain_id=args.chain_id,
            private_key=args.private_key,
            agent_card_uri=args.token_uri,
            did_uri=args.did_uri,
            did_doc_uri=args.did_doc_uri,
            identity_registry=args.registry,
            reputation_registry=args.reputation,
            validation_registry=args.validation,
            agent_registry=args.agent_registry,
        )

    run_server(args.port, args.agent_name, args.did_uri)


if __name__ == "__main__":
    main()
