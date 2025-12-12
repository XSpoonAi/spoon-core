"""
ERC-8004 DID Client (fetch on-chain agent card + ask server)
------------------------------------------------------------

Responsibilities:
- Read ERC-8004 IdentityRegistry to discover an agent's tokenURI + metadata.
- Resolve DID / AgentCard URI (NeoFS/IPFS) if present.
- Send a question to the local server (server_agent.py) and print the answer.

Env / CLI:
    NEOX_RPC_URL              RPC endpoint (default: https://testnet.rpc.banelabs.org)
    NEOX_CHAIN_ID             Chain ID (default: 12227332)
    NEOX_IDENTITY_REGISTRY    IdentityRegistry address (required)
    ERC8004_AGENT_ID          Agent tokenId to inspect (default: 1)
    ERC8004_SERVER_URL        Server base URL (default: http://127.0.0.1:8004)
"""

from __future__ import annotations

import argparse
import json
import os
import time
os.environ.setdefault("WEB3_ENABLE_CKZG", "0")
from typing import Any, Dict

import requests
from web3 import Web3
from web3.middleware import ExtraDataToPOAMiddleware
from spoon_ai.identity.erc8004_abi import IDENTITY_ABI_MIN
from spoon_ai.identity.erc8004_client import ERC8004Client

def _connect_w3(rpc_url: str, chain_id: int) -> Web3:
    w3 = Web3(Web3.HTTPProvider(rpc_url))
    if chain_id in (12227332, 97, 56, 11155111):  # common PoA chains
        w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
    assert w3.is_connected(), f"Cannot connect to {rpc_url}"
    return w3


def fetch_agent_info(registry_addr: str, agent_id: int, rpc_url: str, chain_id: int) -> Dict[str, Any]:
    client = ERC8004Client(
        rpc_url=rpc_url,
        agent_registry_address=os.getenv("NEOX_AGENT_REGISTRY") or "0x0000000000000000000000000000000000000000",
        identity_registry_address=registry_addr,
        reputation_registry_address=os.getenv("NEOX_REPUTATION_REGISTRY") or "0x0000000000000000000000000000000000000000",
        validation_registry_address=os.getenv("NEOX_VALIDATION_REGISTRY") or "0x0000000000000000000000000000000000000000",
        private_key=None,
    )

    # Direct web3 call for totalAgents (no private key needed)
    w3 = _connect_w3(rpc_url, chain_id)
    contract = w3.eth.contract(address=Web3.to_checksum_address(registry_addr), abi=IDENTITY_ABI_MIN)
    total = contract.functions.totalAgents().call()
    if total == 0:
        raise ValueError("registry has no agents (totalAgents=0)")
    if agent_id > total:
        raise ValueError(f"agentId {agent_id} exceeds totalAgents={total}")

    token_uri = client.identity_registry.functions.tokenURI(agent_id).call()
    did_uri_bytes = client.identity_registry.functions.getMetadata(agent_id, "did_uri").call()
    card_uri_bytes = client.identity_registry.functions.getMetadata(agent_id, "card_uri").call()
    return {
        "agent_id": agent_id,
        "token_uri": token_uri,
        "did_uri": did_uri_bytes.decode(errors="ignore") if did_uri_bytes else "",
        "card_uri": card_uri_bytes.decode(errors="ignore") if card_uri_bytes else "",
    }


def ask_server(server_url: str, question: str) -> Dict[str, Any]:
    url = f"{server_url.rstrip('/')}/ask"
    last_err = None
    for _ in range(5):
        try:
            resp = requests.post(url, json={"question": question}, timeout=180)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            last_err = exc
            time.sleep(2)
    raise last_err or Exception("Failed to reach server")


def main() -> None:
    parser = argparse.ArgumentParser(description="ERC-8004 DID client demo")
    parser.add_argument("--rpc", default=os.getenv("NEOX_RPC_URL", "https://testnet.rpc.banelabs.org"))
    parser.add_argument("--chain-id", type=int, default=int(os.getenv("NEOX_CHAIN_ID", "12227332")))
    parser.add_argument("--registry", default=os.getenv("NEOX_IDENTITY_REGISTRY"))
    parser.add_argument("--agent-id", type=int, default=int(os.getenv("ERC8004_AGENT_ID", "3")))
    parser.add_argument("--server", default=os.getenv("ERC8004_SERVER_URL", "http://127.0.0.1:8004"))
    parser.add_argument(
        "--question",
        default="ERC-8004 latest ecosystem? How to register an agent and publish DID on NeoFS?",
    )
    parser.add_argument("--token-uri", default=os.getenv("AGENT_CARD_URI") or os.getenv("AGENT_DID_URI"))
    parser.add_argument("--did-uri", default=os.getenv("AGENT_DID_URI"))
    args = parser.parse_args()

    if not args.registry:
        raise SystemExit("NEOX_IDENTITY_REGISTRY is required (set env or --registry)")

    print(f"Fetching agent {args.agent_id} from IdentityRegistry {args.registry} ...")
    try:
        info = fetch_agent_info(args.registry, args.agent_id, args.rpc, args.chain_id)
    except ValueError as exc:
        if args.token_uri or args.did_uri:
            info = {
                "agent_id": args.agent_id,
                "token_uri": args.token_uri or "",
                "did_uri": args.did_uri or "",
                "card_uri": args.token_uri or "",
                "warning": str(exc),
            }
            print(f"Registry empty or missing agent; using provided URIs. {exc}")
        else:
            raise
    print(json.dumps(info, indent=2))

    print(f"\nAsking server at {args.server} ...")
    result = ask_server(args.server, args.question)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
