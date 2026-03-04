"""
ERC-8004 DID Client
===================

Fetches on-chain agent metadata and asks the server a question.

Responsibilities:
- Read ERC-8004 IdentityRegistry to discover an agent's tokenURI + metadata
- Resolve DID / AgentCard URI (NeoFS/IPFS) if present
- Send a question to the local server (server_agent.py) and print the answer

Usage:
    python -m examples.erc8004_did.client_agent --agent-id 3

Environment variables:
    NEOX_RPC_URL              RPC endpoint (default: https://testnet.rpc.banelabs.org)
    NEOX_CHAIN_ID             Chain ID (default: 12227332)
    NEOX_IDENTITY_REGISTRY    IdentityRegistry address (required)
    ERC8004_AGENT_ID          Agent token ID to query (default: 3)
    ERC8004_SERVER_URL        Server base URL (default: http://127.0.0.1:8004)

Deployed contracts (NeoX Testnet - open registration):
    IdentityRegistry:   0xaB5623F3DD66f2a52027FA06007C78c7b0E63508
    ReputationRegistry: 0x8bb086D12659D6e2c7220b07152255d10b2fB049
    ValidationRegistry: 0x18A9240c99c7283d9332B738f9C6972b5B59aEc2
"""

from __future__ import annotations

import argparse
import json
import os
import time

os.environ.setdefault("WEB3_ENABLE_CKZG", "0")

from typing import Any, Dict, Optional

import requests
from web3 import Web3
from web3.middleware import ExtraDataToPOAMiddleware
from spoon_ai.identity.erc8004_abi import IDENTITY_ABI_MIN
from spoon_ai.identity.erc8004_client import ERC8004Client
from spoon_ai.identity.storage_client import DIDStorageClient
from spoon_ai.identity.did_models import AgentDID, ServiceEndpoint, ServiceType


# Default contract addresses (NeoX Testnet)
DEFAULT_IDENTITY_REGISTRY = "0xaB5623F3DD66f2a52027FA06007C78c7b0E63508"
DEFAULT_REPUTATION_REGISTRY = "0x8bb086D12659D6e2c7220b07152255d10b2fB049"
DEFAULT_VALIDATION_REGISTRY = "0x18A9240c99c7283d9332B738f9C6972b5B59aEc2"
DEFAULT_RPC_URL = "https://testnet.rpc.banelabs.org"
DEFAULT_CHAIN_ID = 12227332


def _connect_w3(rpc_url: str, chain_id: int) -> Web3:
    """Connect to the EVM network with PoA middleware if needed."""
    w3 = Web3(Web3.HTTPProvider(rpc_url))
    if chain_id in (12227332, 97, 56, 11155111):  # common PoA chains
        w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
    assert w3.is_connected(), f"Cannot connect to {rpc_url}"
    return w3


def fetch_agent_info(
    registry_addr: str,
    agent_id: int,
    rpc_url: str,
    chain_id: int,
) -> Dict[str, Any]:
    """
    Fetch agent metadata from the IdentityRegistry.

    Args:
        registry_addr: IdentityRegistry contract address
        agent_id: Agent token ID to query
        rpc_url: RPC endpoint URL
        chain_id: Chain ID

    Returns:
        Dict with agent_id, token_uri, did_uri, did_doc_uri, card_uri
    """
    client = ERC8004Client(
        rpc_url=rpc_url,
        identity_registry_address=registry_addr,
        reputation_registry_address=os.getenv("NEOX_REPUTATION_REGISTRY", DEFAULT_REPUTATION_REGISTRY),
        validation_registry_address=os.getenv("NEOX_VALIDATION_REGISTRY", DEFAULT_VALIDATION_REGISTRY),
        private_key=None,
    )

    # Direct web3 call for totalAgents (no private key needed)
    w3 = _connect_w3(rpc_url, chain_id)
    contract = w3.eth.contract(
        address=Web3.to_checksum_address(registry_addr),
        abi=IDENTITY_ABI_MIN,
    )

    # Retry a few times in case RPC lags after recent writes
    total = 0
    for _ in range(10):
        total = contract.functions.totalAgents().call()
        if total > 0 and agent_id <= total:
            break
        time.sleep(1)

    if total == 0:
        raise ValueError("Registry has no agents (totalAgents=0)")
    if agent_id > total:
        raise ValueError(f"Agent ID {agent_id} exceeds totalAgents={total}")

    # Fetch token URI and metadata
    token_uri = client.identity_registry.functions.tokenURI(agent_id).call()
    did_uri_bytes = client.identity_registry.functions.getMetadata(agent_id, "did_uri").call()
    did_doc_uri_bytes = client.identity_registry.functions.getMetadata(agent_id, "did_doc_uri").call()
    card_uri_bytes = client.identity_registry.functions.getMetadata(agent_id, "card_uri").call()

    return {
        "agent_id": agent_id,
        "token_uri": token_uri,
        "did_uri": did_uri_bytes.decode(errors="ignore") if did_uri_bytes else "",
        "did_doc_uri": did_doc_uri_bytes.decode(errors="ignore") if did_doc_uri_bytes else "",
        "card_uri": card_uri_bytes.decode(errors="ignore") if card_uri_bytes else "",
    }


def fetch_agent_card(uri: str) -> Optional[Dict[str, Any]]:
    """
    Fetch Agent Card from NeoFS or IPFS URI.
    
    Args:
        uri: NeoFS or IPFS URI (e.g., neofs://container/object or ipfs://cid)
        
    Returns:
        Agent Card JSON dict, or None if fetch fails
    """
    if not uri or not (uri.startswith("neofs://") or uri.startswith("ipfs://")):
        return None
    
    # Skip placeholder IPFS URIs (not real CIDs)
    if uri.startswith("ipfs://") and not uri.startswith("ipfs://Qm") and not uri.startswith("ipfs://baf"):
        # Check if it looks like a placeholder (e.g., "ipfs://spoon/...")
        if "/spoon/" in uri or not uri.replace("ipfs://", "").strip().startswith(("Qm", "baf")):
            print(f"Warning: Skipping placeholder IPFS URI: {uri}")
            return None
    
    try:
        # Use minimal storage client (no auth needed for public reads)
        storage = DIDStorageClient(
            neofs_url=os.getenv("NEOFS_BASE_URL", "https://rest.t5.fs.neo.org"),
            neofs_owner=None,  # Public read doesn't need owner
            neofs_private_key=None,
            ipfs_gateway=os.getenv("IPFS_GATEWAY_URL", "https://ipfs.io/ipfs"),
        )
        return storage.fetch_did_document(uri)
    except Exception as e:
        print(f"Warning: Failed to fetch Agent Card from {uri}: {e}")
        return None


def extract_service_endpoint(agent_info: Dict[str, Any]) -> Optional[str]:
    """
    Extract service endpoint from agent metadata.
    
    Tries in order:
    1. DID Document service endpoints (from did_doc_uri)
    2. Agent Card metadata
    3. Returns None if not found
    
    Args:
        agent_info: Dict with token_uri, card_uri, did_doc_uri
        
    Returns:
        Service endpoint URL, or None
    """
    # Try DID Document first (has structured service endpoints)
    did_doc_uri = agent_info.get("did_doc_uri")
    if did_doc_uri:
        try:
            did_doc = fetch_agent_card(did_doc_uri)
            if did_doc:
                # Look for AgentService or APIService endpoints
                services = did_doc.get("service", [])
                for svc in services:
                    if isinstance(svc, dict):
                        svc_type = svc.get("type", "")
                        if svc_type in ("AgentService", "APIService", "MessagingService"):
                            endpoint = svc.get("serviceEndpoint") or svc.get("service_endpoint")
                            if endpoint and endpoint.startswith("http"):
                                return endpoint
        except Exception as e:
            print(f"Warning: Failed to parse DID Document: {e}")
    
    # Fallback: Try Agent Card
    card_uri = agent_info.get("card_uri") or agent_info.get("token_uri")
    if card_uri:
        try:
            card = fetch_agent_card(card_uri)
            if card:
                # Agent Card might have service endpoint in metadata
                metadata = card.get("metadata", {})
                endpoint = metadata.get("service_endpoint") or metadata.get("serviceEndpoint")
                if endpoint and endpoint.startswith("http"):
                    return endpoint
        except Exception as e:
            print(f"Warning: Failed to parse Agent Card: {e}")
    
    return None


def ask_server(server_url: str, question: str) -> Dict[str, Any]:
    """
    Send a question to the ERC-8004 agent server.

    Args:
        server_url: Server base URL (e.g., http://127.0.0.1:8004)
        question: Question to ask

    Returns:
        Server response as dict
    """
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
    parser = argparse.ArgumentParser(
        description="ERC-8004 DID client - fetch agent info and ask questions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Query agent ID 3:
    python -m examples.erc8004_did.client_agent --agent-id 3

    # Custom question:
    python -m examples.erc8004_did.client_agent --agent-id 3 \\
        --question "How does ERC-8004 reputation work?"
        """,
    )

    parser.add_argument(
        "--agent-id",
        type=int,
        default=int(os.getenv("ERC8004_AGENT_ID", "3")),
        help="Agent token ID to query (default: 3)",
    )
    parser.add_argument(
        "--registry",
        default=os.getenv("NEOX_IDENTITY_REGISTRY", DEFAULT_IDENTITY_REGISTRY),
        help="IdentityRegistry contract address",
    )
    parser.add_argument(
        "--rpc",
        default=os.getenv("NEOX_RPC_URL", DEFAULT_RPC_URL),
        help="RPC endpoint URL",
    )
    parser.add_argument(
        "--chain-id",
        type=int,
        default=int(os.getenv("NEOX_CHAIN_ID", str(DEFAULT_CHAIN_ID))),
        help="Chain ID",
    )
    parser.add_argument(
        "--server",
        default=os.getenv("ERC8004_SERVER_URL", "http://127.0.0.1:8004"),
        help="Server base URL",
    )
    parser.add_argument(
        "--question",
        default="What is ERC-8004 and how do I register an agent?",
        help="Question to ask the agent",
    )

    args = parser.parse_args()

    if not args.registry:
        raise SystemExit("NEOX_IDENTITY_REGISTRY is required (set env or --registry)")

    # Fetch agent info from chain
    print(f"Fetching agent {args.agent_id} from IdentityRegistry {args.registry} ...")
    try:
        info = fetch_agent_info(args.registry, args.agent_id, args.rpc, args.chain_id)
        print("\nAgent Info:")
        print(json.dumps(info, indent=2))
    except ValueError as exc:
        print(f"Warning: {exc}")
        info = {"agent_id": args.agent_id, "error": str(exc)}

    # Try to extract service endpoint from Agent Card / DID Document
    service_endpoint = extract_service_endpoint(info)
    
    # Use agent's service endpoint if found, otherwise fallback to provided server URL
    if service_endpoint:
        print(f"\n✓ Found service endpoint in Agent Card: {service_endpoint}")
        server_url = service_endpoint
    else:
        print(f"\n⚠ No service endpoint found in Agent Card, using default: {args.server}")
        print("  (To use agent's own service, ensure Agent Card includes service endpoints)")
        server_url = args.server

    # Ask the server
    print(f"\nAsking server at {server_url} ...")
    print(f"Question: {args.question}\n")

    result = ask_server(server_url, args.question)
    print("Response:")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
