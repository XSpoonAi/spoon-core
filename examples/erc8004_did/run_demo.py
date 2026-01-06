"""
End-to-end launcher:
- Starts the ERC8004SearchAgent server
- Calls the client to fetch on-chain data and ask a question
- Stops the server

Run:
    python -m examples.erc8004_did.run_demo

Prereqs:
    export TAVILY_API_KEY=...
    export NEOX_IDENTITY_REGISTRY=0xaB5623F3DD66f2a52027FA06007C78c7b0E63508  # or your own
    (Optional) ERC8004_AGENT_ID, ERC8004_AGENT_DID_URI, NEOX_RPC_URL, NEOX_CHAIN_ID
"""

from __future__ import annotations

import os
from pathlib import Path
import argparse
import subprocess
import sys
import time

try:
    # Load local dev secrets/config (e.g. PRIVATE_KEY, NEOX_RPC_URL) without printing.
    from dotenv import load_dotenv  # type: ignore

    core_dir = Path(__file__).resolve().parents[2]
    load_dotenv(core_dir / ".env", override=False)
except Exception:
    pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ERC8004 server + client demo")
    parser.add_argument("--register-agent", action="store_true", help="Pre-register agent on IdentityRegistry before serving")
    parser.add_argument("--token-uri", default=os.getenv("AGENT_CARD_URI") or os.getenv("AGENT_DID_URI"))
    parser.add_argument("--did-uri", default=os.getenv("ERC8004_AGENT_DID_URI"))
    parser.add_argument("--did-doc-uri", default=os.getenv("AGENT_DID_DOC_URI") or os.getenv("DID_DOC_URI"))
    parser.add_argument("--registry", default=os.getenv("NEOX_IDENTITY_REGISTRY"))
    parser.add_argument("--reputation", default=os.getenv("NEOX_REPUTATION_REGISTRY"))
    parser.add_argument("--validation", default=os.getenv("NEOX_VALIDATION_REGISTRY"))
    parser.add_argument("--agent-registry", default=os.getenv("NEOX_AGENT_REGISTRY") or os.getenv("AGENT_REGISTRY"))
    parser.add_argument("--rpc", default=os.getenv("NEOX_RPC_URL", "https://testnet.rpc.banelabs.org"))
    parser.add_argument("--chain-id", default=os.getenv("NEOX_CHAIN_ID", "12227332"))
    parser.add_argument(
        "--private-key",
        default=os.getenv("PRIVATE_KEY") or os.getenv("NEOX_PRIVATE_KEY") or os.getenv("REACT_PRIVATE_KEY"),
    )
    args = parser.parse_args()

    server_port = int(os.getenv("ERC8004_AGENT_PORT", "8004"))
    env = os.environ.copy()
    # Ensure core package path on PYTHONPATH so sub-processes can import examples.*
    core_dir = Path(__file__).resolve().parents[2]
    env["PYTHONPATH"] = f"{core_dir}{os.pathsep}{env.get('PYTHONPATH', '')}"
    if args.private_key:
        # Avoid passing secrets via argv (process list / logs).
        env["PRIVATE_KEY"] = str(args.private_key)

    server_cmd = [
        sys.executable,
        "-m",
        "examples.erc8004_did.server_agent",
        "--port",
        str(server_port),
    ]
    if args.register_agent:
        server_cmd += [
            "--register-agent",
            "--registry",
            args.registry or "",
            "--reputation",
            args.reputation or "",
            "--validation",
            args.validation or "",
            "--agent-registry",
            args.agent_registry or "",
            "--rpc",
            args.rpc,
            "--chain-id",
            str(args.chain_id),
        ]
        if args.token_uri:
            server_cmd += ["--token-uri", args.token_uri]
        if args.did_uri:
            server_cmd += ["--did-uri", args.did_uri]
        if args.did_doc_uri:
            server_cmd += ["--did-doc-uri", args.did_doc_uri]

    print("Launching server:", " ".join(server_cmd))
    server = subprocess.Popen(server_cmd, env=env)
    time.sleep(10)  # give server time to start (and finish optional registration)

    client_cmd = [
        sys.executable,
        "-m",
        "examples.erc8004_did.client_agent",
        "--server",
        f"http://127.0.0.1:{server_port}",
    ]
    if args.registry:
        client_cmd += ["--registry", args.registry]
    if args.token_uri:
        client_cmd += ["--token-uri", args.token_uri]
    if args.did_uri:
        client_cmd += ["--did-uri", args.did_uri]
    elif os.getenv("AGENT_DID_URI"):
        client_cmd += ["--did-uri", os.getenv("AGENT_DID_URI")]
    if args.chain_id:
        client_cmd += ["--chain-id", str(args.chain_id)]
    if args.rpc:
        client_cmd += ["--rpc", args.rpc]
    print("Running client:", " ".join(client_cmd), "\n")
    try:
        subprocess.check_call(client_cmd, env=env)
    finally:
        print("Stopping server...")
        server.terminate()
        try:
            server.wait(timeout=5)
        except Exception:
            server.kill()


if __name__ == "__main__":
    main()
