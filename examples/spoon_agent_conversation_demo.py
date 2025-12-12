"""
Spoon Agent conversation demo (no deployment):
- Uses already deployed ERC8004 registries (pass via args or env NEOX_*_REGISTRY).
- Builds AgentCard & DID, registers agent, simulates chat with a real ReActAgent subclass,
  then submits reputation/validation on-chain.
- ABI is embedded here (no artifact path dependency).
"""

from __future__ import annotations

import json
import argparse
import json
import os
from pathlib import Path
from typing import Dict

from eth_account import Account
from web3 import Web3
from web3.middleware import ExtraDataToPOAMiddleware

from spoon_ai.agents.react import ReActAgent
from spoon_ai.chat import ChatBot, Memory
from spoon_ai.schema import AgentState
from spoon_ai.identity.did_models import AgentCard, AgentDID

DEFAULT_RPC = os.getenv("REACT_RPC_URL", "https://testnet.rpc.banelabs.org")
DEFAULT_PRIVATE_KEY = os.getenv("REACT_PRIVATE_KEY")
IDENTITY_ADDR = os.getenv("NEOX_IDENTITY_REGISTRY") or os.getenv("IDENTITY_REGISTRY_ADDRESS")
REPUTATION_ADDR = os.getenv("NEOX_REPUTATION_REGISTRY") or os.getenv("REPUTATION_REGISTRY_ADDRESS")
VALIDATION_ADDR = os.getenv("NEOX_VALIDATION_REGISTRY") or os.getenv("VALIDATION_REGISTRY_ADDRESS")


IDENTITY_ABI = json.loads(
    """
    [
      {"inputs":[{"internalType":"string","name":"tokenURI_","type":"string"}],"name":"register","outputs":[{"internalType":"uint256","name":"agentId","type":"uint256"}],"stateMutability":"nonpayable","type":"function"},
      {"inputs":[{"internalType":"uint256","name":"agentId","type":"uint256"},{"internalType":"string","name":"key","type":"string"}],"name":"getMetadata","outputs":[{"internalType":"bytes","name":"value","type":"bytes"}],"stateMutability":"view","type":"function"},
      {"inputs":[{"internalType":"uint256","name":"agentId","type":"uint256"}],"name":"ownerOf","outputs":[{"internalType":"address","name":"","type":"address"}],"stateMutability":"view","type":"function"},
      {"anonymous":false,"inputs":[{"indexed":true,"internalType":"uint256","name":"agentId","type":"uint256"},{"indexed":false,"internalType":"string","name":"tokenURI","type":"string"},{"indexed":true,"internalType":"address","name":"owner","type":"address"}],"name":"Registered","type":"event"}
    ]
    """
)

REPUTATION_ABI = json.loads(
    """
    [
      {"inputs":[{"internalType":"uint256","name":"agentId","type":"uint256"},{"components":[{"internalType":"uint256","name":"score","type":"uint256"},{"internalType":"uint256","name":"normalizedScore","type":"uint256"},{"internalType":"bytes32","name":"tag1","type":"bytes32"},{"internalType":"bytes32","name":"tag2","type":"bytes32"},{"internalType":"bytes32","name":"tag3","type":"bytes32"},{"internalType":"bytes32","name":"tag4","type":"bytes32"},{"internalType":"string","name":"fileURI","type":"string"},{"internalType":"bytes32","name":"fileHash","type":"bytes32"},{"internalType":"bytes32","name":"paymentProofHash","type":"bytes32"},{"internalType":"uint256","name":"index","type":"uint256"},{"internalType":"bool","name":"isEncrypted","type":"bool"}],"internalType":"struct ISpoonReputationRegistry.Feedback","name":"fb","type":"tuple"},{"internalType":"bytes","name":"auth","type":"bytes"}],"name":"submitFeedback","outputs":[],"stateMutability":"nonpayable","type":"function"},
      {"inputs":[{"internalType":"uint256","name":"agentId","type":"uint256"}],"name":"aggregateScore","outputs":[{"internalType":"uint256","name":"score","type":"uint256"},{"internalType":"uint256","name":"samples","type":"uint256"}],"stateMutability":"view","type":"function"}
    ]
    """
)

VALIDATION_ABI = json.loads(
    """
    [
      {"inputs":[{"internalType":"address","name":"validator","type":"address"},{"internalType":"uint256","name":"agentId","type":"uint256"},{"internalType":"string","name":"requestURI","type":"string"},{"internalType":"bytes32","name":"requestHash","type":"bytes32"},{"internalType":"string","name":"stage","type":"string"}],"name":"validationRequest","outputs":[{"internalType":"bytes32","name":"reqHash","type":"bytes32"}],"stateMutability":"nonpayable","type":"function"},
      {"inputs":[{"internalType":"bytes32","name":"reqHash","type":"bytes32"},{"internalType":"uint256","name":"response","type":"uint256"},{"internalType":"string","name":"stage","type":"string"},{"internalType":"string","name":"responseURI","type":"string"},{"internalType":"bytes32","name":"responseHash","type":"bytes32"},{"internalType":"bytes32","name":"tag","type":"bytes32"},{"internalType":"bytes32","name":"paymentProofHash","type":"bytes32"}],"name":"validationResponse","outputs":[],"stateMutability":"nonpayable","type":"function"},
      {"inputs":[{"internalType":"bytes32","name":"reqHash","type":"bytes32"}],"name":"getResponse","outputs":[{"internalType":"address","name":"","type":"address"},{"internalType":"uint256","name":"","type":"uint256"},{"internalType":"uint256","name":"","type":"uint256"},{"internalType":"bytes32","name":"","type":"bytes32"},{"internalType":"string","name":"","type":"string"},{"internalType":"string","name":"","type":"string"},{"internalType":"bytes32","name":"","type":"bytes32"},{"internalType":"bytes32","name":"","type":"bytes32"},{"internalType":"uint256","name":"","type":"uint256"}],"stateMutability":"view","type":"function"},
      {"anonymous":false,"inputs":[{"indexed":true,"internalType":"bytes32","name":"requestHash","type":"bytes32"},{"indexed":true,"internalType":"uint256","name":"agentId","type":"uint256"},{"indexed":true,"internalType":"address","name":"validator","type":"address"}],"name":"ValidationRequested","type":"event"}
    ]
    """
)


def simple_agent_reply(message: str) -> str:
    """Fallback tiny conversation stub."""
    if "hello" in message.lower():
        return "Hello! I'm Spoon Agent, with on-chain identity."
    if "task" in message.lower():
        return "I'll log this task and anchor evidence on-chain if needed."
    return "Noted. I'll handle it and anchor trust signals."


class OnChainReActAgent(ReActAgent):
    w3: Web3
    identity: Any
    reputation: Any
    validation: Any
    acct: Any
    agent_id: Optional[int] = None
    req_hash: Optional[bytes] = None
    card_uri: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True

    def respond(self, user_message: str) -> str:
        # Lightweight heuristic reply; still records via ReActAgent memory.
        reply = simple_agent_reply(user_message)
        self.memory.add_message("assistant", reply)
        return reply

    async def think(self) -> bool:
        # stop when all three phases done
        if self.agent_id and self.req_hash:
            self.state = AgentState.FINISHED
            return False
        return True

    async def act(self) -> str:
        if not self.agent_id:
            return await self._register()
        if not self.req_hash:
            return await self._feedback_and_validate()
        return "No action"

    async def _register(self) -> str:
        if not self.card_uri:
            raise RuntimeError("card_uri not set for registration")
        tx = self.identity.functions.register(self.card_uri).build_transaction(
            {"from": self.acct.address, "nonce": self.w3.eth.get_transaction_count(self.acct.address), "gas": 800_000}
        )
        signed = self.acct.sign_transaction(tx)
        receipt = self.w3.eth.wait_for_transaction_receipt(self.w3.eth.send_raw_transaction(signed.raw_transaction))
        self.agent_id = self.identity.events.Registered().process_receipt(receipt)[0]["args"]["agentId"]
        await self.add_message("assistant", f"Registered agent id={self.agent_id} uri={self.card_uri}")
        return "Registered"

    async def _feedback_and_validate(self) -> str:
        if self.agent_id is None:
            raise RuntimeError("agent_id not set")
        fb = (
            95,
            95,
            Web3.to_bytes(text="chat").ljust(32, b"\x00"),
            Web3.to_bytes(text="support").ljust(32, b"\x00"),
            Web3.to_bytes(text="conv").ljust(32, b"\x00"),
            Web3.to_bytes(text="demo").ljust(32, b"\x00"),
            os.getenv("EVIDENCE_URI", self.card_uri),
            Web3.keccak(text="evidence"),
            Web3.keccak(text="payproof"),
            0,
            False,
        )
        tx = self.reputation.functions.submitFeedback(self.agent_id, fb, b"").build_transaction(
            {"from": self.acct.address, "nonce": self.w3.eth.get_transaction_count(self.acct.address), "gas": 900_000}
        )
        signed = self.acct.sign_transaction(tx)
        self.w3.eth.wait_for_transaction_receipt(self.w3.eth.send_raw_transaction(signed.raw_transaction))

        req_uri = os.getenv("VALIDATION_REQUEST_URI", self.card_uri or "")
        stage = "stage-1"
        tx = self.validation.functions.validationRequest(
            self.acct.address, self.agent_id, req_uri, Web3.keccak(text="req"), stage
        ).build_transaction(
            {"from": self.acct.address, "nonce": self.w3.eth.get_transaction_count(self.acct.address), "gas": 600_000}
        )
        signed = self.acct.sign_transaction(tx)
        req_receipt = self.w3.eth.wait_for_transaction_receipt(self.w3.eth.send_raw_transaction(signed.raw_transaction))
        self.req_hash = self.validation.events.ValidationRequested().process_receipt(req_receipt)[0]["args"]["requestHash"]

        tx = self.validation.functions.validationResponse(
            self.req_hash,
            97,
            stage,
            os.getenv("VALIDATION_RESPONSE_URI", self.card_uri or ""),
            Web3.keccak(text="resp"),
            Web3.to_bytes(text="pass").ljust(32, b"\x00"),
            Web3.keccak(text="payproof2"),
        ).build_transaction(
            {"from": self.acct.address, "nonce": self.w3.eth.get_transaction_count(self.acct.address), "gas": 600_000}
        )
        signed = self.acct.sign_transaction(tx)
        self.w3.eth.wait_for_transaction_receipt(self.w3.eth.send_raw_transaction(signed.raw_transaction))

        await self.add_message("assistant", "Feedback + validation submitted.")
        return "Feedback+validation done"


def main():
    parser = argparse.ArgumentParser(description="Spoon agent conversation demo (no deploy).")
    parser.add_argument("--rpc", default=DEFAULT_RPC)
    parser.add_argument("--private-key", default=DEFAULT_PRIVATE_KEY, help="EOA for tx signing (env REACT_PRIVATE_KEY)")
    parser.add_argument("--identity", default=IDENTITY_ADDR, help="IdentityRegistry address (required)")
    parser.add_argument("--reputation", default=REPUTATION_ADDR, help="ReputationRegistry address (required)")
    parser.add_argument("--validation", default=VALIDATION_ADDR, help="ValidationRegistry address (required)")
    parser.add_argument("--poa", action="store_true", help="Inject POA middleware (NeoX, etc.)")
    args = parser.parse_args()

    if not args.private_key:
        raise RuntimeError("Set REACT_PRIVATE_KEY or pass --private-key.")
    if not (args.identity and args.reputation and args.validation):
        raise RuntimeError("Provide --identity/--reputation/--validation (or env NEOX_*_REGISTRY).")

    w3 = Web3(Web3.HTTPProvider(args.rpc))
    if "rpc.banelabs.org" in args.rpc or args.poa:
        w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
    acct = Account.from_key(args.private_key)

    identity = w3.eth.contract(address=Web3.to_checksum_address(args.identity), abi=IDENTITY_ABI)
    reputation = w3.eth.contract(address=Web3.to_checksum_address(args.reputation), abi=REPUTATION_ABI)
    validation = w3.eth.contract(address=Web3.to_checksum_address(args.validation), abi=VALIDATION_ABI)

    # Build AgentCard & DID
    card = AgentCard(
        name="Spoon Local Agent",
        description="Conversational agent with ERC8004 identity",
        capabilities=["chat", "log", "submit-onchain-feedback"],
        trust_model="reputation+validation",
        payment_layer="x402",
        memory_primary=os.getenv("NEOFS_DID_CONTAINER", "neofs://CK44Vxzo21RED5LAZcBjQAGVQsPW5cLCxnr36vEBXvLG"),
    )
    did_model = AgentDID(
        id="did:spoon:agent:local-conv",
        controller=[acct.address],
        verification_method=[],
        authentication=[],
        service=[],
        agent_card=card,
    )

    # Register agent on-chain
    card_uri = os.getenv(
        "AGENT_CARD_URI",
        "neofs://CK44Vxzo21RED5LAZcBjQAGVQsPW5cLCxnr36vEBXvLG/QXxz1MnhzuU7sxDiBfUdbWUGtLNpgFBtQrA2mqowzKb",
    )
    agent = OnChainReActAgent(
        name="conv-agent",
        description="Conversational agent using ReActAgent with on-chain anchors",
        system_prompt="You are a helpful Spoon agent; be concise and action-driven.",
        llm=ChatBot(use_llm_manager=False),  # deterministic stub
        memory=Memory(),
        w3=w3,
        identity=identity,
        reputation=reputation,
        validation=validation,
        acct=acct,
    )
    agent.card_uri = card_uri

    # Register on-chain
    tx = identity.functions.register(card_uri).build_transaction(
        {"from": acct.address, "nonce": w3.eth.get_transaction_count(acct.address), "gas": 800_000}
    )
    signed = w3.eth.account.sign_transaction(tx, args.private_key)
    reg_receipt = w3.eth.wait_for_transaction_receipt(w3.eth.send_raw_transaction(signed.raw_transaction))
    agent.agent_id = identity.events.Registered().process_receipt(reg_receipt)[0]["args"]["agentId"]
    print(f"Agent registered with tokenId={agent.agent_id}")

    # Simulate conversation with ReActAgent memory
    user_messages = ["Hello agent", "Give me the task status", "Thanks"]
    for m in user_messages:
        print(f"User: {m}")
        agent.memory.add_message("user", m)
        reply = agent.respond(m)
        print(f"Agent: {reply}")

    # Perform feedback + validation via agent actions
    import asyncio
    asyncio.run(agent.step())
    asyncio.run(agent.step())

    # Aggregate trust signals
    score, samples = reputation.functions.aggregateScore(agent.agent_id).call()
    validation_resp = validation.functions.getResponse(agent.req_hash).call()
    print(f"Reputation aggregate: score={score}, samples={samples}")
    print("Validation response:", validation_resp)
    print("DID document (local):", did_model.to_did_document())


if __name__ == "__main__":
    main()
