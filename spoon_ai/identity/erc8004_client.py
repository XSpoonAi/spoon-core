"""
ERC-8004 Smart Contract Client
Handles on-chain interactions with agent registries (IdentityRegistry only)
"""

from typing import List, Dict, Optional, Tuple, Union
from web3 import Web3
from web3.contract import Contract
from web3.middleware import ExtraDataToPOAMiddleware
from eth_account import Account
from eth_utils import to_checksum_address
from eth_abi import encode as abi_encode
from spoon_ai.identity.erc8004_abi import (
    get_abi,
)


class ERC8004Client:
    """Client for interacting with ERC-8004 agent registries"""

    def __init__(
        self,
        rpc_url: str,
        identity_registry_address: str,
        reputation_registry_address: str,
        validation_registry_address: str,
        private_key: Optional[str] = None
    ):
        self.w3 = Web3(Web3.HTTPProvider(rpc_url))
        if not self.w3.is_connected():
            raise ConnectionError(f"Failed to connect to RPC: {rpc_url}")

        # NeoX / other PoA-style chains may require the extraData middleware.
        try:
            if int(self.w3.eth.chain_id) in (12227332, 97, 56, 11155111):
                self.w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
        except Exception:
            pass

        self.private_key = private_key
        if private_key:
            self.account = Account.from_key(private_key)
        else:
            self.account = None

        # Load contract ABIs (centralized, artifact-free)
        self.identity_registry = self._load_contract(identity_registry_address, "ERC8004IdentityRegistry")
        self.reputation_registry = self._load_contract(reputation_registry_address, "ERC8004ReputationRegistry")
        self.validation_registry = self._load_contract(validation_registry_address, "ERC8004ValidationRegistry")

    def _load_contract(self, address: str, contract_name: str) -> Contract:
        abi = get_abi(contract_name)
        if not abi:
            raise ValueError(f"No ABI found for {contract_name}")
        return self.w3.eth.contract(address=to_checksum_address(address), abi=abi)

    # ---------------- Identity ----------------
    def register_agent(self, token_uri: str, metadata: Optional[List[Tuple[str, bytes]]] = None) -> int:
        """Register agent on IdentityRegistry; returns agentId."""
        if not self.account:
            raise ValueError("Private key required for registration")
        used_batch_register = False
        func = None

        # Prefer the batched register(tokenURI, metadata[]) overload when available.
        if metadata:
            try:
                func = self.identity_registry.functions.register(token_uri, [(k, v) for k, v in metadata])
                used_batch_register = True
            except Exception:
                func = None

        if func is None:
            func = self.identity_registry.functions.register(token_uri)
        tx = func.build_transaction(self._tx_params())
        receipt = self._send_tx(tx)
        # Try to infer agentId (avoid noisy web3 warnings by filtering logs ourselves)
        agent_id: Optional[int] = None
        try:
            registered_topic0 = self.w3.keccak(text="Registered(uint256,string,address)").hex()
            for log in receipt.get("logs", []):
                topics = log.get("topics") or []
                if topics and topics[0].hex() == registered_topic0:
                    decoded = self.identity_registry.events.Registered().process_log(log)
                    agent_id = int(decoded["args"]["agentId"])
                    break
        except Exception:
            agent_id = None

        if agent_id is None:
            agent_id = int(self.identity_registry.functions.totalAgents().call())

        # If we couldn't batch metadata at register-time, apply it post-register.
        if metadata and not used_batch_register:
            for key, value in metadata:
                try:
                    self.set_metadata(agent_id, key, value)
                except Exception:
                    break

        return agent_id

    def resolve_agent(self, agent_id: int) -> Dict:
        """Resolve agent metadata from IdentityRegistry by agentId.

        Returns dict with owner, tokenURI, and common metadata fields.
        """
        exists = self.identity_registry.functions.agentExists(agent_id).call()
        if not exists:
            return {"exists": False}

        owner = self.identity_registry.functions.ownerOf(agent_id).call()
        token_uri = self.identity_registry.functions.tokenURI(agent_id).call()

        # Read common metadata keys (best-effort)
        metadata: Dict[str, str] = {}
        for key in ("did_uri", "did_doc_uri", "card_uri", "displayName"):
            try:
                raw = self.identity_registry.functions.getMetadata(agent_id, key).call()
                if raw:
                    metadata[key] = raw.decode(errors="ignore") if isinstance(raw, (bytes, bytearray)) else str(raw)
            except Exception:
                pass

        return {
            "exists": True,
            "agentId": agent_id,
            "owner": owner,
            "tokenURI": token_uri,
            "metadata": metadata,
        }

    def set_metadata(self, agent_id: int, key: str, value: bytes) -> str:
        if not self.account:
            raise ValueError("Private key required for metadata update")
        tx = self.identity_registry.functions.setMetadata(agent_id, key, value).build_transaction(self._tx_params())
        receipt = self._send_tx(tx)
        return receipt.transactionHash.hex()

    # ---------------- Reputation ----------------
    def give_feedback(
        self,
        agent_id: int,
        score: int,
        tag: bytes,
        stage: bytes,
        uri: str,
        payment_hash: bytes,
        feedback_auth: bytes,
    ) -> str:
        if not self.account:
            raise ValueError("Private key required for feedback")
        tx = self.reputation_registry.functions.giveFeedback(
            agent_id, score, tag, stage, uri, payment_hash, feedback_auth
        ).build_transaction(self._tx_params())
        receipt = self._send_tx(tx)
        return receipt.transactionHash.hex()

    def revoke_feedback(self, agent_id: int, validator: str, index: int) -> str:
        if not self.account:
            raise ValueError("Private key required for revoke")
        tx = self.reputation_registry.functions.revokeFeedback(agent_id, validator, index).build_transaction(
            self._tx_params()
        )
        receipt = self._send_tx(tx)
        return receipt.transactionHash.hex()

    def get_reputation_summary(self, agent_id: int, validators, tag: bytes, stage: bytes):
        return self.reputation_registry.functions.getSummary(agent_id, validators, tag, stage).call()

    # ---------------- Validation ----------------
    def validation_request(self, validator: str, agent_id: int, uri: str, request_hash: bytes) -> str:
        if not self.account:
            raise ValueError("Private key required for validation request")
        tx = self.validation_registry.functions.validationRequest(
            validator, agent_id, uri, request_hash
        ).build_transaction(self._tx_params())
        receipt = self._send_tx(tx)
        return receipt.transactionHash.hex()

    def validation_response(
        self, request_hash: bytes, score: int, uri: str, payment_hash: bytes, response_hash: bytes
    ) -> str:
        if not self.account:
            raise ValueError("Private key required for validation response")
        tx = self.validation_registry.functions.validationResponse(
            request_hash, score, uri, payment_hash, response_hash
        ).build_transaction(self._tx_params())
        receipt = self._send_tx(tx)
        return receipt.transactionHash.hex()

    def get_validation_status(self, request_hash: bytes):
        return self.validation_registry.functions.getValidationStatus(request_hash).call()

    # ---------------- Helpers ----------------
    def _tx_params(self) -> Dict:
        gas_price = self.w3.eth.gas_price
        params: Dict = {
            "from": self.account.address if self.account else None,
            "nonce": self.w3.eth.get_transaction_count(self.account.address),
            "gas": 600000,
        }
        chain_id = int(getattr(self.w3.eth, "chain_id", 0) or 0)

        # NeoX Testnet T4 (12227332) behaves more reliably with legacy gasPrice txs.
        if chain_id == 12227332:
            params["gasPrice"] = gas_price
        else:
            params["maxFeePerGas"] = gas_price
            params["maxPriorityFeePerGas"] = gas_price // 2

        return params

    def _send_tx(self, tx: Dict) -> any:
        signed = self.account.sign_transaction(tx)
        tx_hash = self.w3.eth.send_raw_transaction(signed.raw_transaction)
        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
        if receipt.status != 1:
            raise RuntimeError(f"Transaction failed: {tx_hash.hex()}")
        return receipt
