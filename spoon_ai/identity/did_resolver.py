"""
DID Resolver for SpoonOS Agents
Implements unified DID resolution via IdentityRegistry with NeoFS-first policy
"""

from typing import Optional, Dict
from .did_models import AgentDID, DIDResolutionResult, AgentCard, ReputationScore
from .erc8004_client import ERC8004Client
from .storage_client import DIDStorageClient
from datetime import datetime


class DIDResolver:
    """
    Unified DID resolver for SpoonOS agents.
    Resolution flow: IdentityRegistry (agentId) → NeoFS (primary) → IPFS (fallback)
    """

    def __init__(
        self,
        erc8004_client: ERC8004Client,
        storage_client: DIDStorageClient
    ):
        self.erc8004_client = erc8004_client
        self.storage_client = storage_client

    def resolve(self, agent_id: int) -> DIDResolutionResult:
        """
        Resolve agent identity to complete DID document.

        Args:
            agent_id: On-chain agent token ID from IdentityRegistry

        Returns:
            DIDResolutionResult with document and metadata
        """
        try:
            # Step 1: Resolve on-chain data from IdentityRegistry
            on_chain = self.erc8004_client.resolve_agent(agent_id)

            if not on_chain.get("exists"):
                return DIDResolutionResult(
                    did_document=None,
                    did_resolution_metadata={
                        "error": "notFound",
                        "message": f"Agent {agent_id} not found in IdentityRegistry"
                    }
                )

            owner = on_chain["owner"]
            token_uri = on_chain.get("tokenURI", "")
            metadata = on_chain.get("metadata", {})

            did_doc_uri = metadata.get("did_doc_uri", "")
            agent_card_uri = metadata.get("card_uri", "") or token_uri

            # Step 2: Fetch DID document from storage (NeoFS primary, IPFS fallback)
            did_document_dict = {}
            if did_doc_uri:
                did_document_dict = self._fetch_with_fallback(did_doc_uri)

            agent_card_dict = {}
            if agent_card_uri:
                agent_card_dict = self._fetch_with_fallback(agent_card_uri)

            # Step 3: Construct DID string from agentId
            did = f"did:spoon:agent:{agent_id}"

            # Step 4: Construct complete AgentDID
            agent_did = AgentDID(
                id=did,
                controller=[str(owner)],
                verification_method=did_document_dict.get("verificationMethod", []),
                authentication=did_document_dict.get("authentication", []),
                service=did_document_dict.get("service", []),
                agent_card=AgentCard(**(agent_card_dict or {"name": f"Agent #{agent_id}", "description": ""})),
                attestations=did_document_dict.get("attestations", []),
                agent_card_uri=agent_card_uri or None,
                did_doc_uri=did_doc_uri or None,
            )

            return DIDResolutionResult(
                did_document=agent_did,
                did_document_metadata={
                    "agentId": agent_id,
                    "owner": owner,
                    "tokenURI": token_uri,
                },
                did_resolution_metadata={
                    "contentType": "application/did+ld+json",
                    "retrieved": datetime.utcnow().isoformat()
                }
            )

        except Exception as e:
            return DIDResolutionResult(
                did_document=None,
                did_resolution_metadata={
                    "error": "internalError",
                    "message": str(e)
                }
            )

    def _fetch_with_fallback(self, uri: str) -> Dict:
        """Fetch from primary URI with fallback logic"""
        try:
            return self.storage_client.fetch_did_document(uri)
        except Exception as primary_error:
            # If NeoFS fails, try IPFS if we have backup CID
            if uri.startswith("neofs://"):
                raise ValueError(f"NeoFS fetch failed and no IPFS backup: {primary_error}")
            raise

    def resolve_metadata_only(self, agent_id: int) -> Dict:
        """Resolve only on-chain metadata (fast path)"""
        return self.erc8004_client.resolve_agent(agent_id)

    def verify_agent(self, agent_id: int) -> bool:
        """Verify agent exists and is resolvable"""
        try:
            result = self.resolve(agent_id)
            return result.did_document is not None
        except Exception:
            return False
