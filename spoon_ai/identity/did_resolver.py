"""
DID Resolver for SpoonOS Agents
Implements unified DID resolution with NeoFS-first policy
"""

from typing import Optional, Dict
from .did_models import AgentDID, DIDResolutionResult, AgentCard, ReputationScore
from .erc8004_client import ERC8004Client
from .storage_client import DIDStorageClient
from datetime import datetime


class DIDResolver:
    """
    Unified DID resolver for SpoonOS agents
    Resolution flow: On-chain anchor → NeoFS (primary) → IPFS (fallback)
    """

    def __init__(
        self,
        erc8004_client: ERC8004Client,
        storage_client: DIDStorageClient
    ):
        self.erc8004_client = erc8004_client
        self.storage_client = storage_client

    def resolve(self, did: str) -> DIDResolutionResult:
        """
        Resolve DID to complete DID document

        Args:
            did: DID string (did:spoon:agent:<identifier>)

        Returns:
            DIDResolutionResult with document and metadata
        """
        try:
            # Step 1: Resolve on-chain anchor
            on_chain_metadata = self.erc8004_client.resolve_agent(did)

            if not on_chain_metadata.get("exists"):
                return DIDResolutionResult(
                    did_document=None,
                    did_resolution_metadata={
                        "error": "notFound",
                        "message": f"DID {did} not found in registry"
                    }
                )

            # Step 2: Fetch DID document from storage (NeoFS primary, IPFS fallback)
            did_doc_uri = on_chain_metadata["didDocURI"]
            agent_card_uri = on_chain_metadata["agentCardURI"]

            did_document_dict = self._fetch_with_fallback(did_doc_uri)
            agent_card_dict = self._fetch_with_fallback(agent_card_uri)

            # Step 3: Fetch reputation
            reputation = self._fetch_reputation(did)

            # Step 4: Construct complete AgentDID
            agent_did = AgentDID(
                id=did,
                controller=[str(addr) for addr in on_chain_metadata["controllers"]],
                verification_method=did_document_dict.get("verificationMethod", []),
                authentication=did_document_dict.get("authentication", []),
                service=did_document_dict.get("service", []),
                agent_card=AgentCard(**agent_card_dict),
                reputation=reputation,
                attestations=did_document_dict.get("attestations", []),
                did_hash=self.erc8004_client.calculate_did_hash(did).hex(),
                agent_card_uri=agent_card_uri,
                did_doc_uri=did_doc_uri,
                registered_at=datetime.fromtimestamp(on_chain_metadata["registeredAt"])
            )

            return DIDResolutionResult(
                did_document=agent_did,
                did_document_metadata={
                    "created": on_chain_metadata["registeredAt"],
                    "updated": on_chain_metadata["registeredAt"]
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
                # TODO: Implement backup CID lookup from indexer
                raise ValueError(f"NeoFS fetch failed and no IPFS backup: {primary_error}")
            raise

    def _fetch_reputation(self, did: str) -> Optional[ReputationScore]:
        """Fetch reputation from ERC-8004 reputation registry"""
        try:
            score, total_submissions = self.erc8004_client.get_reputation(did)
            if total_submissions > 0:
                return ReputationScore(
                    score=float(score),
                    total_submissions=total_submissions,
                    last_updated=datetime.utcnow()
                )
        except Exception as e:
            print(f"Reputation fetch warning: {e}")
        return None

    def resolve_metadata_only(self, did: str) -> Dict:
        """Resolve only on-chain metadata (fast path)"""
        return self.erc8004_client.resolve_agent(did)

    def verify_did(self, did: str) -> bool:
        """Verify DID exists and is resolvable"""
        try:
            result = self.resolve(did)
            return result.did_document is not None
        except Exception:
            return False
















