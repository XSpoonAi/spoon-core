"""
Storage clients for DID documents and agent cards
Supports NeoFS (primary) and IPFS (backup replication)
"""

import json
import os
from typing import Dict, Optional, Tuple
from spoon_ai.neofs.client import NeoFSClient
import httpx


class DIDStorageClient:
    """
    Unified storage client for DID documents
    NeoFS primary with IPFS replication
    """

    def __init__(
        self,
        neofs_url: Optional[str] = None,
        neofs_owner: Optional[str] = None,
        neofs_private_key: Optional[str] = None,
        neofs_container: Optional[str] = None,
        ipfs_api: Optional[str] = None,
        ipfs_gateway: Optional[str] = None,
        allow_ipfs_placeholder: Optional[bool] = None,
    ):
        # NeoFS setup
        self.neofs_container = neofs_container or os.getenv("NEOFS_DID_CONTAINER")
        self.neofs_client = None

        if neofs_url and neofs_owner and neofs_private_key:
            try:
                self.neofs_client = NeoFSClient(
                    base_url=neofs_url,
                    owner_address=neofs_owner,
                    private_key_wif=neofs_private_key
                )
            except Exception as e:
                print(f"Warning: NeoFS client initialization failed: {e}")

        # IPFS setup
        self.ipfs_api = ipfs_api or os.getenv("IPFS_API_URL", "http://127.0.0.1:5001")
        self.ipfs_gateway = ipfs_gateway or os.getenv("IPFS_GATEWAY_URL", "http://127.0.0.1:8080")
        env_allow = os.getenv("ALLOW_IPFS_PLACEHOLDER", "false").lower() in {"1", "true", "yes"}
        self.allow_ipfs_placeholder = allow_ipfs_placeholder if allow_ipfs_placeholder is not None else env_allow
        self.ipfs_http_client = httpx.Client(timeout=30.0)

    def publish_did_document(
        self,
        agent_id: str,
        did_document: Dict,
        agent_card: Dict
    ) -> Tuple[str, str]:
        """
        Publish DID document and agent card to storage
        Returns (didDocURI, agentCardURI)
        """
        # Serialize to JSON
        did_doc_json = json.dumps(did_document, indent=2)
        agent_card_json = json.dumps(agent_card, indent=2)

        # Primary: Publish to NeoFS
        neofs_did_uri = None
        neofs_card_uri = None

        if self.neofs_client and self.neofs_container:
            try:
                neofs_did_uri = self._publish_to_neofs(
                    f"dids/agent/{agent_id}/document.json",
                    did_doc_json.encode('utf-8')
                )
                neofs_card_uri = self._publish_to_neofs(
                    f"dids/agent/{agent_id}/agent_card.json",
                    agent_card_json.encode('utf-8')
                )
            except Exception as e:
                print(f"Warning: NeoFS publish failed: {e}")

        # Backup: Replicate to IPFS
        ipfs_did_cid = self._publish_to_ipfs(did_doc_json.encode('utf-8'))
        ipfs_card_cid = self._publish_to_ipfs(agent_card_json.encode('utf-8'))

        # Prefer NeoFS, fallback to IPFS
        did_doc_uri = neofs_did_uri or f"ipfs://{ipfs_did_cid}"
        agent_card_uri = neofs_card_uri or f"ipfs://{ipfs_card_cid}"

        return (did_doc_uri, agent_card_uri)

    def _publish_to_neofs(self, path: str, content: bytes) -> str:
        """Publish content to NeoFS"""
        if not self.neofs_client or not self.neofs_container:
            raise ValueError("NeoFS client not configured")

        # Get bearer token
        bearer_response = self.neofs_client.get_binary_bearer_token()
        bearer_token = bearer_response.token

        # Upload object
        upload_result = self.neofs_client.upload_object(
            container_id=self.neofs_container,
            bearer_token=bearer_token,
            content=content,
            attributes={"FilePath": path}
        )

        return f"neofs://{self.neofs_container}/{upload_result.object_id}"

    def _publish_to_ipfs(self, content: bytes) -> str:
        """Publish content to IPFS, returns CID"""
        try:
            response = self.ipfs_http_client.post(
                f"{self.ipfs_api}/api/v0/add",
                files={"file": content}
            )
            response.raise_for_status()
            result = response.json()
            return result['Hash']
        except Exception as e:
            if self.allow_ipfs_placeholder:
                print(f"IPFS publish warning (placeholder CID emitted): {e}")
                # Return deterministic placeholder CID for tests that explicitly allow it
                import hashlib
                return hashlib.sha256(content).hexdigest()[:46]
            raise RuntimeError(f"Failed to publish to IPFS: {e}") from e

    def fetch_did_document(self, uri: str) -> Dict:
        """Fetch DID document from URI (NeoFS or IPFS)"""
        if uri.startswith("neofs://"):
            content = self._fetch_from_neofs(uri)
        elif uri.startswith("ipfs://"):
            content = self._fetch_from_ipfs(uri)
        else:
            raise ValueError(f"Unsupported URI scheme: {uri}")

        return json.loads(content.decode('utf-8'))

    def _fetch_from_neofs(self, uri: str) -> bytes:
        """Fetch from NeoFS"""
        if not self.neofs_client:
            raise ValueError("NeoFS client not configured")

        # Parse: neofs://<container>/<object_id>
        parts = uri.replace("neofs://", "").split("/", 1)
        container_id = parts[0]
        object_id = parts[1] if len(parts) > 1 else ""

        bearer_response = self.neofs_client.get_binary_bearer_token()
        bearer_token = bearer_response.token

        response = self.neofs_client.download_object_by_id(
            container_id=container_id,
            object_id=object_id,
            bearer_token=bearer_token
        )

        return response.content

    def _fetch_from_ipfs(self, uri: str) -> bytes:
        """Fetch from IPFS"""
        cid = uri.replace("ipfs://", "")

        try:
            response = self.ipfs_http_client.get(f"{self.ipfs_gateway}/ipfs/{cid}")
            response.raise_for_status()
            return response.content
        except Exception as e:
            raise ValueError(f"Failed to fetch from IPFS: {e}")

    def publish_credential(self, agent_id: str, credential: Dict) -> str:
        """Publish verifiable credential"""
        cred_json = json.dumps(credential, indent=2)

        if self.neofs_client and self.neofs_container:
            try:
                return self._publish_to_neofs(
                    f"dids/agent/{agent_id}/credentials/{credential.get('id', 'cred')}.json",
                    cred_json.encode('utf-8')
                )
            except Exception:
                pass

        # Fallback to IPFS
        cid = self._publish_to_ipfs(cred_json.encode('utf-8'))
        return f"ipfs://{cid}"

    def close(self):
        """Close HTTP clients"""
        self.ipfs_http_client.close()
        if self.neofs_client:
            self.neofs_client.http_client.close()
















