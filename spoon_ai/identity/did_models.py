"""
DID Data Models for SpoonOS Agents
Following W3C DID Core specification and ERC-8004 standard
"""

from typing import List, Dict, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum


class VerificationMethodType(str, Enum):
    """Supported verification method types"""
    ECDSA_SECP256K1 = "EcdsaSecp256k1VerificationKey2019"
    ED25519 = "Ed25519VerificationKey2020"
    RSA = "RsaVerificationKey2018"


class ServiceType(str, Enum):
    """Agent service endpoint types"""
    AGENT_SERVICE = "AgentService"
    MESSAGING = "MessagingService"
    CAPABILITY_DISCOVERY = "CapabilityDiscovery"
    API = "APIService"


class VerificationMethod(BaseModel):
    """Cryptographic verification method for DID authentication"""
    id: str = Field(..., description="Method identifier (DID#key-1)")
    type: VerificationMethodType = Field(..., description="Cryptographic suite")
    controller: str = Field(..., description="DID that controls this method")
    public_key_hex: Optional[str] = Field(None, alias="publicKeyHex")
    public_key_base58: Optional[str] = Field(None, alias="publicKeyBase58")
    ethereum_address: Optional[str] = Field(None, alias="ethereumAddress")

    class Config:
        populate_by_name = True


class ServiceEndpoint(BaseModel):
    """Service endpoint for agent interaction"""
    id: str = Field(..., description="Service identifier")
    type: ServiceType = Field(..., description="Service type")
    service_endpoint: str = Field(..., alias="serviceEndpoint", description="URL or URI")
    description: Optional[str] = Field(None, description="Human-readable description")

    class Config:
        populate_by_name = True


class ReputationScore(BaseModel):
    """Aggregated reputation score"""
    score: float = Field(..., description="Average reputation score (-100 to 100)")
    total_submissions: int = Field(0, description="Number of reputation submissions")
    last_updated: datetime = Field(default_factory=datetime.utcnow)


class Attestation(BaseModel):
    """Verifiable attestation about an agent"""
    issuer: str = Field(..., description="DID of attestation issuer")
    subject: str = Field(..., description="DID of subject agent")
    claim: Dict[str, Any] = Field(..., description="Attestation claim data")
    evidence: Optional[str] = Field(None, description="Supporting evidence")
    issued_at: datetime = Field(default_factory=datetime.utcnow, alias="issuedAt")
    signature: Optional[str] = Field(None, description="Issuer's signature")

    class Config:
        populate_by_name = True


class AgentCard(BaseModel):
    """
    Agent Card following Google's A2A protocol
    Provides human-readable agent information
    """
    name: str = Field(..., description="Agent name")
    description: str = Field(..., description="Agent description")
    capabilities: List[str] = Field(default_factory=list, description="Agent capabilities")
    version: str = Field("1.0.0", description="Agent version")
    author: Optional[str] = Field(None, description="Agent author")
    homepage: Optional[str] = Field(None, description="Agent homepage URL")
    tags: List[str] = Field(default_factory=list, description="Agent tags")
    trust_model: Optional[str] = Field(
        None, description="Trust model hint (e.g., reputation/stake/TEE/zkML)"
    )
    payment_layer: Optional[str] = Field(
        None, description="Payment protocol used (e.g., x402, permit2, erc3009)"
    )
    payment_uri: Optional[str] = Field(
        None, description="URI to payment terms or proof policy"
    )
    memory_primary: Optional[str] = Field(
        None, description="Primary long-term memory URI (NeoFS/IPFS/Unibase)"
    )
    memory_backup: Optional[str] = Field(
        None, description="Backup memory URI"
    )


class AgentDID(BaseModel):
    """
    Complete W3C DID Document for SpoonOS Agent
    """
    context: List[str] = Field(
        default=["https://www.w3.org/ns/did/v1"],
        alias="@context",
        description="JSON-LD context"
    )
    id: str = Field(..., description="DID identifier (did:spoon:agent:<id>)")
    controller: List[str] = Field(..., description="DID controllers (Ethereum addresses)")
    verification_method: List[VerificationMethod] = Field(
        default_factory=list,
        alias="verificationMethod",
        description="Verification methods"
    )
    authentication: List[str] = Field(
        default_factory=list,
        description="Authentication method references"
    )
    service: List[ServiceEndpoint] = Field(
        default_factory=list,
        description="Service endpoints"
    )

    # SpoonOS extensions
    agent_card: AgentCard = Field(..., alias="agentCard", description="Agent card metadata")
    reputation: Optional[ReputationScore] = Field(None, description="Reputation score")
    attestations: List[Attestation] = Field(
        default_factory=list,
        description="Verifiable attestations"
    )

    # On-chain anchors
    did_hash: Optional[str] = Field(None, alias="didHash", description="Keccak256 hash of DID")
    agent_card_uri: Optional[str] = Field(None, alias="agentCardURI", description="NeoFS/IPFS URI")
    did_doc_uri: Optional[str] = Field(None, alias="didDocURI", description="NeoFS/IPFS URI")
    registered_at: Optional[datetime] = Field(None, alias="registeredAt")

    class Config:
        populate_by_name = True

    def to_did_document(self) -> Dict[str, Any]:
        """Export as standard W3C DID Document"""
        return self.model_dump(
            by_alias=True,
            exclude_none=True,
            exclude={"did_hash", "agent_card_uri", "did_doc_uri", "registered_at"}
        )

    def to_agent_card(self) -> Dict[str, Any]:
        """Export agent card separately"""
        return self.agent_card.model_dump()


class DIDResolutionResult(BaseModel):
    """Result of DID resolution"""
    did_document: Optional[AgentDID] = Field(None, alias="didDocument")
    did_document_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        alias="didDocumentMetadata"
    )
    did_resolution_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        alias="didResolutionMetadata"
    )

    class Config:
        populate_by_name = True
















