"""
SpoonOS Agent DID Identity Module
Implements ERC-8004 compliant decentralized identity for agents
"""

from .did_models import (
    AgentDID,
    AgentCard,
    VerificationMethod,
    ServiceEndpoint,
    Attestation,
    ReputationScore
)
from .did_resolver import DIDResolver
from .attestation import AttestationManager, TrustScoreCalculator
from .erc8004_client import ERC8004Client

__all__ = [
    "AgentDID",
    "AgentCard",
    "VerificationMethod",
    "ServiceEndpoint",
    "Attestation",
    "ReputationScore",
    "DIDResolver",
    "AttestationManager",
    "TrustScoreCalculator",
    "ERC8004Client"
]
















