"""
Attestation and Trust Score Management
Handles verifiable credentials and reputation calculations
"""

from typing import Dict, List, Optional
from datetime import datetime
from eth_account import Account
from eth_account.messages import encode_defunct
from .did_models import Attestation
from .erc8004_client import ERC8004Client


class AttestationManager:
    """Manages verifiable attestations for agents"""

    def __init__(self, erc8004_client: ERC8004Client, private_key: Optional[str] = None):
        self.erc8004_client = erc8004_client
        self.private_key = private_key
        if private_key:
            self.account = Account.from_key(private_key)
        else:
            self.account = None

    def create_attestation(
        self,
        issuer_did: str,
        subject_did: str,
        claim: Dict,
        evidence: Optional[str] = None
    ) -> Attestation:
        """
        Create a verifiable attestation

        Args:
            issuer_did: DID of the attestation issuer
            subject_did: DID of the agent being attested
            claim: Attestation claim data
            evidence: Optional supporting evidence

        Returns:
            Signed Attestation object
        """
        attestation = Attestation(
            issuer=issuer_did,
            subject=subject_did,
            claim=claim,
            evidence=evidence,
            issued_at=datetime.utcnow()
        )

        # Sign attestation if private key available
        if self.account:
            attestation_data = f"{issuer_did}|{subject_did}|{claim}|{attestation.issued_at.isoformat()}"
            message = encode_defunct(text=attestation_data)
            signed = self.account.sign_message(message)
            attestation.signature = signed.signature.hex()

        return attestation

    def verify_attestation(self, attestation: Attestation) -> bool:
        """Verify attestation signature"""
        if not attestation.signature:
            return False

        try:
            attestation_data = f"{attestation.issuer}|{attestation.subject}|{attestation.claim}|{attestation.issued_at.isoformat()}"
            message = encode_defunct(text=attestation_data)

            # Recover signer address
            recovered_address = Account.recover_message(
                message,
                signature=bytes.fromhex(attestation.signature.replace('0x', ''))
            )

            # Verify issuer controls the DID
            issuer_metadata = self.erc8004_client.resolve_agent(attestation.issuer)
            return recovered_address.lower() in [
                addr.lower() for addr in issuer_metadata.get("controllers", [])
            ]
        except Exception as e:
            print(f"Attestation verification failed: {e}")
            return False

    def submit_reputation_on_chain(
        self,
        subject_did: str,
        score: int,
        evidence: str
    ) -> str:
        """
        Submit reputation score to on-chain registry

        Args:
            subject_did: DID of agent being rated
            score: Score between -100 and 100
            evidence: Evidence for the score

        Returns:
            Transaction hash
        """
        if score < -100 or score > 100:
            raise ValueError("Score must be between -100 and 100")

        return self.erc8004_client.submit_reputation(subject_did, score, evidence)

    def submit_validation_on_chain(
        self,
        subject_did: str,
        is_valid: bool,
        reason: str
    ) -> str:
        """
        Submit validation for an agent

        Args:
            subject_did: DID of agent being validated
            is_valid: Whether agent is valid
            reason: Reason for validation decision

        Returns:
            Transaction hash
        """
        return self.erc8004_client.submit_validation(subject_did, is_valid, reason)


class TrustScoreCalculator:
    """Calculates trust scores for agents"""

    def __init__(self, erc8004_client: ERC8004Client):
        self.erc8004_client = erc8004_client

    def calculate_trust_score(self, did: str) -> Dict:
        """
        Calculate comprehensive trust score

        Returns:
            Dict with trust score components:
            - reputation_score: -100 to 100
            - validation_status: bool
            - trust_level: "high" | "medium" | "low" | "untrusted"
            - confidence: 0 to 1
        """
        # Get reputation (average 0-100) and count
        reputation_score, reputation_submissions = self.erc8004_client.get_reputation(did)

        # Get validation summary (count/avg 0-100) with backward compatibility
        validation_summary = None
        getter = getattr(self.erc8004_client, "get_validation_summary", None)
        if callable(getter):
            try:
                validation_summary = getter(did)
            except Exception:
                validation_summary = None
        if not isinstance(validation_summary, dict) or "isValidated" not in validation_summary:
            validation_summary = {
                "isValidated": False,
                "count": 0,
                "averageResponse": 0,
            }
        is_validated = validation_summary.get("isValidated", False)

        # Calculate confidence based on number of submissions
        confidence = min(1.0, reputation_submissions / 10.0)

        # Determine trust level
        if is_validated and reputation_score > 50:
            trust_level = "high"
        elif is_validated and reputation_score > 0:
            trust_level = "medium"
        elif reputation_score > -50:
            trust_level = "low"
        else:
            trust_level = "untrusted"

        return {
            "reputation_score": reputation_score,
            "reputation_submissions": reputation_submissions,
            "validation_status": validation_summary,
            "trust_level": trust_level,
            "confidence": confidence,
            "calculated_at": datetime.utcnow().isoformat()
        }

    def get_reputation_breakdown(self, did: str, limit: int = 10) -> List[Dict]:
        """Get detailed reputation submissions"""
        agent_id = self.erc8004_client.calculate_did_hash(did)

        try:
            submitters, scores, evidences, timestamps = \
                self.erc8004_client.reputation_registry.functions.getReputationSubmissions(
                    agent_id, 0, limit
                ).call()

            return [
                {
                    "submitter": submitter,
                    "score": score,
                    "evidence": evidence,
                    "timestamp": timestamp
                }
                for submitter, score, evidence, timestamp in zip(
                    submitters, scores, evidences, timestamps
                )
            ]
        except Exception as e:
            print(f"Error fetching reputation breakdown: {e}")
            return []

    def get_validation_breakdown(self, did: str, limit: int = 10) -> List[Dict]:
        """Get detailed validation submissions"""
        agent_id = self.erc8004_client.calculate_did_hash(did)

        try:
            validators, validations, reasons, timestamps = \
                self.erc8004_client.validation_registry.functions.getValidationSubmissions(
                    agent_id, 0, limit
                ).call()

            return [
                {
                    "validator": validator,
                    "is_valid": is_valid,
                    "reason": reason,
                    "timestamp": timestamp
                }
                for validator, is_valid, reason, timestamp in zip(
                    validators, validations, reasons, timestamps
                )
            ]
        except Exception as e:
            print(f"Error fetching validation breakdown: {e}")
            return []
















