from __future__ import annotations

from decimal import Decimal
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, field_validator


class X402PaymentRequest(BaseModel):
    """Describes a payment requirement that should be issued for a resource."""

    amount_usdc: Optional[Decimal] = Field(default=None, description="Amount to charge in USD (will be converted to atomic units)")
    amount_atomic: Optional[int] = Field(default=None, description="Override for atomic units (takes precedence over amount_usdc)")
    currency: Optional[str] = Field(default=None, description="User-facing currency label (e.g. USDC)")
    memo: Optional[str] = Field(default=None, description="Memo or purpose string included in payment metadata")
    payer: Optional[str] = Field(default=None, description="Optional payer identifier for receipts")

    resource: Optional[str] = Field(default=None)
    description: Optional[str] = Field(default=None)
    mime_type: Optional[str] = Field(default=None)
    scheme: Optional[str] = Field(default=None)
    network: Optional[str] = Field(default=None)
    pay_to: Optional[str] = Field(default=None)
    timeout_seconds: Optional[int] = Field(default=None)
    extra: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Structured metadata merged into x402 extra payload")
    output_schema: Optional[Dict[str, Any]] = Field(default=None, description="Structured schema advertised to payers")

    @field_validator("amount_usdc", mode="before")
    @classmethod
    def _coerce_decimal(cls, value):
        if value is None or isinstance(value, Decimal):
            return value
        return Decimal(str(value))

    @field_validator("amount_atomic", mode="before")
    @classmethod
    def _coerce_int(cls, value):
        if value is None or isinstance(value, int):
            return value
        return int(value)


class X402VerifyResult(BaseModel):
    """Captures the facilitator verification response."""

    is_valid: bool
    invalid_reason: Optional[str] = None
    payer: Optional[str] = None


class X402SettleResult(BaseModel):
    """Captures settlement details."""

    success: bool
    error_reason: Optional[str] = None
    transaction: Optional[str] = None
    network: Optional[str] = None
    payer: Optional[str] = None


class X402PaymentOutcome(BaseModel):
    """Aggregates verification and settlement outcomes."""

    verify: X402VerifyResult
    settle: Optional[X402SettleResult] = None


class X402PaymentReceipt(BaseModel):
    """Decoded representation of the X-PAYMENT-RESPONSE header."""

    success: bool
    transaction: Optional[str] = None
    network: Optional[str] = None
    payer: Optional[str] = None
    error_reason: Optional[str] = None
    raw: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("raw", mode="before")
    @classmethod
    def _ensure_dict(cls, value):
        if value is None:
            return {}
        if isinstance(value, dict):
            return value
        raise TypeError("raw payload must be a mapping")
