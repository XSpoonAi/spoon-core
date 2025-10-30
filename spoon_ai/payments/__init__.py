"""
Payment utilities for integrating the SpoonOS core with the x402 payments protocol.

This package wraps the upstream `x402` Python SDK with configuration and service
abstractions that align to SpoonOS conventions (config.json priority, .env overrides,
and async-friendly helper utilities).
"""

from .config import X402Settings, X402PaywallBranding, X402ClientConfig
from .exceptions import (
    X402ConfigurationError,
    X402PaymentError,
    X402VerificationError,
    X402SettlementError,
)
from .models import X402PaymentRequest, X402PaymentOutcome, X402VerifyResult, X402SettleResult, X402PaymentReceipt
from .x402_service import X402PaymentService

__all__ = [
    "X402Settings",
    "X402PaywallBranding",
    "X402ClientConfig",
    "X402PaymentService",
    "X402PaymentRequest",
    "X402PaymentOutcome",
    "X402VerifyResult",
    "X402SettleResult",
    "X402PaymentReceipt",
    "X402ConfigurationError",
    "X402PaymentError",
    "X402VerificationError",
    "X402SettlementError",
]
