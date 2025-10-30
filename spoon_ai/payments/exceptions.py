class X402PaymentError(Exception):
    """Base exception for x402 payment operations."""


class X402ConfigurationError(X402PaymentError):
    """Raised when integration configuration is invalid or incomplete."""


class X402VerificationError(X402PaymentError):
    """Raised when a payment header fails verification against the facilitator."""


class X402SettlementError(X402PaymentError):
    """Raised when settlement fails or returns an error response."""
