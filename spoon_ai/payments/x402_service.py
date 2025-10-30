from __future__ import annotations

import json
from decimal import Decimal, ROUND_DOWN
from typing import Any, Dict, Optional

from eth_account import Account
from eth_account.signers.local import LocalAccount
from x402.clients.base import decode_x_payment_response, x402Client
from x402.common import x402_VERSION
from x402.encoding import safe_base64_decode
from x402.paywall import get_paywall_html, is_browser_request
from x402.types import (
    ListDiscoveryResourcesRequest,
    ListDiscoveryResourcesResponse,
    PaymentPayload,
    PaymentRequirements,
    VerifyResponse,
    SettleResponse,
    x402PaymentRequiredResponse,
)

from .config import X402Settings
from .exceptions import (
    X402ConfigurationError,
    X402PaymentError,
    X402SettlementError,
    X402VerificationError,
)
from .facilitator_client import X402FacilitatorClient
from .models import (
    X402PaymentOutcome,
    X402PaymentRequest,
    X402PaymentReceipt,
    X402SettleResult,
    X402VerifyResult,
)


class X402PaymentService:
    """High level service that aligns the x402 SDK with SpoonOS conventions."""

    def __init__(
        self,
        settings: Optional[X402Settings] = None,
        facilitator: Optional[X402FacilitatorClient] = None,
    ) -> None:
        self.settings = settings or X402Settings.load()
        self.facilitator = facilitator or X402FacilitatorClient(self.settings.facilitator_url)
        self._client_account: Optional[LocalAccount] = None

    # ------------------------------------------------------------------ #
    # Configuration helpers
    # ------------------------------------------------------------------ #
    def _merge_request(self, request: Optional[X402PaymentRequest]) -> X402PaymentRequest:
        default_extra = self.settings.build_asset_extra()
        default_request = X402PaymentRequest(
            amount_usdc=self.settings.max_amount_usdc,
            resource=self.settings.resource,
            description=self.settings.description,
            mime_type=self.settings.mime_type,
            scheme=self.settings.default_scheme,
            network=self.settings.default_network,
            pay_to=self.settings.pay_to,
            timeout_seconds=self.settings.max_timeout_seconds,
            extra=default_extra,
            currency=self.settings.asset_name,
            memo=self.settings.description,
        )

        if not request:
            return default_request

        merged = default_request.model_dump()
        incoming = request.model_dump(exclude_none=True)

        extra = {**default_request.extra, **request.extra}
        metadata = {**default_request.metadata, **request.metadata}
        merged.update({k: v for k, v in incoming.items() if k not in {"extra", "metadata"}})
        merged["extra"] = extra
        merged["metadata"] = metadata
        return X402PaymentRequest(**merged)

    def _prepare_extra(self, request: X402PaymentRequest) -> Dict[str, Any]:
        """Merge structured metadata into the x402 `extra` payload."""
        extra = dict(request.extra or {})
        metadata = request.metadata or {}
        if metadata:
            existing_meta = extra.get("metadata")
            if isinstance(existing_meta, dict):
                extra["metadata"] = {**existing_meta, **metadata}
            else:
                extra["metadata"] = metadata
        if request.currency is not None:
            extra["currency"] = request.currency
        if request.memo is not None:
            extra["memo"] = request.memo
        if request.payer is not None:
            extra["payer"] = request.payer
        return extra

    def build_payment_requirements(self, request: Optional[X402PaymentRequest] = None) -> PaymentRequirements:
        merged = self._merge_request(request)

        amount_atomic = merged.amount_atomic
        if amount_atomic is None:
            if merged.amount_usdc is None:
                amount_atomic = int(self.settings.amount_in_atomic_units)
            else:
                scale_factor = Decimal(10) ** self.settings.asset_decimals
                scaled = (merged.amount_usdc * scale_factor).quantize(Decimal("1"), rounding=ROUND_DOWN)
                amount_atomic = int(scaled)

        extra_payload = self._prepare_extra(merged)
        requirements = PaymentRequirements(
            scheme=merged.scheme or self.settings.default_scheme,
            network=merged.network or self.settings.default_network,
            max_amount_required=str(amount_atomic),
            resource=merged.resource or self.settings.resource,
            description=merged.description or self.settings.description,
            mime_type=merged.mime_type or self.settings.mime_type,
            pay_to=merged.pay_to or self.settings.pay_to,
            max_timeout_seconds=merged.timeout_seconds or self.settings.max_timeout_seconds,
            asset=self.settings.asset,
            extra=extra_payload or None,
            output_schema=merged.output_schema,
        )
        return requirements

    def build_payment_required_response(
        self,
        error: str,
        request: Optional[X402PaymentRequest] = None,
    ) -> x402PaymentRequiredResponse:
        requirements = self.build_payment_requirements(request)
        return x402PaymentRequiredResponse(
            x402_version=x402_VERSION,
            accepts=[requirements],
            error=error,
        )

    async def discover_resources(
        self,
        *,
        resource_type: Optional[str] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> ListDiscoveryResourcesResponse:
        """Query the facilitator discovery endpoint for registered paywalled resources."""
        request_payload: Optional[ListDiscoveryResourcesRequest] = None
        if resource_type or limit is not None or offset is not None:
            request_payload = ListDiscoveryResourcesRequest(type=resource_type, limit=limit, offset=offset)
        return await self.facilitator.list_resources(request_payload)

    def render_paywall_html(
        self,
        error: str,
        request: Optional[X402PaymentRequest] = None,
        headers: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Render the embedded paywall HTML with payment requirements."""
        requirements = self.build_payment_requirements(request)
        branding_dict = self.settings.paywall_branding.model_dump(exclude_none=True)
        html = get_paywall_html(error, [requirements], branding_dict or None)
        if headers and not is_browser_request(headers):
            # For API clients prefer JSON representation
            response = self.build_payment_required_response(error, request).model_dump(by_alias=True)
            return json.dumps(response)
        return html

    # ------------------------------------------------------------------ #
    # Facilitator interactions
    # ------------------------------------------------------------------ #
    def decode_payment_header(self, header_value: str) -> PaymentPayload:
        try:
            decoded = safe_base64_decode(header_value)
            return PaymentPayload.model_validate_json(decoded)
        except Exception as exc:  # pragma: no cover - defensive
            raise X402PaymentError(f"Failed to decode X-PAYMENT header: {exc}") from exc

    async def verify_payment(
        self,
        header_value: str,
        requirements: Optional[PaymentRequirements] = None,
    ) -> X402VerifyResult:
        requirements = requirements or self.build_payment_requirements()
        payload = self.decode_payment_header(header_value)
        try:
            verify_response: VerifyResponse = await self.facilitator.verify(payload, requirements)
        except Exception as exc:  # pragma: no cover - facilitator failure
            raise X402VerificationError(f"Unable to verify payment payload: {exc}") from exc
        return X402VerifyResult(
            is_valid=verify_response.is_valid,
            invalid_reason=verify_response.invalid_reason,
            payer=verify_response.payer,
        )

    async def settle_payment(
        self,
        header_value: str,
        requirements: Optional[PaymentRequirements] = None,
    ) -> X402SettleResult:
        requirements = requirements or self.build_payment_requirements()
        payload = self.decode_payment_header(header_value)
        try:
            settle_response: SettleResponse = await self.facilitator.settle(payload, requirements)
        except Exception as exc:  # pragma: no cover - facilitator failure
            raise X402SettlementError(f"Settlement failed: {exc}") from exc
        return X402SettleResult(
            success=settle_response.success,
            error_reason=settle_response.error_reason,
            transaction=settle_response.transaction,
            network=settle_response.network,
            payer=settle_response.payer,
        )

    async def verify_and_settle(
        self,
        header_value: str,
        requirements: Optional[PaymentRequirements] = None,
        settle: bool = True,
    ) -> X402PaymentOutcome:
        verify = await self.verify_payment(header_value, requirements)
        settle_result = None
        if settle and verify.is_valid:
            settle_result = await self.settle_payment(header_value, requirements)
        return X402PaymentOutcome(verify=verify, settle=settle_result)

    # ------------------------------------------------------------------ #
    # Client-side helpers
    # ------------------------------------------------------------------ #
    def _get_client_account(self) -> LocalAccount:
        if self._client_account:
            return self._client_account
        if not self.settings.client.private_key:
            raise X402ConfigurationError(
                "X402 client private key not configured. Set X402_AGENT_PRIVATE_KEY or configure x402.client.private_key."
            )
        self._client_account = Account.from_key(self.settings.client.private_key)  # type: ignore[arg-type]
        return self._client_account

    def build_payment_header(
        self,
        requirements: PaymentRequirements,
        *,
        max_value: Optional[int] = None,
    ) -> str:
        """Create a signed X-PAYMENT header for outbound requests."""
        if max_value is not None:
            required_value = int(requirements.max_amount_required)
            if required_value > max_value:
                raise X402PaymentError(
                    f"Payment requirement exceeds allowed maximum: required {required_value}, max_value {max_value}."
                )
        account = self._get_client_account()
        client = x402Client(account=account, max_value=max_value)
        return client.create_payment_header(requirements)

    def decode_payment_response(self, header_value: str) -> X402PaymentReceipt:
        """Decode an X-PAYMENT-RESPONSE header into a structured receipt."""
        try:
            payload = decode_x_payment_response(header_value)
        except Exception as exc:  # pragma: no cover - defensive
            raise X402PaymentError(f"Failed to decode X-PAYMENT-RESPONSE header: {exc}") from exc

        receipt_payload = {
            "success": bool(payload.get("success")),
            "transaction": payload.get("transaction"),
            "network": payload.get("network"),
            "payer": payload.get("payer"),
            "error_reason": payload.get("error_reason") or payload.get("errorReason") or payload.get("error"),
            "raw": payload,
        }
        return X402PaymentReceipt.model_validate(receipt_payload)
