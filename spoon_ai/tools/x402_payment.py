from __future__ import annotations

import json
from typing import Any, Dict, Optional

import httpx
from pydantic import Field

from x402.types import PaymentRequirements, x402PaymentRequiredResponse

from spoon_ai.tools.base import BaseTool, ToolResult
from spoon_ai.payments import (
    X402PaymentRequest,
    X402PaymentService,
    X402ConfigurationError,
    X402PaymentError,
)


class X402PaymentHeaderTool(BaseTool):
    """Create a signed X-PAYMENT header for a given resource."""

    name: str = "x402_create_payment"
    description: str = "Generate a signed x402 payment header for a paywalled resource."
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "resource": {"type": "string", "description": "Resource URL being accessed"},
            "description": {"type": "string", "description": "Description for the payment"},
            "mime_type": {"type": "string", "description": "MIME type of the resource"},
            "amount_usdc": {"type": "number", "description": "Amount in USD to authorise"},
            "amount_atomic": {"type": "integer", "description": "Amount override in atomic units"},
            "scheme": {"type": "string", "description": "Payment scheme identifier"},
            "network": {"type": "string", "description": "Blockchain network identifier"},
            "pay_to": {"type": "string", "description": "Wallet receiving the payment"},
            "timeout_seconds": {"type": "integer", "description": "Valid duration of the payment"},
            "max_value": {"type": "integer", "description": "Optional safety limit for payment amount"},
            "currency": {"type": "string", "description": "Override the currency label presented to the payer"},
            "memo": {"type": "string", "description": "Memo or purpose string recorded with the payment"},
            "payer": {"type": "string", "description": "Optional payer identifier stored in metadata"},
            "metadata": {
                "type": "object",
                "description": "Arbitrary key/value metadata merged into the payment extra payload",
                "additionalProperties": True,
            },
            "output_schema": {
                "type": "object",
                "description": "JSON schema advertised alongside the paywalled response shape",
                "additionalProperties": True,
            },
        },
        "additionalProperties": False,
    }

    service: X402PaymentService = Field(exclude=True)

    async def execute(
        self,
        resource: Optional[str] = None,
        description: Optional[str] = None,
        mime_type: Optional[str] = None,
        amount_usdc: Optional[float] = None,
        amount_atomic: Optional[int] = None,
        scheme: Optional[str] = None,
        network: Optional[str] = None,
        pay_to: Optional[str] = None,
        timeout_seconds: Optional[int] = None,
        max_value: Optional[int] = None,
        currency: Optional[str] = None,
        memo: Optional[str] = None,
        payer: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        output_schema: Optional[Dict[str, Any]] = None,
    ) -> ToolResult:
        try:
            request = X402PaymentRequest(
                resource=resource,
                description=description,
                mime_type=mime_type,
                amount_usdc=amount_usdc,
                amount_atomic=amount_atomic,
                scheme=scheme,
                network=network,
                pay_to=pay_to,
                timeout_seconds=timeout_seconds,
                currency=currency,
                memo=memo,
                payer=payer,
                metadata=metadata or {},
                output_schema=output_schema,
            )
            requirements = self.service.build_payment_requirements(request)
            header = self.service.build_payment_header(requirements, max_value=max_value)
            payload = {
                "header": header,
                "requirements": requirements.model_dump(by_alias=True, exclude_none=True),
            }
            return ToolResult(output=payload)
        except X402ConfigurationError as exc:
            return ToolResult(error=f"x402 configuration error: {exc}")
        except Exception as exc:  # pragma: no cover - defensive
            return ToolResult(error=f"Unable to create x402 payment header: {exc}")


class X402PaywalledRequestTool(BaseTool):
    """Fetch a paywalled resource, handling the x402 402 negotiation automatically."""

    name: str = "x402_paywalled_request"
    description: str = "Call an HTTP endpoint protected by x402; automatically signs and retries with payment."
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "Endpoint to call"},
            "method": {"type": "string", "enum": ["GET", "POST", "PUT", "DELETE", "PATCH"], "default": "GET"},
            "headers": {
                "type": "object",
                "description": "Optional headers for the request",
                "additionalProperties": {"type": "string"},
            },
            "body": {"description": "JSON/body payload to send", "type": ["object", "string", "null"]},
            "amount_usdc": {"type": "number", "description": "Optional override for the payment amount"},
            "amount_atomic": {"type": "integer", "description": "Atomic units override"},
            "scheme": {"type": "string", "description": "Preferred payment scheme"},
            "network": {"type": "string", "description": "Preferred network"},
            "pay_to": {"type": "string", "description": "Override pay_to address"},
            "timeout_seconds": {"type": "integer", "description": "Payment validity window"},
            "max_value": {"type": "integer", "description": "Safety limit for signed payments"},
            "timeout": {"type": "number", "description": "HTTP timeout in seconds", "default": 30},
            "currency": {"type": "string", "description": "Currency label override"},
            "memo": {"type": "string", "description": "Memo stored with the payment"},
            "payer": {"type": "string", "description": "Payer identifier recorded in metadata"},
            "metadata": {
                "type": "object",
                "description": "Additional metadata merged into the x402 extra payload",
                "additionalProperties": True,
            },
            "output_schema": {
                "type": "object",
                "description": "Advertised JSON schema for the expected response",
                "additionalProperties": True,
            },
        },
        "required": ["url"],
        "additionalProperties": False,
    }

    service: X402PaymentService = Field(exclude=True)

    async def execute(
        self,
        url: str,
        method: str = "GET",
        headers: Optional[Dict[str, str]] = None,
        body: Optional[Any] = None,
        amount_usdc: Optional[float] = None,
        amount_atomic: Optional[int] = None,
        scheme: Optional[str] = None,
        network: Optional[str] = None,
        pay_to: Optional[str] = None,
        timeout_seconds: Optional[int] = None,
        max_value: Optional[int] = None,
        timeout: float = 30.0,
        currency: Optional[str] = None,
        memo: Optional[str] = None,
        payer: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        output_schema: Optional[Dict[str, Any]] = None,
    ) -> ToolResult:
        async with httpx.AsyncClient(timeout=timeout) as client:
            initial_headers = headers.copy() if headers else {}
            initial_response = await client.request(method.upper(), url, headers=initial_headers, json=body if isinstance(body, (dict, list)) else None, content=body if isinstance(body, (str, bytes)) else None)

            if initial_response.status_code != 402:
                return ToolResult(output=self._format_response(initial_response))

            try:
                paywall_json = initial_response.json()
                paywall = x402PaymentRequiredResponse.model_validate(paywall_json)
            except Exception as exc:
                return ToolResult(error=f"Failed to parse x402 payment requirements: {exc}")

            requirements = self._select_requirements(paywall, scheme or self.service.settings.default_scheme, network or self.service.settings.default_network)
            if any(
                value is not None
                for value in (
                    amount_usdc,
                    amount_atomic,
                    pay_to,
                    timeout_seconds,
                    currency,
                    memo,
                    payer,
                    metadata,
                    output_schema,
                )
            ):
                request_override = X402PaymentRequest(
                    amount_usdc=amount_usdc,
                    amount_atomic=amount_atomic,
                    network=network,
                    scheme=scheme,
                    pay_to=pay_to,
                    timeout_seconds=timeout_seconds,
                    currency=currency,
                    memo=memo,
                    payer=payer,
                    metadata=metadata or {},
                    output_schema=output_schema,
                )
                requirements = self.service.build_payment_requirements(request_override)

            try:
                header = self.service.build_payment_header(requirements, max_value=max_value)
            except X402ConfigurationError as exc:
                return ToolResult(error=f"x402 configuration error: {exc}")
            except X402PaymentError as exc:
                return ToolResult(error=str(exc))
            except Exception as exc:  # pragma: no cover
                return ToolResult(error=f"Failed to sign payment header: {exc}")

            payment_headers = headers.copy() if headers else {}
            payment_headers["X-PAYMENT"] = header
            paid_response = await client.request(
                method.upper(),
                url,
                headers=payment_headers,
                json=body if isinstance(body, (dict, list)) else None,
                content=body if isinstance(body, (str, bytes)) else None,
            )

            output = self._format_response(paid_response)
            output["paymentHeader"] = header
            output["requirements"] = requirements.model_dump(by_alias=True, exclude_none=True)

            payment_response_header = paid_response.headers.get("X-PAYMENT-RESPONSE")
            if payment_response_header:
                try:
                    receipt = self.service.decode_payment_response(payment_response_header)
                    output["paymentResponse"] = receipt.model_dump(exclude_none=True)
                except X402PaymentError as exc:
                    output["paymentResponse"] = {"error": str(exc), "raw": payment_response_header}

            return ToolResult(output=output)

    def _select_requirements(
        self,
        paywall: x402PaymentRequiredResponse,
        scheme: str,
        network: str,
    ) -> PaymentRequirements:
        if not paywall.accepts:
            raise X402PaymentError("Paywall response does not contain any payment requirements.")
        for candidate in paywall.accepts:
            if candidate.scheme == scheme and candidate.network == network:
                return candidate
        for candidate in paywall.accepts:
            if candidate.scheme == scheme:
                return candidate
        return paywall.accepts[0]

    def _format_response(self, response: httpx.Response) -> Dict[str, Any]:
        body: Any
        try:
            body = response.json()
        except json.JSONDecodeError:
            body = response.text
        return {
            "status": response.status_code,
            "headers": dict(response.headers),
            "body": body,
        }
