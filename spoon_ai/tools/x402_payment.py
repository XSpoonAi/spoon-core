from __future__ import annotations

import json
import secrets
import time
from typing import Any, Dict, Optional

import httpx
from pydantic import Field

from x402.encoding import safe_base64_decode
from x402.types import PaymentRequirements, x402PaymentRequiredResponse
from x402.common import x402_VERSION
from x402.chains import get_chain_id
from x402.exact import prepare_payment_header, encode_payment
from eth_account.messages import encode_typed_data

from spoon_ai.tools.base import BaseTool, ToolResult
from spoon_ai.payments import (
    X402PaymentRequest,
    X402PaymentService,
    X402ConfigurationError,
    X402PaymentError,
)


class X402PaymentHeaderTool(BaseTool):
    """Create a signed X-PAYMENT header for a given resource."""

    requires_decrypted_env = True
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

    requires_decrypted_env = True
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
            "body": {"description": "JSON payload string or raw body content.", "type": "string"},
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
                paywall_payload = self._extract_paywall_payload(initial_response)
            except Exception as exc:
                return ToolResult(error=f"Failed to parse x402 payment requirements: {exc}")

            x402_version = paywall_payload.get("x402Version") or paywall_payload.get("x402_version")
            if x402_version == 2:
                try:
                    return await self._execute_v2_paywall(
                        client=client,
                        paywall_payload=paywall_payload,
                        url=url,
                        method=method,
                        headers=headers,
                        body=body,
                        amount_usdc=amount_usdc,
                        amount_atomic=amount_atomic,
                        scheme=scheme,
                        network=network,
                        pay_to=pay_to,
                        timeout_seconds=timeout_seconds,
                        max_value=max_value,
                        currency=currency,
                        memo=memo,
                        payer=payer,
                        metadata=metadata,
                        output_schema=output_schema,
                    )
                except Exception as exc:
                    return ToolResult(error=f"Failed to process x402 v2 paywall: {exc}")

            try:
                paywall = x402PaymentRequiredResponse.model_validate(paywall_payload)
            except Exception as exc:
                return ToolResult(error=f"Failed to parse x402 payment requirements: {exc}")

            selected_requirements = self._select_requirements(
                paywall,
                scheme or self.service.settings.default_scheme,
                network or self.service.settings.default_network,
            )
            req_payload = selected_requirements.model_dump(by_alias=True)
            requirements = PaymentRequirements.model_validate(req_payload)
            candidate_pay_to = None
            if paywall.accepts:
                for candidate_requirement in paywall.accepts:
                    if candidate_requirement.pay_to:
                        candidate_pay_to = candidate_requirement.pay_to
                        break
            if candidate_pay_to:
                requirements.pay_to = candidate_pay_to
            if not requirements.pay_to:
                requirements.pay_to = self.service.settings.pay_to
            pay_to_override = requirements.pay_to
            override_requested = any(
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
            )
            if override_requested:
                paywall_defaults = {
                    "asset": requirements.asset,
                    "pay_to": requirements.pay_to,
                    "resource": requirements.resource,
                    "description": requirements.description,
                    "mime_type": requirements.mime_type,
                    "output_schema": requirements.output_schema,
                    "extra": requirements.extra or {},
                    "max_timeout_seconds": requirements.max_timeout_seconds,
                }

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

                requirements.asset = paywall_defaults["asset"]
                requirements.pay_to = pay_to or paywall_defaults["pay_to"]
                requirements.resource = paywall_defaults["resource"]
                if paywall_defaults["description"]:
                    requirements.description = paywall_defaults["description"]
                if paywall_defaults["mime_type"]:
                    requirements.mime_type = paywall_defaults["mime_type"]
                if paywall_defaults["output_schema"] and requirements.output_schema is None:
                    requirements.output_schema = paywall_defaults["output_schema"]
                requirements.max_timeout_seconds = (
                    timeout_seconds if timeout_seconds is not None else paywall_defaults["max_timeout_seconds"]
                )
                merged_extra: Dict[str, Any] = {**paywall_defaults["extra"], **(requirements.extra or {})}
                requirements.extra = merged_extra or None
                pay_to_override = requirements.pay_to

            try:
                if self.service.settings.client.use_turnkey:
                    header = self.service.build_payment_header(requirements, max_value=max_value)
                else:
                    header = self._build_local_payment_header(requirements, pay_to_override, max_value=max_value)
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

    def _extract_paywall_payload(self, response: httpx.Response) -> Dict[str, Any]:
        """Extract x402 payment requirements from a 402 response.

        x402.org v2 paywalls return the requirements in the PAYMENT-REQUIRED header (base64 JSON),
        while older implementations may return JSON in the response body.
        """
        header_value = response.headers.get("payment-required")
        if header_value:
            decoded = safe_base64_decode(header_value)
            payload = json.loads(decoded)
            if isinstance(payload, dict):
                return payload
            raise ValueError("PAYMENT-REQUIRED header did not decode into an object.")

        payload = response.json()
        if isinstance(payload, dict):
            return payload
        raise ValueError("Paywall response JSON did not decode into an object.")

    async def _execute_v2_paywall(
        self,
        *,
        client: httpx.AsyncClient,
        paywall_payload: Dict[str, Any],
        url: str,
        method: str,
        headers: Optional[Dict[str, str]],
        body: Optional[Any],
        amount_usdc: Optional[float],
        amount_atomic: Optional[int],
        scheme: Optional[str],
        network: Optional[str],
        pay_to: Optional[str],
        timeout_seconds: Optional[int],
        max_value: Optional[int],
        currency: Optional[str],
        memo: Optional[str],
        payer: Optional[str],
        metadata: Optional[Dict[str, Any]],
        output_schema: Optional[Dict[str, Any]],
    ) -> ToolResult:
        """Handle x402 v2 paywalls (PAYMENT-REQUIRED header payloads)."""
        accepts = paywall_payload.get("accepts") or []
        if not isinstance(accepts, list) or not accepts:
            raise X402PaymentError("Paywall response does not contain any payment requirements.")

        selected = self._select_requirements_v2(
            accepts,
            desired_scheme=scheme or self.service.settings.default_scheme,
            desired_network=network or self.service.settings.default_network,
        )

        requirements = self._build_requirements_v2(
            paywall_payload=paywall_payload,
            selected=selected,
            amount_usdc=amount_usdc,
            amount_atomic=amount_atomic,
            pay_to=pay_to,
            timeout_seconds=timeout_seconds,
            currency=currency,
            memo=memo,
            payer=payer,
            metadata=metadata,
            output_schema=output_schema,
        )

        try:
            if self.service.settings.client.use_turnkey:
                payment_header = self._build_turnkey_payment_header_v2(requirements, max_value=max_value)
            else:
                payment_header = self._build_local_payment_header_v2(requirements, max_value=max_value)
        except X402ConfigurationError as exc:
            return ToolResult(error=f"x402 configuration error: {exc}")
        except X402PaymentError as exc:
            return ToolResult(error=str(exc))

        payment_headers = headers.copy() if headers else {}
        # x402 v2 uses modernized HTTP headers (PAYMENT-SIGNATURE / PAYMENT-RESPONSE).
        payment_headers["PAYMENT-SIGNATURE"] = payment_header

        paid_response = await client.request(
            method.upper(),
            url,
            headers=payment_headers,
            json=body if isinstance(body, (dict, list)) else None,
            content=body if isinstance(body, (str, bytes)) else None,
        )

        output = self._format_response(paid_response)
        output["paymentHeader"] = payment_header
        output["requirements"] = requirements

        payment_response_header = (
            paid_response.headers.get("PAYMENT-RESPONSE")
            or paid_response.headers.get("X-PAYMENT-RESPONSE")
        )
        if payment_response_header:
            try:
                receipt = self.service.decode_payment_response(payment_response_header)
                output["paymentResponse"] = receipt.model_dump(exclude_none=True)
            except X402PaymentError as exc:
                output["paymentResponse"] = {"error": str(exc), "raw": payment_response_header}

        return ToolResult(output=output)

    def _select_requirements_v2(
        self,
        accepts: list[Any],
        *,
        desired_scheme: str,
        desired_network: str,
    ) -> Dict[str, Any]:
        candidates: list[Dict[str, Any]] = []
        for entry in accepts:
            if isinstance(entry, dict):
                candidates.append(entry)

        if not candidates:
            raise X402PaymentError("Paywall response does not contain any usable payment requirements.")

        by_scheme = [c for c in candidates if c.get("scheme") == desired_scheme]
        if not by_scheme:
            by_scheme = candidates

        # Prefer exact network match.
        for candidate in by_scheme:
            if candidate.get("network") == desired_network:
                return candidate

        # Prefer EVM chain-id match (supports CAIP-2 `eip155:<id>`).
        desired_chain_id = self._resolve_evm_chain_id(desired_network)
        if desired_chain_id is not None:
            for candidate in by_scheme:
                candidate_chain_id = self._resolve_evm_chain_id(str(candidate.get("network") or ""))
                if candidate_chain_id is not None and candidate_chain_id == desired_chain_id:
                    return candidate

        return by_scheme[0]

    def _build_requirements_v2(
        self,
        *,
        paywall_payload: Dict[str, Any],
        selected: Dict[str, Any],
        amount_usdc: Optional[float],
        amount_atomic: Optional[int],
        pay_to: Optional[str],
        timeout_seconds: Optional[int],
        currency: Optional[str],
        memo: Optional[str],
        payer: Optional[str],
        metadata: Optional[Dict[str, Any]],
        output_schema: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Normalize a selected v2 requirement into the dict we use for signing."""
        x402_version = paywall_payload.get("x402Version") or 2
        network_value = selected.get("network")
        scheme_value = selected.get("scheme")
        asset_value = selected.get("asset")
        pay_to_value = pay_to or selected.get("payTo") or selected.get("pay_to") or self.service.settings.pay_to
        timeout_value = timeout_seconds if timeout_seconds is not None else selected.get("maxTimeoutSeconds") or 0

        extra_value: Dict[str, Any] = {}
        if isinstance(selected.get("extra"), dict):
            extra_value.update(selected["extra"])

        if metadata:
            extra_meta = extra_value.get("metadata")
            if isinstance(extra_meta, dict):
                extra_value["metadata"] = {**extra_meta, **metadata}
            else:
                extra_value["metadata"] = metadata
        if currency is not None:
            extra_value["currency"] = currency
        if memo is not None:
            extra_value["memo"] = memo
        if payer is not None:
            extra_value["payer"] = payer
        if output_schema is not None:
            extra_value["outputSchema"] = output_schema

        if amount_atomic is not None:
            amount_value = str(int(amount_atomic))
        elif amount_usdc is not None:
            derived = self.service.build_payment_requirements(X402PaymentRequest(amount_usdc=amount_usdc))
            amount_value = derived.max_amount_required
        else:
            amount_value = str(selected.get("amount") or selected.get("maxAmountRequired") or selected.get("max_amount_required") or "0")

        if not scheme_value:
            raise X402PaymentError("v2 payment requirement missing scheme.")
        if not network_value:
            raise X402PaymentError("v2 payment requirement missing network.")
        if not asset_value:
            raise X402PaymentError("v2 payment requirement missing asset.")
        if not pay_to_value:
            raise X402PaymentError("v2 payment requirement missing payTo.")

        return {
            "x402Version": int(x402_version),
            "scheme": str(scheme_value),
            "network": str(network_value),
            "amount": str(amount_value),
            "asset": str(asset_value),
            "payTo": str(pay_to_value),
            "maxTimeoutSeconds": int(timeout_value),
            "extra": extra_value or None,
            "resource": paywall_payload.get("resource"),
        }

    def _resolve_evm_chain_id(self, network_value: str) -> Optional[int]:
        if not network_value:
            return None
        if network_value.startswith("eip155:"):
            _, _, chain_id_str = network_value.partition(":")
            try:
                return int(chain_id_str)
            except ValueError:
                return None
        try:
            return int(get_chain_id(network_value))
        except Exception:
            return None

    def _build_local_payment_header_v2(self, requirements: Dict[str, Any], *, max_value: Optional[int] = None) -> str:
        account = self.service._get_client_account()

        required_value = int(requirements.get("amount") or 0)
        if max_value is not None and required_value > max_value:
            raise X402PaymentError(
                f"Payment requirement exceeds allowed maximum: required {required_value}, max_value {max_value}."
            )

        pay_to = requirements.get("payTo")
        if not pay_to:
            raise X402PaymentError("Paywalled resource did not specify a payTo address.")

        nonce_hex = secrets.token_hex(32)
        valid_after = int(time.time()) - 60
        valid_before = int(time.time()) + int(requirements.get("maxTimeoutSeconds") or 0)

        authorization = {
            "from": account.address,
            "to": pay_to,
            "value": str(required_value),
            "validAfter": str(valid_after),
            "validBefore": str(valid_before),
            "nonce": "0x" + nonce_hex,
        }

        signature = self._sign_eip712_authorization_v2(
            account=account,
            requirements=requirements,
            authorization=authorization,
            nonce_hex=nonce_hex,
        )
        accepted = {
            "scheme": requirements["scheme"],
            "network": requirements["network"],
            "amount": str(requirements.get("amount") or required_value),
            "asset": requirements["asset"],
            "payTo": pay_to,
            "maxTimeoutSeconds": int(requirements.get("maxTimeoutSeconds") or 0),
        }
        if requirements.get("extra"):
            accepted["extra"] = requirements["extra"]

        payment_payload: Dict[str, Any] = {
            "x402Version": int(requirements.get("x402Version") or 2),
            "resource": requirements.get("resource"),
            "accepted": accepted,
            "payload": {"signature": signature, "authorization": authorization},
        }
        if requirements.get("extensions"):
            payment_payload["extensions"] = requirements["extensions"]

        return encode_payment(payment_payload)

    def _build_turnkey_payment_header_v2(self, requirements: Dict[str, Any], *, max_value: Optional[int] = None) -> str:
        client_cfg = self.service.settings.client
        if not client_cfg.turnkey_sign_with:
            raise X402ConfigurationError(
                "Turnkey signing identity missing. Set X402_TURNKEY_SIGN_WITH or TURNKEY_SIGN_WITH."
            )
        if not client_cfg.turnkey_address:
            raise X402ConfigurationError(
                "Turnkey payer address missing. Set X402_TURNKEY_ADDRESS or TURNKEY_ADDRESS."
            )

        required_value = int(requirements.get("amount") or 0)
        if max_value is not None and required_value > max_value:
            raise X402PaymentError(
                f"Payment requirement exceeds allowed maximum: required {required_value}, max_value {max_value}."
            )

        nonce_hex = secrets.token_hex(32)
        valid_after = int(time.time()) - 60
        valid_before = int(time.time()) + int(requirements.get("maxTimeoutSeconds") or 0)

        authorization = {
            "from": client_cfg.turnkey_address,
            "to": requirements["payTo"],
            "value": str(required_value),
            "validAfter": str(valid_after),
            "validBefore": str(valid_before),
            "nonce": "0x" + nonce_hex,
        }

        typed_data = self._build_typed_data_v2(
            requirements=requirements,
            authorization=authorization,
        )
        response = self.service._get_turnkey_client().sign_typed_data(
            sign_with=client_cfg.turnkey_sign_with,
            typed_data=typed_data,
        )
        signature = self.service._extract_turnkey_signature(response)
        if not signature:
            raise X402PaymentError("Turnkey did not return a usable signature payload.")

        accepted = {
            "scheme": requirements["scheme"],
            "network": requirements["network"],
            "amount": str(requirements.get("amount") or required_value),
            "asset": requirements["asset"],
            "payTo": requirements["payTo"],
            "maxTimeoutSeconds": int(requirements.get("maxTimeoutSeconds") or 0),
        }
        if requirements.get("extra"):
            accepted["extra"] = requirements["extra"]

        payment_payload: Dict[str, Any] = {
            "x402Version": int(requirements.get("x402Version") or 2),
            "resource": requirements.get("resource"),
            "accepted": accepted,
            "payload": {"signature": signature, "authorization": authorization},
        }
        if requirements.get("extensions"):
            payment_payload["extensions"] = requirements["extensions"]

        return encode_payment(payment_payload)

    def _build_typed_data_v2(
        self,
        *,
        requirements: Dict[str, Any],
        authorization: Dict[str, Any],
    ) -> Dict[str, Any]:
        extra = requirements.get("extra") or {}
        chain_id = self._resolve_evm_chain_id(str(requirements.get("network") or ""))
        if chain_id is None:
            raise X402PaymentError(f"Unsupported EVM network: {requirements.get('network')}")

        domain: Dict[str, Any] = {
            "name": extra.get("name") or self.service.settings.asset_name,
            "version": extra.get("version") or "2",
            "chainId": int(chain_id),
            "verifyingContract": requirements["asset"],
        }
        if extra.get("salt"):
            domain["salt"] = extra["salt"]

        message = {
            "from": authorization["from"],
            "to": authorization["to"],
            "value": int(authorization["value"]),
            "validAfter": int(authorization["validAfter"]),
            "validBefore": int(authorization["validBefore"]),
            "nonce": authorization["nonce"],
        }

        return {
            "types": {
                "TransferWithAuthorization": [
                    {"name": "from", "type": "address"},
                    {"name": "to", "type": "address"},
                    {"name": "value", "type": "uint256"},
                    {"name": "validAfter", "type": "uint256"},
                    {"name": "validBefore", "type": "uint256"},
                    {"name": "nonce", "type": "bytes32"},
                ]
            },
            "primaryType": "TransferWithAuthorization",
            "domain": domain,
            "message": message,
        }

    def _sign_eip712_authorization_v2(
        self,
        *,
        account: "LocalAccount",
        requirements: Dict[str, Any],
        authorization: Dict[str, Any],
        nonce_hex: str,
    ) -> str:
        extra = requirements.get("extra") or {}
        chain_id = self._resolve_evm_chain_id(str(requirements.get("network") or ""))
        if chain_id is None:
            raise X402PaymentError(f"Unsupported EVM network: {requirements.get('network')}")

        domain: Dict[str, Any] = {
            "name": extra.get("name") or self.service.settings.asset_name,
            "version": extra.get("version") or "2",
            "chainId": int(chain_id),
            "verifyingContract": requirements["asset"],
        }
        if extra.get("salt"):
            domain["salt"] = extra["salt"]

        message_types = {
            "TransferWithAuthorization": [
                {"name": "from", "type": "address"},
                {"name": "to", "type": "address"},
                {"name": "value", "type": "uint256"},
                {"name": "validAfter", "type": "uint256"},
                {"name": "validBefore", "type": "uint256"},
                {"name": "nonce", "type": "bytes32"},
            ]
        }

        message_data = {
            "from": authorization["from"],
            "to": authorization["to"],
            "value": int(authorization["value"]),
            "validAfter": int(authorization["validAfter"]),
            "validBefore": int(authorization["validBefore"]),
            "nonce": bytes.fromhex(nonce_hex),
        }

        signable = encode_typed_data(domain, message_types, message_data)
        signature = account.sign_message(signable).signature.hex()
        return signature if signature.startswith("0x") else f"0x{signature}"

    def _build_local_payment_header(
        self,
        requirements: PaymentRequirements,
        pay_to_override: str,
        max_value: Optional[int] = None,
    ) -> str:
        account = self.service._get_client_account()

        required_value = int(requirements.max_amount_required)
        if max_value is not None and required_value > max_value:
            raise X402PaymentError(
                f"Payment requirement exceeds allowed maximum: required {required_value}, max_value {max_value}."
            )

        pay_to = pay_to_override or requirements.pay_to
        if not pay_to:
            raise X402PaymentError("Paywalled resource did not specify a pay_to address.")

        requirements_copy = PaymentRequirements.model_validate(requirements.model_dump())
        requirements_copy.pay_to = pay_to

        unsigned_header = prepare_payment_header(account.address, x402_VERSION, requirements_copy)
        nonce_bytes = unsigned_header["payload"]["authorization"]["nonce"]
        unsigned_header["payload"]["authorization"]["nonce"] = "0x" + nonce_bytes.hex()

        signature = self._sign_eip712_authorization(
            account,
            requirements_copy,
            unsigned_header["payload"]["authorization"],
        )
        unsigned_header["payload"]["signature"] = signature

        return encode_payment(unsigned_header)

    def _sign_eip712_authorization(
        self,
        account: "LocalAccount",
        requirements: PaymentRequirements,
        authorization: Dict[str, Any],
    ) -> str:
        extra = requirements.extra or {}
        domain: Dict[str, Any] = {
            "name": extra.get("name") or self.service.settings.asset_name,
            "version": extra.get("version") or "2",
            "chainId": int(get_chain_id(requirements.network)),
            "verifyingContract": requirements.asset,
        }
        if extra.get("salt"):
            domain["salt"] = extra["salt"]

        message_types = {
            "TransferWithAuthorization": [
                {"name": "from", "type": "address"},
                {"name": "to", "type": "address"},
                {"name": "value", "type": "uint256"},
                {"name": "validAfter", "type": "uint256"},
                {"name": "validBefore", "type": "uint256"},
                {"name": "nonce", "type": "bytes32"},
            ]
        }

        message_data = {
            "from": authorization["from"],
            "to": authorization["to"],
            "value": int(authorization["value"]),
            "validAfter": int(authorization["validAfter"]),
            "validBefore": int(authorization["validBefore"]),
            "nonce": bytes.fromhex(authorization["nonce"][2:]),
        }

        signable = encode_typed_data(domain, message_types, message_data)
        signature = account.sign_message(signable).signature.hex()
        print(
            "[x402-debug] signing typed data",
            {
                "domain": domain,
                "message": message_data,
                "signature": signature,
            },
        )
        return signature if signature.startswith("0x") else f"0x{signature}"

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
