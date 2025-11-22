import json
import os
from typing import Any, Dict

import pytest
from fastapi import FastAPI
from httpx import AsyncClient, ASGITransport, Response, Request

from eth_account import Account

from spoon_ai.payments import (
    X402PaymentOutcome,
    X402PaymentRequest,
    X402PaymentService,
    X402SettleResult,
    X402Settings,
    X402PaymentError,
    X402VerifyResult,
)
from spoon_ai.payments.config import X402ClientConfig
from spoon_ai.payments.server import create_paywalled_router
from spoon_ai.tools import x402_payment

from x402.encoding import safe_base64_decode, safe_base64_encode
from x402.chains import get_chain_id
from x402.types import (
    ListDiscoveryResourcesResponse,
    SettleResponse,
    VerifyResponse,
)


class StubFacilitator:
    def __init__(self, discovery_response: ListDiscoveryResourcesResponse | None = None):
        self.discovery_response = discovery_response

    async def verify(self, payment, requirements) -> VerifyResponse:
        return VerifyResponse(isValid=True, invalidReason=None, payer="0xabc123")

    async def settle(self, payment, requirements) -> SettleResponse:
        return SettleResponse(
            success=True,
            transaction="0xsettled",
            network=requirements.network,
            payer="0xabc123",
        )

    async def list_resources(self, request=None) -> ListDiscoveryResourcesResponse:
        if self.discovery_response is not None:
            return self.discovery_response
        return ListDiscoveryResourcesResponse.model_validate(
            {
                "x402Version": 1,
                "items": [],
                "pagination": {"limit": 0, "offset": 0, "total": 0},
            }
        )


@pytest.fixture(autouse=True)
def x402_private_key(monkeypatch):
    key = Account.create().key.hex()
    if not key.startswith("0x"):
        key = "0x" + key
    monkeypatch.delenv("X402_AGENT_PRIVATE_KEY", raising=False)
    monkeypatch.setenv("PRIVATE_KEY", key)
    monkeypatch.setenv("X402_RECEIVER_ADDRESS", "0x1234567890abcdef1234567890abcdef12345678")
    return key


def test_client_config_turnkey_auto(monkeypatch):
    monkeypatch.delenv("PRIVATE_KEY", raising=False)
    monkeypatch.delenv("X402_AGENT_PRIVATE_KEY", raising=False)
    monkeypatch.setenv("TURNKEY_SIGN_WITH", "wallet:auto")
    monkeypatch.setenv("TURNKEY_ADDRESS", "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
    config = X402ClientConfig.from_raw({})
    assert config.private_key is None
    assert config.use_turnkey is True
    assert config.turnkey_sign_with == "wallet:auto"
    assert config.turnkey_address == "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"


def test_settings_loads_from_env(monkeypatch):
    monkeypatch.setenv("X402_FACILITATOR_URL", "https://www.x402.org/facilitator")
    monkeypatch.setenv("X402_DEFAULT_AMOUNT_USDC", "0.01")
    monkeypatch.setenv("X402_DEFAULT_SCHEME", "exact")
    monkeypatch.setenv("X402_DEFAULT_NETWORK", "base-sepolia")
    monkeypatch.setenv("X402_DEFAULT_ASSET", "0xa063B8d5ada3bE64A24Df594F96aB75F0fb78160")
    monkeypatch.setenv("X402_RECEIVER_ADDRESS", "0x1234567890abcdef1234567890abcdef12345678")
    monkeypatch.setenv("TURNKEY_SIGN_WITH", "wallet:prefer")
    monkeypatch.setenv("TURNKEY_ADDRESS", "0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb")
    settings = X402Settings.load()
    assert settings.facilitator_url == "https://www.x402.org/facilitator"
    assert settings.resource.startswith("https://")
    assert settings.default_network == "base-sepolia"
    assert settings.amount_in_atomic_units == str(int(0.01 * 10**settings.asset_decimals))
    assert settings.pay_to == "0x1234567890abcdef1234567890abcdef12345678"
    assert settings.client.private_key == os.getenv("PRIVATE_KEY")
    assert settings.client.use_turnkey is False
    assert settings.client.private_key == os.getenv("PRIVATE_KEY")
    assert settings.client.use_turnkey is False


def test_build_payment_requirements_amount_conversion():
    service = X402PaymentService(facilitator=StubFacilitator())
    request = X402PaymentRequest(
        amount_usdc=0.01,
        resource="https://www.x402.org/protected",
        description="x402 demo resource",
    )
    requirements = service.build_payment_requirements(request)
    print("--------------------------------")
    print(requirements)
    print("--------------------------------")
    assert requirements.resource == "https://www.x402.org/protected"
    assert requirements.max_amount_required == str(int(0.01 * 10**service.settings.asset_decimals))
    assert requirements.pay_to == service.settings.pay_to
    assert requirements.description == "x402 demo resource"


def test_build_payment_requirements_includes_metadata():
    service = X402PaymentService(facilitator=StubFacilitator())
    request = X402PaymentRequest(
        amount_usdc=0.02,
        memo="Demo payment via x402",
        currency="USDC",
        metadata={"service": "analysis", "category": "shopping"},
        extra={"priority": "high"},
        output_schema={"type": "object", "properties": {"result": {"type": "string"}}},
    )
    requirements = service.build_payment_requirements(request)
    print("--------------------------------")
    print(requirements)
    print("--------------------------------")
    assert requirements.output_schema == {"type": "object", "properties": {"result": {"type": "string"}}}
    assert requirements.extra["currency"] == "USDC"
    assert requirements.extra["memo"] == "Demo payment via x402"
    assert requirements.extra["metadata"]["service"] == "analysis"
    assert requirements.extra["priority"] == "high"


def test_build_payment_header_respects_max_value():
    service = X402PaymentService(facilitator=StubFacilitator())
    requirements = service.build_payment_requirements(X402PaymentRequest(amount_usdc=0.05))
    with pytest.raises(X402PaymentError):
        service.build_payment_header(requirements, max_value=10)


@pytest.mark.asyncio
async def test_verify_and_settle(monkeypatch):
    service = X402PaymentService(facilitator=StubFacilitator())
    requirements = service.build_payment_requirements(X402PaymentRequest(amount_usdc=0.01))
    print("--------------------------------")
    print(requirements)
    print("--------------------------------")
    header = service.build_payment_header(requirements)
    print("--------------------------------")
    print(header)
    print("--------------------------------")

    outcome = await service.verify_and_settle(header, requirements)
    assert outcome.verify.is_valid
    assert outcome.settle is not None
    assert outcome.settle.success


@pytest.mark.asyncio
async def test_discover_resources_returns_stubbed_items():
    stub = StubFacilitator()
    service = X402PaymentService(facilitator=stub)
    requirements = service.build_payment_requirements()
    stub.discovery_response = ListDiscoveryResourcesResponse.model_validate(
        {
            "x402Version": 1,
            "items": [
                {
                    "resource": service.settings.resource,
                    "type": "http",
                    "x402Version": 1,
                    "accepts": [requirements.model_dump(by_alias=True)],
                    "lastUpdated": "2025-10-01T00:00:00Z",
                    "metadata": {"category": "demo"},
                }
            ],
            "pagination": {"limit": 1, "offset": 0, "total": 1},
        }
    )
    response = await service.discover_resources()
    print("--------------------------------")
    print(response)
    print("--------------------------------")
    assert response.items
    assert response.items[0].metadata["category"] == "demo"
    assert response.items[0].accepts[0].resource == service.settings.resource


def test_decode_payment_response_parses_header():
    service = X402PaymentService(facilitator=StubFacilitator())
    payload = {"success": True, "transaction": "0xdeadbeef", "network": "base-sepolia", "payer": "0xabc123"}
    header = safe_base64_encode(json.dumps(payload))
    receipt = service.decode_payment_response(header)
    assert receipt.success is True
    assert receipt.transaction == "0xdeadbeef"
    assert receipt.network == "base-sepolia"
    assert receipt.raw["payer"] == "0xabc123"


@pytest.mark.asyncio
async def test_paywall_router_returns_402_json(monkeypatch):
    service = X402PaymentService(facilitator=StubFacilitator())
    app = FastAPI()
    app.include_router(create_paywalled_router(service=service, agent_factory=lambda name: None))  # type: ignore[arg-type]

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/x402/invoke/demo", json={"prompt": "hi"}, headers={"accept": "application/json"})

    assert resp.status_code == 402
    data = resp.json()
    assert "accepts" in data
    assert data["accepts"][0]["scheme"] == service.settings.default_scheme


@pytest.mark.asyncio
async def test_paywall_router_processes_valid_payment(monkeypatch):
    service = X402PaymentService(facilitator=StubFacilitator())

    async def agent_factory(name: str):
        class StubAgent:
            async def initialize(self):
                return None

            async def run(self, prompt: str):
                return f"echo:{prompt}"

        agent = StubAgent()
        await agent.initialize()
        return agent

    async def fake_verify_and_settle(header_value: str, requirements=None, settle: bool = True):
        return X402PaymentOutcome(
            verify=X402VerifyResult(is_valid=True, payer="0xabc123"),
            settle=X402SettleResult(success=True, transaction="0xdeadbeef", network="base", payer="0xabc123"),
        )

    monkeypatch.setattr(service, "verify_and_settle", fake_verify_and_settle)  # type: ignore[assignment]

    app = FastAPI()
    app.include_router(create_paywalled_router(service=service, agent_factory=agent_factory))  # type: ignore[arg-type]

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/x402/invoke/demo",
            json={"prompt": "hello"},
            headers={"X-PAYMENT": "ZmFrZS1wYXltZW50"},
        )

    assert resp.status_code == 200
    assert resp.json()["result"] == "echo:hello"
    assert "X-PAYMENT-RESPONSE" in resp.headers


@pytest.mark.asyncio
async def test_paywalled_request_override_preserves_paywall_asset(monkeypatch):
    service = X402PaymentService(facilitator=StubFacilitator())
    tool = x402_payment.X402PaywalledRequestTool(service=service)
    url = "https://www.x402.org/protected"
    paywall_body = {
        "x402Version": 1,
        "error": "X-PAYMENT header is required",
        "accepts": [
            {
                "scheme": "exact",
                "network": "base-sepolia",
                "maxAmountRequired": "10000",
                "resource": url,
                "description": "Protected content",
                "mimeType": "application/json",
                "payTo": "0x209693Bc6afc0C5328bA36FaF03C514EF312287C",
                "maxTimeoutSeconds": 300,
                "asset": "0x036CbD53842c5426634e7929541eC2318f3dCF7e",
                "extra": {"name": "USDC", "version": "2"},
            }
        ],
    }

    responses = [
        Response(
            402,
            request=Request("GET", url),
            json=paywall_body,
            headers={"content-type": "application/json"},
        ),
        Response(
            200,
            request=Request("GET", url),
            json={"ok": True},
            headers={"content-type": "application/json"},
        ),
    ]

    captured: dict[str, Any] = {}
    original_sign = x402_payment.X402PaywalledRequestTool._sign_eip712_authorization

    def capture_signature(self, account, requirements, authorization):
        captured["asset"] = requirements.asset
        captured["extra"] = requirements.extra
        return original_sign(self, account, requirements, authorization)

    monkeypatch.setattr(
        x402_payment.X402PaywalledRequestTool,
        "_sign_eip712_authorization",
        capture_signature,
    )

    class StubAsyncClient:
        def __init__(self, responses_list, **kwargs):
            self._responses = list(responses_list)
            self.requests = []

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def request(self, method, request_url, headers=None, json=None, content=None):
            self.requests.append({"method": method, "url": request_url, "headers": headers})
            if not self._responses:
                raise AssertionError("No more responses configured for StubAsyncClient")
            return self._responses.pop(0)

    monkeypatch.setattr(
        x402_payment.httpx,
        "AsyncClient",
        lambda *args, **kwargs: StubAsyncClient(responses, **kwargs),
    )

    result = await tool.execute(url=url, amount_usdc=0.01)

    assert result.error is None
    assert captured["asset"] == "0x036CbD53842c5426634e7929541eC2318f3dCF7e"
    assert captured["extra"]["name"] == "USDC"
    assert captured["extra"]["version"] == "2"


@pytest.mark.asyncio
async def test_build_payment_header_with_turnkey(monkeypatch):
    settings = X402Settings(
        pay_to="0xfeedfeedfeedfeedfeedfeedfeedfeedfeedfeed",
        client=X402ClientConfig(
            use_turnkey=True,
            turnkey_sign_with="wallet:abc123",
            turnkey_address="0xfeedfeedfeedfeedfeedfeedfeedfeedfeedfeed",
            private_key=None,
        ),
    )
    service = X402PaymentService(settings=settings, facilitator=StubFacilitator())

    stub_signature = "0x" + "ab" * 65
    captured: dict[str, Any] = {}

    class StubTurnkey:
        def sign_typed_data(self, sign_with: str, typed_data: Dict[str, Any]):
            captured["sign_with"] = sign_with
            captured["typed_data"] = typed_data
            return {
                "activity": {
                    "result": {
                        "signRawPayloadResult": {
                            "signatures": [
                                {"signature": stub_signature}
                            ]
                        }
                    }
                }
            }

    stub_client = StubTurnkey()
    monkeypatch.setattr(service, "_turnkey_client", stub_client)
    monkeypatch.setattr(service, "_get_turnkey_client", lambda: stub_client)

    requirements = service.build_payment_requirements(X402PaymentRequest(amount_usdc=0.01))
    header_b64 = service.build_payment_header(requirements)
    payload = json.loads(safe_base64_decode(header_b64))

    assert payload["payload"]["signature"] == stub_signature
    assert payload["payload"]["authorization"]["from"] == settings.client.turnkey_address
    assert payload["payload"]["authorization"]["nonce"].startswith("0x")
    assert captured["sign_with"] == "wallet:abc123"
    assert captured["typed_data"]["domain"]["chainId"] == int(get_chain_id(requirements.network))
    assert captured["typed_data"]["message"]["nonce"].startswith("0x")
