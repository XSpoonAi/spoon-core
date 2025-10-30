import json
import os
from typing import Any

import pytest
from fastapi import FastAPI
from httpx import AsyncClient, ASGITransport

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
from spoon_ai.payments.server import create_paywalled_router

from x402.encoding import safe_base64_encode
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
    monkeypatch.setenv("X402_AGENT_PRIVATE_KEY", key)
    monkeypatch.setenv("X402_RECEIVER_ADDRESS", "0x1234567890abcdef1234567890abcdef12345678")
    return key


def test_settings_loads_from_env(monkeypatch):
    monkeypatch.setenv("X402_FACILITATOR_URL", "https://www.x402.org/facilitator")
    monkeypatch.setenv("X402_DEFAULT_AMOUNT_USDC", "0.01")
    monkeypatch.setenv("X402_DEFAULT_SCHEME", "exact")
    monkeypatch.setenv("X402_DEFAULT_NETWORK", "base-sepolia")
    monkeypatch.setenv("X402_DEFAULT_ASSET", "0xa063B8d5ada3bE64A24Df594F96aB75F0fb78160")
    monkeypatch.setenv("X402_RECEIVER_ADDRESS", "0x1234567890abcdef1234567890abcdef12345678")
    settings = X402Settings.load()
    assert settings.facilitator_url == "https://www.x402.org/facilitator"
    assert settings.resource.startswith("https://")
    assert settings.default_network == "base-sepolia"
    assert settings.amount_in_atomic_units == str(int(0.01 * 10**settings.asset_decimals))
    assert settings.pay_to == "0x1234567890abcdef1234567890abcdef12345678"


def test_build_payment_requirements_amount_conversion():
    service = X402PaymentService(facilitator=StubFacilitator())
    request = X402PaymentRequest(
        amount_usdc=0.01,
        resource="https://www.x402.org/protected",
        description="x402 demo resource",
    )
    requirements = service.build_payment_requirements(request)
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
    header = service.build_payment_header(requirements)

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
