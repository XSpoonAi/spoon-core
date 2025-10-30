from __future__ import annotations

from typing import Awaitable, Callable, Dict, Optional

from x402.facilitator import FacilitatorClient as SDKFacilitatorClient
from x402.types import (
    ListDiscoveryResourcesRequest,
    ListDiscoveryResourcesResponse,
    PaymentPayload,
    PaymentRequirements,
    SettleResponse,
    VerifyResponse,
)


CreateHeadersCallable = Callable[[], Awaitable[Dict[str, Dict[str, str]]]]


class X402FacilitatorClient:
    """Thin wrapper over the upstream facilitator client with async header hooks."""

    def __init__(
        self,
        base_url: str,
        create_headers: Optional[CreateHeadersCallable] = None,
    ) -> None:
        config: Dict[str, object] = {"url": base_url}
        if create_headers:
            config["create_headers"] = create_headers
        self._client = SDKFacilitatorClient(config)  # type: ignore[arg-type]

    async def verify(self, payment: PaymentPayload, requirements: PaymentRequirements) -> VerifyResponse:
        return await self._client.verify(payment, requirements)

    async def settle(self, payment: PaymentPayload, requirements: PaymentRequirements) -> SettleResponse:
        return await self._client.settle(payment, requirements)

    async def list_resources(
        self, request: Optional[ListDiscoveryResourcesRequest] = None
    ) -> ListDiscoveryResourcesResponse:
        return await self._client.list(request)
