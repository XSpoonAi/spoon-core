from __future__ import annotations

import json
from typing import Any, Awaitable, Callable, Optional

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response

from x402.encoding import safe_base64_encode

from spoon_ai.agents.spoon_react import SpoonReactAI
from spoon_ai.payments import X402PaymentOutcome, X402PaymentRequest, X402PaymentService


AgentFactory = Callable[[str], Awaitable[SpoonReactAI]]


async def _default_agent_factory(agent_name: str) -> SpoonReactAI:
    agent = SpoonReactAI(name=agent_name)
    await agent.initialize()
    return agent


def create_paywalled_router(
    service: Optional[X402PaymentService] = None,
    agent_factory: AgentFactory = _default_agent_factory,
    payment_message: str = "Payment required to invoke this agent.",
) -> APIRouter:
    """
    Build a FastAPI router that protects agent invocations behind an x402 paywall.

    Args:
        service: Optional pre-configured payment service.
        agent_factory: Coroutine that returns an initialized agent given its name.
        payment_message: Message displayed when payment is required.

    Returns:
        APIRouter: Router with `/invoke/{agent_name}` endpoint ready to mount.
    """
    payment_service = service or X402PaymentService()
    router = APIRouter(prefix="/x402", tags=["x402"])

    @router.get("/requirements")
    async def fetch_requirements():
        requirements = payment_service.build_payment_requirements()
        return requirements.model_dump(by_alias=True, exclude_none=True)

    @router.post("/invoke/{agent_name}")
    async def invoke_agent(agent_name: str, payload: dict[str, Any], request: Request):
        header_value = request.headers.get("X-PAYMENT")

        if not header_value:
            html_response = payment_service.render_paywall_html(payment_message, headers=dict(request.headers))
            if request.headers.get("accept", "").lower().find("text/html") != -1:
                return Response(content=html_response, status_code=402, media_type="text/html")
            else:
                return JSONResponse(
                    payment_service.build_payment_required_response(payment_message).model_dump(by_alias=True),
                    status_code=402,
                )

        outcome: X402PaymentOutcome = await payment_service.verify_and_settle(header_value)
        if not outcome.verify.is_valid:
            detail = {
                "error": outcome.verify.invalid_reason or "Invalid payment payload",
                "payer": outcome.verify.payer,
            }
            return JSONResponse(detail, status_code=402)

        agent = await agent_factory(agent_name)
        prompt = payload.get("prompt") or payload.get("input") or payload.get("message")
        if not prompt:
            return JSONResponse({"error": "Missing prompt in payload"}, status_code=400)

        result = await agent.run(prompt)

        response_payload = {
            "result": result,
            "payer": outcome.verify.payer,
        }

        settlement = outcome.settle.model_dump(exclude_none=True) if outcome.settle else {"success": True}
        encoded_settlement = safe_base64_encode(json.dumps(settlement))

        response = JSONResponse(response_payload, status_code=200)
        response.headers["X-PAYMENT-RESPONSE"] = encoded_settlement
        return response

    return router
