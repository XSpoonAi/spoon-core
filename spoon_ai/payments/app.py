from __future__ import annotations

import uvicorn
from fastapi import FastAPI

from spoon_ai.payments.server import create_paywalled_router

app = FastAPI(
    title="SpoonOS x402 Agent Gateway",
    description="Expose SpoonOS agents behind an x402 payment paywall.",
    version="0.1.0",
)

app.include_router(create_paywalled_router())


@app.get("/health", tags=["health"])
async def health() -> dict[str, str]:
    return {"status": "ok"}


def run(host: str = "0.0.0.0", port: int = 8090) -> None:
    uvicorn.run("spoon_ai.payments.app:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    run()
