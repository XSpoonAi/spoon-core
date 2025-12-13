from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from typing import Iterable, List, Optional

import os
import requests


class EmbeddingClient(ABC):
    @abstractmethod
    def embed(self, texts: Iterable[str]) -> List[List[float]]:
        raise NotImplementedError


class OpenAIEmbeddingClient(EmbeddingClient):
    def __init__(self, api_key: str, model: str = "text-embedding-3-small", base_url: Optional[str] = None):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url or "https://api.openai.com/v1"

    def embed(self, texts: Iterable[str]) -> List[List[float]]:
        url = f"{self.base_url.rstrip('/')}/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        data = {"input": list(texts), "model": self.model}
        resp = requests.post(url, headers=headers, data=json.dumps(data), timeout=60)
        resp.raise_for_status()
        payload = resp.json()
        return [d["embedding"] for d in payload.get("data", [])]


class AnyRouteEmbeddingClient(EmbeddingClient):
    def __init__(self, api_key: str, base_url: str, model: Optional[str] = None):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model

    def embed(self, texts: Iterable[str]) -> List[List[float]]:
        # AnyRoute is OpenAI-compatible; use the same /embeddings route
        url = f"{self.base_url}/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        # If model is provided, pass it; else rely on server default
        payload = {"input": list(texts)}
        if self.model:
            payload["model"] = self.model
        resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=60)
        resp.raise_for_status()
        data = resp.json()
        return [d["embedding"] for d in data.get("data", [])]


class HashEmbeddingClient(EmbeddingClient):
    """Deterministic offline embedding via hashing.

    Produces fixed-length vectors in [0,1] normalized range. Not semantically meaningful
    but stable for tests and offline demos.
    """

    def __init__(self, dim: int = 256):
        self.dim = dim

    def _hash_to_vec(self, text: str) -> List[float]:
        h = hashlib.sha256(text.encode("utf-8")).digest()
        # Expand to required dim by repeated hashing
        vals: List[float] = []
        seed = h
        while len(vals) < self.dim:
            for b in seed:
                vals.append(b / 255.0)
                if len(vals) >= self.dim:
                    break
            seed = hashlib.sha256(seed).digest()
        # L2 normalize
        norm = sum(v * v for v in vals) ** 0.5 or 1.0
        return [v / norm for v in vals]

    def embed(self, texts: Iterable[str]) -> List[List[float]]:
        return [self._hash_to_vec(t) for t in texts]


def get_embedding_client(
    provider: str,
    *,
    openai_api_key: Optional[str] = None,
    openai_model: str = "text-embedding-3-small",
    anyroute_api_key: Optional[str] = None,
    anyroute_base_url: Optional[str] = None,
    anyroute_model: Optional[str] = None,
) -> EmbeddingClient:
    provider = (provider or "").lower()
    if provider == "openai":
        key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("OPENAI_API_KEY not configured for OpenAI embeddings")
        return OpenAIEmbeddingClient(api_key=key, model=openai_model)
    if provider == "anyroute":
        key = anyroute_api_key or os.getenv("ANYROUTE_API_KEY")
        base = anyroute_base_url or os.getenv("ANYROUTE_BASE_URL")
        model = anyroute_model or os.getenv("ANYROUTE_MODEL")
        if not key or not base:
            raise ValueError("ANYROUTE_API_KEY and ANYROUTE_BASE_URL required for AnyRoute embeddings")
        return AnyRouteEmbeddingClient(api_key=key, base_url=base, model=model)
    # default deterministic
    return HashEmbeddingClient()

