from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from typing import Dict, Iterable, List, Optional

import os
import requests


class EmbeddingClient(ABC):
    @abstractmethod
    def embed(self, texts: Iterable[str]) -> List[List[float]]:
        raise NotImplementedError


class OpenAIEmbeddingClient(EmbeddingClient):
    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        base_url: Optional[str] = None,
        custom_headers: Optional[Dict[str, str]] = None,
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url or "https://api.openai.com/v1"
        self.custom_headers = custom_headers or {}

    def embed(self, texts: Iterable[str]) -> List[List[float]]:
        url = f"{self.base_url.rstrip('/')}/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.custom_headers:
            headers.update(self.custom_headers)
        data = {"input": list(texts), "model": self.model}
        resp = requests.post(url, headers=headers, data=json.dumps(data), timeout=60)
        resp.raise_for_status()
        payload = resp.json()
        return [d["embedding"] for d in payload.get("data", [])]


class OpenAICompatibleEmbeddingClient(EmbeddingClient):
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: Optional[str] = None,
        custom_headers: Optional[Dict[str, str]] = None,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.custom_headers = custom_headers or {}

    def embed(self, texts: Iterable[str]) -> List[List[float]]:
        # OpenAI-compatible; use the same /embeddings route
        url = f"{self.base_url}/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.custom_headers:
            headers.update(self.custom_headers)
        # If model is provided, pass it; else rely on server default
        payload = {"input": list(texts)}
        if self.model:
            payload["model"] = self.model
        resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=60)
        resp.raise_for_status()
        data = resp.json()
        return [d["embedding"] for d in data.get("data", [])]


class GeminiEmbeddingClient(EmbeddingClient):
    def __init__(
        self,
        api_key: str,
        model: str,
    ):
        try:
            from google import genai  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "Gemini embeddings require the google-genai package to be installed."
            ) from exc

        self.client = genai.Client(api_key=api_key)
        self.model = model

    def embed(self, texts: Iterable[str]) -> List[List[float]]:
        contents = list(texts)
        response = self.client.models.embed_content(
            model=self.model,
            contents=contents,
        )
        embeddings = getattr(response, "embeddings", None) or []
        return [e.values for e in embeddings]


class OllamaEmbeddingClient(EmbeddingClient):
    def __init__(
        self,
        *,
        base_url: str = "http://localhost:11434",
        model: str,
        timeout: int = 60,
        batch_size: int = 32,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.batch_size = max(1, int(batch_size))

    def embed(self, texts: Iterable[str]) -> List[List[float]]:
        inputs = list(texts)
        if not inputs:
            return []

        # Preferred (batch) endpoint
        embed_url = f"{self.base_url}/api/embed"
        if self.batch_size > 1 and len(inputs) > self.batch_size:
            out: List[List[float]] = []
            for i in range(0, len(inputs), self.batch_size):
                batch = inputs[i : i + self.batch_size]
                resp = requests.post(
                    embed_url,
                    json={"model": self.model, "input": batch},
                    timeout=self.timeout,
                )
                if resp.status_code == 404:
                    out = []
                    break
                resp.raise_for_status()
                payload = resp.json()
                embeddings = payload.get("embeddings")
                if not isinstance(embeddings, list) or len(embeddings) != len(batch):
                    raise RuntimeError("Ollama /api/embed returned unexpected embeddings payload")
                out.extend(embeddings)
            if out:
                return out

        resp = requests.post(
            embed_url,
            json={"model": self.model, "input": inputs},
            timeout=self.timeout,
        )
        if resp.status_code != 404:
            resp.raise_for_status()
            payload = resp.json()
            embeddings = payload.get("embeddings")
            if isinstance(embeddings, list):
                return embeddings

        # Fallback (legacy) endpoint: one request per input
        legacy_url = f"{self.base_url}/api/embeddings"
        out: List[List[float]] = []
        for text in inputs:
            r = requests.post(
                legacy_url,
                json={"model": self.model, "prompt": text},
                timeout=self.timeout,
            )
            r.raise_for_status()
            data = r.json()
            out.append(data.get("embedding") or [])
        return out


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
    provider: Optional[str],
    *,
    openai_api_key: Optional[str] = None,
    model: Optional[str] = None,
    openai_model: Optional[str] = None,  # Deprecated: use 'model' instead. Kept for backward compatibility.
) -> EmbeddingClient:
    """Create an embedding client.

    Args:
        provider: Embedding provider name ("openai", "gemini", "ollama", etc.) or None/"auto" for auto-detection.
        openai_api_key: Optional OpenAI API key (for OpenAI provider only).
        model: Model name/ID for the embedding provider. This is a generic parameter that works for all providers.
        openai_model: Deprecated alias for 'model'. Use 'model' instead.

    Provider selection rules:
    - provider is None/"auto": pick the first configured embeddings provider using a dedicated
      priority order (OpenAI > OpenRouter > Gemini).
    - provider is "openai" / "openrouter" / "gemini" / "ollama": force that provider (uses core env config when applicable).
    - provider is "openai_compatible": use OpenAI-compatible embeddings via RAG_EMBEDDINGS_* env vars.
    - otherwise: deterministic hash embeddings (offline).
    """
    # Handle backward compatibility: if openai_model is provided but model is not, use openai_model
    if model is None and openai_model is not None:
        model = openai_model
    # Note: We don't set a default model here. Each provider handles None appropriately:
    # - OpenAI: uses OpenAIEmbeddingClient's default "text-embedding-3-small"
    # - Gemini: uses "models/embedding-001" (see line 339)
    # - OpenRouter: uses _derive_openrouter_embedding_model which returns "openai/text-embedding-3-small"
    # - Ollama: raises error (requires explicit model)

    def _normalize(value: Optional[str]) -> str:
        return (value or "").strip().lower()

    def _derive_openrouter_embedding_model(base_model: str) -> str:
        base_model = (base_model or "").strip()
        if not base_model:
            return "openai/text-embedding-3-small"
        # OpenRouter uses namespaced model IDs (e.g. openai/text-embedding-3-small)
        if "/" in base_model:
            return base_model
        return f"openai/{base_model}"

    provider_norm = _normalize(provider)

    if provider_norm in ("", "auto"):
        # Auto: pick the first configured embeddings provider using a dedicated priority
        # order (OpenAI > OpenRouter > Gemini). This is intentionally independent from
        # the chat LLM provider and its fallback chain.
        try:
            from spoon_ai.llm.config import ConfigurationManager

            cm = ConfigurationManager()
            available = set(cm.list_configured_providers())
            for p in ("openai", "openrouter", "gemini"):
                if p in available:
                    provider_norm = p
                    break

            # Finally, allow a custom OpenAI-compatible embeddings endpoint if explicitly configured.
            # This is checked after Gemini to match the desired priority.
            if provider_norm in ("", "auto") and os.getenv("RAG_EMBEDDINGS_BASE_URL"):
                try:
                    cm.load_provider_config("rag_embeddings")
                except Exception:
                    pass
                else:
                    provider_norm = "openai_compatible"
        except Exception:
            # If core config is unavailable/misconfigured, fall back to offline embeddings.
            provider_norm = "hash"

    supported = {"", "auto", "hash", "openai", "openrouter", "gemini", "openai_compatible", "ollama"}
    if provider_norm not in supported:
        raise ValueError(
            f"Unsupported embeddings provider '{provider_norm}'. "
            "Supported: auto, openai, openrouter, gemini, openai_compatible, ollama, hash."
        )

    if provider_norm == "hash":
        return HashEmbeddingClient()

    if provider_norm == "openai":
        # Use core provider config to honor env priority and base_url overrides.
        try:
            from spoon_ai.llm.config import ConfigurationManager

            cm = ConfigurationManager()
            cfg = cm.load_provider_config("openai")
            key = openai_api_key or cfg.api_key
            base_url = cfg.base_url or "https://api.openai.com/v1"
        except Exception:
            key = openai_api_key or os.getenv("OPENAI_API_KEY")
            base_url = os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"

        if not key:
            raise ValueError("OPENAI_API_KEY not configured for OpenAI embeddings")

        # Allow passing OpenRouter-style namespaced IDs (e.g. openai/text-embedding-3-small)
        # while keeping OpenAI's expected model id (text-embedding-3-small).
        model_name = model.split("/", 1)[-1] if "/" in (model or "") else model
        return OpenAIEmbeddingClient(api_key=key, model=model_name, base_url=base_url)

    if provider_norm == "openrouter":
        # OpenRouter is OpenAI-compatible for embeddings.
        from spoon_ai.llm.config import ConfigurationManager

        cm = ConfigurationManager()
        cfg = cm.load_provider_config("openrouter")

        # If OPENROUTER_MODEL is an embedding model, use it; otherwise default to OpenAI embeddings via OpenRouter.
        if cfg.model and "embedding" in cfg.model.lower():
            model_name = cfg.model
        else:
            model_name = _derive_openrouter_embedding_model(model)

        if not cfg.api_key:
            raise ValueError("OPENROUTER_API_KEY not configured for OpenRouter embeddings")

        return OpenAICompatibleEmbeddingClient(
            api_key=cfg.api_key,
            base_url=cfg.base_url or "https://openrouter.ai/api/v1",
            model=model_name,
            custom_headers=cfg.custom_headers,
        )

    if provider_norm == "gemini":
        # Gemini embeddings are handled via google-genai SDK. The embedding model must be
        # provided via RAG_EMBEDDINGS_MODEL (passed as model parameter here).
        from spoon_ai.llm.config import ConfigurationManager

        cm = ConfigurationManager()
        cfg = cm.load_provider_config("gemini")
        if not cfg.api_key:
            raise ValueError("GEMINI_API_KEY not configured for Gemini embeddings")

        model_name = (model or "").strip()
        # If model is empty or is OpenAI's default model name, use Gemini's default
        # if not model_name or model_name == "text-embedding-3-small":
        if not model_name:
            # Use Gemini's embedding model. embedding-001 is the standard Gemini embedding model
            model_name = "models/embedding-001"
        
        # Ensure model has 'models/' prefix if not already present
        if not model_name.startswith("models/"):
            model_name = f"models/{model_name}"

        return GeminiEmbeddingClient(api_key=cfg.api_key, model=model_name)

    if provider_norm == "openai_compatible":
        # Custom OpenAI-compatible embeddings endpoint configured via:
        # - RAG_EMBEDDINGS_API_KEY
        # - RAG_EMBEDDINGS_BASE_URL
        # - RAG_EMBEDDINGS_MODEL (optional; defaults to model parameter)
        from spoon_ai.llm.config import ConfigurationManager

        cm = ConfigurationManager()
        try:
            cfg = cm.load_provider_config("rag_embeddings")
        except Exception as exc:
            raise ValueError(
                "RAG_EMBEDDINGS_API_KEY and RAG_EMBEDDINGS_BASE_URL must be set when "
                "RAG_EMBEDDINGS_PROVIDER=openai_compatible."
            ) from exc

        if not cfg.base_url:
            raise ValueError(
                "RAG_EMBEDDINGS_BASE_URL must be set when RAG_EMBEDDINGS_PROVIDER=openai_compatible."
            )

        model_name = cfg.model or model
        return OpenAICompatibleEmbeddingClient(
            api_key=cfg.api_key,
            base_url=cfg.base_url,
            model=model_name,
            custom_headers=cfg.custom_headers,
        )

    if provider_norm == "ollama":
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").strip() or "http://localhost:11434"
        model_name = (model or "").strip()
        if not model_name:
            raise ValueError("RAG_EMBEDDINGS_MODEL is required for Ollama embeddings")
        return OllamaEmbeddingClient(base_url=base_url, model=model_name)

    # Default deterministic offline embedding
    return HashEmbeddingClient()

