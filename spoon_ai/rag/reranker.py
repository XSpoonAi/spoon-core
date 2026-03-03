from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Iterable, Dict
import os
import requests
import json
import logging

logger = logging.getLogger(__name__)

class RerankClient(ABC):
    @abstractmethod
    def rerank(self, query: str, docs: List[str]) -> List[float]:
        """
        Rerank a list of documents based on a query.
        Returns a list of scores (higher is better) corresponding to each doc.
        """
        pass

class NoOpRerankClient(RerankClient):
    """Pass-through reranker (does nothing, returns 0.0 scores or keeps original order implicitly)."""
    def rerank(self, query: str, docs: List[str]) -> List[float]:
        return [0.0] * len(docs)

class LLMRerankClient(RerankClient):
    """
    Uses an LLM (via OpenAI-compatible API) to score documents.
    This is dependency-free (uses requests) and flexible.
    """
    def __init__(
        self, 
        api_key: str, 
        base_url: str, 
        model: str,
        timeout: int = 10
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        
    def _score_single(self, query: str, doc: str) -> float:
        # Prompt engineering for scoring
        prompt = (
            f"Query: {query}\n"
            f"Document: {doc[:4000]}...\n\n"  # Truncate to avoid context unexpected limits
            "Rate the relevance of the document to the query on a continuous scale from 0.0 (irrelevant) to 10.0 (exact match).\n"
            "Output ONLY the number, nothing else."
        )
        
        try:
            url = f"{self.base_url}/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            if "openrouter" in self.base_url:
                 # Optional headers for OpenRouter
                headers["HTTP-Referer"] = "https://spoon.ai"
                headers["X-Title"] = "Spoon AI Reranker"



            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "You are a helpful relevance ranking assistant. Output only a float score."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.0,
                "max_tokens": 10
            }
            
            resp = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"].strip()
            
            # Simple parsing
            try:
                score = float(content)
                return score
            except ValueError:
                # Fallback heuristic parsing if model explains itself
                import re
                match = re.search(r"(\d+(\.\d+)?)", content)
                if match:
                    return float(match.group(1))
                return 0.0
                
        except Exception as e:
            logger.warning(f"Rerank failed for doc: {e}")
            return 0.0

    def rerank(self, query: str, docs: List[str]) -> List[float]:
        # Optimization: In a real prod scenario, we might batch this or use a specific Rerank API endpoint.
        # For now, we do sequential or simple parallel calls.
        # Given "chunk + BM25" context, we assume we rerank only Top N (e.g. 5-10).
        # We process sequentially here for safety and simplicity, or we could use ThreadPoolExecutor.
        # Let's use simple sequential for now to avoid complexity issues with threads/requests.
        
        scores = []
        for doc in docs:
            scores.append(self._score_single(query, doc))
        return scores

def get_rerank_client(
    provider: Optional[str],
    *,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None
) -> RerankClient:
    """
    Factory to get a rerank client.
    """
    url = base_url
    if not provider or provider == "none":
        return NoOpRerankClient()
        
    if provider in ("openai", "openrouter", "openai_compatible", "deepseek"):
        # Resolve config with similar logic to EmbeddingClient
        # Ideally we'd reuse a centralized config manager, but to keep RAG standalone-ish:
        
        # Defaults
        if provider == "openai":
            key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("RERANK_API_KEY")
            url = url or "https://api.openai.com/v1"
        elif provider == "openrouter":
            key = api_key or os.getenv("OPENROUTER_API_KEY") or os.getenv("RERANK_API_KEY") or os.getenv("OPENAI_API_KEY")
            url = url or "https://openrouter.ai/api/v1"
        else:
             # Generice fallback
             key = api_key or os.getenv("RERANK_API_KEY") or os.getenv("OPENAI_API_KEY")

        final_model = model or os.getenv("RERANK_MODEL") or "gpt-4o-mini" # default to a cheap fast model
        
        if not key:
            logger.warning("No API key found for reranker, disabling.")
            return NoOpRerankClient()
            
        if not url:
             logger.warning("No Base URL found for reranker, disabling.")
             return NoOpRerankClient()
             
        return LLMRerankClient(api_key=key, base_url=url, model=final_model)

    return NoOpRerankClient()
