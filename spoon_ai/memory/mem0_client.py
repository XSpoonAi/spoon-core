import asyncio
import logging
import os
from typing import Any, Dict, List, Optional

from spoon_ai.schema import Message

try:
    from mem0 import MemoryClient  # type: ignore
except ImportError:
    MemoryClient = None  # type: ignore

logger = logging.getLogger(__name__)

class SpoonMem0:
    """Lightweight wrapper around Mem0's MemoryClient with safe defaults."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.api_key = self.config.get("api_key") or os.getenv("MEM0_API_KEY")
        self.user_id = self.config.get("user_id") or self.config.get("agent_id")
        self.agent_id = self.config.get("agent_id")
        self.collection = self.config.get("collection") or self.config.get("namespace")
        self.limit = self.config.get("limit")
        self.filters = self.config.get("filters") or {}
        self.metadata = self._build_metadata()
        if MemoryClient is None:
            logger.warning("mem0ai is not installed. Memory features will be disabled.")
            self.client = None
        else:
            self.client = self._initialize_client()

    def _build_metadata(self) -> Dict[str, Any]:
        metadata = dict(self.config.get("metadata") or {})
        if self.agent_id and "agent_id" not in metadata:
            metadata["agent_id"] = self.agent_id
        if self.user_id and "user_id" not in metadata:
            metadata["user_id"] = self.user_id
        return metadata

    def _initialize_client(self):
        if MemoryClient is None:
            logger.warning("mem0ai package not installed; long-term memory is disabled.")
            return None

        if not self.api_key:
            logger.warning("MEM0_API_KEY is not set; long-term memory is disabled.")
            return None

        try:
            client_kwargs = self.config.get("client_kwargs") or {}
            return MemoryClient(api_key=self.api_key, **client_kwargs)
        except Exception as exc:  # pragma: no cover - protective logging
            logger.error("Failed to initialize Mem0 client: %s", exc)
            return None

    def is_ready(self) -> bool:
        return self.client is not None

    def search_memory(self, query: str, user_id: Optional[str] = None, limit: Optional[int] = None) -> List[str]:
        if not query or not self.client:
            return []

        search_kwargs: Dict[str, Any] = {}
        active_user = user_id or self.user_id
        if active_user:
            search_kwargs["user_id"] = active_user
        if self.collection:
            search_kwargs["collection"] = self.collection

        base_filters = dict(self.filters) if self.filters else {}
        if active_user and "user_id" not in base_filters:
            base_filters["user_id"] = active_user
        if base_filters:
            search_kwargs["filters"] = base_filters

        if limit is not None:
            search_kwargs["limit"] = limit
        elif self.limit:
            search_kwargs["limit"] = self.limit

        try:
            results = self.client.search(query=query, **search_kwargs)
        except TypeError:
            # Retry with the minimal set of arguments Mem0 accepts.
            fallback_kwargs: Dict[str, Any] = {}
            if active_user:
                fallback_kwargs["user_id"] = active_user
            try:
                results = self.client.search(query=query, **fallback_kwargs)
            except Exception as exc:
                logger.warning("Mem0 search failed after fallback: %s", exc)
                return []
        except Exception as exc:
            logger.warning("Mem0 search failed: %s", exc)
            return []

        return self._extract_memories(results)

    async def asearch_memory(self, query: str, user_id: Optional[str] = None) -> List[str]:
        return await asyncio.to_thread(self.search_memory, query, user_id)

    def add_memory(self, messages: List[Any], user_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> None:
        if not messages or not self.client:
            return

        active_user = user_id or self.user_id
        formatted_messages = self._format_messages(messages)

        add_kwargs: Dict[str, Any] = {}
        if active_user:
            add_kwargs["user_id"] = active_user
        if self.collection:
            add_kwargs["collection"] = self.collection
        if self.metadata or metadata:
            combined_meta = dict(self.metadata or {})
            if metadata:
                combined_meta.update(metadata)
            add_kwargs["metadata"] = combined_meta
        # Default to synchronous writes so searches immediately see the memory
        add_kwargs["async_mode"] = self.config.get("async_mode", False)
        base_filters = dict(self.filters) if self.filters else {}
        if active_user and "user_id" not in base_filters:
            base_filters["user_id"] = active_user
        if base_filters:
            add_kwargs["filters"] = base_filters

        try:
            self.client.add(formatted_messages, **add_kwargs)
        except TypeError:
            combined_text = "\n".join(
                f"{m['role']}: {m['content']}"
                for m in formatted_messages
                if m.get("content")
            )
            try:
                self.client.add(combined_text, **add_kwargs)
            except Exception as exc:
                logger.warning("Mem0 add failed after fallback: %s", exc)
        except Exception as exc:
            logger.warning("Mem0 add failed: %s", exc)

    def add_text(self, data: str, user_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Convenience helper for adding a single text memory."""
        self.add_memory([{"role": "user", "content": data}], user_id=user_id, metadata=metadata)

    async def aadd_memory(self, messages: List[Any], user_id: Optional[str] = None) -> None:
        await asyncio.to_thread(self.add_memory, messages, user_id)

    def _format_messages(self, messages: List[Any]) -> List[Dict[str, str]]:
        formatted: List[Dict[str, str]] = []
        for message in messages:
            role: Optional[str] = None
            content: Optional[str] = None

            if isinstance(message, Message):
                role = str(message.role)
                content = message.content
            elif isinstance(message, dict):
                role = str(message.get("role"))
                content = message.get("content")

            if role and content:
                formatted.append({"role": role, "content": content})

        return formatted

    def _extract_memories(self, results: Any) -> List[str]:
        memories: List[str] = []

        # v2 API returns {"results": [...]}
        if isinstance(results, dict) and "results" in results:
            items = results.get("results") or []
        else:
            items = results

        if isinstance(items, list):
            for item in items:
                if isinstance(item, str):
                    memories.append(item)
                elif isinstance(item, dict):
                    text = (
                        item.get("memory")
                        or item.get("text")
                        or item.get("content")
                        or item.get("value")
                    )
                    if text:
                        memories.append(str(text))
        elif isinstance(items, dict):
            text = (
                items.get("memory")
                or items.get("text")
                or items.get("content")
                or items.get("value")
            )
            if text:
                memories.append(str(text))

        return memories

    def get_all_memory(self, user_id: Optional[str] = None, limit: Optional[int] = None) -> List[str]:
        """Retrieve all memories for a user (subject to backend limits)."""
        if not self.client:
            return []

        active_user = user_id or self.user_id
        params: Dict[str, Any] = {}
        if active_user:
            params["user_id"] = active_user
        if self.collection:
            params["collection"] = self.collection

        base_filters = dict(self.filters) if self.filters else {}
        if active_user and "user_id" not in base_filters:
            base_filters["user_id"] = active_user
        if base_filters:
            params["filters"] = base_filters

        if limit is not None:
            params["limit"] = limit
        elif self.limit:
            params["limit"] = self.limit

        try:
            results = self.client.get_all(**params)
        except TypeError:
            fallback_params = {}
            if active_user:
                fallback_params["user_id"] = active_user
            try:
                results = self.client.get_all(**fallback_params)
            except Exception as exc:
                logger.warning("Mem0 get_all failed after fallback: %s", exc)
                return []
        except Exception as exc:
            logger.warning("Mem0 get_all failed: %s", exc)
            return []

        return self._extract_memories(results)
