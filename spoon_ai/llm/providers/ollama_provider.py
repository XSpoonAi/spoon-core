"""Ollama Provider implementation for the unified LLM interface.

Ollama runs locally and exposes an HTTP API (default: http://localhost:11434).
This provider supports chat, completion, and streaming.

Notes:
- Ollama does not require an API key; the configuration layer may still provide
  a placeholder api_key value for consistency.
- Tool calling is supported via /api/chat (tools + tool_calls).
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from logging import getLogger
from typing import Any, AsyncIterator, Dict, List, Optional
from uuid import uuid4

import httpx

from spoon_ai.callbacks.base import BaseCallbackHandler
from spoon_ai.schema import Function, LLMResponseChunk, Message, ToolCall

from ..errors import NetworkError, ProviderError
from ..interface import LLMProviderInterface, LLMResponse, ProviderCapability, ProviderMetadata
from ..registry import register_provider

logger = getLogger(__name__)


@register_provider(
    "ollama",
    [
        ProviderCapability.CHAT,
        ProviderCapability.COMPLETION,
        ProviderCapability.TOOLS,
        ProviderCapability.STREAMING,
    ],
)
class OllamaProvider(LLMProviderInterface):
    """Local Ollama provider via HTTP."""

    def __init__(self):
        self.client: Optional[httpx.AsyncClient] = None
        self.config: Dict[str, Any] = {}
        self.model: str = "llama3.2"
        self.base_url: str = "http://localhost:11434"
        self.timeout: float = 30.0
        self.max_tokens: int = 4096
        self.temperature: float = 0.3

    async def initialize(self, config: Dict[str, Any]) -> None:
        self.config = config or {}
        self.model = (self.config.get("model") or self.model).strip()
        self.base_url = (self.config.get("base_url") or self.base_url).rstrip("/")
        self.timeout = float(self.config.get("timeout", self.timeout))
        self.max_tokens = int(self.config.get("max_tokens", self.max_tokens))
        self.temperature = float(self.config.get("temperature", self.temperature))

        # Lazily create client
        if self.client is None:
            self.client = httpx.AsyncClient(timeout=self.timeout)

        logger.info("Ollama provider initialized with model: %s", self.model)

    @staticmethod
    def _normalize_tool_arguments_dict(arguments: Any) -> Dict[str, Any]:
        """Normalize tool arguments into a JSON-object dict for Ollama payloads."""
        if arguments is None:
            return {}
        if isinstance(arguments, dict):
            return arguments
        if isinstance(arguments, str):
            s = arguments.strip()
            if not s:
                return {}
            try:
                parsed = json.loads(s)
            except json.JSONDecodeError:
                return {}
            return parsed if isinstance(parsed, dict) else {}
        return {}

    @classmethod
    def _convert_tool_calls_for_payload(cls, tool_calls: Any) -> List[Dict[str, Any]]:
        """Convert internal tool call objects to Ollama tool_calls payload format."""
        out: List[Dict[str, Any]] = []
        if not tool_calls:
            return out

        for tc in tool_calls:
            func_name: Optional[str] = None
            func_args: Any = None

            if isinstance(tc, ToolCall):
                func_name = tc.function.name
                func_args = tc.function.arguments
            elif isinstance(tc, dict):
                func = tc.get("function") if isinstance(tc.get("function"), dict) else None
                if func:
                    func_name = func.get("name")
                    func_args = func.get("arguments")
                else:
                    func_name = tc.get("name")
                    func_args = tc.get("arguments")

            if not func_name:
                func_name = "unknown"

            out.append(
                {
                    "function": {
                        "name": func_name,
                        "arguments": cls._normalize_tool_arguments_dict(func_args),
                    }
                }
            )

        return out

    def _convert_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        tool_call_id_to_name: Dict[str, str] = {}

        # Build a best-effort map from tool_call_id -> tool name for tool result messages.
        for msg in messages:
            if (getattr(msg, "role", None) or "").strip() != "assistant":
                continue
            tcs = getattr(msg, "tool_calls", None)
            if not tcs:
                continue
            for tc in tcs:
                tc_id: Optional[str] = None
                tc_name: Optional[str] = None
                if isinstance(tc, ToolCall):
                    tc_id = tc.id
                    tc_name = tc.function.name
                elif isinstance(tc, dict):
                    if isinstance(tc.get("id"), str):
                        tc_id = tc.get("id")
                    func = tc.get("function") if isinstance(tc.get("function"), dict) else None
                    if func and isinstance(func.get("name"), str):
                        tc_name = func.get("name")
                if tc_id and tc_name:
                    tool_call_id_to_name[tc_id] = tc_name

        for m in messages:
            role = (getattr(m, "role", None) or "").strip()
            if not role:
                continue
            content = getattr(m, "content", None)
            if isinstance(content, list):
                # Ollama is text-first; fall back to extracted text.
                content = getattr(m, "text_content", "") or ""
            elif content is None:
                content = ""
            elif not isinstance(content, str):
                try:
                    content = json.dumps(content, ensure_ascii=False)
                except Exception:
                    content = str(content)

            msg: Dict[str, Any] = {"role": role, "content": content}

            # Carry assistant tool calls forward so the model can see prior tool selections.
            tool_calls = getattr(m, "tool_calls", None)
            if tool_calls:
                msg["tool_calls"] = self._convert_tool_calls_for_payload(tool_calls)

            # Tool result messages: Ollama expects tool_name on role=tool messages.
            if role == "tool":
                tool_name = (getattr(m, "name", None) or "").strip()
                if not tool_name:
                    tool_call_id = getattr(m, "tool_call_id", None)
                    if isinstance(tool_call_id, str) and tool_call_id:
                        tool_name = tool_call_id_to_name.get(tool_call_id, "")
                if tool_name:
                    msg["tool_name"] = tool_name

            out.append(msg)
        return out

    def _build_options(self, **kwargs: Any) -> Dict[str, Any]:
        """Map common parameters to Ollama options."""
        options: Dict[str, Any] = {}

        temperature = kwargs.get("temperature", self.temperature)
        if temperature is not None:
            options["temperature"] = float(temperature)

        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        if max_tokens is not None:
            # Ollama uses num_predict for max output tokens
            options["num_predict"] = int(max_tokens)

        return options

    @staticmethod
    def _extract_content(data: Dict[str, Any]) -> str:
        msg = data.get("message")
        if isinstance(msg, dict):
            content = msg.get("content")
            if isinstance(content, str):
                return content
        resp = data.get("response")
        if isinstance(resp, str):
            return resp
        return ""

    @staticmethod
    def _extract_usage(data: Dict[str, Any]) -> Optional[Dict[str, int]]:
        prompt = data.get("prompt_eval_count")
        completion = data.get("eval_count")
        if prompt is None and completion is None:
            return None
        prompt_i = int(prompt or 0)
        completion_i = int(completion or 0)
        return {
            "prompt_tokens": prompt_i,
            "completion_tokens": completion_i,
            "total_tokens": prompt_i + completion_i,
        }

    @staticmethod
    def _extract_tool_calls(data: Dict[str, Any]) -> List[ToolCall]:
        """Extract tool calls from an Ollama /api/chat response."""
        msg = data.get("message")
        if not isinstance(msg, dict):
            return []

        raw_tool_calls = msg.get("tool_calls")
        if not isinstance(raw_tool_calls, list):
            return []

        tool_calls: List[ToolCall] = []
        for tc in raw_tool_calls:
            if not isinstance(tc, dict):
                continue
            func = tc.get("function")
            if not isinstance(func, dict):
                continue

            name = func.get("name") if isinstance(func.get("name"), str) else None
            if not name:
                name = "unknown"

            args = func.get("arguments")
            if isinstance(args, dict):
                args_str = json.dumps(args, ensure_ascii=False)
            elif isinstance(args, str):
                args_str = args
            else:
                args_str = "{}"

            tc_id = tc.get("id") if isinstance(tc.get("id"), str) else f"call_{uuid4().hex}"
            tc_type = tc.get("type") if isinstance(tc.get("type"), str) else "function"

            tool_calls.append(
                ToolCall(
                    id=tc_id,
                    type=tc_type,
                    function=Function(name=name, arguments=args_str),
                )
            )

        return tool_calls

    async def chat(self, messages: List[Message], **kwargs) -> LLMResponse:
        if not self.client:
            raise ProviderError("ollama", "Provider not initialized")

        model = (kwargs.get("model") or self.model).strip()
        payload: Dict[str, Any] = {
            "model": model,
            "messages": self._convert_messages(messages),
            "stream": False,
            "options": self._build_options(**kwargs),
        }

        url = f"{self.base_url}/api/chat"
        start = asyncio.get_event_loop().time()

        try:
            resp = await self.client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
        except httpx.TimeoutException as exc:
            raise NetworkError("ollama", "Request timeout", original_error=exc)
        except httpx.HTTPError as exc:
            raise ProviderError("ollama", f"Request failed: {exc}", original_error=exc)
        except Exception as exc:  # pragma: no cover
            raise ProviderError("ollama", f"Unexpected error: {exc}", original_error=exc)

        duration = asyncio.get_event_loop().time() - start
        content = self._extract_content(data) if isinstance(data, dict) else ""
        tool_calls = self._extract_tool_calls(data) if isinstance(data, dict) else []
        done_reason = (data.get("done_reason") if isinstance(data, dict) else None) or "stop"

        return LLMResponse(
            content=content,
            provider="ollama",
            model=model,
            finish_reason="tool_calls" if tool_calls else "stop",
            native_finish_reason=str(done_reason),
            tool_calls=tool_calls,
            usage=self._extract_usage(data) if isinstance(data, dict) else None,
            metadata={"raw": data} if isinstance(data, dict) else {},
            request_id=str(data.get("id", "")) if isinstance(data, dict) else "",
            duration=duration,
            timestamp=datetime.now(),
        )

    async def chat_stream(
        self,
        messages: List[Message],
        callbacks: Optional[List[BaseCallbackHandler]] = None,
        **kwargs,
    ) -> AsyncIterator[LLMResponseChunk]:
        if not self.client:
            raise ProviderError("ollama", "Provider not initialized")

        model = (kwargs.get("model") or self.model).strip()
        payload: Dict[str, Any] = {
            "model": model,
            "messages": self._convert_messages(messages),
            "stream": True,
            "options": self._build_options(**kwargs),
        }

        url = f"{self.base_url}/api/chat"
        full_content = ""
        chunk_index = 0

        try:
            async with self.client.stream("POST", url, json=payload) as resp:
                resp.raise_for_status()

                async for line in resp.aiter_lines():
                    if not line:
                        continue

                    try:
                        data = json.loads(line)
                    except Exception:
                        continue

                    if not isinstance(data, dict):
                        continue

                    delta = self._extract_content(data)
                    if delta:
                        full_content += delta

                    tool_calls = self._extract_tool_calls(data)
                    done = bool(data.get("done"))
                    native_finish = data.get("done_reason") if done else None
                    finish_reason = str(native_finish) if done else None

                    usage = self._extract_usage(data) if done else None

                    raw_msg = data.get("message")
                    yield LLMResponseChunk(
                        content=full_content,
                        delta=delta,
                        provider="ollama",
                        model=model,
                        finish_reason=("tool_calls" if tool_calls else finish_reason),
                        tool_calls=tool_calls,
                        tool_call_chunks=(
                            raw_msg.get("tool_calls")
                            if tool_calls and isinstance(raw_msg, dict)
                            else None
                        ),
                        usage=usage,
                        metadata={"done": done},
                        chunk_index=chunk_index,
                        timestamp=datetime.now().isoformat(),
                    )
                    chunk_index += 1

        except httpx.TimeoutException as exc:
            raise NetworkError("ollama", "Request timeout", original_error=exc)
        except httpx.HTTPError as exc:
            raise ProviderError("ollama", f"Request failed: {exc}", original_error=exc)

    async def completion(self, prompt: str, **kwargs) -> LLMResponse:
        if not self.client:
            raise ProviderError("ollama", "Provider not initialized")

        model = (kwargs.get("model") or self.model).strip()
        payload: Dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": self._build_options(**kwargs),
        }

        url = f"{self.base_url}/api/generate"
        start = asyncio.get_event_loop().time()

        try:
            resp = await self.client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
        except httpx.TimeoutException as exc:
            raise NetworkError("ollama", "Request timeout", original_error=exc)
        except httpx.HTTPError as exc:
            raise ProviderError("ollama", f"Request failed: {exc}", original_error=exc)

        duration = asyncio.get_event_loop().time() - start
        text = data.get("response", "") if isinstance(data, dict) else ""
        done_reason = (data.get("done_reason") if isinstance(data, dict) else None) or "stop"

        return LLMResponse(
            content=text,
            provider="ollama",
            model=model,
            finish_reason="stop",
            native_finish_reason=str(done_reason),
            tool_calls=[],
            usage=self._extract_usage(data) if isinstance(data, dict) else None,
            metadata={"raw": data} if isinstance(data, dict) else {},
            request_id=str(data.get("id", "")) if isinstance(data, dict) else "",
            duration=duration,
            timestamp=datetime.now(),
        )

    async def chat_with_tools(self, messages: List[Message], tools: List[Dict], **kwargs) -> LLMResponse:
        if not self.client:
            raise ProviderError("ollama", "Provider not initialized")

        tool_choice_raw = kwargs.get("tool_choice", "auto")
        if isinstance(tool_choice_raw, str):
            tool_choice = tool_choice_raw
        else:
            tool_choice = getattr(tool_choice_raw, "value", None) or str(tool_choice_raw)
        tool_choice = (tool_choice or "auto").strip().lower()

        include_tools = tool_choice != "none"
        if tool_choice == "required" and not tools:
            raise ProviderError("ollama", "tool_choice=required but no tools provided")

        model = (kwargs.get("model") or self.model).strip()
        payload: Dict[str, Any] = {
            "model": model,
            "messages": self._convert_messages(messages),
            "stream": False,
            "options": self._build_options(**kwargs),
        }
        if include_tools and tools:
            payload["tools"] = tools

        url = f"{self.base_url}/api/chat"
        start = asyncio.get_event_loop().time()

        try:
            resp = await self.client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
        except httpx.TimeoutException as exc:
            raise NetworkError("ollama", "Request timeout", original_error=exc)
        except httpx.HTTPError as exc:
            raise ProviderError("ollama", f"Request failed: {exc}", original_error=exc)
        except Exception as exc:  # pragma: no cover
            raise ProviderError("ollama", f"Unexpected error: {exc}", original_error=exc)

        duration = asyncio.get_event_loop().time() - start
        content = self._extract_content(data) if isinstance(data, dict) else ""
        tool_calls = self._extract_tool_calls(data) if isinstance(data, dict) else []
        done_reason = (data.get("done_reason") if isinstance(data, dict) else None) or "stop"

        return LLMResponse(
            content=content,
            provider="ollama",
            model=model,
            finish_reason="tool_calls" if tool_calls else "stop",
            native_finish_reason=str(done_reason),
            tool_calls=tool_calls,
            usage=self._extract_usage(data) if isinstance(data, dict) else None,
            metadata={"raw": data} if isinstance(data, dict) else {},
            request_id=str(data.get("id", "")) if isinstance(data, dict) else "",
            duration=duration,
            timestamp=datetime.now(),
        )

    def get_metadata(self) -> ProviderMetadata:
        return ProviderMetadata(
            name="ollama",
            version="1.0.0",
            capabilities=[
                ProviderCapability.CHAT,
                ProviderCapability.COMPLETION,
                ProviderCapability.TOOLS,
                ProviderCapability.STREAMING,
            ],
            max_tokens=self.max_tokens,
            supports_system_messages=True,
            rate_limits={},
        )

    async def health_check(self) -> bool:
        if not self.client:
            return False

        try:
            resp = await self.client.get(f"{self.base_url}/api/tags")
            return resp.status_code == 200
        except Exception:
            return False

    async def cleanup(self) -> None:
        if self.client is not None:
            await self.client.aclose()
            self.client = None

