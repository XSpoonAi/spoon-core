"""
OpenAI Compatible Provider base class for providers that use OpenAI-compatible APIs.
This includes OpenAI, OpenRouter, DeepSeek, and other providers with similar interfaces.
"""

import asyncio
from typing import List, Dict, Any, Optional, AsyncGenerator, AsyncIterator
from logging import getLogger
from uuid import uuid4
from datetime import datetime

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion

from spoon_ai.schema import Message, ToolCall, Function, LLMResponseChunk
from ..interface import LLMProviderInterface, LLMResponse, ProviderMetadata, ProviderCapability
from ..errors import ProviderError, AuthenticationError, RateLimitError, ModelNotFoundError, NetworkError
from spoon_ai.callbacks.base import BaseCallbackHandler
from spoon_ai.callbacks.manager import CallbackManager

logger = getLogger(__name__)


class OpenAICompatibleProvider(LLMProviderInterface):
    """Base class for OpenAI-compatible providers."""

    def __init__(self):
        self.client: Optional[AsyncOpenAI] = None
        self.config: Dict[str, Any] = {}
        self.model: str = ""
        self.max_tokens: int = 4096
        self.temperature: float = 0.3
        self.provider_name: str = "openai_compatible"
        self.default_base_url: str = "https://api.openai.com/v1"
        self.default_model: str = "gpt-4.1"

    def _uses_completion_token_param(self, model: str) -> bool:
        """Whether this model expects max_completion_tokens instead of max_tokens.

        Only the new OpenAI `gpt-5*` and `o*` models use the completion-only
        parameter. Namespaces like OpenRouter's `openai/gpt-3.5-turbo` should
        still use `max_tokens`, so we check the final segment only.
        """
        model_lower = (model or "").lower()
        tail = model_lower.split("/")[-1]  # strip any provider prefix like openrouter
        return tail.startswith("gpt-5") or tail.startswith("o1") or tail.startswith("o3") or tail.startswith("o4")

    def _max_token_kwargs(self, model: str, max_tokens: int, overrides: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build the correct token-limit argument for the OpenAI API.

        - gpt-5* / o* models require `max_completion_tokens`
        - older models keep using `max_tokens`
        - explicit overrides in kwargs take precedence
        """
        if "max_completion_tokens" in overrides:
            return {"max_completion_tokens": overrides["max_completion_tokens"]}

        if "max_tokens" in overrides:
            max_tokens = overrides["max_tokens"]

        if self._uses_completion_token_param(model):
            return {"max_completion_tokens": max_tokens}

        return {"max_tokens": max_tokens}

    def get_provider_name(self) -> str:
        """Get the provider name. Should be overridden by subclasses."""
        return self.provider_name

    def get_default_base_url(self) -> str:
        """Get the default base URL. Should be overridden by subclasses."""
        return self.default_base_url

    def get_default_model(self) -> str:
        """Get the default model. Should be overridden by subclasses."""
        return self.default_model

    def get_additional_headers(self, config: Dict[str, Any]) -> Dict[str, str]:
        """Get additional headers for the provider. Can be overridden by subclasses."""
        return {}

    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the provider with configuration."""
        try:
            self.config = config
            self.model = config.get('model', self.get_default_model())
            self.max_tokens = config.get('max_tokens', 4096)
            self.temperature = config.get('temperature', 0.3)

            api_key = config.get('api_key')
            if not api_key:
                raise AuthenticationError(self.get_provider_name(), context={"config": config})

            base_url = config.get('base_url', self.get_default_base_url())
            timeout = config.get('timeout', 30)

            # Get provider-specific headers
            additional_headers = self.get_additional_headers(config)

            self.client = AsyncOpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=timeout,
                default_headers=additional_headers if additional_headers else None
            )

            logger.info(f"{self.get_provider_name()} provider initialized with model: {self.model}")

        except Exception as e:
            if isinstance(e, (AuthenticationError, ProviderError)):
                raise
            raise ProviderError(self.get_provider_name(), f"Failed to initialize: {str(e)}", original_error=e)

    def _convert_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """Convert Message objects to OpenAI-compatible format with validation."""
        openai_messages = []

        for i, message in enumerate(messages):
            msg_dict = {"role": message.role}

            if message.content:
                msg_dict["content"] = message.content

            if message.tool_calls:
                msg_dict["tool_calls"] = [
                    {
                        "id": tool_call.id,
                        "type": tool_call.type,
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments
                        }
                    }
                    for tool_call in message.tool_calls
                ]

            if message.name:
                msg_dict["name"] = message.name

            if message.tool_call_id:
                msg_dict["tool_call_id"] = message.tool_call_id

            # Validate tool message placement - but don't skip, let the validation fix it later
            if message.role == "tool":
                # Log if there's a potential issue but don't skip - let validation fix it
                if i == 0 or openai_messages[-1]["role"] != "assistant" or "tool_calls" not in openai_messages[-1]:
                    logger.debug(f"Tool message at index {i} may need validation adjustment.")
                    # Don't skip - add it and let _validate_and_fix_message_sequence handle it

            openai_messages.append(msg_dict)

        return self._validate_and_fix_message_sequence(openai_messages)

    def _validate_and_fix_message_sequence(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate and fix message sequence to comply with OpenAI API requirements."""
        if not messages:
            return messages

        fixed_messages = []
        i = 0

        while i < len(messages):
            current_msg = messages[i]

            # Handle tool messages
            if current_msg["role"] == "tool":
                # Find the preceding assistant message with tool_calls
                assistant_msg_idx = -1
                for j in range(len(fixed_messages) - 1, -1, -1):
                    if fixed_messages[j]["role"] == "assistant" and "tool_calls" in fixed_messages[j]:
                        assistant_msg_idx = j
                        break

                if assistant_msg_idx == -1:
                    # No preceding assistant message with tool_calls found, but don't skip
                    # This might be a tool response that should be kept
                    logger.debug(f"Tool message without preceding assistant tool_calls - keeping: {current_msg.get('tool_call_id', 'unknown')}")
                else:
                    # Check if this tool message corresponds to any tool_call in the assistant message
                    assistant_msg = fixed_messages[assistant_msg_idx]
                    tool_call_ids = [tc["id"] for tc in assistant_msg.get("tool_calls", [])]

                    if current_msg.get("tool_call_id") not in tool_call_ids:
                        logger.debug(f"Tool message tool_call_id {current_msg.get('tool_call_id')} not found in assistant tool_calls - keeping anyway.")

            # Handle system messages - they should be at the beginning
            elif current_msg["role"] == "system":
                # If we already have messages and this is not the first, move it to the beginning
                if fixed_messages and fixed_messages[0]["role"] != "system":
                    # Insert at the beginning, but after any existing system messages
                    insert_idx = 0
                    while insert_idx < len(fixed_messages) and fixed_messages[insert_idx]["role"] == "system":
                        insert_idx += 1
                    fixed_messages.insert(insert_idx, current_msg)
                    i += 1
                    continue

            # Handle consecutive messages with the same role (except tool messages)
            elif (fixed_messages and
                  fixed_messages[-1]["role"] == current_msg["role"] and
                  current_msg["role"] != "tool"):

                # Merge content if both have content
                if (fixed_messages[-1].get("content") and current_msg.get("content")):
                    fixed_messages[-1]["content"] += "\n" + current_msg["content"]
                    i += 1
                    continue
                # If current message has content but previous doesn't, replace
                elif current_msg.get("content") and not fixed_messages[-1].get("content"):
                    fixed_messages[-1] = current_msg
                    i += 1
                    continue

            fixed_messages.append(current_msg)
            i += 1

        return fixed_messages

    def _convert_response(self, response: ChatCompletion, duration: float) -> LLMResponse:
        """Convert OpenAI-compatible response to standardized LLMResponse."""
        choice = response.choices[0]
        message = choice.message

        # Convert tool calls
        tool_calls = []
        if message.tool_calls:
            for tool_call in message.tool_calls:
                tool_calls.append(ToolCall(
                    id=tool_call.id,
                    type=tool_call.type,
                    function=Function(
                        name=tool_call.function.name,
                        arguments=tool_call.function.arguments
                    )
                ))

        # Map finish reasons
        finish_reason = choice.finish_reason
        if finish_reason == "stop":
            standardized_finish_reason = "stop"
        elif finish_reason == "length":
            standardized_finish_reason = "length"
        elif finish_reason == "tool_calls":
            standardized_finish_reason = "tool_calls"
        elif finish_reason == "content_filter":
            standardized_finish_reason = "content_filter"
        else:
            standardized_finish_reason = finish_reason

        # Extract usage information
        usage = None
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }

        return LLMResponse(
            content=message.content or "",
            provider=self.get_provider_name(),
            model=response.model,
            finish_reason=standardized_finish_reason,
            native_finish_reason=finish_reason,
            tool_calls=tool_calls,
            usage=usage,
            duration=duration,
            metadata={
                "response_id": response.id,
                "created": response.created,
                "system_fingerprint": getattr(response, 'system_fingerprint', None)
            }
        )

    async def chat(self, messages: List[Message], **kwargs) -> LLMResponse:
        """Send chat request to the provider."""
        if not self.client:
            raise ProviderError(self.get_provider_name(), "Provider not initialized")

        try:
            start_time = asyncio.get_event_loop().time()

            openai_messages = self._convert_messages(messages)

            # Extract parameters
            model = kwargs.get('model', self.model)
            max_tokens = kwargs.get('max_tokens', self.max_tokens)
            temperature = kwargs.get('temperature', self.temperature)

            tools = kwargs.get('tools')
            tool_choice = kwargs.get('tool_choice', 'auto')

            request_kwargs: Dict[str, Any] = {
                "model": model,
                "messages": openai_messages,
                "temperature": temperature,
                "stream": False,
            }
            request_kwargs.update(self._max_token_kwargs(model, max_tokens, kwargs))

            if tools:
                request_kwargs["tools"] = tools
                request_kwargs["tool_choice"] = tool_choice

            extra_keys = {'model', 'max_tokens', 'max_completion_tokens', 'temperature', 'tools', 'tool_choice'}
            request_kwargs.update({k: v for k, v in kwargs.items() if k not in extra_keys})

            response = await self.client.chat.completions.create(**request_kwargs)

            duration = asyncio.get_event_loop().time() - start_time
            return self._convert_response(response, duration)

        except Exception as e:
            await self._handle_error(e)

    async def chat_stream(self,messages: List[Message],callbacks: Optional[List[BaseCallbackHandler]] = None,**kwargs) -> AsyncIterator[LLMResponseChunk]:
        """Send streaming chat request with full callback support.
        Yields:
            LLMResponseChunk: Structured streaming response chunks
        """
        if not self.client:
            raise ProviderError(self.get_provider_name(), "Provider not initialized")

        # Create callback manager
        callback_manager = CallbackManager.from_callbacks(callbacks)
        run_id = uuid4()

        try:
            openai_messages = self._convert_messages(messages)

            # Extract parameters
            model = kwargs.get('model', self.model)
            max_tokens = kwargs.get('max_tokens', self.max_tokens)
            temperature = kwargs.get('temperature', self.temperature)

            # Trigger on_llm_start callback
            await callback_manager.on_llm_start(run_id=run_id,messages=messages,model=model,provider=self.get_provider_name())

            tools = kwargs.get('tools')
            tool_choice = kwargs.get('tool_choice', 'auto')

            request_kwargs: Dict[str, Any] = {
                "model": model,
                "messages": openai_messages,
                "temperature": temperature,
                "stream": True,
                "stream_options": {"include_usage": True},
            }
            request_kwargs.update(self._max_token_kwargs(model, max_tokens, kwargs))

            if tools:
                request_kwargs["tools"] = tools
                request_kwargs["tool_choice"] = tool_choice

            extra_keys = {'model', 'max_tokens', 'max_completion_tokens', 'temperature', 'callbacks', 'tools', 'tool_choice'}
            request_kwargs.update({k: v for k, v in kwargs.items() if k not in extra_keys})

            stream = await self.client.chat.completions.create(**request_kwargs)
            # Process streaming response
            full_content = ""
            chunk_index = 0
            tool_call_accumulator = {}  # For accumulating tool calls
            finish_reason = None  # Initialize finish_reason outside loop
            tool_calls = []  # Initialize tool_calls outside loop
            
            async for chunk in stream:
                # Skip chunks without choices (e.g., final chunk with only usage stats)
                if not hasattr(chunk, 'choices') or not chunk.choices:
                    # Extract usage stats from chunks without choices
                    if hasattr(chunk, 'usage') and chunk.usage:
                        usage = {
                            "prompt_tokens": chunk.usage.prompt_tokens,
                            "completion_tokens": chunk.usage.completion_tokens,
                            "total_tokens": chunk.usage.total_tokens
                        }
                        # Yield a final chunk with usage info
                        response_chunk = LLMResponseChunk(
                            content=full_content,
                            delta="",
                            provider=self.get_provider_name(),
                            model=model,
                            finish_reason=finish_reason,
                            tool_calls=tool_calls,
                            tool_call_chunks=None,
                            usage=usage,
                            metadata={
                                "chunk_id": chunk.id if hasattr(chunk, 'id') else None,
                                "created": chunk.created if hasattr(chunk, 'created') else None
                            },
                            chunk_index=chunk_index,
                            timestamp=datetime.now().isoformat()
                        )
                        yield response_chunk
                        chunk_index += 1
                    continue

                choice = chunk.choices[0]
                delta = choice.delta

                # Extract token/content
                token = delta.content or ""
                if token:
                    full_content += token

                # Extract finish_reason (preserve once set, don't let None overwrite it)
                if choice.finish_reason is not None:
                    finish_reason = choice.finish_reason

                # Handle tool calls (OpenAI streams them incrementally)
                tool_call_chunks = None

                if delta.tool_calls:
                    tool_call_chunks = []
                    for tc_chunk in delta.tool_calls:
                        tc_id = tc_chunk.id or f"call_{tc_chunk.index}"

                        # Initialize accumulator if needed
                        if tc_id not in tool_call_accumulator:
                            tool_call_accumulator[tc_id] = {
                                "id": tc_id,
                                "type": tc_chunk.type or "function",
                                "function": {
                                    "name": "",
                                    "arguments": ""
                                }
                            }

                        # Accumulate function name and arguments
                        if tc_chunk.function:
                            if tc_chunk.function.name:
                                tool_call_accumulator[tc_id]["function"]["name"] += tc_chunk.function.name
                            if tc_chunk.function.arguments:
                                tool_call_accumulator[tc_id]["function"]["arguments"] += tc_chunk.function.arguments

                        tool_call_chunks.append({
                            "index": tc_chunk.index,
                            "id": tc_chunk.id,
                            "type": tc_chunk.type,
                            "function": tc_chunk.function.model_dump() if tc_chunk.function else None
                        })
                
                # Convert accumulated tool calls to ToolCall objects (build from accumulator each time)
                tool_calls = [
                    ToolCall(
                        id=tc["id"],
                        type=tc["type"],
                        function=Function(
                            name=tc["function"]["name"],
                            arguments=tc["function"]["arguments"]
                        )
                    )
                    for tc in tool_call_accumulator.values()
                ]
                
                # Extract usage stats (typically in final chunk)
                usage = None
                if hasattr(chunk, 'usage') and chunk.usage:
                    usage = {
                        "prompt_tokens": chunk.usage.prompt_tokens,
                        "completion_tokens": chunk.usage.completion_tokens,
                        "total_tokens": chunk.usage.total_tokens
                    }

                # Build response chunk
                response_chunk = LLMResponseChunk(
                    content=full_content,
                    delta=token,
                    provider=self.get_provider_name(),
                    model=model,
                    finish_reason=finish_reason,
                    tool_calls=tool_calls,
                    tool_call_chunks=tool_call_chunks,
                    usage=usage,
                    metadata={
                        "chunk_id": chunk.id if hasattr(chunk, 'id') else None,
                        "created": chunk.created if hasattr(chunk, 'created') else None
                    },
                    chunk_index=chunk_index,
                    timestamp=datetime.now().isoformat()
                )

                # Trigger on_llm_new_token callback
                if token:  # Only trigger if there's actual content
                    await callback_manager.on_llm_new_token(
                        token=token,
                        chunk=response_chunk,
                        run_id=run_id
                    )

                # Yield chunk
                yield response_chunk
                chunk_index += 1

            # Trigger on_llm_end callback
            final_response = LLMResponse(
                content=full_content,
                provider=self.get_provider_name(),
                model=model,
                finish_reason=finish_reason or "stop",
                native_finish_reason=finish_reason or "stop",
                tool_calls=tool_calls,
                usage=usage,
                metadata={}
            )
            await callback_manager.on_llm_end(
                response=final_response,
                run_id=run_id
            )

        except Exception as e:
            await callback_manager.on_llm_error(
                error=e,
                run_id=run_id
            )
            await self._handle_error(e)

    async def completion(self, prompt: str, **kwargs) -> LLMResponse:
        """Send completion request to the provider."""
        # Convert to chat format
        messages = [Message(role="user", content=prompt)]
        return await self.chat(messages, **kwargs)

    async def chat_with_tools(self, messages: List[Message], tools: List[Dict], **kwargs) -> LLMResponse:
        """Send chat request with tools to the provider."""
        if not self.client:
            raise ProviderError(self.get_provider_name(), "Provider not initialized")

        try:
            start_time = asyncio.get_event_loop().time()

            openai_messages = self._convert_messages(messages)

            # Extract parameters
            model = kwargs.get('model', self.model)
            max_tokens = kwargs.get('max_tokens', self.max_tokens)
            temperature = kwargs.get('temperature', self.temperature)
            tool_choice = kwargs.get('tool_choice', 'auto')

            request_kwargs: Dict[str, Any] = {
                "model": model,
                "messages": openai_messages,
                "temperature": temperature,
                "stream": False,
            }
            request_kwargs.update(self._max_token_kwargs(model, max_tokens, kwargs))

            if tools:
                request_kwargs["tools"] = tools
                request_kwargs["tool_choice"] = tool_choice

            extra_keys = {'model', 'max_tokens', 'max_completion_tokens', 'temperature', 'tool_choice'}
            request_kwargs.update({k: v for k, v in kwargs.items() if k not in extra_keys})

            response = await self.client.chat.completions.create(**request_kwargs)

            duration = asyncio.get_event_loop().time() - start_time
            return self._convert_response(response, duration)

        except Exception as e:
            await self._handle_error(e)

    def get_metadata(self) -> ProviderMetadata:
        """Get provider metadata. Should be overridden by subclasses."""
        return ProviderMetadata(
            name=self.get_provider_name(),
            version="1.0.0",
            capabilities=[
                ProviderCapability.CHAT,
                ProviderCapability.COMPLETION,
                ProviderCapability.TOOLS,
                ProviderCapability.STREAMING
            ],
            max_tokens=4096,
            supports_system_messages=True,
            rate_limits={}
        )

    async def health_check(self) -> bool:
        """Check if provider is healthy."""
        if not self.client:
            return False

        try:
            # Simple test request
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=1
            )
            return True
        except Exception as e:
            logger.warning(f"{self.get_provider_name()} health check failed: {e}")
            return False

    async def cleanup(self) -> None:
        """Cleanup provider resources."""
        if self.client:
            await self.client.close()
            self.client = None
        logger.info(f"{self.get_provider_name()} provider cleaned up")

    async def _handle_error(self, error: Exception) -> None:
        """Handle and convert provider errors to standardized errors."""
        error_str = str(error).lower()
        provider_name = self.get_provider_name()

        if "authentication" in error_str or "api key" in error_str or "unauthorized" in error_str:
            raise AuthenticationError(provider_name, context={"original_error": str(error)})
        elif "rate limit" in error_str or "quota" in error_str:
            raise RateLimitError(provider_name, context={"original_error": str(error)})
        elif "model" in error_str and ("not found" in error_str or "not available" in error_str):
            raise ModelNotFoundError(provider_name, self.model, context={"original_error": str(error)})
        elif "timeout" in error_str or "connection" in error_str:
            raise NetworkError(provider_name, "Network error", original_error=error)
        else:
            raise ProviderError(provider_name, f"Request failed: {str(error)}", original_error=error)