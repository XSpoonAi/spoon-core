"""
OpenAI Compatible Provider base class for providers that use OpenAI-compatible APIs.
This includes OpenAI, OpenRouter, DeepSeek, and other providers with similar interfaces.
"""

import asyncio
import base64
from typing import List, Dict, Any, Optional, AsyncGenerator, AsyncIterator
from logging import getLogger
from uuid import uuid4
from datetime import datetime

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion

from spoon_ai.schema import (
    Message, ToolCall, Function, LLMResponseChunk,
    TextContent, ImageContent, ImageUrlContent, DocumentContent, FileContent, ContentBlock
)
from ..interface import LLMProviderInterface, LLMResponse, ProviderMetadata, ProviderCapability
from ..errors import ProviderError, AuthenticationError, RateLimitError, ModelNotFoundError, NetworkError
from spoon_ai.callbacks.base import BaseCallbackHandler
from spoon_ai.callbacks.manager import CallbackManager

logger = getLogger(__name__)

# Maximum file size for inline base64 upload (4MB)
# Files larger than this will be uploaded via Files API
MAX_INLINE_FILE_SIZE = 4 * 1024 * 1024  # 4MB in bytes


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

    def _supports_temperature(self, model: str) -> bool:
        """Whether this model supports custom temperature values.

        Some newer OpenAI models (gpt-5.1*, o1*, o3*, o4*) only support the 
        default temperature value (1.0) and will error on other values.
        """
        model_lower = (model or "").lower()
        tail = model_lower.split("/")[-1]  # strip any provider prefix
        # gpt-5.1 and reasoning models don't support custom temperature
        return not (tail.startswith("gpt-5.1") or tail.startswith("o1") or tail.startswith("o3") or tail.startswith("o4"))

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

            # Treat explicit None/empty base_url as "use provider default".
            base_url = config.get('base_url') or self.get_default_base_url()
            # Increase default timeout to 300s (5 minutes) for better support of large documents (PDFs, images)
            # Use httpx.Timeout for granular control: read timeout is most important for large document processing
            timeout_config = config.get('timeout', 300)
            try:
                import httpx
                HTTPX_AVAILABLE = True
            except ImportError:
                HTTPX_AVAILABLE = False
            
            if HTTPX_AVAILABLE and isinstance(timeout_config, (int, float)):
                # Use httpx.Timeout for better control: read timeout is critical for large documents
                timeout = httpx.Timeout(
                    connect=10.0,  # Connection timeout: 10 seconds
                    read=float(timeout_config),  # Read timeout: configurable (default 300s for large docs)
                    write=30.0,  # Write timeout: 30 seconds
                    pool=10.0  # Pool timeout: 10 seconds
                )
            else:
                # Fallback to simple timeout if httpx not available or timeout is already a Timeout object
                timeout = timeout_config

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

    async def _upload_file_to_openai(self, file_data: bytes, filename: str, purpose: str = "user_data") -> str:
        """Upload a file to OpenAI Files API and return the file ID.

        This is used for large files that exceed the inline base64 size limit.

        Args:
            file_data: Raw bytes of the file
            filename: Name of the file
            purpose: File purpose (default: "user_data" for chat inputs)

        Returns:
            File ID from OpenAI
        """
        if not self.client:
            raise ProviderError(self.get_provider_name(), "Client not initialized")

        try:
            # Create a file-like object from bytes
            from io import BytesIO
            file_obj = BytesIO(file_data)
            file_obj.name = filename

            # Upload to OpenAI Files API with extended timeout for large files
            # Large files (4MB+) may need up to 600 seconds (10 minutes) to upload
            # Use httpx.Timeout for granular control: write timeout is critical for file uploads
            try:
                import httpx
                # For file uploads, we need longer write timeout (uploading data) and read timeout (waiting for response)
                upload_timeout = httpx.Timeout(
                    connect=30.0,  # Connection timeout: 30 seconds
                    read=600.0,  # Read timeout: 10 minutes (waiting for server response)
                    write=600.0,  # Write timeout: 10 minutes (uploading file data) - CRITICAL for large files
                    pool=30.0  # Pool timeout: 30 seconds
                )
                logger.debug(f"[File Upload] Using granular timeout: connect=30s, read=600s, write=600s, pool=30s")
            except ImportError:
                # Fallback to simple timeout if httpx not available
                upload_timeout = 600.0  # 10 minutes
                logger.debug(f"[File Upload] httpx not available, using simple timeout: {upload_timeout}s")
            
            # Use with_options to temporarily increase timeout for this specific call
            # This overrides the client's default timeout (300s) with a longer timeout for file uploads
            response = await self.client.with_options(timeout=upload_timeout).files.create(
                file=file_obj,
                purpose=purpose
            )

            file_id = response.id
            logger.info(f"Uploaded file '{filename}' to OpenAI, file_id: {file_id}")
            return file_id

        except Exception as e:
            logger.error(f"Failed to upload file to OpenAI: {e}")
            raise ProviderError(self.get_provider_name(), f"File upload failed: {str(e)}", original_error=e)

    async def _cleanup_uploaded_files(self, file_ids: List[str]) -> None:
        """Clean up uploaded files from OpenAI.

        Call this after the chat completion to remove temporary files.
        This method is request-scoped to ensure concurrent requests don't
        interfere with each other's file cleanup.

        Args:
            file_ids: List of file IDs to delete (scoped to the current request)
        """
        if not self.client or not file_ids:
            return

        for file_id in file_ids:
            try:
                await self.client.files.delete(file_id)
                logger.debug(f"Deleted uploaded file: {file_id}")
            except Exception as e:
                logger.warning(f"Failed to delete uploaded file {file_id}: {e}")

    async def _prepare_large_files(self, messages: List[Dict[str, Any]]) -> tuple:
        """Process messages and upload any large files to OpenAI Files API.

        Finds file blocks marked with _pending_upload and uploads them,
        replacing the placeholder with the actual file_id.

        Args:
            messages: Converted OpenAI-format messages

        Returns:
            Tuple of (messages, uploaded_file_ids) where:
            - messages: Messages with large files uploaded and file_ids filled in
            - uploaded_file_ids: List of file IDs uploaded in this request (for cleanup)
        """
        uploaded_file_ids: List[str] = []

        for msg in messages:
            content = msg.get("content")
            if not isinstance(content, list):
                continue

            for i, block in enumerate(content):
                if block.get("type") != "file":
                    continue

                file_info = block.get("file", {})
                if not file_info.get("_pending_upload"):
                    continue

                # Upload the file
                try:
                    b64_data = file_info.get("_base64_data", "")
                    filename = file_info.get("_filename", "document.pdf")
                    media_type = file_info.get("_media_type", "application/pdf")

                    # Decode base64 to bytes
                    file_bytes = base64.b64decode(b64_data)

                    # Upload to OpenAI
                    file_id = await self._upload_file_to_openai(file_bytes, filename)
                    uploaded_file_ids.append(file_id)

                    # Replace the placeholder with the actual file reference
                    content[i] = {
                        "type": "file",
                        "file": {
                            "file_id": file_id
                        }
                    }
                    logger.info(f"Large file '{filename}' uploaded successfully, file_id: {file_id}")

                except Exception as e:
                    logger.error(f"Failed to upload large file: {e}")
                    # Fallback: try inline anyway (may fail with API error)
                    file_data = f"data:{media_type};base64,{b64_data}"
                    content[i] = {
                        "type": "file",
                        "file": {
                            "filename": filename,
                            "file_data": file_data
                        }
                    }

        return messages, uploaded_file_ids

    def _convert_content_block(self, block: ContentBlock) -> Dict[str, Any]:
        """Convert a content block to OpenAI-compatible format.

        Args:
            block: A content block (TextContent, ImageContent, ImageUrlContent, etc.)

        Returns:
            Dict in OpenAI multimodal content format
        """
        if isinstance(block, TextContent):
            return {"type": "text", "text": block.text}

        elif isinstance(block, ImageUrlContent):
            # OpenAI expects image_url format
            result = {
                "type": "image_url",
                "image_url": {"url": block.image_url.url}
            }
            if block.image_url.detail:
                result["image_url"]["detail"] = block.image_url.detail
            return result

        elif isinstance(block, ImageContent):
            # Convert base64 image to OpenAI's data URL format
            data_url = f"data:{block.source.media_type};base64,{block.source.data}"
            return {
                "type": "image_url",
                "image_url": {"url": data_url}
            }

        elif isinstance(block, DocumentContent):
            # OpenAI supports PDF via type: "file" with file_data as data URL
            # https://platform.openai.com/docs/guides/pdf-files
            # For non-PDF documents, convert to text content
            if block.source.media_type == "application/pdf":
                # Use native PDF support
                filename = block.filename or "document.pdf"

                # Check if file size exceeds inline limit (4MB)
                # Base64 data size ≈ original size * 4/3
                decoded_size = len(block.source.data) * 3 // 4  # Approximate original size

                if decoded_size > MAX_INLINE_FILE_SIZE:
                    # Large file - mark for async upload via Files API
                    # The actual upload will be handled in _prepare_large_files()
                    logger.info(f"Document '{filename}' ({decoded_size / (1024*1024):.2f} MB) exceeds inline limit, will use Files API")
                    return {
                        "type": "file",
                        "file": {
                            "file_id": None,  # Will be filled by _prepare_large_files()
                            "_pending_upload": True,
                            "_base64_data": block.source.data,
                            "_media_type": block.source.media_type,
                            "_filename": filename
                        }
                    }
                else:
                    # Small file - use inline base64 data URL
                    # Format: data:application/pdf;base64,<base64_data>
                    file_data = f"data:{block.source.media_type};base64,{block.source.data}"
                    return {
                        "type": "file",
                        "file": {
                            "filename": filename,
                            "file_data": file_data
                        }
                    }
            else:
                # For non-PDF documents (text/plain, application/json, etc.),
                # decode the base64 content and include as text
                try:
                    decoded_bytes = base64.b64decode(block.source.data)
                    # Try to decode as UTF-8 text
                    try:
                        text_content = decoded_bytes.decode("utf-8")
                    except UnicodeDecodeError:
                        # If not valid UTF-8, return as base64 representation
                        text_content = f"[Binary document: {block.source.media_type}, size: {len(decoded_bytes)} bytes]"

                    # Include filename in text if available
                    if block.filename:
                        text_content = f"[Document: {block.filename} ({block.source.media_type})]\n\n{text_content}"

                    logger.debug(f"Converting non-PDF document ({block.source.media_type}) to text content for OpenAI")
                    return {"type": "text", "text": text_content}
                except Exception as e:
                    logger.warning(f"Failed to decode document content: {e}, converting to text placeholder")
                    return {
                        "type": "text",
                        "text": f"[Document: {block.filename or 'unknown'} ({block.source.media_type}) - Failed to decode]"
                    }

        elif isinstance(block, FileContent):
            # File content - include as text with path info (OpenAI doesn't support files directly)
            logger.warning(f"FileContent not fully supported, converting to text: {block.file_path}")
            return {"type": "text", "text": f"[File: {block.file_path}]"}

        elif isinstance(block, dict):
            # Already in dict format, pass through
            return block

        else:
            # Fallback for unknown content types
            logger.warning(f"Unknown content block type: {type(block)}")
            return {"type": "text", "text": str(block)}

    def _convert_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """Convert Message objects to OpenAI-compatible format with validation.

        Handles both text-only and multimodal messages seamlessly.
        """
        openai_messages = []

        for i, message in enumerate(messages):
            msg_dict = {"role": message.role}

            if message.content is not None:
                if isinstance(message.content, str):
                    # Simple text content
                    msg_dict["content"] = message.content
                elif isinstance(message.content, list):
                    # Multimodal content - convert each block
                    msg_dict["content"] = [
                        self._convert_content_block(block) for block in message.content
                    ]
                else:
                    # Fallback for unexpected types
                    msg_dict["content"] = str(message.content)

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
                    prev_content = fixed_messages[-1]["content"]
                    curr_content = current_msg["content"]
                    
                    # Handle different content types
                    if isinstance(prev_content, list) and isinstance(curr_content, list):
                        # Both are lists (multimodal) - merge lists
                        fixed_messages[-1]["content"] = prev_content + curr_content
                    elif isinstance(prev_content, list) and isinstance(curr_content, str):
                        # Previous is list, current is string - convert string to text object and append
                        fixed_messages[-1]["content"] = prev_content + [{"type": "text", "text": curr_content}]
                    elif isinstance(prev_content, str) and isinstance(curr_content, list):
                        # Previous is string, current is list - convert string to text object and prepend
                        fixed_messages[-1]["content"] = [{"type": "text", "text": prev_content}] + curr_content
                    else:
                        # Both are strings - simple concatenation
                        fixed_messages[-1]["content"] = prev_content + "\n" + curr_content
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
                # Defensive defaults for providers that return nulls
                func_name = getattr(tool_call.function, "name", None) or "unknown"
                func_args = getattr(tool_call.function, "arguments", None)
                if func_args is None:
                    func_args = "{}"

                tool_calls.append(ToolCall(
                    id=tool_call.id,
                    type=tool_call.type,
                    function=Function(
                        name=func_name,
                        arguments=func_args
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

        uploaded_file_ids: List[str] = []
        try:
            start_time = asyncio.get_event_loop().time()

            openai_messages = self._convert_messages(messages)

            # Handle large files by uploading to Files API (request-scoped)
            openai_messages, uploaded_file_ids = await self._prepare_large_files(openai_messages)

            # Extract parameters
            model = kwargs.get('model', self.model)
            max_tokens = kwargs.get('max_tokens', self.max_tokens)
            temperature = kwargs.get('temperature', self.temperature)

            tools = kwargs.get('tools')
            tool_choice = kwargs.get('tool_choice', 'auto')

            request_kwargs: Dict[str, Any] = {
                "model": model,
                "messages": openai_messages,
                "stream": False,
            }
            # Only add temperature for models that support it
            if self._supports_temperature(model):
                request_kwargs["temperature"] = temperature
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
        finally:
            # Clean up any uploaded files (request-scoped)
            await self._cleanup_uploaded_files(uploaded_file_ids)

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
        uploaded_file_ids: List[str] = []

        try:
            openai_messages = self._convert_messages(messages)

            # Handle large files by uploading to Files API (request-scoped)
            openai_messages, uploaded_file_ids = await self._prepare_large_files(openai_messages)

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
                "stream": True,
                "stream_options": {"include_usage": True},
            }
            # Only add temperature for models that support it
            if self._supports_temperature(model):
                request_kwargs["temperature"] = temperature
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
        finally:
            # Clean up any uploaded files (request-scoped)
            await self._cleanup_uploaded_files(uploaded_file_ids)

    async def completion(self, prompt: str, **kwargs) -> LLMResponse:
        """Send completion request to the provider."""
        # Convert to chat format
        messages = [Message(role="user", content=prompt)]
        return await self.chat(messages, **kwargs)

    async def chat_with_tools(self, messages: List[Message], tools: List[Dict], **kwargs) -> LLMResponse:
        """Send chat request with tools to the provider."""
        if not self.client:
            raise ProviderError(self.get_provider_name(), "Provider not initialized")

        uploaded_file_ids: List[str] = []
        try:
            start_time = asyncio.get_event_loop().time()

            openai_messages = self._convert_messages(messages)

            # Handle large files by uploading to Files API (request-scoped)
            openai_messages, uploaded_file_ids = await self._prepare_large_files(openai_messages)

            # Extract parameters
            model = kwargs.get('model', self.model)
            max_tokens = kwargs.get('max_tokens', self.max_tokens)
            temperature = kwargs.get('temperature', self.temperature)
            tool_choice = kwargs.get('tool_choice', 'auto')

            request_kwargs: Dict[str, Any] = {
                "model": model,
                "messages": openai_messages,
                "stream": False,
            }
            # Only add temperature for models that support it
            if self._supports_temperature(model):
                request_kwargs["temperature"] = temperature
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
        finally:
            # Clean up any uploaded files (request-scoped)
            await self._cleanup_uploaded_files(uploaded_file_ids)

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
