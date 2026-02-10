"""
Gemini Provider implementation for the unified LLM interface.
"""

import asyncio
import json
import os
import time
import uuid
from typing import List, Dict, Any, Optional, AsyncIterator
from logging import getLogger
from uuid import uuid4
import mimetypes

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    try:
        import requests
        REQUESTS_AVAILABLE = True
        HTTPX_AVAILABLE = False
    except ImportError:
        REQUESTS_AVAILABLE = False
        HTTPX_AVAILABLE = False

from google import genai
from google.genai import types

from spoon_ai.schema import (
    Message, ToolCall, Function, LLMResponseChunk,
    TextContent, ImageContent, ImageUrlContent, DocumentContent, FileContent, ContentBlock
)
import base64
import re
from spoon_ai.callbacks.manager import CallbackManager
from ..interface import LLMProviderInterface, LLMResponse, ProviderMetadata, ProviderCapability
from ..errors import ProviderError, AuthenticationError, RateLimitError, ModelNotFoundError, NetworkError
from ..registry import register_provider

logger = getLogger(__name__)


def _normalize_mime_type(mime_type: str) -> str:
    """Normalize MIME type for Gemini API compatibility.
    
    Only converts image/jpg to image/jpeg (Gemini doesn't accept image/jpg directly).
    All other MIME types are passed through without validation.
    
    Args:
        mime_type: Original MIME type string
        
    Returns:
        Normalized MIME type string (image/jpg -> image/jpeg, others unchanged)
    """
    # Only normalize image/jpg to image/jpeg for Gemini compatibility
    # All other MIME types are passed through as-is
    if mime_type == "image/jpg":
        return "image/jpeg"
    return mime_type


@register_provider("gemini", [
    ProviderCapability.CHAT,
    ProviderCapability.COMPLETION,
    ProviderCapability.STREAMING,
    ProviderCapability.TOOLS,
    ProviderCapability.IMAGE_GENERATION,
    ProviderCapability.VISION
])
class GeminiProvider(LLMProviderInterface):
    """Gemini provider implementation."""

    def __init__(self):
        self.client: Optional[genai.Client] = None
        self.config: Dict[str, Any] = {}
        self.model: str = ""
        self.max_tokens: int = 4096
        self.temperature: float = 0.3
        self.api_key: str = ""

    @staticmethod
    def _safe_get_response_text(response: Any) -> str:
        """Best-effort extraction of plain text from a Gemini response object.

        The google-genai SDK can sometimes return responses where candidates exist but
        the candidate content has empty/missing ``parts`` while the convenience
        ``response.text`` is still populated. We use a defensive strategy:
        - collect any part.text values if present
        - fall back to response.text if available
        """
        try:
            parts_text: List[str] = []
            if hasattr(response, "candidates") and response.candidates:
                candidate = response.candidates[0]
                content = getattr(candidate, "content", None)
                parts = getattr(content, "parts", None) if content is not None else None
                if parts:
                    for part in parts:
                        text = getattr(part, "text", None)
                        if text:
                            parts_text.append(text)
            joined = "\n".join(parts_text).strip()
            if joined:
                return joined

            text_attr = getattr(response, "text", None)
            if isinstance(text_attr, str) and text_attr.strip():
                return text_attr.strip()
        except Exception:
            pass
        return ""

    @staticmethod
    def _is_gemini_model_name(model: str) -> bool:
        if not isinstance(model, str):
            return False
        normalized = model.strip().lower()
        if not normalized:
            return False
        return "gemini" in normalized

    def _resolve_model_name(self, requested_model: Optional[str]) -> str:
        """Resolve requested model name for Gemini.

        Graph snippets/examples may pass OpenAI/Anthropic model names (e.g. "gpt-4")
        even when the runtime provider is Gemini. In that case we fall back to the
        configured Gemini model to avoid hard failures.
        """
        if requested_model and self._is_gemini_model_name(requested_model):
            return requested_model.strip()
        if requested_model and not self._is_gemini_model_name(requested_model):
            logger.info(
                "Gemini provider received non-Gemini model '%s'; falling back to '%s'",
                requested_model,
                self.model,
            )
        return self.model

    @staticmethod
    def _thinking_budget_min_for_model(model: str) -> Optional[int]:
        """Return minimum thinking_budget for models that support/require thinking_config."""
        if not isinstance(model, str):
            return None
        normalized = model.strip().lower()
        if not normalized:
            return None

        # Gemini 3 preview: must be in thinking mode; small budgets still work.
        if "gemini-3" in normalized:
            return 24

        # Gemini 2.5 Pro: supports thinking_config but rejects small budgets.
        if "gemini-2.5-pro" in normalized:
            return 128

        return None

    @staticmethod
    def _min_output_tokens_for_model(model: str) -> int:
        """Return a safe minimum max_output_tokens for models that can otherwise return empty output.

        Empirically, some Gemini "pro" / thinking-oriented models may return empty
        visible content when max_output_tokens is too small (even for short answers).
        """
        if not isinstance(model, str):
            return 0
        normalized = model.strip().lower()
        if not normalized:
            return 0

        # Gemini 3 preview (thinking) and Gemini 2.5 Pro frequently need a higher
        # output budget to include visible text (otherwise finish_reason=MAX_TOKENS and text=None).
        if "gemini-3" in normalized or "gemini-2.5-pro" in normalized:
            return 256

        return 0

    def _apply_thinking_defaults(
        self,
        *,
        model: str,
        requested_max_tokens: int,
        kwargs: Dict[str, Any],
    ) -> tuple[int, Optional[types.ThinkingConfig]]:
        """Apply safe defaults for Gemini models to avoid empty outputs."""
        max_tokens = requested_max_tokens

        min_output_tokens = self._min_output_tokens_for_model(model)
        if min_output_tokens and max_tokens < min_output_tokens:
            logger.info(
                "Gemini model '%s' requested max_tokens=%s; bumping to %s to avoid empty output",
                model,
                requested_max_tokens,
                min_output_tokens,
            )
            max_tokens = min_output_tokens

        min_thinking_budget = self._thinking_budget_min_for_model(model)
        if min_thinking_budget is None:
            return max_tokens, None

        # Allow callers to pass an explicit ThinkingConfig or thinking_budget.
        thinking_cfg = kwargs.get("thinking_config")
        if isinstance(thinking_cfg, dict):
            try:
                thinking_cfg = types.ThinkingConfig(**thinking_cfg)
            except Exception:
                thinking_cfg = None
        if isinstance(thinking_cfg, types.ThinkingConfig):
            # Normalize invalid/empty budgets for known models.
            budget = getattr(thinking_cfg, "thinking_budget", None)
            if budget is None:
                budget_int = min_thinking_budget
            else:
                try:
                    budget_int = int(budget)
                except Exception:
                    budget_int = min_thinking_budget

            if budget_int < min_thinking_budget:
                budget_int = min_thinking_budget

            return max_tokens, types.ThinkingConfig(
                include_thoughts=getattr(thinking_cfg, "include_thoughts", None),
                thinking_level=getattr(thinking_cfg, "thinking_level", None),
                thinking_budget=budget_int,
            )

        thinking_budget = kwargs.get("thinking_budget")
        if thinking_budget is None:
            # Default tuned to reliably yield visible text output.
            thinking_budget = 32 if min_thinking_budget <= 32 else min_thinking_budget
        try:
            thinking_budget_int = int(thinking_budget)
        except Exception:
            thinking_budget_int = 32 if min_thinking_budget <= 32 else min_thinking_budget

        # Enforce model-specific minimums.
        if thinking_budget_int < min_thinking_budget:
            thinking_budget_int = min_thinking_budget

        return max_tokens, types.ThinkingConfig(thinking_budget=thinking_budget_int)

    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the Gemini provider with configuration."""
        try:
            self.config = config
            self.model = config.get('model', 'models/gemini-3-pro-preview')
            self.max_tokens = config.get('max_tokens', 4096)
            self.temperature = config.get('temperature', 0.3)

            api_key = config.get('api_key')
            if not api_key:
                raise AuthenticationError("gemini", context={"config": config})

            # Keep api_key only; create/close clients per request to avoid
            # shutdown warnings from google-genai about pending aclose tasks.
            self.api_key = str(api_key)
            self.client = None

            logger.info(f"Gemini provider initialized with model: {self.model}")

        except Exception as e:
            if isinstance(e, (AuthenticationError, ProviderError)):
                raise
            raise ProviderError("gemini", f"Failed to initialize: {str(e)}", original_error=e)

    def _convert_content_block_to_part(self, block: ContentBlock) -> types.Part:
        """Convert a content block to a Gemini Part.

        Args:
            block: A content block (TextContent, ImageContent, ImageUrlContent, etc.)

        Returns:
            Gemini types.Part object
        """
        if isinstance(block, TextContent):
            return types.Part.from_text(text=block.text)

        elif isinstance(block, ImageContent):
            # Base64 image data - convert to bytes
            try:
                image_data = base64.b64decode(block.source.data)
            except Exception as e:
                raise ValueError(
                    f"Invalid base64 image data: {str(e)}. "
                    f"Please ensure the image data is properly base64 encoded."
                )
            # Normalize MIME type (e.g., image/jpg -> image/jpeg)
            normalized_mime_type = _normalize_mime_type(block.source.media_type)
            return types.Part.from_bytes(
                data=image_data,
                mime_type=normalized_mime_type
            )

        elif isinstance(block, ImageUrlContent):
            url = block.image_url.url
            # Check if it's a data URL (base64)
            if url.startswith("data:"):
                # Parse data URL: data:image/png;base64,<data>
                # More robust regex: case-insensitive, validates base64 characters
                match = re.match(r"data:([^;]+);base64,([A-Za-z0-9+/=]+)", url, re.IGNORECASE)
                if not match:
                    raise ValueError(f"Invalid data URL format: {url[:100]}...")
                mime_type, b64_data = match.groups()
                # Strip whitespace from base64 data
                b64_data = b64_data.strip()
                try:
                    image_data = base64.b64decode(b64_data)
                except Exception as e:
                    raise ValueError(
                        f"Invalid base64 data in data URL: {str(e)}. "
                        f"Please ensure the data URL contains valid base64 encoded image data."
                    )
                # Normalize MIME type (e.g., image/jpg -> image/jpeg)
                normalized_mime_type = _normalize_mime_type(mime_type)
                return types.Part.from_bytes(
                    data=image_data,
                    mime_type=normalized_mime_type
                )
            # For regular URLs, download the image and convert to bytes
            if not HTTPX_AVAILABLE and not REQUESTS_AVAILABLE:
                logger.error(
                    "httpx or requests is required to download external image URLs. "
                    "Please install one with: pip install httpx (or pip install requests)"
                )
                raise ValueError(
                    f"Cannot process external image URL: {url}. "
                    "httpx or requests is required but not installed."
                )
            
            try:
                # Download the image using available library
                if HTTPX_AVAILABLE:
                    # Use httpx (preferred, supports async but we use sync here)
                    with httpx.Client(timeout=10.0) as client:
                        response = client.get(url)
                        response.raise_for_status()
                        image_bytes = response.content
                        content_type = response.headers.get("Content-Type")
                else:
                    # Fallback to requests
                    import requests
                    response = requests.get(url, timeout=10.0)
                    response.raise_for_status()
                    image_bytes = response.content
                    content_type = response.headers.get("Content-Type")
                
                # Detect MIME type from URL or Content-Type header
                if not content_type:
                    mime_type, _ = mimetypes.guess_type(url)
                    content_type = mime_type or "image/jpeg"
                
                # Extract MIME type (remove charset if present)
                mime_type = content_type.split(";")[0].strip()
                # Normalize MIME type (e.g., image/jpg -> image/jpeg)
                normalized_mime_type = _normalize_mime_type(mime_type)
                
                logger.debug(f"Downloaded image from URL: {url[:50]}... (size: {len(image_bytes)} bytes, type: {normalized_mime_type})")
                
                return types.Part.from_bytes(
                    data=image_bytes,
                    mime_type=normalized_mime_type
                )
            except Exception as e:
                logger.error(f"Failed to download image from URL {url}: {e}")
                raise ValueError(f"Cannot process image URL: {url}. Download failed: {e}")

        elif isinstance(block, DocumentContent):
            # Gemini supports PDFs natively, but other document types should be converted to text
            # Check if it's a PDF
            if block.source.media_type == "application/pdf":
                # Use native PDF support
                document_data = base64.b64decode(block.source.data)
                return types.Part.from_bytes(
                    data=document_data,
                    mime_type=block.source.media_type
                )
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
                    
                    logger.debug(f"Converting non-PDF document ({block.source.media_type}) to text content for Gemini")
                    return types.Part.from_text(text=text_content)
                except Exception as e:
                    logger.warning(f"Failed to decode document content: {e}, converting to text placeholder")
                    return types.Part.from_text(
                        text=f"[Document: {block.filename or 'unknown'} ({block.source.media_type}) - Failed to decode]"
                    )

        elif isinstance(block, FileContent):
            # File content - include as text (Gemini doesn't support files directly without upload)
            logger.warning(f"FileContent not fully supported, converting to text: {block.file_path}")
            return types.Part.from_text(text=f"[File: {block.file_path}]")

        elif isinstance(block, dict):
            # Handle dict format content blocks
            block_type = block.get("type", "text")
            if block_type == "text":
                return types.Part.from_text(text=block.get("text", ""))
            elif block_type == "image":
                source = block.get("source", {})
                if source.get("type") == "base64":
                    image_data = base64.b64decode(source.get("data", ""))
                    # Normalize MIME type (e.g., image/jpg -> image/jpeg)
                    normalized_mime_type = _normalize_mime_type(source.get("media_type", "image/png"))
                    return types.Part.from_bytes(
                        data=image_data,
                        mime_type=normalized_mime_type
                    )
            elif block_type == "image_url":
                url = block.get("image_url", {}).get("url", "")
                if url.startswith("data:"):
                    match = re.match(r"data:([^;]+);base64,(.+)", url)
                    if match:
                        mime_type, b64_data = match.groups()
                        image_data = base64.b64decode(b64_data)
                        # Normalize MIME type (e.g., image/jpg -> image/jpeg)
                        normalized_mime_type = _normalize_mime_type(mime_type)
                        return types.Part.from_bytes(
                            data=image_data,
                            mime_type=normalized_mime_type
                        )
                return types.Part.from_text(text=f"[Image URL: {url}]")
            return types.Part.from_text(text=str(block))

        else:
            # Fallback for unknown content types
            logger.warning(f"Unknown content block type: {type(block)}")
            return types.Part.from_text(text=str(block))

    def _convert_message_content_to_parts(self, content) -> List[types.Part]:
        """Convert message content to list of Gemini Parts.

        Args:
            content: Message content (str or List[ContentBlock])

        Returns:
            List of Gemini types.Part objects
        """
        if content is None:
            return [types.Part.from_text(text="")]
        if isinstance(content, str):
            return [types.Part.from_text(text=content)]
        elif isinstance(content, list):
            return [self._convert_content_block_to_part(block) for block in content]
        else:
            return [types.Part.from_text(text=str(content))]

    def _convert_messages(self, messages: List[Message]) -> tuple[Optional[str], Any]:
        """Convert Message objects to Gemini format for simple chat.

        Handles both text-only and multimodal messages seamlessly.
        """
        system_content = ""
        user_parts = []

        # Extract system messages
        for message in messages:
            if message.role == "system":
                msg_text = message.text_content if message.is_multimodal else message.content
                if system_content:
                    system_content += " " + msg_text
                else:
                    system_content = msg_text or ""

        # Get the last user message - can be text or multimodal
        for message in reversed(messages):
            if message.role == "user":
                user_parts = self._convert_message_content_to_parts(message.content)
                break

        # If no user message found, use a default
        if not user_parts:
            user_parts = [types.Part.from_text(text="Hello")]

        return system_content, user_parts

    def _convert_messages_for_tools(self, messages: List[Message]) -> tuple[Optional[str], List]:
        """Convert Message objects to Gemini format for tool calling.

        Handles both text-only and multimodal messages seamlessly.
        """
        system_content = ""
        gemini_messages = []

        for message in messages:
            if message.role == "system":
                msg_text = message.text_content if message.is_multimodal else message.content
                if system_content:
                    system_content += " " + msg_text
                else:
                    system_content = msg_text or ""
            elif message.role == "user":
                # Handle both text and multimodal user messages
                parts = self._convert_message_content_to_parts(message.content)
                gemini_messages.append(types.Content(
                    role="user",
                    parts=parts
                ))
            elif message.role == "assistant":
                if message.tool_calls:
                    # Gemini tool-calling sequencing is strict:
                    # function_call turn must be standalone and followed immediately
                    # by a function_response turn. Mixing assistant text + function_call
                    # in the same turn can trigger INVALID_ARGUMENT ordering errors.

                    # Emit assistant text as a separate model turn first (if any)
                    if message.content:
                        gemini_messages.append(types.Content(
                            role="model",
                            parts=[types.Part.from_text(text=message.content)]
                        ))

                    # Emit each function_call as its own model turn
                    for tool_call in message.tool_calls:
                        args = tool_call.function.get_arguments_dict()
                        gemini_messages.append(types.Content(
                            role="model",
                            parts=[types.Part.from_function_call(
                                name=tool_call.function.name,
                                args=args
                            )]
                        ))
                else:
                    gemini_messages.append(types.Content(
                        role="model",
                        parts=[types.Part.from_text(text=message.content)]
                    ))
            elif message.role == "tool":
                # Convert tool response to Gemini format
                # Gemini requires a non-empty name for function_response
                tool_name = message.name
                if not tool_name:
                    # Fallback: try to extract tool name from tool_call_id or use a default
                    if message.tool_call_id:
                        # Try to find the corresponding tool call in previous messages
                        for prev_msg in reversed(messages):
                            if prev_msg.role == "assistant" and prev_msg.tool_calls:
                                for tool_call in prev_msg.tool_calls:
                                    if tool_call.id == message.tool_call_id:
                                        tool_name = tool_call.function.name
                                        break
                                if tool_name:
                                    break

                    # If still no name found, use a default
                    if not tool_name:
                        tool_name = "unknown_function"

                gemini_messages.append(types.Content(
                    role="user",
                    parts=[types.Part.from_function_response(
                        name=tool_name,
                        response={"result": message.content}
                    )]
                ))

        return system_content, gemini_messages

    def _sanitize_gemini_schema(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Remove fields that are not supported by Gemini function declarations."""
        if not isinstance(schema, dict):
            return schema

        clean_schema = schema.copy()
        # Remove fields that Gemini API doesn't support
        for key in ["additionalProperties", "additional_properties", "title", "$schema"]:
            clean_schema.pop(key, None)

        if "properties" in clean_schema and isinstance(clean_schema["properties"], dict):
            clean_schema["properties"] = {
                name: self._sanitize_gemini_schema(value) for name, value in clean_schema["properties"].items()
            }
        if "items" in clean_schema:
            clean_schema["items"] = self._sanitize_gemini_schema(clean_schema["items"])

        return clean_schema

    def _convert_tools_to_gemini(self, tools: List[Dict]) -> List:
        """Convert OpenAI/Anthropic tool format to Gemini format."""
        gemini_tools = []

        if tools:
            function_declarations = []
            for tool in tools:
                if 'function' in tool:
                    func = tool['function']
                    parameters = self._sanitize_gemini_schema(func.get('parameters', {}))
                    function_declarations.append(types.FunctionDeclaration(
                        name=func.get('name'),
                        description=func.get('description'),
                        parameters=parameters
                    ))

            if function_declarations:
                gemini_tools.append(types.Tool(function_declarations=function_declarations))

        return gemini_tools

    async def chat(self, messages: List[Message], **kwargs) -> LLMResponse:
        """Send chat request to Gemini."""
        if not self.api_key:
            raise ProviderError("gemini", "Provider not initialized")

        try:
            start_time = asyncio.get_event_loop().time()

            system_content, user_parts = self._convert_messages(messages)

            # Extract parameters
            model = self._resolve_model_name(kwargs.get('model'))
            max_tokens_raw = kwargs.get('max_tokens', self.max_tokens)
            try:
                max_tokens = int(max_tokens_raw)
            except Exception:
                max_tokens = int(self.max_tokens)
            temperature = kwargs.get('temperature', self.temperature)
            response_modalities = kwargs.get('response_modalities')

            max_tokens, thinking_config = self._apply_thinking_defaults(
                model=model,
                requested_max_tokens=max_tokens,
                kwargs=kwargs,
            )

            # Build request content - user_parts is already a list of Parts
            contents = user_parts

            # Generate configuration
            generate_config = types.GenerateContentConfig(
                max_output_tokens=max_tokens,
                temperature=temperature
            )

            if thinking_config is not None:
                generate_config.thinking_config = thinking_config

            # Add system instruction if available
            if system_content:
                generate_config.system_instruction = system_content

            # Add response modalities if specified
            if response_modalities:
                generate_config.response_modalities = response_modalities

            # Check for structured output requirements
            if "IMPORTANT INSTRUCTION" in system_content and "JSON format" in system_content:
                schema = {
                    "type": "object",
                    "properties": {
                        "response": {
                            "type": "string",
                            "description": "Response content to the user"
                        },
                        "should_hand_off": {
                            "type": "boolean",
                            "description": "Whether the conversation should be handed off to a design expert"
                        }
                    },
                    "required": ["response", "should_hand_off"],
                    "propertyOrdering": ["should_hand_off", "response"]
                }
                generate_config.response_schema = schema
                generate_config.response_mime_type = 'application/json'

            client = genai.Client(api_key=self.api_key)
            try:
                # Send request
                response = client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=generate_config
                )
            finally:
                # Close both sync + async clients to prevent pending-task warnings.
                try:
                    client.close()
                except Exception:
                    pass
                try:
                    await client.aio.aclose()
                except Exception:
                    pass

            duration = asyncio.get_event_loop().time() - start_time
            return self._convert_response(response, duration, model=model)

        except Exception as e:
            await self._handle_error(e)

    async def chat_stream(self, messages: List[Message],callbacks: Optional[List] = None, **kwargs) -> AsyncIterator[LLMResponseChunk]:
        """Send streaming chat request to Gemini with callback support.
        Yields:
            LLMResponseChunk: Structured streaming response chunks
        """
        if not self.api_key:
            raise ProviderError("gemini", "Provider not initialized")

        # Create callback manager
        callback_manager = CallbackManager.from_callbacks(callbacks)
        run_id = uuid4()

        try:
            system_content, user_parts = self._convert_messages(messages)

            # Extract parameters
            model = self._resolve_model_name(kwargs.get('model'))
            max_tokens_raw = kwargs.get('max_tokens', self.max_tokens)
            try:
                max_tokens = int(max_tokens_raw)
            except Exception:
                max_tokens = int(self.max_tokens)
            temperature = kwargs.get('temperature', self.temperature)

            max_tokens, thinking_config = self._apply_thinking_defaults(
                model=model,
                requested_max_tokens=max_tokens,
                kwargs=kwargs,
            )

            # Trigger on_llm_start callback
            await callback_manager.on_llm_start(run_id=run_id,messages=messages,model=model,provider="gemini")

            # Build request content - user_parts is already a list of Parts
            contents = user_parts

            # Generate configuration
            generate_config = types.GenerateContentConfig(
                max_output_tokens=max_tokens,
                temperature=temperature
            )

            if thinking_config is not None:
                generate_config.thinking_config = thinking_config

            if system_content:
                generate_config.system_instruction = system_content

            # Process streaming response
            full_content = ""
            chunk_index = 0
            finish_reason = None
            usage_data = None

            # Send streaming request
            # Filter out parameters that generate_content_stream doesn't accept
            filtered_kwargs = {k: v for k, v in kwargs.items()
                               if k not in ['model', 'max_tokens', 'temperature', 'callbacks', 'timeout']}
            client = genai.Client(api_key=self.api_key)
            try:
                stream = client.models.generate_content_stream(
                    model=model,
                    contents=contents,
                    config=generate_config,
                    **filtered_kwargs
                )

                for part_response in stream:
                    chunk = ""
                    try:
                        if (
                            hasattr(part_response, "candidates")
                            and part_response.candidates
                            and getattr(part_response.candidates[0], "content", None) is not None
                            and getattr(part_response.candidates[0].content, "parts", None)
                        ):
                            parts = part_response.candidates[0].content.parts
                            chunk = "".join(
                                [p.text for p in parts if getattr(p, "text", None)]
                            )
                    except Exception:
                        chunk = ""

                    # Fallback: some SDK responses expose streaming text via `part_response.text`
                    if not chunk:
                        maybe_text = getattr(part_response, "text", None)
                        if isinstance(maybe_text, str):
                            chunk = maybe_text

                    if not chunk:
                        continue

                    full_content += chunk

                    # Trigger on_llm_new_token callback
                    await callback_manager.on_llm_new_token(
                        token=chunk,
                        run_id=run_id
                    )

                    # Extract finish reason
                    if (
                        hasattr(part_response, "candidates")
                        and part_response.candidates
                        and part_response.candidates[0].finish_reason
                    ):
                        finish_reason = str(part_response.candidates[0].finish_reason)

                    # Extract usage stats if available
                    if hasattr(part_response, 'usage_metadata') and part_response.usage_metadata:
                        usage_data = {
                            "prompt_tokens": part_response.usage_metadata.prompt_token_count,
                            "completion_tokens": part_response.usage_metadata.candidates_token_count,
                            "total_tokens": part_response.usage_metadata.total_token_count
                        }

                    # Build response chunk
                    response_chunk = LLMResponseChunk(
                        content=full_content,
                        delta=chunk,
                        provider="gemini",
                        model=model,
                        finish_reason=finish_reason,
                        tool_calls=[],
                        usage=usage_data,
                        metadata={
                            "chunk_index": chunk_index,
                            "finish_reason": finish_reason
                        },
                        chunk_index=chunk_index
                    )
                    chunk_index += 1
                    yield response_chunk
            finally:
                try:
                    client.close()
                except Exception:
                    pass
                try:
                    await client.aio.aclose()
                except Exception:
                    pass

            # Trigger on_llm_end callback
            final_response = LLMResponse(
                content=full_content,
                provider="gemini",
                model=model,
                finish_reason=finish_reason or "stop",
                native_finish_reason=finish_reason or "stop",
                tool_calls=[],
                usage=usage_data,
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
        """Send completion request to Gemini."""
        # Convert to chat format
        messages = [Message(role="user", content=prompt)]
        return await self.chat(messages, **kwargs)

    async def chat_with_tools(self, messages: List[Message], tools: List[Dict], **kwargs) -> LLMResponse:
        """Send chat request with tools to Gemini using native function calling."""
        if not self.api_key:
            raise ProviderError("gemini", "Provider not initialized")

        try:
            start_time = asyncio.get_event_loop().time()

            # Convert messages to Gemini format
            system_content, gemini_messages = self._convert_messages_for_tools(messages)

            # Convert tools to Gemini format
            gemini_tools = self._convert_tools_to_gemini(tools)

            # Extract parameters
            model = self._resolve_model_name(kwargs.get('model'))
            max_tokens_raw = kwargs.get('max_tokens', self.max_tokens)
            try:
                max_tokens = int(max_tokens_raw)
            except Exception:
                max_tokens = int(self.max_tokens)
            temperature = kwargs.get('temperature', self.temperature)

            max_tokens, thinking_config = self._apply_thinking_defaults(
                model=model,
                requested_max_tokens=max_tokens,
                kwargs=kwargs,
            )

            # Generate configuration
            generate_config = types.GenerateContentConfig(
                max_output_tokens=max_tokens,
                temperature=temperature,
                tools=gemini_tools
            )

            if thinking_config is not None:
                generate_config.thinking_config = thinking_config

            # Add system instruction if available
            if system_content:
                generate_config.system_instruction = system_content

            # Send request
            client = genai.Client(api_key=self.api_key)
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=gemini_messages,
                    config=generate_config
                )
            finally:
                try:
                    client.close()
                except Exception:
                    pass
                try:
                    await client.aio.aclose()
                except Exception:
                    pass

            duration = asyncio.get_event_loop().time() - start_time
            return self._convert_tool_response(response, duration, model=model)

        except Exception as e:
            await self._handle_error(e)

    @staticmethod
    def _clean_json_response(content: str) -> str:
        """Clean JSON response by removing markdown code blocks and extra text.
        
        Gemini sometimes returns JSON wrapped in markdown code blocks or with extra text.
        This method extracts the JSON content.
        """
        if not content:
            return content
        
        import re
        
        # Remove markdown code block markers if present
        cleaned = re.sub(r'^```(?:json)?\s*', '', content, flags=re.MULTILINE)
        cleaned = re.sub(r'\s*```$', '', cleaned, flags=re.MULTILINE)
        cleaned = cleaned.strip()
        
        # Try to find JSON object in the response (handles cases with extra text)
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', cleaned)
        if json_match:
            # If JSON is found, return it (user can parse it themselves if needed)
            # But for now, we keep the original content and let users handle JSON extraction
            # This is a conservative approach to avoid breaking existing code
            pass
        
        return cleaned


    def _convert_response(self, response, duration: float, *, model: Optional[str] = None) -> LLMResponse:
        """Convert Gemini response to standardized LLMResponse."""
        content = ""
        response_text = ""
        image_paths = []
        tool_calls = []

        # Check if there are candidate results
        if hasattr(response, "candidates") and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, "content") and candidate.content:
                # Iterate through all parts
                parts = getattr(candidate.content, "parts", None) or []
                for part in parts:
                    # Check if there is text content
                    if hasattr(part, "text") and part.text:
                        if content:
                            content += "\n" + part.text
                        else:
                            content = part.text
                        response_text = content
                    # Check if there is image content
                    elif hasattr(part, "inline_data"):
                        # Save image data
                        image_data = {
                            "mime_type": getattr(part.inline_data, "mime_type", "image/jpeg"),
                            "data": getattr(part.inline_data, "data", b"")
                        }

                        # Save image to local storage
                        output_dir = "runtime-image-output"
                        os.makedirs(output_dir, exist_ok=True)

                        timestamp = int(time.time())
                        ext = image_data['mime_type'].split('/')[-1]
                        filename = f"gemini_image_{timestamp}.{ext}"
                        filepath = os.path.join(output_dir, filename)

                        try:
                            with open(filepath, "wb") as f:
                                f.write(image_data['data'])

                            image_paths.append({
                                "filepath": filepath,
                                "mime_type": image_data['mime_type']
                            })

                            # Add to tool calls for compatibility
                            tool_calls.append({
                                "type": "image",
                                "image": {
                                    "mime_type": image_data["mime_type"],
                                    "data": image_data["data"]
                                }
                            })

                            # Update content
                            if content:
                                content += f"\n[Generated image saved to: {filepath}]"
                            else:
                                content = f"[Generated image saved to: {filepath}]"
                            response_text = content

                        except Exception as e:
                            logger.error(f"Failed to save image: {str(e)}")

        # If no text content was obtained, fall back to response.text (if available).
        # Some Gemini SDK responses keep `response.text` populated even when
        # `candidates[0].content.parts` is empty.
        if not content:
            fallback_text = self._safe_get_response_text(response)
            if fallback_text:
                response_text = fallback_text
                content = fallback_text

        # Default content for image responses
        if not content and image_paths:
            content = "【Image response】"

        # Clean JSON responses (remove markdown code blocks and extra text)
        # This helps when LLM is asked to return JSON but wraps it in markdown
        cleaned_content = self._clean_json_response(content or "")

        return LLMResponse(
            content=cleaned_content,
            provider="gemini",
            model=model or self.model,
            finish_reason="stop",  # Gemini doesn't provide detailed finish reasons
            native_finish_reason="stop",
            tool_calls=[],  # Will be populated by _convert_tool_response for tool calls
            duration=duration,
            metadata={
                "image_paths": image_paths,
                "has_images": len(image_paths) > 0
            }
        )

    def _convert_tool_response(self, response, duration: float, *, model: Optional[str] = None) -> LLMResponse:
        """Convert Gemini tool response to standardized LLMResponse."""
        content = ""
        tool_calls = []
        finish_reason = "stop"

        # Check if there are candidate results
        if hasattr(response, "candidates") and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, "content") and candidate.content:
                # Iterate through all parts
                parts = getattr(candidate.content, "parts", None) or []
                for part in parts:
                    # Check if there is text content
                    if hasattr(part, "text") and part.text:
                        if content:
                            content += "\n" + part.text
                        else:
                            content = part.text
                    # Check if there is a function call
                    elif hasattr(part, "function_call") and part.function_call:
                        # Convert Gemini function call to our ToolCall format
                        function_call = part.function_call

                        # Generate a unique ID for the tool call
                        import uuid
                        tool_call_id = f"call_{uuid.uuid4().hex[:8]}"

                        # Convert arguments to JSON string
                        import json
                        arguments_json = json.dumps(function_call.args) if function_call.args else "{}"

                        tool_call = ToolCall(
                            id=tool_call_id,
                            type="function",
                            function=Function(
                                name=function_call.name,
                                arguments=arguments_json
                            )
                        )
                        tool_calls.append(tool_call)
                        finish_reason = "tool_calls"

        # If no content was obtained, fall back to response.text (if available).
        if not content:
            fallback_text = self._safe_get_response_text(response)
            if fallback_text:
                content = fallback_text

        return LLMResponse(
            content=content or "",
            provider="gemini",
            model=model or self.model,
            finish_reason=finish_reason,
            native_finish_reason=finish_reason,
            tool_calls=tool_calls,
            duration=duration,
            metadata={}
        )

    def get_metadata(self) -> ProviderMetadata:
        """Get Gemini provider metadata."""
        return ProviderMetadata(
            name="gemini",
            version="1.0.0",
            capabilities=[
                ProviderCapability.CHAT,
                ProviderCapability.COMPLETION,
                ProviderCapability.STREAMING,
                ProviderCapability.TOOLS,
                ProviderCapability.IMAGE_GENERATION,
                ProviderCapability.VISION
            ],
            max_tokens=self.max_tokens,
            supports_system_messages=True,
            rate_limits={
                "requests_per_minute": 60,
                "tokens_per_minute": 32000
            }
        )

    async def health_check(self) -> bool:
        """Check if Gemini provider is healthy."""
        if not self.client:
            return False

        try:
            # Simple test request
            contents = [types.Part.from_text(text="test")]
            config = types.GenerateContentConfig(max_output_tokens=1)

            response = self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=config
            )
            return True
        except Exception as e:
            logger.warning(f"Gemini health check failed: {e}")
            return False

    async def cleanup(self) -> None:
        """Cleanup Gemini provider resources."""
        # No persistent client is kept; nothing to clean up.
        self.client = None
        self.api_key = ""

        logger.info("Gemini provider cleaned up")

    async def _handle_error(self, error: Exception) -> None:
        """Handle and convert Gemini errors to standardized errors."""
        error_str = str(error).lower()

        if "authentication" in error_str or "api key" in error_str:
            raise AuthenticationError("gemini", context={"original_error": str(error)})
        elif "quota" in error_str or "rate limit" in error_str:
            raise RateLimitError("gemini", context={"original_error": str(error)})
        elif "model" in error_str and "not found" in error_str:
            raise ModelNotFoundError("gemini", self.model, context={"original_error": str(error)})
        elif "timeout" in error_str or "connection" in error_str:
            raise NetworkError("gemini", "Network error", original_error=error)
        elif "unsupported mime type" in error_str.lower() or "mime type" in error_str.lower():
            # Extract the MIME type from error message if possible
            mime_match = None
            import re
            mime_match = re.search(r'mime type[:\s]+([^\s\'"]+)', error_str, re.IGNORECASE)
            mime_type = mime_match.group(1) if mime_match else "unknown"
            raise ProviderError(
                "gemini",
                f"Gemini API does not support MIME type: {mime_type}. "
                f"This is a limitation of the Gemini API, not the spoon-core framework. "
            )
        else:
            raise ProviderError("gemini", f"Request failed: {str(error)}", original_error=error)
