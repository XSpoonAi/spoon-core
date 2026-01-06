"""
Anthropic Provider implementation for the unified LLM interface.
"""

import asyncio
import json
from typing import List, Dict, Any, Optional, AsyncIterator
from logging import getLogger
from uuid import uuid4

from anthropic import AsyncAnthropic
from httpx import AsyncClient

from spoon_ai.schema import (
    Message, ToolCall, Function, LLMResponseChunk,
    TextContent, ImageContent, ImageUrlContent, DocumentContent, FileContent, ContentBlock
)
from spoon_ai.callbacks.manager import CallbackManager
from ..interface import LLMProviderInterface, LLMResponse, ProviderMetadata, ProviderCapability
from ..errors import ProviderError, AuthenticationError, RateLimitError, ModelNotFoundError, NetworkError
from ..registry import register_provider

logger = getLogger(__name__)


@register_provider("anthropic", [
    ProviderCapability.CHAT,
    ProviderCapability.COMPLETION,
    ProviderCapability.TOOLS,
    ProviderCapability.STREAMING
])
class AnthropicProvider(LLMProviderInterface):
    """Anthropic provider implementation."""
    
    def __init__(self):
        self.client: Optional[AsyncAnthropic] = None
        self.config: Dict[str, Any] = {}
        self.model: str = ""
        self.max_tokens: int = 4096
        self.temperature: float = 0.3
        self.enable_prompt_cache: bool = True
        self.cache_metrics = {
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "total_input_tokens": 0
        }
        
    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the Anthropic provider with configuration."""
        try:
            self.config = config
            self.model = config.get('model', 'claude-sonnet-4-20250514')
            self.max_tokens = config.get('max_tokens', 4096)
            self.temperature = config.get('temperature', 0.3)
            self.enable_prompt_cache = config.get('enable_prompt_cache', True)
            
            api_key = config.get('api_key')
            if not api_key:
                raise AuthenticationError("anthropic", context={"config": config})
            
            timeout = config.get('timeout', 30)
            http_client = AsyncClient(follow_redirects=True, timeout=timeout)
            
            self.client = AsyncAnthropic(
                api_key=api_key,
                http_client=http_client
            )
            
            logger.info(f"Anthropic provider initialized with model: {self.model}")
            
        except Exception as e:
            if isinstance(e, (AuthenticationError, ProviderError)):
                raise
            raise ProviderError("anthropic", f"Failed to initialize: {str(e)}", original_error=e)
    
    def _convert_content_block(self, block: ContentBlock) -> Dict[str, Any]:
        """Convert a content block to Anthropic-compatible format.

        Args:
            block: A content block (TextContent, ImageContent, ImageUrlContent, etc.)

        Returns:
            Dict in Anthropic multimodal content format
        """
        if isinstance(block, TextContent):
            return {"type": "text", "text": block.text}

        elif isinstance(block, ImageContent):
            # Anthropic's native base64 image format
            # Validate base64 format
            import re
            if not re.match(r'^[A-Za-z0-9+/]*={0,2}$', block.source.data):
                raise ValueError(f"Invalid base64 format for image data")
            
            return {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": block.source.media_type,
                    "data": block.source.data
                }
            }

        elif isinstance(block, ImageUrlContent):
            # Convert image URL to Anthropic format
            url = block.image_url.url
            # Check if it's a data URL (base64)
            if url.startswith("data:"):
                # Parse data URL: data:image/png;base64,<data>
                # More robust regex: case-insensitive, validates base64 characters
                import re
                match = re.match(r"data:([^;]+);base64,([A-Za-z0-9+/=]+)", url, re.IGNORECASE)
                if not match:
                    raise ValueError(f"Invalid data URL format: {url[:100]}...")
                media_type, data = match.groups()
                # Strip whitespace from base64 data
                data = data.strip()
                # Validate base64 format
                if not re.match(r'^[A-Za-z0-9+/]*={0,2}$', data):
                    raise ValueError(f"Invalid base64 data in data URL")
                
                return {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": data
                    }
                }
            # Otherwise, use URL source (Anthropic supports URLs)
            return {
                "type": "image",
                "source": {
                    "type": "url",
                    "url": url
                }
            }

        elif isinstance(block, DocumentContent):
            # Anthropic only supports PDF documents natively
            # https://docs.anthropic.com/en/docs/build-with-claude/pdf-support
            if block.source.media_type == "application/pdf":
                # Use native PDF support
                return {
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "media_type": block.source.media_type,
                        "data": block.source.data
                    }
                }
            else:
                # For non-PDF documents (text/plain, application/json, etc.),
                # decode the base64 content and include as text
                import base64
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
                    
                    logger.debug(f"Converting non-PDF document ({block.source.media_type}) to text content for Anthropic")
                    return {"type": "text", "text": text_content}
                except Exception as e:
                    logger.warning(f"Failed to decode document content: {e}, converting to text placeholder")
                    return {
                        "type": "text",
                        "text": f"[Document: {block.filename or 'unknown'} ({block.source.media_type}) - Failed to decode]"
                    }

        elif isinstance(block, FileContent):
            # File content - include as text (Anthropic doesn't support files directly)
            logger.warning(f"FileContent not fully supported, converting to text: {block.file_path}")
            return {"type": "text", "text": f"[File: {block.file_path}]"}

        elif isinstance(block, dict):
            # Already in dict format, pass through
            return block

        else:
            # Fallback for unknown content types
            logger.warning(f"Unknown content block type: {type(block)}")
            return {"type": "text", "text": str(block)}

    def _convert_message_content(self, content) -> Any:
        """Convert message content (string or list of blocks) to Anthropic format.

        Args:
            content: Message content (str or List[ContentBlock])

        Returns:
            String or list of content dicts for Anthropic API
        """
        if content is None:
            return None
        if isinstance(content, str):
            return content
        elif isinstance(content, list):
            return [self._convert_content_block(block) for block in content]
        else:
            return str(content)

    def _convert_messages(self, messages: List[Message]) -> tuple[Optional[str], List[Dict[str, Any]]]:
        """Convert Message objects to Anthropic format, separating system messages.

        Handles both text-only and multimodal messages seamlessly.
        """
        system_content = None
        anthropic_messages = []

        for message in messages:
            if message.role == "system":
                # Handle system messages separately
                # Get text content from message
                msg_content = message.text_content if message.is_multimodal else message.content

                # Only apply cache_control to system message if it's large enough
                if (self.enable_prompt_cache and
                    msg_content and len(msg_content) >= 4000):
                    system_content = [
                        {
                            "type": "text",
                            "text": msg_content,
                            "cache_control": {"type": "ephemeral"}
                        }
                    ]
                    logger.info(f"Applied cache_control to system message ({len(msg_content)} chars)")
                else:
                    # Use string format for simple system messages
                    system_content = msg_content
            elif message.role == "tool":
                # Convert tool messages to user messages with tool_result
                tool_content = message.text_content if message.is_multimodal else message.content
                anthropic_messages.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": message.tool_call_id,
                        "content": tool_content
                    }]
                })
            elif message.role == "assistant":
                content = []

                # Add text/multimodal content if present
                if message.content:
                    if isinstance(message.content, str):
                        content.append({
                            "type": "text",
                            "text": message.content
                        })
                    elif isinstance(message.content, list):
                        # Handle multimodal content
                        for block in message.content:
                            content.append(self._convert_content_block(block))

                # Add tool calls if present
                if message.tool_calls:
                    for tool_call in message.tool_calls:
                        try:
                            arguments = json.loads(tool_call.function.arguments) if isinstance(tool_call.function.arguments, str) else tool_call.function.arguments
                        except json.JSONDecodeError:
                            arguments = {}

                        content.append({
                            "type": "tool_use",
                            "id": tool_call.id,
                            "name": tool_call.function.name,
                            "input": arguments
                        })

                anthropic_messages.append({
                    "role": "assistant",
                    "content": content if content else (message.text_content if message.is_multimodal else message.content)
                })
            elif message.role == "user":
                # Handle user messages - can be text or multimodal
                converted_content = self._convert_message_content(message.content)
                anthropic_messages.append({
                    "role": "user",
                    "content": converted_content
                })

        return system_content, anthropic_messages
    
    def _convert_tools(self, tools: List[Dict]) -> List[Dict]:
        """Convert tools to Anthropic format."""
        anthropic_tools = []
        cache_control_count = 0
        max_cache_control_blocks = 3  # Leave room for system message cache_control
        
        for i, tool in enumerate(tools):
            anthropic_tool = {
                "name": tool["function"]["name"],
                "description": tool["function"]["description"],
                "input_schema": tool["function"]["parameters"]
            }
            
            # Only add cache control to the first few tools to stay within Anthropic's limit
            # Prioritize tools with larger schemas or descriptions
            tool_size = len(str(tool["function"]["parameters"])) + len(tool["function"]["description"])
            should_cache = (
                self.enable_prompt_cache and 
                cache_control_count < max_cache_control_blocks and
                len(tools) > 4 and  # Only use cache_control when there are many tools
                (tool_size > 500 or i < 2)  # Cache large tools or first 2 tools
            )
            
            if should_cache:
                anthropic_tool["cache_control"] = {"type": "ephemeral"}
                cache_control_count += 1
                logger.debug(f"Applied cache_control to tool: {tool['function']['name']} (size: {tool_size})")
            
            anthropic_tools.append(anthropic_tool)
        
        if cache_control_count > 0:
            logger.info(f"Applied cache_control to {cache_control_count}/{len(tools)} tools")
        
        return anthropic_tools
    
    def _log_cache_metrics(self, usage_data) -> None:
        """Log cache metrics from Anthropic API response usage data."""
        if self.enable_prompt_cache and usage_data:
            if hasattr(usage_data, 'cache_creation_input_tokens') and usage_data.cache_creation_input_tokens:
                self.cache_metrics["cache_creation_input_tokens"] += usage_data.cache_creation_input_tokens
                logger.info(f"Cache creation tokens: {usage_data.cache_creation_input_tokens}")
            if hasattr(usage_data, 'cache_read_input_tokens') and usage_data.cache_read_input_tokens:
                self.cache_metrics["cache_read_input_tokens"] += usage_data.cache_read_input_tokens
                logger.info(f"Cache read tokens: {usage_data.cache_read_input_tokens}")
            if hasattr(usage_data, 'input_tokens') and usage_data.input_tokens:
                self.cache_metrics["total_input_tokens"] += usage_data.input_tokens
    
    def get_cache_metrics(self) -> Dict[str, int]:
        """Get current cache performance metrics."""
        return self.cache_metrics.copy()
    
    async def chat(self, messages: List[Message], **kwargs) -> LLMResponse:
        """Send chat request to Anthropic."""
        if not self.client:
            raise ProviderError("anthropic", "Provider not initialized")
        
        try:
            start_time = asyncio.get_event_loop().time()
            
            system_content, anthropic_messages = self._convert_messages(messages)
            
            # Extract parameters
            model = kwargs.get('model', self.model)
            max_tokens = kwargs.get('max_tokens', self.max_tokens)
            temperature = kwargs.get('temperature', self.temperature)
            tool_choice = kwargs.get('tool_choice')
            # Guard against missing/None model overrides
            if not model:
                model = self.model or "claude-sonnet-4-20250514"
            
            # Prepare request parameters
            request_params = {
                'model': model,
                'max_tokens': max_tokens,
                'temperature': temperature,
                'messages': anthropic_messages,
                **{k: v for k, v in kwargs.items() if k not in ['model', 'max_tokens', 'temperature']}
            }
            
            # Only add system parameter if we have system content
            if system_content is not None:
                request_params['system'] = system_content
            
            response = await self.client.messages.create(**request_params)
            
            duration = asyncio.get_event_loop().time() - start_time
            
            # Log cache metrics
            if hasattr(response, 'usage'):
                self._log_cache_metrics(response.usage)
            
            return self._convert_response(response, duration)
            
        except Exception as e:
            await self._handle_error(e)
    
    async def chat_stream(self, messages: List[Message],callbacks: Optional[List] = None, **kwargs) -> AsyncIterator[LLMResponseChunk]:
        """Send streaming chat request to Anthropic with callback support.
        Yields:
            LLMResponseChunk: Structured streaming response chunks
        """
        if not self.client:
            raise ProviderError("anthropic", "Provider not initialized")

        # Create callback manager
        callback_manager = CallbackManager.from_callbacks(callbacks)
        run_id = uuid4()
        
        try:
            system_content, anthropic_messages = self._convert_messages(messages)
            
            # Extract parameters
            model = kwargs.get('model', self.model)
            max_tokens = kwargs.get('max_tokens', self.max_tokens)
            temperature = kwargs.get('temperature', self.temperature)
            
            # Trigger on_llm_start callback
            await callback_manager.on_llm_start(
                run_id=run_id,
                messages=messages,
                model=model,
                provider="anthropic"
            )
            
            # Prepare request parameters
            request_params = {
                'model': model,
                'max_tokens': max_tokens,
                'temperature': temperature,
                'messages': anthropic_messages,
                **{k: v for k, v in kwargs.items() 
                   if k not in ['model', 'max_tokens', 'temperature', 'callbacks']}
            }
            
            # Only add system parameter if we have system content
            if system_content is not None:
                request_params['system'] = system_content
            
            # Process streaming response
            full_content = ""
            chunk_index = 0
            finish_reason = None
            usage_data = None
            
            async with self.client.messages.stream(**request_params) as stream:
                async for chunk in stream:
                    # Handle different chunk types
                    if chunk.type == "content_block_delta" and chunk.delta.type == "text_delta":
                        token = chunk.delta.text
                        full_content += token
                        
                        # Trigger on_llm_new_token callback
                        await callback_manager.on_llm_new_token(
                            token=token,
                            run_id=run_id
                        )
                        
                        # Build response chunk
                        response_chunk = LLMResponseChunk(
                            content=full_content,
                            delta=token,
                            provider="anthropic",
                            model=model,
                            finish_reason=finish_reason,
                            tool_calls=[],
                            usage=usage_data,
                            metadata={
                                "chunk_index": chunk_index,
                                "chunk_type": chunk.type
                            },
                            chunk_index=chunk_index
                        )
                        chunk_index += 1
                        yield response_chunk
                        
                    elif chunk.type == "message_start":
                        if hasattr(chunk, 'message') and hasattr(chunk.message, 'usage'):
                            self._log_cache_metrics(chunk.message.usage)
                            
                    elif chunk.type == "message_delta":
                        # Extract finish_reason
                        if hasattr(chunk, 'delta') and hasattr(chunk.delta, 'stop_reason'):
                            finish_reason = chunk.delta.stop_reason
                        # Extract final usage stats
                        if hasattr(chunk, 'usage'):
                            usage_data = {
                                "input_tokens": chunk.usage.input_tokens,
                                "output_tokens": chunk.usage.output_tokens
                            }
                            if hasattr(chunk.usage, 'cache_creation_input_tokens'):
                                usage_data["cache_creation_input_tokens"] = chunk.usage.cache_creation_input_tokens
                            if hasattr(chunk.usage, 'cache_read_input_tokens'):
                                usage_data["cache_read_input_tokens"] = chunk.usage.cache_read_input_tokens
                
            # Trigger on_llm_end callback
            final_response = LLMResponse(
                content=full_content,
                provider="anthropic",
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
        """Send completion request to Anthropic."""
        # Convert to chat format
        messages = [Message(role="user", content=prompt)]
        return await self.chat(messages, **kwargs)
    
    async def chat_with_tools(self, messages: List[Message], tools: List[Dict], **kwargs) -> LLMResponse:
        """Send chat request with tools to Anthropic."""
        if not self.client:
            raise ProviderError("anthropic", "Provider not initialized")
        
        try:
            start_time = asyncio.get_event_loop().time()
            
            system_content, anthropic_messages = self._convert_messages(messages)
            anthropic_tools = self._convert_tools(tools)
            
            # Extract parameters
            model = kwargs.get('model', self.model)
            max_tokens = kwargs.get('max_tokens', self.max_tokens)
            temperature = kwargs.get('temperature', self.temperature)
            tool_choice = kwargs.get('tool_choice')
            # Guard against missing/None model overrides
            if not model:
                model = self.model or "claude-sonnet-4-20250514"
            
            # Use streaming for tool calls to handle complex responses
            content = ""
            buffer = ""
            buffer_type = ""
            current_tool = None
            tool_calls = []
            finish_reason = None
            native_finish_reason = None
            
            # Prepare request parameters
            request_params = {
                'model': model,
                'max_tokens': max_tokens,
                'temperature': temperature,
                'messages': anthropic_messages,
                'tools': anthropic_tools,
                **{k: v for k, v in kwargs.items() if k not in ['model', 'max_tokens', 'temperature', 'tool_choice']}
            }

            # Anthropic expects tool_choice as an object, not a plain string/enum
            if tool_choice:
                if isinstance(tool_choice, str):
                    request_params['tool_choice'] = {"type": tool_choice}
                elif isinstance(tool_choice, dict):
                    request_params['tool_choice'] = tool_choice
            
            # Only add system parameter if we have system content
            if system_content is not None:
                request_params['system'] = system_content
            
            async with self.client.messages.stream(**request_params) as stream:
                async for chunk in stream:
                    if chunk.type == "message_start":
                        # Log cache metrics from streaming message_start event
                        if hasattr(chunk, 'message') and hasattr(chunk.message, 'usage'):
                            self._log_cache_metrics(chunk.message.usage)
                        continue
                    elif chunk.type == "message_delta":
                        # Extract finish_reason from message delta
                        if hasattr(chunk, 'delta') and hasattr(chunk.delta, 'stop_reason'):
                            finish_reason = chunk.delta.stop_reason
                            native_finish_reason = chunk.delta.stop_reason
                        continue
                    elif chunk.type == "message_stop":
                        # Extract finish_reason from message stop
                        if hasattr(chunk, 'message') and hasattr(chunk.message, 'stop_reason'):
                            finish_reason = chunk.message.stop_reason
                            native_finish_reason = chunk.message.stop_reason
                        continue
                    elif chunk.type in ["text", "input_json"]:
                        continue
                    elif chunk.type == "content_block_start":
                        buffer_type = chunk.content_block.type
                        if buffer_type == "tool_use":
                            current_tool = {
                                "id": chunk.content_block.id,
                                "function": {
                                    "name": chunk.content_block.name,
                                    "arguments": {}
                                }
                            }
                        continue
                    elif chunk.type == "content_block_delta" and chunk.delta.type == "text_delta":
                        buffer += chunk.delta.text
                        continue
                    elif chunk.type == "content_block_delta" and chunk.delta.type == "input_json_delta":
                        buffer += chunk.delta.partial_json
                    elif chunk.type == "content_block_stop":
                        content += buffer
                        if buffer_type == "tool_use" and current_tool:
                            current_tool["function"]["arguments"] = buffer
                            tool_call = ToolCall(
                                id=current_tool["id"],
                                type="function",
                                function=Function(
                                    name=current_tool["function"]["name"],
                                    arguments=current_tool["function"]["arguments"]
                                )
                            )
                            tool_calls.append(tool_call)
                        buffer = ""
                        buffer_type = ""
                        current_tool = None
            
            duration = asyncio.get_event_loop().time() - start_time
            
            # Map Anthropic stop reasons to standard finish reasons
            if finish_reason == "end_turn":
                finish_reason = "stop"
            elif finish_reason == "max_tokens":
                finish_reason = "length"
            elif finish_reason == "tool_use":
                finish_reason = "tool_calls"
            
            return LLMResponse(
                content=content,
                provider="anthropic",
                model=model,
                finish_reason=finish_reason,
                native_finish_reason=native_finish_reason,
                tool_calls=tool_calls,
                duration=duration,
                metadata={"cache_metrics": self.cache_metrics}
            )
            
        except Exception as e:
            await self._handle_error(e)
    
    def _convert_response(self, response, duration: float) -> LLMResponse:
        """Convert Anthropic response to standardized LLMResponse."""
        content = ""
        if response.content:
            for block in response.content:
                if hasattr(block, 'text'):
                    content += block.text
        
        # Extract usage information
        usage = None
        if hasattr(response, 'usage'):
            usage = {
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens
            }
        
        # Map stop reasons
        finish_reason = response.stop_reason
        if finish_reason == "end_turn":
            standardized_finish_reason = "stop"
        elif finish_reason == "max_tokens":
            standardized_finish_reason = "length"
        elif finish_reason == "tool_use":
            standardized_finish_reason = "tool_calls"
        else:
            standardized_finish_reason = finish_reason
        
        return LLMResponse(
            content=content,
            provider="anthropic",
            model=response.model,
            finish_reason=standardized_finish_reason,
            native_finish_reason=response.stop_reason,
            usage=usage,
            duration=duration,
            metadata={
                "response_id": response.id,
                "cache_metrics": self.cache_metrics
            }
        )
    
    def get_metadata(self) -> ProviderMetadata:
        """Get Anthropic provider metadata."""
        return ProviderMetadata(
            name="anthropic",
            version="1.0.0",
            capabilities=[
                ProviderCapability.CHAT,
                ProviderCapability.COMPLETION,
                ProviderCapability.TOOLS,
                ProviderCapability.STREAMING
            ],
            max_tokens=200000,  # Claude-3 context limit
            supports_system_messages=True,
            rate_limits={
                "requests_per_minute": 1000,
                "tokens_per_minute": 40000
            }
        )
    
    async def health_check(self) -> bool:
        """Check if Anthropic provider is healthy."""
        if not self.client:
            return False
        
        try:
            # Simple test request
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=1,
                messages=[{"role": "user", "content": "test"}]
            )
            return True
        except Exception as e:
            logger.warning(f"Anthropic health check failed: {e}")
            return False
    
    async def cleanup(self) -> None:
        """Cleanup Anthropic provider resources."""
        if self.client:
            await self.client.close()
            self.client = None
        logger.info("Anthropic provider cleaned up")
    
    async def _handle_error(self, error: Exception) -> None:
        """Handle and convert Anthropic errors to standardized errors."""
        error_str = str(error).lower()
        
        if "authentication" in error_str or "api key" in error_str:
            raise AuthenticationError("anthropic", context={"original_error": str(error)})
        elif "rate limit" in error_str:
            raise RateLimitError("anthropic", context={"original_error": str(error)})
        elif "model" in error_str and "not found" in error_str:
            raise ModelNotFoundError("anthropic", self.model, context={"original_error": str(error)})
        elif "timeout" in error_str or "connection" in error_str or "network" in error_str:
            # Provide more helpful error message for network errors
            error_msg = f"Network error: {str(error)}. This may be a temporary issue. Please try again."
            raise NetworkError("anthropic", error_msg, original_error=error)
        else:
            raise ProviderError("anthropic", f"Request failed: {str(error)}", original_error=error)
