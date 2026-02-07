"""Anthropic Prompt Caching Middleware.

Adds cache_control markers to system prompts and messages for Anthropic models,
enabling prompt caching to reduce costs and latency for repeated content.

How Anthropic Prompt Caching Works:
- Content marked with cache_control: {"type": "ephemeral"} is cached for ~5 minutes
- Subsequent requests within the cache window reuse the cached content
- This reduces input token costs and speeds up responses
- Only works with Claude models (claude-3-*, claude-2-*, etc.)

Compatible with LangChain DeepAgents AnthropicPromptCachingMiddleware interface.

Usage:
    from spoon_ai.middleware.prompt_caching import AnthropicPromptCachingMiddleware

    agent = ToolCallAgent(
        middleware=[AnthropicPromptCachingMiddleware()],
        ...
    )
"""

import logging
from typing import Any, Callable, Dict, List, Literal, Optional
from copy import deepcopy

from spoon_ai.middleware.base import (
    AgentMiddleware,
    AgentRuntime,
    ModelRequest,
    ModelResponse,
)
from spoon_ai.schema import Message, Role

logger = logging.getLogger(__name__)


# ============================================================================
# Constants
# ============================================================================

# Models that support prompt caching
SUPPORTED_MODELS = [
    "claude-3",
    "claude-sonnet",
    "claude-opus",
    "claude-haiku",
    "claude-2",
    "claude-instant",
]

# Minimum content length to cache (in characters)
MIN_CACHE_LENGTH = 1024


# ============================================================================
# Cache Control Utilities
# ============================================================================

def is_anthropic_model(model_name: Optional[str]) -> bool:
    """Check if a model name indicates an Anthropic Claude model.

    Args:
        model_name: Model identifier string

    Returns:
        True if this is an Anthropic model that supports caching
    """
    if not model_name:
        return False

    model_lower = model_name.lower()
    return any(prefix in model_lower for prefix in SUPPORTED_MODELS)


def add_cache_control(content: Any) -> Any:
    """Add cache_control marker to content.

    Anthropic expects content to be a list of content blocks, where each
    block can have a cache_control field.

    Args:
        content: Content to add cache control to (string or list of blocks)

    Returns:
        Content with cache_control added
    """
    if isinstance(content, str):
        # Convert string to content block with cache control
        return [
            {
                "type": "text",
                "text": content,
                "cache_control": {"type": "ephemeral"}
            }
        ]
    elif isinstance(content, list):
        # Add cache_control to the last block
        if not content:
            return content

        result = deepcopy(content)
        last_block = result[-1]

        if isinstance(last_block, dict) and "cache_control" not in last_block:
            last_block["cache_control"] = {"type": "ephemeral"}

        return result
    elif isinstance(content, dict):
        # Single content block
        result = deepcopy(content)
        if "cache_control" not in result:
            result["cache_control"] = {"type": "ephemeral"}
        return [result]

    return content


def should_cache_content(content: Any) -> bool:
    """Determine if content is worth caching based on length.

    Args:
        content: Content to evaluate

    Returns:
        True if content should be cached
    """
    if isinstance(content, str):
        return len(content) >= MIN_CACHE_LENGTH
    elif isinstance(content, list):
        total_length = sum(
            len(block.get("text", "")) if isinstance(block, dict) else len(str(block))
            for block in content
        )
        return total_length >= MIN_CACHE_LENGTH
    elif isinstance(content, dict):
        return len(content.get("text", "")) >= MIN_CACHE_LENGTH

    return False


# ============================================================================
# Anthropic Prompt Caching Middleware
# ============================================================================

class AnthropicPromptCachingMiddleware(AgentMiddleware):
    """Middleware that adds cache control markers for Anthropic prompt caching.

    When using Anthropic Claude models, this middleware:
    1. Adds cache_control to system prompts (if long enough)
    2. Optionally adds cache_control to tool definitions
    3. Skips caching for non-Anthropic models

    Benefits:
    - Reduces input token costs for repeated content
    - Speeds up response time for cached prompts
    - Automatic cache invalidation after ~5 minutes

    Example:
        ```python
        from spoon_ai.middleware.prompt_caching import AnthropicPromptCachingMiddleware

        # Basic usage - caches system prompt
        middleware = AnthropicPromptCachingMiddleware()

        # With options
        middleware = AnthropicPromptCachingMiddleware(
            cache_system_prompt=True,
            cache_tools=True,
            min_cache_length=1024,
            unsupported_model_behavior="ignore",  # or "warn"
        )

        agent = ToolCallAgent(
            middleware=[middleware],
            ...
        )
        ```
    """

    def __init__(
        self,
        cache_system_prompt: bool = True,
        cache_tools: bool = True,
        min_cache_length: int = MIN_CACHE_LENGTH,
        unsupported_model_behavior: Literal["ignore", "warn"] = "ignore",
    ):
        """Initialize Anthropic prompt caching middleware.

        Args:
            cache_system_prompt: Whether to cache the system prompt (default: True)
            cache_tools: Whether to cache tool definitions (default: True)
            min_cache_length: Minimum content length to cache (default: 1024 chars)
            unsupported_model_behavior: What to do for non-Anthropic models:
                - "ignore": Silently skip caching
                - "warn": Log a warning and skip caching
        """
        super().__init__()
        self._cache_system_prompt = cache_system_prompt
        self._cache_tools = cache_tools
        self._min_cache_length = min_cache_length
        self._unsupported_model_behavior = unsupported_model_behavior

        # Statistics
        self._cache_hits = 0
        self._cache_misses = 0
        self._cached_prompts = 0

    def _should_cache(self, content: Any) -> bool:
        """Check if content should be cached based on length."""
        if isinstance(content, str):
            return len(content) >= self._min_cache_length
        elif isinstance(content, list):
            total = sum(
                len(block.get("text", "")) if isinstance(block, dict) else len(str(block))
                for block in content
            )
            return total >= self._min_cache_length
        return False

    def _add_cache_control_to_system(self, system_prompt: str) -> Any:
        """Add cache control to system prompt.

        Returns the system prompt in Anthropic's expected format with cache control.
        """
        if not system_prompt or not self._should_cache(system_prompt):
            return system_prompt

        # Anthropic expects system as a list of content blocks for caching
        return [
            {
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"}
            }
        ]

    def _add_cache_control_to_tools(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Add cache control to tool definitions.

        Adds cache_control to the last tool in the list to cache all tools together.
        """
        if not tools:
            return tools

        # Only cache if tools are substantial
        import json
        tools_json = json.dumps(tools)
        if len(tools_json) < self._min_cache_length:
            return tools

        result = deepcopy(tools)

        # Add cache_control to the last tool
        if result:
            last_tool = result[-1]
            if isinstance(last_tool, dict):
                last_tool["cache_control"] = {"type": "ephemeral"}

        return result

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable
    ) -> ModelResponse:
        """Add cache control markers before model call."""
        # Get model name to check if it's Anthropic
        model_name = None
        if request.runtime and hasattr(request.runtime, '_agent_instance'):
            agent = request.runtime._agent_instance
            if hasattr(agent, 'llm') and hasattr(agent.llm, 'model'):
                model_name = agent.llm.model
            elif hasattr(agent, 'model'):
                model_name = agent.model

        # Check if model supports caching
        if not is_anthropic_model(model_name):
            if self._unsupported_model_behavior == "warn":
                logger.warning(
                    f"AnthropicPromptCachingMiddleware: Model '{model_name}' "
                    "does not support Anthropic prompt caching, skipping"
                )
            return await handler(request)

        # Apply cache control
        new_system_prompt = request.system_prompt
        new_tools = request.tools

        # Cache system prompt
        if self._cache_system_prompt and request.system_prompt:
            if self._should_cache(request.system_prompt):
                new_system_prompt = self._add_cache_control_to_system(request.system_prompt)
                self._cached_prompts += 1
                logger.debug("Added cache_control to system prompt")

        # Cache tools
        if self._cache_tools and request.tools:
            new_tools = self._add_cache_control_to_tools(request.tools)
            logger.debug(f"Added cache_control to {len(request.tools)} tools")

        # Create new request if changes were made
        if new_system_prompt != request.system_prompt or new_tools != request.tools:
            request = ModelRequest(
                messages=request.messages,
                system_prompt=new_system_prompt,
                tools=new_tools,
                phase=request.phase,
                runtime=request.runtime,
            )

        return await handler(request)

    def get_stats(self) -> Dict[str, int]:
        """Get caching statistics."""
        return {
            "cached_prompts": self._cached_prompts,
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
        }


# ============================================================================
# Factory Function
# ============================================================================

def create_prompt_caching_middleware(
    cache_system_prompt: bool = True,
    cache_tools: bool = True,
    min_cache_length: int = MIN_CACHE_LENGTH,
    unsupported_model_behavior: Literal["ignore", "warn"] = "ignore",
) -> AnthropicPromptCachingMiddleware:
    """Create an Anthropic prompt caching middleware.

    Args:
        cache_system_prompt: Whether to cache system prompts
        cache_tools: Whether to cache tool definitions
        min_cache_length: Minimum content length to cache
        unsupported_model_behavior: How to handle non-Anthropic models

    Returns:
        Configured AnthropicPromptCachingMiddleware
    """
    return AnthropicPromptCachingMiddleware(
        cache_system_prompt=cache_system_prompt,
        cache_tools=cache_tools,
        min_cache_length=min_cache_length,
        unsupported_model_behavior=unsupported_model_behavior,
    )
