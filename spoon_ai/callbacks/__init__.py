"""
Callback system for streaming and event handling in Spoon AI.

This module provides a comprehensive callback system similar to LangChain's callbacks,
enabling real-time monitoring and event handling for LLM calls, agent execution,
tool invocation, and graph workflows.
"""

from spoon_ai.callbacks.base import (
    BaseCallbackHandler,
    AsyncCallbackHandler,
    LLMManagerMixin,
    ChainManagerMixin,
    ToolManagerMixin,
    RetrieverManagerMixin,
    PromptManagerMixin,
)
from spoon_ai.callbacks.manager import (
    CallbackManager,
    AsyncCallbackManager,
)
from spoon_ai.callbacks.streaming_stdout import (
    StreamingStdOutCallbackHandler,
)
from spoon_ai.callbacks.stream_event import StreamEventCallbackHandler
from spoon_ai.callbacks.statistics import StreamingStatisticsCallback
from spoon_ai.callbacks.skill_callback import (
    SkillCallbackHandler,
    LoggingSkillCallback,
    MetricsSkillCallback,
)

__all__ = [
    # Base handlers
    "BaseCallbackHandler",
    "AsyncCallbackHandler",

    # Mixins
    "LLMManagerMixin",
    "ChainManagerMixin",
    "ToolManagerMixin",
    "RetrieverManagerMixin",
    "PromptManagerMixin",

    # Managers
    "CallbackManager",
    "AsyncCallbackManager",

    # Built-in handlers
    "StreamingStdOutCallbackHandler",
    "StreamEventCallbackHandler",
    "StreamingStatisticsCallback",

    # Skill handlers
    "SkillCallbackHandler",
    "LoggingSkillCallback",
    "MetricsSkillCallback",
]
