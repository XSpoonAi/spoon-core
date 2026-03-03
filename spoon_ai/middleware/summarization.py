"""Summarization Middleware - Context Compression for Long Conversations.

Automatically summarizes conversation history when token limits are approached,
preserving recent messages and maintaining context continuity by ensuring
AI/Tool message pairs remain together.

Compatible with LangChain DeepAgents SummarizationMiddleware interface.

Usage:
    from spoon_ai.middleware.summarization import SummarizationMiddleware

    agent = ToolCallAgent(
        middleware=[SummarizationMiddleware(
            model=llm,
            trigger=("fraction", 0.85),  # Trigger at 85% of max tokens
            keep=("messages", 20),       # Keep last 20 messages
        )],
        ...
    )
"""

import logging
import uuid
import warnings
from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Tuple, Union, cast

from spoon_ai.middleware.base import (
    AgentMiddleware,
    AgentRuntime,
    ModelRequest,
    ModelResponse,
)
from spoon_ai.schema import Message, Role
from spoon_ai.chat import ChatBot

logger = logging.getLogger(__name__)


# ============================================================================
# Types - Compatible with LangChain ContextSize
# ============================================================================

ContextFraction = Tuple[Literal["fraction"], float]
"""Fraction of model's maximum input tokens.

Example:
    To specify 50% of the model's max input tokens:
    ```python
    ("fraction", 0.5)
    ```
"""

ContextTokens = Tuple[Literal["tokens"], int]
"""Absolute number of tokens.

Example:
    To specify 3000 tokens:
    ```python
    ("tokens", 3000)
    ```
"""

ContextMessages = Tuple[Literal["messages"], int]
"""Absolute number of messages.

Example:
    To specify 50 messages:
    ```python
    ("messages", 50)
    ```
"""

ContextSize = Union[ContextFraction, ContextTokens, ContextMessages]
"""Union type for context size specifications.

Can be either:
- ContextFraction: A fraction of the model's maximum input tokens.
- ContextTokens: An absolute number of tokens.
- ContextMessages: An absolute number of messages.

Depending on use with `trigger` or `keep` parameters, this type indicates either
when to trigger summarization or how much context to retain.

Example:
    ```python
    # ContextFraction
    context_size: ContextSize = ("fraction", 0.5)

    # ContextTokens
    context_size: ContextSize = ("tokens", 3000)

    # ContextMessages
    context_size: ContextSize = ("messages", 50)
    ```
"""

TokenCounter = Callable[[Iterable[Message]], int]


# ============================================================================
# Default Constants
# ============================================================================

_DEFAULT_MESSAGES_TO_KEEP = 20
_DEFAULT_TRIM_TOKEN_LIMIT = 4000
_DEFAULT_FALLBACK_MESSAGE_COUNT = 15

DEFAULT_SUMMARY_PROMPT = """<role>
Context Extraction Assistant
</role>

<primary_objective>
Your sole objective in this task is to extract the highest quality/most relevant context from the conversation history below.
</primary_objective>

<objective_information>
You're nearing the total number of input tokens you can accept, so you must extract the highest quality/most relevant pieces of information from your conversation history.
This context will then overwrite the conversation history presented below. Because of this, ensure the context you extract is only the most important information to your overall goal.
</objective_information>

<instructions>
The conversation history below will be replaced with the context you extract in this step. Because of this, you must do your very best to extract and record all of the most important context from the conversation history.
You want to ensure that you don't repeat any actions you've already completed, so the context you extract from the conversation history should be focused on the most important information to your overall goal.
</instructions>

The user will message you with the full message history you'll be extracting context from, to then replace. Carefully read over it all, and think deeply about what information is most important to your overall goal that should be saved:

With all of this in mind, please carefully read over the entire conversation history, and extract the most important and relevant context to replace it so that you can free up space in the conversation history.
Respond ONLY with the extracted context. Do not include any additional information, or text before or after the extracted context.

<messages>
Messages to summarize:
{messages}
</messages>"""


# ============================================================================
# Token Counting Utilities
# ============================================================================

def count_tokens_approximately(
    messages: Iterable[Message],
    chars_per_token: float = 4.0,
) -> int:
    """Approximate token counter aligned with LangChain semantics.

    Args:
        messages: Iterable of messages to count tokens for.
        chars_per_token: Characters per token ratio (default: 4.0).

    Returns:
        Estimated token count.
    """
    extra_tokens_per_message = 3.0
    token_count = 0.0

    for message in messages:
        message_chars = 0

        content = message.content
        if isinstance(content, str):
            message_chars += len(content)
        elif content is not None:
            message_chars += len(repr(content))

        # Tool calls on assistant messages
        if (
            message.role in (Role.ASSISTANT, "assistant")
            and hasattr(message, 'tool_calls')
            and message.tool_calls
            and not isinstance(message.content, list)
        ):
            message_chars += len(repr(message.tool_calls))

        # Tool call ID on tool messages
        if message.role in (Role.TOOL, "tool") and hasattr(message, 'tool_call_id') and message.tool_call_id:
            message_chars += len(message.tool_call_id)

        # Role
        role_str = message.role.value if hasattr(message.role, 'value') else str(message.role)
        message_chars += len(role_str or "")

        # Name
        if hasattr(message, 'name') and message.name:
            message_chars += len(message.name)

        token_count += message_chars / chars_per_token
        token_count += extra_tokens_per_message

    return max(1, int(token_count + 0.5))


def _get_approximate_token_counter(model: Optional[ChatBot]) -> TokenCounter:
    """Tune parameters of approximate token counter based on model type."""
    if model is None:
        return count_tokens_approximately

    # Get model name/type
    model_type = None
    if hasattr(model, 'model'):
        model_type = model.model
    elif hasattr(model, '_llm_type'):
        model_type = model._llm_type

    if model_type and isinstance(model_type, str):
        model_lower = model_type.lower()
        # Anthropic models use ~3.3 chars per token
        if any(x in model_lower for x in ['claude', 'anthropic']):
            return partial(count_tokens_approximately, chars_per_token=3.3)

    return count_tokens_approximately


# ============================================================================
# RemoveMessage - Compatible with LangChain
# ============================================================================

REMOVE_ALL_MESSAGES = "__remove_all__"


class RemoveMessage:
    """Marker class indicating a message should be removed.

    Compatible with LangChain's RemoveMessage pattern.
    """

    def __init__(self, id: str):
        self.id = id


# ============================================================================
# Summarization Middleware
# ============================================================================

class SummarizationMiddleware(AgentMiddleware):
    """Summarizes conversation history when token limits are approached.

    This middleware monitors message token counts and automatically summarizes older
    messages when a threshold is reached, preserving recent messages and maintaining
    context continuity by ensuring AI/Tool message pairs remain together.

    Compatible with LangChain DeepAgents SummarizationMiddleware interface.

    Example:
        ```python
        from spoon_ai.middleware.summarization import SummarizationMiddleware

        # Basic usage with fraction trigger
        middleware = SummarizationMiddleware(
            model=llm,
            trigger=("fraction", 0.85),
            keep=("messages", 20),
        )

        # With multiple trigger conditions
        middleware = SummarizationMiddleware(
            model=llm,
            trigger=[("fraction", 0.8), ("messages", 100)],
            keep=("tokens", 3000),
        )

        agent = ToolCallAgent(
            middleware=[middleware],
            ...
        )
        ```
    """

    def __init__(
        self,
        model: Optional[ChatBot] = None,
        *,
        trigger: Optional[Union[ContextSize, List[ContextSize]]] = None,
        keep: ContextSize = ("messages", _DEFAULT_MESSAGES_TO_KEEP),
        token_counter: Optional[TokenCounter] = None,
        summary_prompt: str = DEFAULT_SUMMARY_PROMPT,
        trim_tokens_to_summarize: Optional[int] = _DEFAULT_TRIM_TOKEN_LIMIT,
        max_context_tokens: Optional[int] = None,
        **deprecated_kwargs: Any,
    ) -> None:
        """Initialize summarization middleware.

        Args:
            model: The language model to use for generating summaries.
                If None, uses agent's LLM.
            trigger: One or more thresholds that trigger summarization.
                Provide a single ContextSize tuple or a list of tuples.
                Summarization runs when any threshold is met.

                Examples:
                    - ("messages", 50): Trigger at 50 messages
                    - ("tokens", 3000): Trigger at 3000 tokens
                    - ("fraction", 0.8): Trigger at 80% of max input tokens
                    - [("fraction", 0.8), ("messages", 100)]: Multiple conditions

            keep: Context retention policy applied after summarization.
                Provide a ContextSize tuple to specify how much history to preserve.
                Defaults to keeping the most recent 20 messages.

                Examples:
                    - ("messages", 20): Keep last 20 messages
                    - ("tokens", 3000): Keep last 3000 tokens worth
                    - ("fraction", 0.3): Keep last 30% of max input tokens

            token_counter: Function to count tokens in messages.
                Defaults to model-aware approximate counter.
            summary_prompt: Prompt template for generating summaries.
            trim_tokens_to_summarize: Maximum tokens to keep when preparing
                messages for summarization. Pass None to skip trimming.
            max_context_tokens: Maximum context tokens for fraction calculations.
                If None, attempts to get from model profile.
        """
        # Handle deprecated parameters
        if "max_tokens_before_summary" in deprecated_kwargs:
            value = deprecated_kwargs["max_tokens_before_summary"]
            warnings.warn(
                "max_tokens_before_summary is deprecated. Use trigger=('tokens', value) instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if trigger is None and value is not None:
                trigger = ("tokens", value)

        if "messages_to_keep" in deprecated_kwargs:
            value = deprecated_kwargs["messages_to_keep"]
            warnings.warn(
                "messages_to_keep is deprecated. Use keep=('messages', value) instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if keep == ("messages", _DEFAULT_MESSAGES_TO_KEEP):
                keep = ("messages", value)

        super().__init__()

        self._model = model
        self._max_context_tokens = max_context_tokens

        # Process trigger configuration
        if trigger is None:
            self.trigger: Optional[Union[ContextSize, List[ContextSize]]] = None
            self._trigger_conditions: List[ContextSize] = []
        elif isinstance(trigger, list):
            validated_list = [self._validate_context_size(item, "trigger") for item in trigger]
            self.trigger = validated_list
            self._trigger_conditions = validated_list
        else:
            validated = self._validate_context_size(trigger, "trigger")
            self.trigger = validated
            self._trigger_conditions = [validated]

        self.keep = self._validate_context_size(keep, "keep")

        # Set up token counter
        if token_counter is not None:
            self.token_counter = token_counter
        else:
            self.token_counter = _get_approximate_token_counter(model)

        self.summary_prompt = summary_prompt
        self.trim_tokens_to_summarize = trim_tokens_to_summarize

        # Validate fraction-based configs have max_context_tokens available
        requires_profile = any(condition[0] == "fraction" for condition in self._trigger_conditions)
        if self.keep[0] == "fraction":
            requires_profile = True
        if requires_profile and self._get_max_input_tokens() is None:
            msg = (
                "Model profile information or max_context_tokens is required to use "
                "fractional token limits. Please either:\n"
                "1. Pass max_context_tokens parameter explicitly\n"
                "2. Use absolute token counts instead of fractions\n"
                "3. Use a model that provides profile information"
            )
            raise ValueError(msg)

        # Statistics
        self._summary_count = 0
        self._total_tokens_saved = 0

    def _validate_context_size(self, context: ContextSize, parameter_name: str) -> ContextSize:
        """Validate context configuration tuples."""
        if not isinstance(context, tuple) or len(context) != 2:
            raise ValueError(f"{parameter_name} must be a tuple of (type, value)")

        kind, value = context
        if kind == "fraction":
            if not isinstance(value, (int, float)) or not 0 < value <= 1:
                msg = f"Fractional {parameter_name} values must be between 0 and 1, got {value}."
                raise ValueError(msg)
        elif kind in {"tokens", "messages"}:
            if not isinstance(value, int) or value <= 0:
                msg = f"{parameter_name} thresholds must be greater than 0, got {value}."
                raise ValueError(msg)
        else:
            msg = f"Unsupported context size type '{kind}' for {parameter_name}. Use 'fraction', 'tokens', or 'messages'."
            raise ValueError(msg)
        return context

    def _get_max_input_tokens(self) -> Optional[int]:
        """Retrieve max input token limit from model profile or config."""
        # First check explicit config
        if self._max_context_tokens is not None:
            return self._max_context_tokens

        # Try to get from model profile
        if self._model is not None:
            try:
                if hasattr(self._model, 'profile'):
                    profile = self._model.profile
                    if isinstance(profile, dict):
                        max_tokens = profile.get("max_input_tokens")
                        if isinstance(max_tokens, int):
                            return max_tokens
            except Exception:
                pass

            # Try common model attributes
            for attr in ['max_tokens', 'max_context_length', 'context_window']:
                if hasattr(self._model, attr):
                    val = getattr(self._model, attr)
                    if isinstance(val, int):
                        return val

        return None

    def _ensure_message_ids(self, messages: List[Message]) -> None:
        """Ensure all messages have unique IDs for tracking."""
        for msg in messages:
            if not getattr(msg, 'id', None):
                msg.id = str(uuid.uuid4())

    def _should_summarize(self, messages: List[Message], total_tokens: int) -> bool:
        """Determine whether summarization should run for the current token usage."""
        if not self._trigger_conditions:
            return False

        for kind, value in self._trigger_conditions:
            if kind == "messages" and len(messages) >= value:
                return True
            if kind == "tokens" and total_tokens >= value:
                return True
            if kind == "fraction":
                max_input_tokens = self._get_max_input_tokens()
                if max_input_tokens is None:
                    continue
                threshold = int(max_input_tokens * value)
                if threshold <= 0:
                    threshold = 1
                if total_tokens >= threshold:
                    return True
        return False

    def _determine_cutoff_index(self, messages: List[Message]) -> int:
        """Choose cutoff index respecting retention configuration."""
        kind, value = self.keep
        if kind in {"tokens", "fraction"}:
            token_based_cutoff = self._find_token_based_cutoff(messages)
            if token_based_cutoff is not None:
                return token_based_cutoff
            # Fallback to message count if token-based fails
            return self._find_safe_cutoff(messages, _DEFAULT_MESSAGES_TO_KEEP)
        return self._find_safe_cutoff(messages, cast(int, value))

    def _find_token_based_cutoff(self, messages: List[Message]) -> Optional[int]:
        """Find cutoff index based on target token retention using binary search."""
        if not messages:
            return 0

        kind, value = self.keep
        if kind == "fraction":
            max_input_tokens = self._get_max_input_tokens()
            if max_input_tokens is None:
                return None
            target_token_count = int(max_input_tokens * value)
        elif kind == "tokens":
            target_token_count = int(value)
        else:
            return None

        if target_token_count <= 0:
            target_token_count = 1

        if self.token_counter(messages) <= target_token_count:
            return 0

        # Binary search to find earliest index that keeps suffix within budget
        left, right = 0, len(messages)
        cutoff_candidate = len(messages)
        max_iterations = len(messages).bit_length() + 1

        for _ in range(max_iterations):
            if left >= right:
                break

            mid = (left + right) // 2
            if self.token_counter(messages[mid:]) <= target_token_count:
                cutoff_candidate = mid
                right = mid
            else:
                left = mid + 1

        if cutoff_candidate == len(messages):
            cutoff_candidate = left

        if cutoff_candidate >= len(messages):
            if len(messages) == 1:
                return 0
            cutoff_candidate = len(messages) - 1

        # Advance past any ToolMessages to avoid splitting AI/Tool pairs
        return self._find_safe_cutoff_point(messages, cutoff_candidate)

    def _find_safe_cutoff(self, messages: List[Message], messages_to_keep: int) -> int:
        """Find safe cutoff point that preserves AI/Tool message pairs.

        Returns the index where messages can be safely cut without separating
        related AI and Tool messages. Returns 0 if no safe cutoff is found.
        """
        if len(messages) <= messages_to_keep:
            return 0

        target_cutoff = len(messages) - messages_to_keep
        return self._find_safe_cutoff_point(messages, target_cutoff)

    def _find_safe_cutoff_point(self, messages: List[Message], cutoff_index: int) -> int:
        """Find a safe cutoff point that doesn't split AI/Tool message pairs.

        If the message at cutoff_index is a ToolMessage, advance until we find
        a non-ToolMessage. This ensures we never cut in the middle of parallel
        tool call responses.
        """
        while cutoff_index < len(messages):
            msg = messages[cutoff_index]
            role = msg.role.value if hasattr(msg.role, 'value') else str(msg.role)
            if role.lower() == "tool":
                cutoff_index += 1
            else:
                break
        return cutoff_index

    def _partition_messages(
        self,
        messages: List[Message],
        cutoff_index: int,
    ) -> Tuple[List[Message], List[Message]]:
        """Partition messages into those to summarize and those to preserve."""
        messages_to_summarize = messages[:cutoff_index]
        preserved_messages = messages[cutoff_index:]
        return messages_to_summarize, preserved_messages

    def _trim_messages_for_summary(self, messages: List[Message]) -> List[Message]:
        """Trim messages to fit within summary generation limits."""
        if self.trim_tokens_to_summarize is None:
            return messages

        try:
            total_tokens = self.token_counter(messages)
            if total_tokens <= self.trim_tokens_to_summarize:
                return messages

            # Trim from the start, keeping more recent messages
            result = []
            current_tokens = 0

            for msg in reversed(messages):
                msg_tokens = self.token_counter([msg])
                if current_tokens + msg_tokens > self.trim_tokens_to_summarize and result:
                    break
                result.insert(0, msg)
                current_tokens += msg_tokens

            return result if result else messages[-_DEFAULT_FALLBACK_MESSAGE_COUNT:]
        except Exception:
            return messages[-_DEFAULT_FALLBACK_MESSAGE_COUNT:]

    def _format_messages_for_summary(self, messages: List[Message]) -> str:
        """Format messages as text for the summary prompt."""
        formatted = []
        for msg in messages:
            role = msg.role.value if hasattr(msg.role, 'value') else str(msg.role)
            content = msg.text_content if hasattr(msg, 'text_content') else str(msg.content)

            # Include tool call info
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                tool_names = []
                for tc in msg.tool_calls:
                    if isinstance(tc, dict):
                        tool_names.append(tc.get('function', {}).get('name', 'unknown'))
                    elif hasattr(tc, 'function'):
                        tool_names.append(tc.function.name)
                    elif hasattr(tc, 'name'):
                        tool_names.append(tc.name)
                if tool_names:
                    content = f"{content}\n[Tool calls: {', '.join(tool_names)}]"

            formatted.append(f"[{role}]: {content}")

        return "\n\n".join(formatted)

    async def _create_summary(
        self,
        messages: List[Message],
        llm: ChatBot,
    ) -> str:
        """Generate summary for the given messages."""
        if not messages:
            return "No previous conversation history."

        trimmed_messages = self._trim_messages_for_summary(messages)
        if not trimmed_messages:
            return "Previous conversation was too long to summarize."

        try:
            formatted = self._format_messages_for_summary(trimmed_messages)
            prompt = self.summary_prompt.format(messages=formatted)

            response = await llm.ask(
                messages=[Message(role=Role.USER, content=prompt)],
                system_msg="You are a helpful assistant that extracts and preserves important context."
            )

            if hasattr(response, 'content'):
                return response.content.strip()
            elif hasattr(response, 'text'):
                return response.text.strip()
            else:
                return str(response).strip()
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return f"Error generating summary: {e!s}"

    def _build_summary_message(self, summary: str) -> Message:
        """Build a message containing the conversation summary."""
        return Message(
            role=Role.USER,
            content=f"Here is a summary of the conversation to date:\n\n{summary}",
            id=f"summary-{uuid.uuid4()}"
        )

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable
    ) -> ModelResponse:
        """Process messages before model call, potentially triggering summarization."""
        messages = list(request.messages) if request.messages else []

        if not messages:
            return await handler(request)

        self._ensure_message_ids(messages)

        total_tokens = self.token_counter(messages)
        if not self._should_summarize(messages, total_tokens):
            return await handler(request)

        logger.info(
            f"SummarizationMiddleware: Threshold reached "
            f"({total_tokens} tokens, {len(messages)} messages), summarizing..."
        )

        cutoff_index = self._determine_cutoff_index(messages)

        if cutoff_index <= 0:
            return await handler(request)

        messages_to_summarize, preserved_messages = self._partition_messages(messages, cutoff_index)

        # Get LLM for summarization
        llm = self._model
        if llm is None and request.runtime and hasattr(request.runtime, '_agent_instance'):
            agent = request.runtime._agent_instance
            if hasattr(agent, 'llm'):
                llm = agent.llm

        if llm is None:
            logger.warning("No LLM available for summarization, skipping")
            return await handler(request)

        # Generate summary
        summary = await self._create_summary(messages_to_summarize, llm)
        summary_message = self._build_summary_message(summary)

        # Build new message list
        new_messages = [summary_message] + preserved_messages

        # Track stats
        old_tokens = total_tokens
        new_tokens = self.token_counter(new_messages)
        self._summary_count += 1
        self._total_tokens_saved += (old_tokens - new_tokens)

        logger.info(
            f"SummarizationMiddleware: Compressed {len(messages_to_summarize)} messages, "
            f"saved {old_tokens - new_tokens} tokens ({old_tokens} -> {new_tokens})"
        )

        # Create new request with summarized messages
        new_request = ModelRequest(
            messages=new_messages,
            system_prompt=request.system_prompt,
            tools=request.tools,
            phase=request.phase,
            runtime=request.runtime,
        )

        # Also update agent's memory if accessible
        if request.runtime and hasattr(request.runtime, '_agent_instance'):
            agent = request.runtime._agent_instance
            if hasattr(agent, 'memory') and hasattr(agent.memory, 'messages'):
                agent.memory.messages = new_messages
                logger.debug("Updated agent memory with summarized messages")

        return await handler(new_request)

    def get_stats(self) -> Dict[str, Any]:
        """Get summarization statistics."""
        return {
            "summary_count": self._summary_count,
            "total_tokens_saved": self._total_tokens_saved,
            "trigger_config": self.trigger,
            "keep_config": self.keep,
        }


# ============================================================================
# Factory Function
# ============================================================================

def create_summarization_middleware(
    model: Optional[ChatBot] = None,
    trigger: Optional[Union[ContextSize, List[ContextSize]]] = None,
    keep: ContextSize = ("messages", _DEFAULT_MESSAGES_TO_KEEP),
    max_context_tokens: Optional[int] = None,
    **kwargs: Any,
) -> SummarizationMiddleware:
    """Create a summarization middleware.

    Args:
        model: LLM for summarization
        trigger: When to trigger summarization
        keep: What to keep after summarization
        max_context_tokens: Maximum context tokens for fraction calculations
        **kwargs: Additional arguments passed to SummarizationMiddleware

    Returns:
        Configured SummarizationMiddleware
    """
    return SummarizationMiddleware(
        model=model,
        trigger=trigger,
        keep=keep,
        max_context_tokens=max_context_tokens,
        **kwargs,
    )
