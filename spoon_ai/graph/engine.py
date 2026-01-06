"""
Graph engine: StateGraph, CompiledGraph, and interrupt API implementation.
"""
import asyncio
import logging
import uuid
import inspect
import time
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Union, Pattern, TypeVar, Generic, TypedDict, Literal, Iterable
from abc import ABC, abstractmethod

from .exceptions import (
    GraphExecutionError,
    NodeExecutionError,
    GraphConfigurationError,
    EdgeRoutingError,
    InterruptError,
    CheckpointError,
)
from .types import (
    NodeContext,
    NodeResult,
    RouterResult,
    ParallelBranchConfig,
    Command,
    StateSnapshot,
)
from .reducers import (
    merge_dicts,
    add_messages,
)
from .decorators import node_decorator
from .checkpointer import InMemoryCheckpointer
from .cache import BaseCache, compute_cache_key
from spoon_ai.schema import (
    Message, MessageContent, ContentBlock,
    TextContent, ImageContent, ImageUrlContent, ImageSource, ImageUrlSource,
    DocumentContent, DocumentSource
)
from .config import GraphConfig, ParallelGroupConfig, ParallelRetryPolicy, RouterConfig

logger = logging.getLogger(__name__)


# =============================================================================
# Multimodal Message Utilities for Graph Workflows
# =============================================================================

def create_multimodal_message(
    role: str,
    text: str,
    image_url: Optional[str] = None,
    image_data: Optional[str] = None,
    image_media_type: str = "image/png",
    detail: Literal["auto", "low", "high"] = "auto"
) -> Message:
    """Create a multimodal message for use in graph state.

    Supports both URL-based and base64-encoded images.

    Args:
        role: Message role (user, assistant, system)
        text: Text content
        image_url: URL of the image (including data URLs)
        image_data: Base64-encoded image data (alternative to image_url)
        image_media_type: MIME type for base64 images
        detail: Image detail level (auto, low, high)

    Returns:
        Message: A multimodal message ready for graph state

    Example:
        ```python
        # In a graph node function
        async def analyze_image(state: State) -> dict:
            msg = create_multimodal_message(
                "user",
                "Analyze this chart",
                image_url="https://example.com/chart.png"
            )
            return {"messages": [msg]}
        ```
    """
    if not image_url and not image_data:
        # Text-only message
        return Message(role=role, content=text)

    content_blocks: List[ContentBlock] = [TextContent(text=text)]

    if image_url:
        content_blocks.append(
            ImageUrlContent(image_url=ImageUrlSource(url=image_url, detail=detail))
        )
    elif image_data:
        content_blocks.append(
            ImageContent(source=ImageSource(
                type="base64",
                media_type=image_media_type,
                data=image_data
            ))
        )

    return Message(role=role, content=content_blocks)


def create_vision_user_message(
    text: str,
    images: List[Dict[str, str]]
) -> Message:
    """Create a user message with multiple images.

    Args:
        text: Text prompt
        images: List of image specs, each with either:
            - {"url": "https://..."} for URL-based images
            - {"data": "<base64>", "media_type": "image/png"} for base64 images

    Returns:
        Message: A multimodal message with multiple images

    Example:
        ```python
        msg = create_vision_user_message(
            "Compare these two charts",
            images=[
                {"url": "https://example.com/chart1.png"},
                {"url": "https://example.com/chart2.png"}
            ]
        )
        ```
    """
    content_blocks: List[ContentBlock] = [TextContent(text=text)]

    for img in images:
        if "url" in img:
            content_blocks.append(
                ImageUrlContent(image_url=ImageUrlSource(
                    url=img["url"],
                    detail=img.get("detail", "auto")
                ))
            )
        elif "data" in img:
            content_blocks.append(
                ImageContent(source=ImageSource(
                    type="base64",
                    media_type=img.get("media_type", "image/png"),
                    data=img["data"]
                ))
            )

    return Message(role="user", content=content_blocks)


def create_pdf_message(
    role: str,
    text: str,
    pdf_data: str,
    filename: Optional[str] = None
) -> Message:
    """Create a message with a PDF document for use in graph state.

    Args:
        role: Message role (user, assistant, system)
        text: Text content
        pdf_data: Base64-encoded PDF data
        filename: Optional filename for the PDF

    Returns:
        Message: A multimodal message with PDF ready for graph state

    Example:
        ```python
        # In a graph node function
        async def analyze_document(state: State) -> dict:
            msg = create_pdf_message(
                "user",
                "Summarize this whitepaper",
                pdf_data="<base64_encoded_pdf>",
                filename="bitcoin.pdf"
            )
            return {"messages": [msg]}
        ```
    """
    content_blocks: List[ContentBlock] = [TextContent(text=text)]
    content_blocks.append(
        DocumentContent(
            source=DocumentSource(
                type="base64",
                media_type="application/pdf",
                data=pdf_data
            ),
            filename=filename
        )
    )

    return Message(role=role, content=content_blocks)


def create_document_message(
    role: str,
    text: str,
    document_data: str,
    media_type: str = "application/pdf",
    filename: Optional[str] = None
) -> Message:
    """Create a message with a document for use in graph state.

    Supports various document types including PDF, text files, etc.

    Args:
        role: Message role (user, assistant, system)
        text: Text content
        document_data: Base64-encoded document data
        media_type: MIME type of the document (default: application/pdf)
        filename: Optional filename for the document

    Returns:
        Message: A multimodal message with document ready for graph state

    Example:
        ```python
        # In a graph node function
        async def process_report(state: State) -> dict:
            msg = create_document_message(
                "user",
                "Extract key metrics from this report",
                document_data="<base64_encoded_data>",
                media_type="application/pdf",
                filename="annual_report.pdf"
            )
            return {"messages": [msg]}
        ```
    """
    content_blocks: List[ContentBlock] = [TextContent(text=text)]
    content_blocks.append(
        DocumentContent(
            source=DocumentSource(
                type="base64",
                media_type=media_type,
                data=document_data
            ),
            filename=filename
        )
    )

    return Message(role=role, content=content_blocks)


# Type variables for generic state handling
State = TypeVar('State')
ConfigurableFieldSpec = Dict[str, Any]

START = "__start__"
END = "__end__"

class BaseNode(ABC, Generic[State]):
    """Base class for all graph nodes"""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    async def __call__(self, state: State, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute the node logic"""
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"


class RunnableNode(BaseNode[State]):
    """Runnable node that wraps a function"""

    def __init__(self, name: str, func: Callable[[State], Any]):
        super().__init__(name)
        self.func = func

    async def __call__(self, state: State, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute the wrapped function"""
        try:
            if asyncio.iscoroutinefunction(self.func):
                result = await self.func(state)
            else:
                result = self.func(state)

            # Handle different return types
            if isinstance(result, dict):
                return result
            elif isinstance(result, (list, tuple)) and len(result) == 2:
                # Handle (updates, next_node) format
                updates, next_node = result
                return {"updates": updates, "next_node": next_node}
            else:
                return {"result": result}

        except InterruptError:
            # Human-in-the-loop: allow interrupts to propagate to the graph engine.
            # Let InterruptError propagate to be handled in invoke()
            raise
        except Exception as e:
            logger.error(f"Node {self.name} execution failed: {e}")
            raise NodeExecutionError(f"Node '{self.name}' failed", node_name=self.name, original_error=e, state=state) from e


class ToolNode(BaseNode[State]):
    """Tool node for executing tools"""

    def __init__(self, name: str, tools: List[Any]):
        super().__init__(name)
        self.tools = tools

    async def __call__(self, state: State, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute tools based on state"""
        # Extract tool calls from state
        tool_calls = state.get("tool_calls", [])
        results = []

        for tool_call in tool_calls:
            tool_name = tool_call.get("name")
            tool_args = tool_call.get("args", {})

            # Find the tool
            tool = next((t for t in self.tools if getattr(t, 'name', None) == tool_name), None)
            if not tool:
                raise NodeExecutionError(f"Tool '{tool_name}' not found", node_name=self.name, state=state)

            # Execute tool
            try:
                if hasattr(tool, 'execute') and asyncio.iscoroutinefunction(tool.execute):
                    result = await tool.execute(**tool_args)
                elif hasattr(tool, 'execute'):
                    result = tool.execute(**tool_args)
                else:
                    # Direct function call
                    result = tool(**tool_args)

                results.append({
                    "tool_call": tool_call,
                    "result": result,
                    "success": True
                })
            except Exception as e:
                results.append({
                    "tool_call": tool_call,
                    "result": None,
                    "success": False,
                    "error": str(e)
                })

        return {"tool_results": results}


class ConditionNode(BaseNode[State]):
    """Conditional node for routing decisions"""

    def __init__(self, name: str, condition_func: Callable[[State], str]):
        super().__init__(name)
        self.condition_func = condition_func

    async def __call__(self, state: State, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute condition and return routing decision"""
        try:
            result = self.condition_func(state)
            if asyncio.iscoroutinefunction(self.condition_func):
                result = await self.condition_func(state)

            return {"condition_result": result, "next_node": result}
        except InterruptError:
            # Human-in-the-loop: allow interrupts to propagate to the graph engine.
            raise
        except Exception as e:
            logger.error(f"Condition node {self.name} failed: {e}")
            raise NodeExecutionError(f"Condition '{self.name}' failed", node_name=self.name, original_error=e, state=state) from e


def interrupt(data: Dict[str, Any]) -> Any:
    """Interrupt execution and wait for human input."""
    raise InterruptError(data)


class RouteRule:
    """Advanced routing rule for automatic path selection"""

    def __init__(self, condition: Union[str, Callable, Pattern], target: str, priority: int = 0):
        self.condition = condition
        self.target = target
        self.priority = priority

    def matches(self, state: Dict[str, Any], query: str = "") -> bool:
        """Check if this rule matches the current state/query"""
        if isinstance(self.condition, str):
            # Simple string matching
            return self.condition.lower() in (query + str(state)).lower()
        elif isinstance(self.condition, Pattern):
            # Regex pattern matching
            return bool(self.condition.search(query + str(state)))
        elif callable(self.condition):
            # Custom function matching
            return self.condition(state, query)
        return False


@dataclass
class RunningSummary:
    """Rolling conversation summary used by the summarisation node."""

    summary: str = ""
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": self.summary,
            "last_updated": self.last_updated.isoformat(),
        }

    @classmethod
    def from_value(cls, value: Any) -> "RunningSummary":
        if isinstance(value, cls):
            return value
        if isinstance(value, dict):
            summary = value.get("summary", "")
            last_updated_val = value.get("last_updated")
            if isinstance(last_updated_val, datetime):
                last_updated = last_updated_val
            elif isinstance(last_updated_val, str):
                last_updated = datetime.fromisoformat(last_updated_val)
            else:
                last_updated = datetime.now(timezone.utc)
            return cls(summary=summary, last_updated=last_updated)
        return cls()


class SummarizationNode(BaseNode[Dict[str, Any]]):
    """Node that summarises conversation history before model invocation."""

    def __init__(
        self,
        name: str,
        *,
        llm_manager,
        max_tokens: int,
        messages_to_keep: int = 5,
        summary_model: Optional[str] = None,
        summary_key: str = "summary_context",
        output_messages_key: str = "llm_messages",
        manager: Optional["ShortTermMemoryManager"] = None,
    ) -> None:
        super().__init__(name)
        self.llm_manager = llm_manager
        self.max_tokens = max_tokens
        self.messages_to_keep = messages_to_keep
        self.summary_model = summary_model
        self.summary_key = summary_key
        self.output_messages_key = output_messages_key
        if manager is None:
            from spoon_ai.memory.short_term_manager import (
                ShortTermMemoryManager,
                MessageTokenCounter,
            )

            manager = ShortTermMemoryManager(token_counter=MessageTokenCounter())
        self.manager = manager

    async def __call__(self, state: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        messages: List[Message] = state.get("messages", [])
        if not messages:
            return {self.output_messages_key: []}

        running_summary = RunningSummary.from_value(state.get(self.summary_key))

        llm_messages, removals, summary_text = await self.manager.summarize_messages(
            messages=messages,
            max_tokens_before_summary=self.max_tokens,
            messages_to_keep=self.messages_to_keep,
            summary_model=self.summary_model,
            llm_manager=self.llm_manager,
            existing_summary=running_summary.summary,
        )

        updates: Dict[str, Any] = {self.output_messages_key: llm_messages}

        if summary_text:
            running_summary.summary = summary_text
            running_summary.last_updated = datetime.now(timezone.utc)
            updates[self.summary_key] = running_summary.to_dict()

        if removals:
            updates.setdefault("messages", removals)

        return updates


class StateGraph(Generic[State]):
    def __init__(self, state_schema: type, checkpointer: Optional[Any] = None, config_schema: Optional[type] = None):
        self.state_schema = state_schema
        self.config_schema = config_schema
        # Default to in-memory checkpointer if none provided
        self.checkpointer = checkpointer or InMemoryCheckpointer()

        # Node storage
        self.nodes: Dict[str, BaseNode[State]] = {}
        self.node_functions: Dict[str, Callable] = {}  # For backward compatibility

        # Edge management
        self.edges: Dict[str, List[tuple]] = {}  # (end_node, condition_func)
        self.conditional_edges: Dict[str, Dict[str, Callable[[State], bool]]] = {}

        # Special handling for START and END
        self._entry_point: Optional[str] = None
        self._compiled = False

        # Enhanced features
        self.routing_rules: Dict[str, List[RouteRule]] = {}
        self.intelligent_router: Optional[Callable[[Dict[str, Any], str], str]] = None

        # LLM Router
        self.llm_router: Optional[Callable[[Dict[str, Any], str], Union[str, None, asyncio.Future]]] = None
        self.llm_router_config: Dict[str, Any] = {
            "model": "gpt-4",
            "temperature": 0.1,
            "max_tokens": 100,
            "timeout": 8,
        }

        # Parallel execution
        self.parallel_branches: Dict[str, List[str]] = {}
        self.parallel_groups: Dict[str, List[str]] = {}
        self.parallel_group_configs: Dict[str, Dict[str, Any]] = {}
        self.node_to_group: Dict[str, str] = {}
        self.parallel_entry_nodes: List[str] = []

        # Configuration
        self.config: GraphConfig = GraphConfig()
        self.stream_mode: str = "values"
        self.stream_channels: List[str] = ["values", "updates", "debug"]

        # Monitoring
        self.monitoring_enabled: bool = False
        self.monitoring_metrics: List[str] = []

        # Optional hooks
        self.state_cleanup: Optional[Callable[[Dict[str, Any]], None]] = None
        self.state_validator: Optional[Callable[[Dict[str, Any]], None]] = None

    def enable_monitoring(self, metrics: Optional[List[str]] = None) -> "StateGraph":
        self.monitoring_enabled = True
        if metrics:
            self.monitoring_metrics = metrics
        return self

    def add_node(self, node_name: str, node: Union[BaseNode[State], Callable[[State], Any]]) -> "StateGraph":
        """Add a node to the graph"""
        if node_name in [START, END]:
            raise GraphConfigurationError(f"Node name '{node_name}' is reserved", component="node")

        if isinstance(node, BaseNode):
            self.nodes[node_name] = node
        elif callable(node):
            # Wrap function in RunnableNode
            self.nodes[node_name] = RunnableNode(node_name, node)
            self.node_functions[node_name] = node  # For backward compatibility
        else:
            raise GraphConfigurationError(f"Node must be callable or BaseNode instance", component="node")

        return self

    def add_edge(self, start_node: str, end_node: str, condition: Optional[Callable[[State], bool]] = None) -> "StateGraph":
        """Add an edge. When condition is provided, edge becomes conditional."""
        if start_node not in self.nodes and start_node != START:
            raise GraphConfigurationError(f"Start node '{start_node}' does not exist", component="edge")
        if end_node not in self.nodes and end_node != END:
            raise GraphConfigurationError(f"End node '{end_node}' does not exist", component="edge")

        if condition is not None and not callable(condition):
            raise GraphConfigurationError("Edge condition must be callable", component="edge")

        if start_node not in self.edges:
            self.edges[start_node] = []
        self.edges[start_node].append((end_node, condition))
        return self

    def add_conditional_edges(self, start_node: str, condition: Callable[[State], str],
                             path_map: Dict[str, str]) -> "StateGraph":
        """Add conditional edges"""
        if start_node not in self.nodes and start_node != START:
            raise GraphConfigurationError(f"Start node '{start_node}' does not exist", component="conditional_edge")

        # Validate path_map destinations
        for path, target in path_map.items():
            if target not in self.nodes and target != END:
                raise GraphConfigurationError(f"Target node '{target}' does not exist", component="conditional_edge")

        if start_node not in self.edges:
            self.edges[start_node] = []
        self.edges[start_node].append((condition, path_map))
        return self

    def set_entry_point(self, node_name: str) -> "StateGraph":
        """Set the entry point"""
        if node_name not in self.nodes and node_name != START:
            raise GraphConfigurationError(f"Entry point node '{node_name}' does not exist", component="entry_point")
        self._entry_point = node_name
        return self

    def add_tool_node(self, tools: List[Any], name: str = "tools") -> "StateGraph":
        """Add a tool node"""
        tool_node = ToolNode(name, tools)
        return self.add_node(name, tool_node)

    def add_conditional_node(self, condition_func: Callable[[State], str], name: str = "condition") -> "StateGraph":
        """Add a conditional node"""
        condition_node = ConditionNode(name, condition_func)
        return self.add_node(name, condition_node)

    def add_parallel_group(self, group_name: str, nodes: List[str],
                          config: Optional[Union[Dict[str, Any], ParallelGroupConfig]] = None) -> "StateGraph":
        """Add a parallel execution group"""
        # Validate that all nodes exist
        for node_name in nodes:
            if node_name not in self.nodes and node_name not in [START, END]:
                raise GraphConfigurationError(f"Node '{node_name}' does not exist", component="parallel_group")

        self.parallel_groups[group_name] = nodes
        if isinstance(config, ParallelGroupConfig):
            group_cfg = config
        elif isinstance(config, dict):
            group_cfg = ParallelGroupConfig(**config)
        else:
            group_cfg = self.config.parallel_groups.get(group_name, ParallelGroupConfig())

        self.parallel_group_configs[group_name] = group_cfg

        # Mark nodes as belonging to this group
        for node_name in nodes:
            self.node_to_group[node_name] = group_name

        # Mark first node as entry point for parallel execution
        if nodes:
            self.parallel_entry_nodes.append(nodes[0])

        return self

    def add_routing_rule(self, source_node: str, condition: Union[str, Callable[[State, str], bool]],
                        target_node: str, priority: int = 0) -> "StateGraph":
        """Add an intelligent routing rule"""
        if source_node not in self.nodes and source_node != START and source_node != "__start__":
            raise GraphConfigurationError(f"Source node '{source_node}' does not exist", component="routing_rule")
        if target_node not in self.nodes and target_node != END and target_node != "__end__":
            raise GraphConfigurationError(f"Target node '{target_node}' does not exist", component="routing_rule")

        if source_node not in self.routing_rules:
            self.routing_rules[source_node] = []

        rule = RouteRule(condition, target_node, priority)
        self.routing_rules[source_node].append(rule)

        # Sort rules by priority (highest first)
        self.routing_rules[source_node].sort(key=lambda r: r.priority, reverse=True)

        return self

    def get_state(self, config: Optional[Dict[str, Any]] = None) -> Optional[StateSnapshot]:
        """Fetch the latest (or specified) checkpoint snapshot for a thread."""
        if not self.checkpointer:
            raise CheckpointError("No checkpointer configured for this graph", operation="get_state")

        config = config or {}
        configurable = config.get("configurable", {})
        thread_id = configurable.get("thread_id")
        checkpoint_id = configurable.get("checkpoint_id")

        if not thread_id:
            raise CheckpointError("thread_id is required to fetch state", operation="get_state")

        return self.checkpointer.get_checkpoint(thread_id, checkpoint_id)

    def get_state_history(self, config: Optional[Dict[str, Any]] = None) -> Iterable[StateSnapshot]:
        """Return all checkpoints for the given thread, ordered by creation time."""
        if not self.checkpointer:
            raise CheckpointError("No checkpointer configured for this graph", operation="state_history")

        config = config or {}
        configurable = config.get("configurable", {})
        thread_id = configurable.get("thread_id")

        if not thread_id:
            raise CheckpointError("thread_id is required to fetch state history", operation="state_history")

        return list(self.checkpointer.list_checkpoints(thread_id))

    def add_pattern_routing(self, source_node: str, pattern: str, target_node: str,
                           priority: int = 0) -> "StateGraph":
        """Add pattern-based routing rule"""
        try:
            compiled_pattern = re.compile(pattern, re.IGNORECASE)
        except re.error as e:
            raise GraphConfigurationError(f"Invalid regex pattern: {pattern}", component="pattern_routing")

        return self.add_routing_rule(source_node, compiled_pattern, target_node, priority)

    def set_intelligent_router(self, router_func: Callable[[Dict[str, Any], str], str]) -> "StateGraph":
        """Set the intelligent router function"""
        self.intelligent_router = router_func
        return self

    def set_llm_router(self, router_func: Optional[Callable[[Dict[str, Any], str], str]] = None,
                       config: Optional[Dict[str, Any]] = None) -> "StateGraph":
        """Set the LLM-powered router function

        Args:
            router_func: Custom LLM router function. If None, uses default LLM router.
            config: Configuration for LLM router (model, temperature, max_tokens, etc.)
        """
        if config:
            self.llm_router_config.update(config)

        if router_func:
            self.llm_router = router_func
        else:
            # Use default LLM router - will be created when needed
            self.llm_router = self._create_default_llm_router

        return self

    async def _create_default_llm_router(self, state: Dict[str, Any], query: str) -> str:
        """Create and use default LLM router for natural language routing"""
        try:
            # Lazy import to avoid circular dependencies
            from spoon_ai.llm.manager import get_llm_manager

            llm_manager = get_llm_manager()
            config = self.llm_router_config

            routing_prompt = f"""
            Analyze this user query and determine the BEST next step in our workflow.
            Return ONLY the step name from these options:

            Available steps:
            {self._get_available_steps_for_routing()}

            Query: "{query}"

            Consider the current state and conversation context.
            Return ONLY the step name (lowercase, no explanation).
            """

            messages = [Message(role="user", content=routing_prompt)]

            # Use LLM to determine next step
            response = await llm_manager.chat(messages, provider=None, **config)
            next_step = response.content.strip().lower()

            # Validate the response
            available_steps = self._get_available_step_names()
            logger.info(f"LLM Router - Query: '{query}'")
            logger.info(f"LLM Router - Available steps: {available_steps}")
            logger.info(f"LLM Router - LLM response: '{response.content}'")
            logger.info(f"LLM Router - Parsed step: '{next_step}'")

            if next_step not in available_steps:
                logger.warning(f"LLM returned invalid step '{next_step}', available: {available_steps}")
                # Instead of defaulting to entry point, try to find a reasonable fallback
                if "price" in query.lower() or "market" in query.lower():
                    if "fetch_market_data" in available_steps:
                        return "fetch_market_data"
                elif "buy" in query.lower() or "sell" in query.lower() or "trade" in query.lower():
                    if "execute_trade" in available_steps:
                        return "execute_trade"
                elif "analyze" in query.lower():
                    if "analyze_market" in available_steps:
                        return "analyze_market"
                else:
                    if "generate_response" in available_steps:
                        return "generate_response"

            logger.info(f"LLM Router decided: {next_step}")
            return next_step

        except Exception as e:
            logger.error(f"LLM routing failed: {e}")
            # Fallback to entry point
            return self._entry_point or list(self.nodes.keys())[0]

    def _get_available_steps_for_routing(self) -> str:
        """Get available steps description for LLM routing"""
        steps = []
        for node_name in self.nodes.keys():
            if node_name not in [START, END]:
                steps.append(f"- {node_name}: Execute {node_name.replace('_', ' ')}")

        if not steps:
            return "- analyze_intent: Analyze user intent"

        return "\n".join(steps)

    def _get_available_step_names(self) -> List[str]:
        """Get list of available step names for validation"""
        return [name for name in self.nodes.keys() if name not in [START, END]]

    def enable_llm_routing(self, config: Optional[Dict[str, Any]] = None) -> "StateGraph":
        """Enable LLM-powered natural language routing

        This automatically sets up LLM routing for the graph entry point.
        """
        if config:
            self.llm_router_config.update(config)

        # Enabling LLM routing implies allowing the router to call the LLM.
        # Keep this best-effort to avoid surprising failures if config is not a GraphConfig.
        try:
            if isinstance(self.config, GraphConfig):
                self.config.router.allow_llm = True
        except Exception:
            pass

        # Set LLM router as the intelligent router
        self.set_llm_router()
        return self

    def compile(
        self,
        checkpointer: Optional[Any] = None,
        cache: Optional[BaseCache] = None,
    ) -> "CompiledGraph":
        """Compile the graph for execution.

        Args:
            checkpointer: Optional checkpointer for session persistence.
                If not provided, uses the graph's default checkpointer.
            cache: Optional cache for node-level caching.
                When provided, node outputs are cached based on their inputs,
                allowing repeated calls with the same inputs to skip execution.

        Returns:
            CompiledGraph ready for execution.

        Raises:
            GraphConfigurationError: If graph configuration is invalid.

        Example:
            ```python
            from spoon_ai.graph import StateGraph, InMemoryCache, InMemoryCheckpointer

            graph = StateGraph(MyState)
            graph.add_node("process", process_node)
            graph.add_edge(START, "process")
            graph.add_edge("process", END)

            # Compile with both checkpointer and cache
            compiled = graph.compile(
                checkpointer=InMemoryCheckpointer(),
                cache=InMemoryCache(max_entries=100),
            )

            # First call - executes node
            result1 = await compiled.ainvoke({"input": "hello"})

            # Second call with same input - uses cache
            result2 = await compiled.ainvoke({"input": "hello"})
            ```
        """
        errors: List[str] = []

        if not self._entry_point:
            errors.append("Graph must have an entry point")

        if not self.nodes:
            errors.append("Graph must have at least one node")

        if errors:
            raise GraphConfigurationError(
                f"Graph compilation failed: {'; '.join(errors)}",
                component="compilation",
                details={"errors": errors}
            )

        self._compiled = True
        return CompiledGraph(self, checkpointer, cache)

    def get_graph(self) -> Dict[str, Any]:
        """Get graph structure for visualization/debugging"""
        return {
            "nodes": list(self.nodes.keys()),
            "edges": self.edges,
            "entry_point": self._entry_point,
            "config": self.config,
        }




class CompiledGraph(Generic[State]):
    """Compiled graph for execution.

    Supports:
    - Async and sync execution via ainvoke/invoke
    - Session persistence via checkpointer
    - Node-level caching via cache
    - Interrupt/resume for human-in-the-loop workflows
    """

    def __init__(
        self,
        graph: StateGraph[State],
        checkpointer: Optional[Any] = None,
        cache: Optional[BaseCache] = None,
    ):
        """Initialize compiled graph.

        Args:
            graph: The StateGraph to execute
            checkpointer: Optional checkpointer for session persistence
            cache: Optional cache for node-level caching
        """
        self.graph = graph
        self.checkpointer = checkpointer or graph.checkpointer
        self.cache = cache

        # Execution state
        self.execution_history: List[Dict[str, Any]] = []
        self.max_execution_history = 1000

        # Resume functionality
        self._resume_thread_id: Optional[str] = None
        self._resume_checkpoint_id: Optional[str] = None

        # Cache statistics
        self._cache_hits = 0
        self._cache_misses = 0

        # Current execution context
        self._current_node: Optional[str] = None
        self._iteration: int = 0

        # Execution metrics
        self.execution_metrics = {
            "total_executions": 0,
            "success_rate": 0.0,
            "average_execution_time": 0.0,
            "routing_performance": {}
        }

    def _find_matching_route(self, current_node: str, state: Dict[str, Any]) -> Optional[str]:
        """Find matching routing rule for the current node and state"""
        if current_node not in self.graph.routing_rules:
            return None

        query = state.get("user_query", "").lower()

        # Check routing rules in priority order
        for rule in self.graph.routing_rules[current_node]:
            if rule.matches(state, query):
                return rule.target

        return None

    def _find_edge_target(self, current_node: str, state: Dict[str, Any]) -> Optional[str]:
        if current_node in self.graph.edges:
            for edge_target, edge_condition in self.graph.edges[current_node]:
                if edge_condition is None and isinstance(edge_target, str) and (edge_target in self.graph.nodes or edge_target == END):
                    return edge_target
                if callable(edge_target) and isinstance(edge_condition, dict):
                    try:
                        cond_key = edge_target(state)
                        if isinstance(cond_key, str) and cond_key in edge_condition:
                            return edge_condition[cond_key]
                    except Exception as e:
                        logger.warning(f"Conditional map evaluation failed: {e}")
                if isinstance(edge_target, str) and callable(edge_condition):
                    try:
                        if edge_condition(state):
                            return edge_target
                    except Exception as e:
                        logger.warning(f"Predicate condition failed: {e}")
        return None

    async def _determine_next_node(self, current_node: str, state: Dict[str, Any]) -> Optional[str]:
        """Determine the next node to execute (async to support async LLM router)."""
        query = state.get("user_query", "")
        logger.info(f"Getting next node from '{current_node}' for query: '{query}'")

        graph_cfg = self.graph.config if isinstance(self.graph.config, GraphConfig) else GraphConfig()
        router_cfg: RouterConfig = graph_cfg.router

        # Priority 1: Explicit edges
        logger.info("Checking explicit edges...")
        explicit_target = self._find_edge_target(current_node, state)
        if explicit_target:
            return explicit_target

        # Priority 2: Intelligent routing rules
        logger.info("Trying routing rules...")
        matching_route = self._find_matching_route(current_node, state)
        if matching_route:
            logger.info(f"Routing rule matched: {matching_route}")
            return matching_route

        # Priority 3: Intelligent router function
        if self.graph.intelligent_router:
            logger.info("Trying intelligent router...")
            try:
                router = self.graph.intelligent_router
                next_node = await router(state, query) if asyncio.iscoroutinefunction(router) else router(state, query)
                if next_node and next_node != current_node:
                    if router_cfg.allowed_targets and next_node not in router_cfg.allowed_targets:
                        logger.warning(f"Intelligent router returned disallowed target '{next_node}'")
                    else:
                        logger.info(f"Intelligent router selected: {next_node}")
                        return next_node
            except Exception as e:
                logger.warning(f"Intelligent router failed: {e}")

        # Priority 4: LLM Router (if available)
        if router_cfg.allow_llm and self.graph.llm_router:
            try:
                router = self.graph.llm_router
                if asyncio.iscoroutinefunction(router):
                    next_node = await asyncio.wait_for(router(state, query), timeout=router_cfg.llm_timeout)
                else:
                    next_node = router(state, query)
                if next_node and next_node != current_node and next_node in self.graph.nodes:
                    if router_cfg.allowed_targets and next_node not in router_cfg.allowed_targets:
                        logger.warning(f"LLM router returned disallowed target '{next_node}'")
                    else:
                        logger.info(f"LLM Router selected: {next_node}")
                        return next_node
            except Exception as e:
                logger.warning(f"LLM router failed: {e}")

        # Priority 5: Default target
        if router_cfg.enable_fallback_to_default and router_cfg.default_target:
            target = router_cfg.default_target
            if router_cfg.allowed_targets and target not in router_cfg.allowed_targets:
                logger.warning(f"Default target '{target}' not in allowed targets")
            elif target in self.graph.nodes:
                logger.info(f"Using default router target: {target}")
                return target

        logger.debug("No valid next node found - graph execution complete")
        return None

    async def invoke(self, initial_state: Optional[Dict[str, Any]] = None, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        config = config or {}
        thread_id = config.get("configurable", {}).get("thread_id", str(uuid.uuid4()))

        # Handle Command object (for resume/goto/update)
        if isinstance(initial_state, Command):
            cmd = initial_state
            # Handle resume: load checkpoint and merge updates
            if cmd.resume is not None:
                checkpoint = self.graph.checkpointer.get_checkpoint(thread_id)
                if checkpoint:
                    state = checkpoint.values.copy()
                    # Merge resume data into state
                    if isinstance(cmd.resume, dict):
                        state.update(cmd.resume)
                    current_node = checkpoint.next[0] if checkpoint.next else self.graph._entry_point
                    iteration = checkpoint.metadata.get("iteration", 0) + 1
                else:
                    raise GraphExecutionError(f"No checkpoint found for thread_id '{thread_id}' to resume from")
            # Handle goto: jump to specific node
            elif cmd.goto:
                state = self._initialize_state(cmd.update or {})
                current_node = cmd.goto
                iteration = 0
            # Handle update: update state and continue
            elif cmd.update:
                checkpoint = self.graph.checkpointer.get_checkpoint(thread_id)
                if checkpoint:
                    state = checkpoint.values.copy()
                    state.update(cmd.update)
                    current_node = checkpoint.next[0] if checkpoint.next else self.graph._entry_point
                    iteration = checkpoint.metadata.get("iteration", 0) + 1
                else:
                    state = self._initialize_state(cmd.update)
                    current_node = self.graph._entry_point
                    iteration = 0
            else:
                state = self._initialize_state({})
                current_node = self.graph._entry_point
                iteration = 0
        # Handle resume from checkpoint (legacy)
        elif self._resume_thread_id:
            thread_id = self._resume_thread_id
            checkpoint = self.graph.checkpointer.get_checkpoint(thread_id, self._resume_checkpoint_id)
            if checkpoint:
                state = checkpoint.values.copy()
                current_node = checkpoint.next[0] if checkpoint.next else self.graph.entry_point
                iteration = checkpoint.metadata.get("iteration", 0)
            else:
                state = self._initialize_state(initial_state)
                current_node = self.graph._entry_point
                iteration = 0
        else:
            state = self._initialize_state(initial_state)
            current_node = self.graph._entry_point
            iteration = 0

        max_iterations = int(config.get("max_iterations", 100) or 100)
        self._current_thread_id = thread_id
        try:
            while current_node and iteration < max_iterations:
                self._current_iteration = iteration
                iteration += 1
                # checkpoint (best-effort)
                try:
                    snapshot = StateSnapshot(values=state.copy(), next=(current_node,), config=config, metadata={"iteration": iteration, "node": current_node}, created_at=datetime.now())
                    self.graph.checkpointer.save_checkpoint(thread_id, snapshot)
                except Exception:
                    pass
                # execute current node or parallel group
                try:
                    # Check if current node is part of a parallel group
                    if current_node in self.graph.node_to_group:
                        group_name = self.graph.node_to_group[current_node]
                        logger.info(f"Executing parallel group: {group_name}")
                        await self._execute_parallel_group(group_name, state)
                    else:
                        result = await self._execute_node(current_node, state)
                        if isinstance(result, dict):
                            self._update_state_with_reducers(state, result)
                            self._maybe_cleanup_state(state)
                            # optional validation
                            try:
                                if callable(self.graph.state_validator):
                                    self.graph.state_validator(state)
                                # Also support GraphConfig.state_validators (list of callables)
                                if isinstance(self.graph.config, GraphConfig):
                                    for validator in self.graph.config.state_validators:
                                        if callable(validator):
                                            validator(state)
                            except Exception as e:
                                raise GraphExecutionError(f"State validation failed: {e}", node=current_node, iteration=iteration)
                except InterruptError as e:
                    # record interrupt + checkpoint
                    try:
                        interrupt_snapshot = StateSnapshot(values=state.copy(), next=(current_node,), config=config, metadata={"iteration": iteration, "node": current_node, "interrupt_id": e.interrupt_id, "interrupt_data": e.interrupt_data, "status": "interrupted"}, created_at=datetime.now())
                        self.graph.checkpointer.save_checkpoint(thread_id, interrupt_snapshot)
                    except Exception:
                        pass
                    return {**state, "__interrupt__": [{"interrupt_id": e.interrupt_id, "value": e.interrupt_data, "node": current_node, "iteration": iteration}]}

                # next - use intelligent routing
                next_node = await self._determine_next_node(current_node, state)
                if next_node == current_node:
                    break
                current_node = next_node
                if current_node == END or current_node is None:
                    break
            if iteration >= max_iterations:
                raise GraphExecutionError(f"Graph execution exceeded maximum iterations ({max_iterations})", node=current_node, iteration=iteration)
            return state
        except GraphExecutionError:
            raise
        except Exception as e:
            raise GraphExecutionError(f"Graph execution failed: {e}", node=current_node, iteration=iteration) from e


    def _initialize_state(self, initial_state: State) -> State:
        """Initialize state for execution"""
        if initial_state is None:
            # Create default state based on schema
            if hasattr(self.graph.state_schema, '__annotations__'):
                state = {}
                for field_name, field_type in self.graph.state_schema.__annotations__.items():
                    if hasattr(field_type, '__origin__') and field_type.__origin__ is list:
                        state[field_name] = []
                    elif hasattr(field_type, '__origin__') and field_type.__origin__ is dict:
                        state[field_name] = {}
                    else:
                        state[field_name] = None
                return state
            else:
                return {}
        return initial_state

    async def _execute_node(self, node_name: str, state: State, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute a node and return its result.

        If cache is configured, checks for cached result before execution
        and stores result in cache after execution.

        Args:
            node_name: Name of the node to execute
            state: Current graph state
            config: Optional execution configuration

        Returns:
            Node execution result as a dictionary
        """
        node = self.graph.nodes.get(node_name)
        if not node:
            raise GraphExecutionError(f"Node '{node_name}' not found")

        # Check cache if available
        cache_key = None
        if self.cache is not None:
            try:
                # Compute cache key from node name and state
                cache_key = compute_cache_key(node_name, dict(state), config)
                cached_result = self.cache.get(cache_key)
                if cached_result is not None:
                    self._cache_hits += 1
                    logger.debug(f"Cache hit for node '{node_name}' (key: {cache_key[:16]}...)")
                    return cached_result
                self._cache_misses += 1
            except Exception as e:
                logger.warning(f"Cache lookup failed for node '{node_name}': {e}")

        try:
            start_dt = datetime.now()
            # Call the node with proper parameters
            if hasattr(node, '__call__'):
                if config is not None:
                    result = await node(state, config)
                else:
                    result = await node(state)
            else:
                # Fallback for old-style nodes
                result = await node(state)
            end_dt = datetime.now()

            # Normalize result to dict
            result = result if isinstance(result, dict) else {"result": result}

            # Store in cache if available
            if self.cache is not None and cache_key is not None:
                try:
                    self.cache.set(cache_key, result, node_name=node_name)
                    logger.debug(f"Cached result for node '{node_name}' (key: {cache_key[:16]}...)")
                except Exception as e:
                    logger.warning(f"Cache store failed for node '{node_name}': {e}")

            # Record metrics
            try:
                self._record_execution_metrics(node_name, start_dt, end_dt, True, metadata={})
            except Exception:
                pass

            return result
        except InterruptError:
            # Human-in-the-loop: allow interrupts to propagate to the graph engine.
            # Let InterruptError propagate to be handled in invoke()
            raise
        except Exception as e:
            logger.error(f"Node {node_name} execution failed: {e}")
            try:
                self._record_execution_metrics(node_name, start_dt, datetime.now(), False, error=str(e))
            except Exception:
                pass
            raise NodeExecutionError(f"Node '{node_name}' failed", node_name=node_name, original_error=e, state=state) from e

    def _update_state(self, state: State, updates: Dict[str, Any]) -> None:
        """Update state with node results"""
        if not updates:
            return

        for key, value in updates.items():
            if key not in state:
                state[key] = value
            elif key == "messages" and isinstance(value, list):
                existing = state.get(key) or []
                state[key] = add_messages(existing, value)
            elif isinstance(state[key], dict) and isinstance(value, dict):
                # Deep merge for dicts
                self._deep_merge(state[key], value)
            elif isinstance(state[key], list) and isinstance(value, list):
                # Append for lists
                state[key].extend(value)
            else:
                # Replace for other types
                state[key] = value

    def _deep_merge(self, target: Dict[str, Any], source: Dict[str, Any]) -> None:
        """Deep merge two dictionaries"""
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._deep_merge(target[key], value)
            else:
                target[key] = value

    def _get_next_node(self, current_node: str, state: State, node_result: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Determine the next node to execute"""
        # Check edges from current node
        if current_node in self.graph.edges:
            for edge_target, edge_condition in self.graph.edges[current_node]:
                # Case A: Unconditional edge
                if edge_condition is None and isinstance(edge_target, str):
                    return edge_target

                # Case B: conditional edges stored as (condition_func, path_map)
                if callable(edge_target) and isinstance(edge_condition, dict):
                    try:
                        condition_key = edge_target(state)
                        if isinstance(condition_key, str) and condition_key in edge_condition:
                            selected = edge_condition[condition_key]
                            return selected
                    except Exception as e:
                        logger.warning(f"Conditional map evaluation failed for {current_node}: {e}")

                # Case C: Simple predicate condition with explicit target
                if isinstance(edge_target, str) and callable(edge_condition):
                    try:
                        cond = edge_condition(state)
                        if cond is True:
                            return edge_target
                    except Exception as e:
                        logger.warning(f"Predicate condition failed for {current_node}->{edge_target}: {e}")

        # Check for explicit next_node in result
        if node_result and "next_node" in node_result:
            next_node = node_result["next_node"]
            if next_node in self.graph.nodes or next_node == END:
                return next_node

        # No next node found
        return None

    def _maybe_cleanup_state(self, state: Dict[str, Any]) -> None:
        try:
            if self.graph.state_cleanup:
                self.graph.state_cleanup(state)
        except Exception:
            pass

    def _record_execution_metrics(self, node_name: str, start_time: datetime, end_time: datetime, success: bool, error: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> None:
        if not self.graph.monitoring_enabled:
            return
        try:
            execution_time = (end_time - start_time).total_seconds()
            record = {
                "node_name": node_name,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "execution_time": execution_time,
                "success": success,
                "error": error,
                "metadata": metadata or {}
            }
            self.execution_history.append(record)
            # Ring buffer: keep only last max_execution_history entries
            if len(self.execution_history) > self.max_execution_history:
                self.execution_history = self.execution_history[-self.max_execution_history:]
        except Exception:
            pass  # Don't let monitoring break execution

    def get_execution_metrics(self) -> Dict[str, Any]:
        """Get aggregated execution metrics"""
        if not self.execution_history:
            return {"total_executions": 0, "avg_execution_time": 0, "success_rate": 0, "node_stats": {}}

        total = len(self.execution_history)
        successful = sum(1 for h in self.execution_history if h["success"])
        total_time = sum(h["execution_time"] for h in self.execution_history)

        node_stats = {}
        for h in self.execution_history:
            node = h["node_name"]
            if node not in node_stats:
                node_stats[node] = {"count": 0, "total_time": 0, "errors": 0}
            node_stats[node]["count"] += 1
            node_stats[node]["total_time"] += h["execution_time"]
            if not h["success"]:
                node_stats[node]["errors"] += 1

        for node, stats in node_stats.items():
            stats["avg_time"] = stats["total_time"] / stats["count"]
            stats["error_rate"] = stats["errors"] / stats["count"]

        return {
            "total_executions": total,
            "avg_execution_time": total_time / total,
            "success_rate": successful / total,
            "node_stats": node_stats
        }

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache hit/miss counts and hit rate.
            Returns empty stats if no cache is configured.

        Example:
            ```python
            compiled = graph.compile(cache=InMemoryCache())
            result = await compiled.ainvoke({"input": "hello"})
            stats = compiled.get_cache_stats()
            print(f"Cache hit rate: {stats['hit_rate']:.1%}")
            ```
        """
        if self.cache is None:
            return {
                "enabled": False,
                "hits": 0,
                "misses": 0,
                "hit_rate": 0.0,
            }

        total = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total if total > 0 else 0.0

        return {
            "enabled": True,
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "hit_rate": hit_rate,
            "cache_type": self.cache.__class__.__name__,
        }

    async def _execute_parallel_group(self, group_name: str, state: Dict[str, Any]) -> None:
        nodes = self.graph.parallel_groups.get(group_name, [])
        if not nodes:
            return
        graph_cfg = self.graph.config if isinstance(self.graph.config, GraphConfig) else GraphConfig()
        group_cfg = self.graph.parallel_group_configs.get(group_name)
        if not group_cfg:
            group_cfg = graph_cfg.parallel_groups.get(group_name, ParallelGroupConfig())

        join_strategy = group_cfg.join_strategy
        timeout = group_cfg.timeout
        error_strategy = group_cfg.error_strategy
        join_condition = group_cfg.join_condition

        # create tasks
        loop = asyncio.get_event_loop()
        tasks: Dict[str, asyncio.Task] = {}
        for n in nodes:
            tasks[n] = loop.create_task(self._execute_node(n, state))

        completed_nodes: List[str] = []
        updates_to_merge: List[Dict[str, Any]] = []
        errors: List[Dict[str, Any]] = []

        async def handle_done(done_set):
            for t in done_set:
                # find node name by task
                node_name = next((name for name, task in tasks.items() if task is t), None)
                try:
                    result = t.result()
                    if isinstance(result, Command):
                        if result.update:
                            updates_to_merge.append(result.update)
                    elif isinstance(result, dict):
                        updates_to_merge.append(result)
                    elif isinstance(result, RouterResult):
                        # RouterResult in parallel branch is unusual; ignore routing but record metadata
                        updates_to_merge.append({"__router__": {"node": node_name, "next": result.next_node}})
                    completed_nodes.append(node_name or "")
                except Exception as e:
                    err_info = {"node": node_name, "error": str(e)}
                    errors.append(err_info)
                    if error_strategy == "fail_fast":
                        # cancel all other tasks
                        for task in tasks.values():
                            if not task.done():
                                task.cancel()
                        raise e

        try:
            if join_strategy == "any":
                done, pending = await asyncio.wait(tasks.values(), timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
                await handle_done(done)
                # cancel pending
                for p in pending:
                    p.cancel()
            else:
                # all_complete or quorum
                if join_strategy == "quorum":
                    needed = 0
                    total = len(tasks)
                    quorum = group_cfg.quorum or 0.5
                    if quorum >= 1:
                        needed = min(total, int(quorum))
                    else:
                        needed = max(1, int(total * quorum + 0.9999))
                    remaining = set(tasks.values())
                    start = datetime.now()
                    while remaining and len(completed_nodes) < needed:
                        to_wait_timeout = None
                        if timeout is not None:
                            elapsed = (datetime.now() - start).total_seconds()
                            left = max(0.0, timeout - elapsed)
                            to_wait_timeout = left
                            if left == 0:
                                break
                        done, pending = await asyncio.wait(remaining, timeout=to_wait_timeout, return_when=asyncio.FIRST_COMPLETED)
                        remaining = pending
                        if done:
                            await handle_done(done)
                    # timeout reached or quorum met: cancel remaining
                    for p in remaining:
                        p.cancel()
                else:
                    # all_complete
                    done, pending = await asyncio.wait(tasks.values(), timeout=timeout, return_when=asyncio.ALL_COMPLETED)
                    await handle_done(done)
                    for p in pending:
                        p.cancel()
        except asyncio.CancelledError:
            # Surface cancellation upwards
            raise
        except Exception:
            if error_strategy == "collect_errors":
                # merge successful updates and attach errors into state
                for upd in updates_to_merge:
                    self._update_state_with_reducers(state, upd)
                self._update_state_with_reducers(state, {"__errors__": errors})
                return
            raise

        # join_condition hook: allow custom early merge decision
        if callable(join_condition):
            try:
                allow_merge = await join_condition(state, completed_nodes) if asyncio.iscoroutinefunction(join_condition) else join_condition(state, completed_nodes)
                if allow_merge is False:
                    # skip merging if condition blocks it
                    for task in tasks.values():
                        if not task.done():
                            task.cancel()
                    return
            except Exception:
                # ignore join_condition errors and proceed to merge
                pass

        # finally merge accumulated updates
        for upd in updates_to_merge:
            self._update_state_with_reducers(state, upd)
        if errors:
            if error_strategy in {"ignore_errors", "collect_errors"}:
                self._update_state_with_reducers(state, {"__errors__": errors})
            elif error_strategy == "fail_fast":
                raise GraphExecutionError(
                    f"Parallel group '{group_name}' failed", node=group_name, iteration=self._current_iteration
                )
        # optional cleanup per group
        self._maybe_cleanup_state(state)


    async def stream(self, initial_state: Optional[Dict[str, Any]] = None, config: Optional[Dict[str, Any]] = None, stream_mode: str = "values"):
        config = config or {}
        state = self._initialize_state(initial_state)
        current_node = self.graph._entry_point
        iteration = 0
        max_iterations = int(config.get("max_iterations", 100) or 100)
        while current_node and iteration < max_iterations:
            iteration += 1
            try:
                # If current node is a parallel group entry, stream merged results after group finishes
                if current_node in self.graph.node_to_group and current_node in self.graph.parallel_entry_nodes:
                    await self._execute_parallel_group(self.graph.node_to_group[current_node], state)
                    if stream_mode == "values":
                        yield state.copy()
                else:
                    result = await self._execute_node(current_node, state)
                    if isinstance(result, Command):
                        if result.update:
                            self._update_state_with_reducers(state, result.update)
                            if stream_mode == "values":
                                yield state.copy()
                        if result.goto:
                            current_node = result.goto
                            continue
                    elif isinstance(result, dict):
                        self._update_state_with_reducers(state, result)
                        if stream_mode == "values":
                            yield state.copy()
                        # optional validation (mirrors invoke)
                        try:
                            if callable(self.graph.state_validator):
                                self.graph.state_validator(state)
                            if isinstance(self.graph.config, GraphConfig):
                                for validator in self.graph.config.state_validators:
                                    if callable(validator):
                                        validator(state)
                        except Exception as e:
                            raise GraphExecutionError(f"State validation failed: {e}", node=current_node, iteration=iteration)
            except InterruptError as e:
                yield {"type": "interrupt", "node": current_node, "interrupt_id": e.interrupt_id, "interrupt_data": e.interrupt_data, "state": state.copy()}
                return
            next_node = await self._determine_next_node(current_node, state)
            if next_node == END or next_node is None:
                if stream_mode == "values":
                    yield state.copy()
                break
            current_node = next_node

    def _initialize_state(self, initial_state: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        state: Dict[str, Any] = {}
        # fill defaults for Annotated list types to [] for reducer usage
        if hasattr(self.graph.state_schema, "__annotations__"):
            for field_name, field_type in self.graph.state_schema.__annotations__.items():
                # heuristic for list-like fields
                if "List" in str(field_type) or "list" in str(field_type):
                    state[field_name] = []
                else:
                    state[field_name] = None
        if initial_state:
            state.update(initial_state)
        return state



    def _update_state_with_reducers(self, state: Dict[str, Any], updates: Dict[str, Any]) -> None:
        for key, value in updates.items():
            if key not in state:
                state[key] = value
                continue
            # merge dicts deeply, else replace
            if isinstance(state[key], dict) and isinstance(value, dict):
                state[key] = merge_dicts(state[key], value)
            elif key == "messages" and isinstance(value, list):
                existing = state.get(key) or []
                state[key] = add_messages(existing, value)
            elif isinstance(state[key], list) and isinstance(value, list):
                # Cap list growth to avoid MemoryError
                combined = state[key] + value
                # keep only last 100 entries to bound memory
                state[key] = combined[-100:]
            else:
                state[key] = value
