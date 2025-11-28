import asyncio
import logging
import uuid
import json
import datetime
from pathlib import Path
from abc import ABC
from contextlib import asynccontextmanager
from typing import Literal, Optional, List, Union, Dict, Any, cast
import threading
import time

from spoon_ai.schema import Message, Role
from pydantic import BaseModel, Field

from spoon_ai.chat import ChatBot, Memory
from spoon_ai.schema import AgentState, ToolCall
from spoon_ai.callbacks.base import BaseCallbackHandler
from spoon_ai.callbacks.manager import CallbackManager

logger = logging.getLogger(__name__)
DEBUG = False

def debug_log(message):
    if DEBUG:
        logger.info(f"DEBUG: {message}\n")

class ThreadSafeOutputQueue:
    """Thread-safe output queue with fair access and timeout protection"""

    def __init__(self, maxsize: int = 0):
        self._queue = asyncio.Queue(maxsize=maxsize)
        self._consumers = set()
        self._consumer_lock = asyncio.Lock()
        self._fair_access_enabled = True

    async def put(self, item: Any) -> None:
        await self._queue.put(item)

    async def get(self, timeout: Optional[float] = 30.0) -> Any:
        """Get item with timeout and fair access"""
        consumer_id = id(asyncio.current_task())

        async with self._consumer_lock:
            self._consumers.add(consumer_id)

        try:
            if timeout is None:
                return await self._queue.get()
            else:
                return await asyncio.wait_for(self._queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning(f"Queue consumer {consumer_id} timed out after {timeout}s")
            raise
        finally:
            async with self._consumer_lock:
                self._consumers.discard(consumer_id)

    def empty(self) -> bool:
        return self._queue.empty()

    def qsize(self) -> int:
        return self._queue.qsize()


class BaseAgent(BaseModel, ABC):
    """
    Thread-safe base class for all agents with proper concurrency handling.
    """
    name: str = Field(..., description="The name of the agent")
    description: Optional[str] = Field(None, description="The description of the agent")
    system_prompt: Optional[str] = Field(None, description="The system prompt for the agent")
    next_step_prompt: Optional[str] = Field(None, description="Prompt for determining next action")

    llm: ChatBot = Field(..., description="The LLM to use for the agent")
    memory: Memory = Field(default_factory=Memory, description="The memory to use for the agent")
    enable_long_term_memory: bool = Field(default=False, description="Enable Mem0-based long-term memory")
    mem0_config: Dict[str, Any] = Field(default_factory=dict, description="Mem0 configuration passed to the LLM client")
    state: AgentState = Field(default=AgentState.IDLE, description="The state of the agent")

    max_steps: int = Field(default=10, description="The maximum number of steps the agent can take")
    current_step: int = Field(default=0, description="The current step of the agent")

    # Thread-safe replacements
    output_queue: ThreadSafeOutputQueue = Field(default_factory=ThreadSafeOutputQueue, description="Thread-safe output queue")
    task_done: asyncio.Event = Field(default_factory=asyncio.Event, description="The signal of agent run done")
    
    # Callback system
    callbacks: List[BaseCallbackHandler] = Field(default_factory=list, description="Callback handlers for monitoring")

    model_config = {
        "arbitrary_types_allowed": True,
        "extra": "allow"
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.state = AgentState.IDLE

        # Thread safety primitives
        self._state_lock = asyncio.Lock()
        self._memory_lock = asyncio.Lock()
        self._run_lock = asyncio.Lock()
        self._step_lock = asyncio.Lock()

        # State transition tracking
        self._state_transition_history = []
        self._max_history = 100

        # Timeout configurations
        self._default_timeout = 30.0
        self._state_transition_timeout = 5.0
        self._memory_operation_timeout = 10.0

        # Concurrency control
        self._active_operations = set()
        self._shutdown_event = asyncio.Event()

        # Long-term memory configuration (Mem0)
        self.mem0_config = self.mem0_config or {}
        if self.name:
            self.mem0_config.setdefault("agent_id", self.name)
        if isinstance(self.llm, ChatBot):
            try:
                self.llm.update_mem0_config(self.mem0_config, enable=self.enable_long_term_memory)
            except Exception as exc:
                logger.warning("Unable to configure Mem0 for agent %s: %s", self.name, exc)
        
        # Initialize callback manager
        self._callback_manager = CallbackManager.from_callbacks(self.callbacks)

    async def add_message(
        self,
        role: Literal["user", "assistant", "tool"],
        content: str,
        tool_call_id: Optional[str] = None,
        tool_calls: Optional[List[ToolCall]] = None,
        tool_name: Optional[str] = None,
        timeout: Optional[float] = None
    ) -> None:
        """Thread-safe message addition with timeout protection"""
        if role not in ["user", "assistant", "tool"]:
            raise ValueError(f"Invalid role: {role}")

        timeout = timeout or self._memory_operation_timeout
        operation_id = str(uuid.uuid4())

        try:
            self._active_operations.add(operation_id)

            async with asyncio.timeout(timeout):
                async with self._memory_lock:
                    if role == "user":
                        message = Message(role=Role.USER, content=content)
                    elif role == "assistant":
                        if tool_calls:
                            formatted_tool_calls = [
                                {
                                    "id": toolcall.id,
                                    "type": "function",
                                    "function": (
                                        toolcall.function.model_dump()
                                        if isinstance(toolcall.function, BaseModel)
                                        else toolcall.function
                                    )
                                }
                                for toolcall in tool_calls
                            ]
                            message = Message(
                                role=Role.ASSISTANT,
                                content=content,
                                tool_calls=formatted_tool_calls
                            )
                        else:
                            message = Message(role=Role.ASSISTANT, content=content)
                    elif role == "tool":
                        message = Message(
                            role=Role.TOOL,
                            content=content,
                            tool_call_id=tool_call_id,
                            name=tool_name
                        )

                    # Atomic memory operation
                    self.memory.add_message(message)

        except asyncio.TimeoutError:
            logger.error(f"Memory operation timed out after {timeout}s for agent {self.name}")
            raise RuntimeError(f"Memory operation timed out after {timeout}s")
        except Exception as e:
            logger.error(f"Error adding message to agent {self.name}: {e}")
            raise
        finally:
            self._active_operations.discard(operation_id)

    @asynccontextmanager
    async def state_context(self, new_state: AgentState, timeout: Optional[float] = None):
        """Thread-safe state context manager with deadlock prevention.
        Acquires the state lock only to perform quick transitions, not for the
        duration of the work inside the context, avoiding long-held locks and
        false timeouts during network calls.
        """
        if not isinstance(new_state, AgentState):
            raise ValueError(f"Invalid state: {new_state}")

        timeout = timeout or self._state_transition_timeout
        # If we're about to execute tools, allow a longer transition window
        # to accommodate MCP session setup/teardown
        if getattr(self, 'tool_calls', None):
            # Bump transition timeout modestly when tool calls exist
            timeout = max(timeout, 10.0)
        operation_id = str(uuid.uuid4())

        # Capture old_state before mutation (will validate under lock)
        old_state = self.state

        try:
            self._active_operations.add(operation_id)

            # Set new state under lock with timeout using context manager approach
            try:
                async with asyncio.timeout(timeout):
                    async with self._state_lock:
                        old_state = self.state
                        transition = {
                            'from': old_state,
                            'to': new_state,
                            'timestamp': time.time(),
                            'operation_id': operation_id
                        }
                        self.state = new_state
                        self._record_state_transition(transition)
                        logger.debug(f"Agent {self.name}: State {old_state} -> {new_state}")
            except asyncio.TimeoutError:
                logger.error(f"State transition acquire timed out after {timeout}s for agent {self.name}")
                raise RuntimeError("State transition timed out - potential deadlock detected")

            # Execute the wrapped block without holding the state lock
            try:
                yield
            except Exception as e:
                logger.error(f"Exception in state context for agent {self.name}: {e}")
                # Attempt to set ERROR state under lock with timeout protection
                try:
                    async with asyncio.timeout(timeout):
                        async with self._state_lock:
                            if self.state != AgentState.ERROR:
                                self.state = AgentState.ERROR
                                self._record_state_transition({
                                    'from': new_state,
                                    'to': AgentState.ERROR,
                                    'timestamp': time.time(),
                                    'operation_id': operation_id,
                                    'error': str(e)
                                })
                except asyncio.TimeoutError:
                    logger.error(f"Failed to set ERROR state due to timeout for agent {self.name}")
                raise
            finally:
                # Restore previous state under lock unless changed elsewhere
                try:
                    async with asyncio.timeout(timeout):
                        async with self._state_lock:
                            if self.state == new_state:
                                self.state = old_state
                                self._record_state_transition({
                                    'from': new_state,
                                    'to': old_state,
                                    'timestamp': time.time(),
                                    'operation_id': operation_id,
                                    'restore': True
                                })
                except asyncio.TimeoutError:
                    logger.error(f"State restoration timed out after {timeout}s for agent {self.name}")
        finally:
            self._active_operations.discard(operation_id)

    def _record_state_transition(self, transition: Dict[str, Any]) -> None:
        """Record state transition for debugging"""
        self._state_transition_history.append(transition)
        if len(self._state_transition_history) > self._max_history:
            self._state_transition_history.pop(0)

    async def run(self, request: Optional[str] = None, timeout: Optional[float] = None) -> str:
        """Thread-safe run method with proper concurrency control and callback support."""
        timeout = timeout or self._default_timeout
        run_id = uuid.uuid4()

        # Use run lock to prevent multiple concurrent run() calls
        try:
            async with asyncio.timeout(1.0):  # Quick timeout for run lock
                async with self._run_lock:
                    # Double-check state after acquiring lock
                    if self.state != AgentState.IDLE:
                        raise RuntimeError(
                            f"Agent {self.name} is not in the IDLE state (currently: {self.state})"
                        )

                    # Set running state atomically
                    self.state = AgentState.RUNNING

        except asyncio.TimeoutError:
            raise RuntimeError(f"Agent {self.name} is busy - another run() operation is in progress")

        if request is not None:
            await self.add_message("user", request)

        results: List[str] = []
        operation_id = str(uuid.uuid4())

        try:
            self._active_operations.add(operation_id)

            async with asyncio.timeout(timeout):
                async with self.state_context(AgentState.RUNNING):
                    while (
                        self.current_step < self.max_steps and
                        self.state == AgentState.RUNNING and
                        not self._shutdown_event.is_set()
                    ):
                        self.current_step += 1
                        logger.info(f"Agent {self.name} is running step {self.current_step}/{self.max_steps}")

                        # Execute step with timeout protection
                        try:
                            step_result = await asyncio.wait_for(
                                self.step(run_id=run_id),  # Pass run_id to step
                                timeout=min(timeout / self.max_steps, 30.0)
                            )
                        except asyncio.TimeoutError:
                            step_result = f"Step {self.current_step} timed out"
                            logger.warning(f"Agent {self.name} step {self.current_step} timed out")

                        if await self.is_stuck():
                            await self.handle_stuck_state()

                        results.append(f"Step {self.current_step}: {step_result}")
                        logger.info(f"Step {self.current_step}: {step_result}")

                    if self.current_step >= self.max_steps:
                        results.append(f"Step {self.current_step}: Reached maximum steps. Stopping.")

            final_output = "\n".join(results) if results else "No results"
            return final_output

        except asyncio.TimeoutError as e:
            logger.error(f"Agent {self.name} run() timed out after {timeout}s")
            
            raise RuntimeError(f"Agent run timed out after {timeout}s")
            
        except Exception as e:
            logger.error(f"Error during agent run: {e}")
            
            raise
        finally:
            self._active_operations.discard(operation_id)

            # Always reset to IDLE state safely
            async with self._state_lock:
                if self.state != AgentState.IDLE:
                    logger.info(f"Resetting agent {self.name} state from {self.state} to IDLE")
                    self.state = AgentState.IDLE
                    self.current_step = 0

    async def step(self, run_id: Optional[uuid.UUID] = None) -> str:
        """Override this method in subclasses - now with step-level locking and callback support."""
        async with self._step_lock:
            # Subclasses should implement this
            raise NotImplementedError("Subclasses must implement this method")

    async def is_stuck(self) -> bool:
        """Thread-safe stuck detection"""
        async with self._memory_lock:
            messages = self.memory.get_messages()
            if len(messages) < 2:
                return False

            last_message = messages[-1]
            if not last_message.content:
                return False

            duplicate_count = sum(
                1
                for msg in reversed(messages[:-1])
                if msg.role == Role.ASSISTANT and msg.content == last_message.content
            )
            return duplicate_count >= 2

    async def handle_stuck_state(self):
        """Thread-safe stuck state handling"""
        logger.warning(f"Agent {self.name} is stuck. Applying mitigation.")
        stuck_prompt = (
            "Observed duplicate response. Consider new strategies and "
            "avoid repeating ineffective paths already attempted."
        )

        # Thread-safe prompt update
        if self.next_step_prompt:
            self.next_step_prompt = f"{stuck_prompt}\n\n{self.next_step_prompt}"
        else:
            self.next_step_prompt = stuck_prompt

        logger.warning(f"Added stuck prompt: {stuck_prompt}")
    # Basic retrieval compatibility: allow loading documents even if agent doesn't use RAG
    def add_documents(self, documents) -> None:
        """Store documents on the agent so CLI load-docs works without RAG mixin.

        This default implementation keeps the documents in-memory under
        self._loaded_documents. Agents that support retrieval should override
        this method to index documents into their vector store.
        """
        try:
            setattr(self, "_loaded_documents", documents)
            logger.info(f"Loaded {len(documents) if hasattr(documents, '__len__') else 'N'} document chunks into agent {self.name}")
        except Exception as e:
            logger.error(f"Failed to store documents on agent {self.name}: {e}")
            raise

    def save_chat_history(self):
        """Thread-safe chat history saving"""
        history_dir = Path('chat_logs')
        history_dir.mkdir(exist_ok=True)

        history_file = history_dir / f'{self.name}_history.json'
        now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Safe access to chat history
        messages = []
        if hasattr(self, 'chat_history'):
            if isinstance(self.chat_history, list):
                messages = self.chat_history.copy()
            elif isinstance(self.chat_history, dict) and 'messages' in self.chat_history:
                messages = self.chat_history['messages'].copy()

        save_data = {
            'metadata': {
                'agent_name': self.name,
                'created_at': now,
                'updated_at': now,
                'state_transitions': len(self._state_transition_history),
                'active_operations': len(self._active_operations)
            },
            'messages': messages
        }

        try:
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)
            debug_log(f"Saved chat history with {len(messages)} messages")
        except Exception as e:
            debug_log(f"Error saving chat history: {e}")

    async def stream(self, timeout: Optional[float] = None):
        """Thread-safe streaming with proper cleanup and timeout"""
        timeout = timeout or self._default_timeout
        stream_id = str(uuid.uuid4())

        try:
            self._active_operations.add(stream_id)

            while not (self.task_done.is_set() or self.output_queue.empty()):
                try:
                    # Create tasks for queue and done event
                    queue_task = asyncio.create_task(
                        self.output_queue.get(timeout=min(timeout, 5.0))
                    )
                    task_done_task = asyncio.create_task(self.task_done.wait())

                    # Wait for either task to complete
                    done, pending = await asyncio.wait(
                        [queue_task, task_done_task],
                        return_when=asyncio.FIRST_COMPLETED,
                        timeout=timeout
                    )

                    # Clean up pending tasks
                    for task in pending:
                        task.cancel()
                        try:
                            await task
                        except asyncio.CancelledError:
                            pass

                    if not done:  # Timeout occurred
                        logger.warning(f"Stream timeout after {timeout}s for agent {self.name}")
                        break

                    completed_task = done.pop()

                    if completed_task == task_done_task:
                        # Task is done, drain remaining queue items
                        while not self.output_queue.empty():
                            try:
                                item = await asyncio.wait_for(
                                    self.output_queue.get(timeout=1.0),
                                    timeout=1.0
                                )
                                yield item
                            except asyncio.TimeoutError:
                                break
                        break
                    else:
                        # Got item from queue
                        token = completed_task.result()
                        yield token

                except asyncio.TimeoutError:
                    logger.warning(f"Queue get timeout for agent {self.name}")
                    break
                except Exception as e:
                    logger.error(f"Error in stream for agent {self.name}: {e}")
                    break

        finally:
            self._active_operations.discard(stream_id)

    async def process_mcp_message(
        self,
        content: Any,
        sender: str,
        message: Dict[str, Any],
        agent_id: str,
        timeout: Optional[float] = None
    ):
        """Thread-safe MCP message processing with timeout protection"""
        timeout = timeout or self._default_timeout

        # Parse message content
        if isinstance(content, dict) and "text" in content:
            text_content = content["text"]
        elif isinstance(content, str):
            text_content = content
        else:
            text_content = str(content)

        # Record message to agent's memory safely
        await self.add_message("user", text_content)

        # Get metadata safely
        metadata = {}
        if isinstance(content, dict) and "metadata" in content:
            metadata = content.get("metadata", {})

        # Get message topic
        topic = message.get("topic", "general")

        logger.info(
            f"Agent {self.name} received message from {sender}: "
            f"{text_content[:50]}{'...' if len(text_content) > 50 else ''}"
        )

        # Check if streaming is requested
        request_stream = metadata.get("request_stream", False) if isinstance(content, dict) else False

        # Process message and return result with timeout
        try:
            if request_stream:
                logger.info(f"Streaming response requested for agent {self.name}")

                # Reset task_done event and clear output queue safely
                self.task_done.clear()
                while not self.output_queue.empty():
                    try:
                        await asyncio.wait_for(self.output_queue.get(timeout=0.1), timeout=0.1)
                    except asyncio.TimeoutError:
                        break

                # Start the run task in background
                asyncio.create_task(self._run_and_signal_done(request=text_content, timeout=timeout))

                # Return the stream generator
                return self.stream(timeout=timeout)
            else:
                # Standard synchronous response with timeout
                return await self.run(request=text_content, timeout=timeout)

        except Exception as e:
            logger.error(f"Agent {self.name} error processing message: {str(e)}")
            return f"Error processing message: {str(e)}"

    async def _run_and_signal_done(self, request: Optional[str] = None, timeout: Optional[float] = None):
        """Helper method to run the agent and signal when done for streaming"""
        try:
            await self.run(request=request, timeout=timeout)
        except Exception as e:
            logger.error(f"Error in streaming run: {str(e)}")
            # Put error message in queue for streaming
            try:
                await self.output_queue.put(f"Error: {str(e)}")
            except Exception as queue_error:
                logger.error(f"Failed to put error in queue: {queue_error}")
        finally:
            # Signal that the task is done
            self.task_done.set()

            # Reset state safely
            async with self._state_lock:
                if self.state != AgentState.IDLE:
                    logger.info(f"Resetting agent {self.name} state from {self.state} to IDLE")
                    self.state = AgentState.IDLE
                    self.current_step = 0

    async def shutdown(self, timeout: float = 30.0):
        """Graceful shutdown with cleanup of active operations"""
        logger.info(f"Shutting down agent {self.name}...")

        # Signal shutdown
        self._shutdown_event.set()

        # Wait for active operations to complete
        start_time = time.time()
        while self._active_operations and (time.time() - start_time) < timeout:
            logger.info(f"Waiting for {len(self._active_operations)} active operations to complete...")
            await asyncio.sleep(0.5)

        if self._active_operations:
            logger.warning(f"Agent {self.name} shutdown with {len(self._active_operations)} operations still active")

        # Final state cleanup
        async with self._state_lock:
            self.state = AgentState.IDLE
            self.current_step = 0

        logger.info(f"Agent {self.name} shutdown complete")

    def get_diagnostics(self) -> Dict[str, Any]:
        """Get diagnostic information about the agent's state"""
        return {
            'name': self.name,
            'state': self.state.value if hasattr(self.state, 'value') else str(self.state),
            'current_step': self.current_step,
            'max_steps': self.max_steps,
            'active_operations': len(self._active_operations),
            'state_transitions': len(self._state_transition_history),
            'queue_size': self.output_queue.qsize(),
            'queue_empty': self.output_queue.empty(),
            'shutdown_requested': self._shutdown_event.is_set(),
            'memory_messages': len(self.memory.get_messages()) if hasattr(self.memory, 'get_messages') else 0
        }
