import asyncio
import logging
import uuid
import json
import datetime
from pathlib import Path
from abc import ABC
from contextlib import asynccontextmanager
from typing import Literal, Optional, List, Union, Dict, Any, cast

from spoon_ai.schema import Message, Role
from pydantic import BaseModel, Field

from spoon_ai.chat import ChatBot, Memory
from spoon_ai.schema import AgentState, ToolCall

logger = logging.getLogger(__name__)
DEBUG = False
def debug_log(message):
    if DEBUG:
        logger.info(f"DEBUG: {message}\n")

class BaseAgent(BaseModel, ABC):
    """
    Base class for all agents.
    """
    name: str = Field(..., description="The name of the agent")
    description: Optional[str] = Field(None, description="The description of the agent")
    system_prompt: Optional[str] = Field(None, description="The system prompt for the agent")
    next_step_prompt: Optional[str] = Field(None, description="Prompt for determining next action")
    
    llm: ChatBot = Field(..., description="The LLM to use for the agent")
    memory: Memory = Field(default_factory=Memory, description="The memory to use for the agent")
    state: AgentState = Field(default=AgentState.IDLE, description="The state of the agent")
    
    max_steps: int = Field(default=10, description="The maximum number of steps the agent can take")
    current_step: int = Field(default=0, description="The current step of the agent")

    output_queue: asyncio.Queue = Field(default_factory=asyncio.Queue, description="The queue to store the output of the agent")
    task_done: asyncio.Event = Field(default_factory=asyncio.Event, description="The signal of agent run done")
    
    class Config:
        arbitrary_types_allowed = True
        extra = "allow"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.state = AgentState.IDLE
    
    def add_message(self, role: Literal["user", "assistant", "tool"], content: str, tool_call_id: Optional[str] = None, tool_calls: Optional[List[ToolCall]] = None, tool_name: Optional[str] = None):
        if role not in ["user", "assistant", "tool"]:
            raise ValueError(f"Invalid role: {role}")
        
        if role == "user":
            self.memory.add_message(Message(role=Role.USER, content=content))
        elif role == "assistant":
            if tool_calls:
                self.memory.add_message(Message(role=Role.ASSISTANT, content=content, tool_calls=[{"id": toolcall.id, "type": "function", "function": toolcall.function.model_dump() if isinstance(toolcall.function, BaseModel) else toolcall.function} for toolcall in tool_calls]))
            else:
                self.memory.add_message(Message(role=Role.ASSISTANT, content=content))
        elif role == "tool":
            self.memory.add_message(Message(role=Role.TOOL, content=content, tool_call_id=tool_call_id, name=tool_name))
    
    @asynccontextmanager
    async def state_context(self, new_state: AgentState):
        if not isinstance(new_state, AgentState):
            raise ValueError(f"Invalid state: {new_state}")
        
        old_state = self.state
        self.state = new_state
        try:
            yield
        except Exception as e:
            self.state = AgentState.ERROR
            raise e
        finally:
            self.state = old_state
    
    async def run(self, request: Optional[str] = None) -> str:
        if self.state != AgentState.IDLE:
            raise RuntimeError(f"Agent {self.name} is not in the IDLE state")
        
        self.state = AgentState.RUNNING
        
        if request is not None:
            self.memory.add_message(Message(role=Role.USER, content=request))
        results: List[str] = []
        try:
            async with self.state_context(AgentState.RUNNING):
                while (
                    self.current_step < self.max_steps and
                    self.state == AgentState.RUNNING
                ):
                    self.current_step += 1
                    logger.info(f"Agent {self.name} is running step {self.current_step}/{self.max_steps}")
                    
                    step_result = await self.step()
                    if self.is_stuck():
                        self.handle_struck_state()
                    
                    results.append(f"Step {self.current_step}: {step_result}")
                    logger.info(f"Step {self.current_step}: {step_result}")
                
                if self.current_step >= self.max_steps:
                    results.append(f"Step {self.current_step}: Stuck in loop. Resetting state.")
            
            return "\n".join(results) if results else "No results"
        except Exception as e:
            logger.error(f"Error during agent run: {e}")
            raise
        finally:
            # Always reset to IDLE state after run completes or fails
            if self.state != AgentState.IDLE:
                logger.info(f"Resetting agent {self.name} state from {self.state} to IDLE")
                self.state = AgentState.IDLE
                self.current_step = 0
    
    async def step(self) -> str:
        raise NotImplementedError("Subclasses must implement this method")
    
    def is_stuck(self) -> bool:
        if len(self.memory.get_messages()) < 2:
            return False
        
        last_message = self.memory.get_messages()[-1]
        if not last_message.content:
            return False
        
        duplicate_count = sum(
            1
            for msg in reversed(self.memory.get_messages()[:-1])
            if msg.role == Role.ASSISTANT and msg.content == last_message.content
        )
        return duplicate_count >= 2
    
    def handle_struck_state(self):
        logger.warning(f"Agent {self.name} is stuck. Resetting state.")
        struck_prompt = "Observed duplicate response. Consider new strategies and avoid repeating ineffective paths already attempted."
        self.next_step_prompt = f"{struck_prompt}\n\n{self.next_step_prompt}"
        logger.warning(f"Added struck prompt: {struck_prompt}")
    
    def save_chat_history(self):
        history_dir = Path('chat_logs')
        history_dir.mkdir(exist_ok=True)
        
        history_file = history_dir / f'{self.name}_history.json'
        
        now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        if isinstance(self.chat_history, list):
            save_data = {
                'metadata': {
                    'agent_name': self.name,
                    'created_at': now,
                    'updated_at': now
                },
                'messages': self.chat_history
            }
        elif isinstance(self.chat_history, dict) and 'metadata' in self.chat_history:
            save_data = self.chat_history
            save_data['metadata']['updated_at'] = now
        else:
            save_data = {
                'metadata': {
                    'agent_name': self.name,
                    'created_at': now,
                    'updated_at': now
                },
                'messages': []
            }
        
        try:
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)
            debug_log(f"Saved chat history with {len(save_data.get('messages', []))} messages")
        except Exception as e:
            debug_log(f"Error saving chat history: {e}")

    async def stream(self):
        while not (self.task_done.is_set() or self.output_queue.empty()):
            queue_task = asyncio.create_task(self.output_queue.get())
            task_done_task = asyncio.create_task(self.task_done.wait())

            done, pending = await asyncio.wait(queue_task, task_done_task, return_when=asyncio.FIRST_COMPLETED)

            if pending:
                pending.pop().cancel()
            
            token_or_done = cast(Union[str, Literal[True]], done.pop().result())
            if token_or_done is True:
                while not self.output_queue.empty():
                    yield await self.output_queue.get()
                break
            yield token_or_done

    async def process_mcp_message(self, content: Any, sender: str, message: Dict[str, Any], agent_id: str):
        """
        Process messages from the MCP system
        
        Args:
            content: Message content
            sender: Sender ID
            message: Complete message
            agent_id: Agent ID
            
        Returns:
            The result of processing the message, either as a complete string
            or as a generator for streaming responses
        """
        # Parse message content
        if isinstance(content, dict) and "text" in content:
            text_content = content["text"]
        elif isinstance(content, str):
            text_content = content
        else:
            text_content = str(content)
        
        # Record message to agent's memory
        self.add_message("user", text_content)
        
        # Get message metadata
        metadata = {}
        if isinstance(content, dict) and "metadata" in content:
            metadata = content.get("metadata", {})
            
        # Get message topic
        topic = message.get("topic", "general")
        
        logger.info(f"Agent {self.name} received message from {sender}: {text_content[:50]}{'...' if len(text_content) > 50 else ''}")
        
        # Check if streaming is requested
        request_stream = False
        if isinstance(content, dict) and "metadata" in content:
            request_stream = metadata.get("request_stream", False)
        
        # Process message and return result
        try:
            if request_stream:
                logger.info(f"Streaming response requested for agent {self.name}")
                # Reset task_done event and clear output queue
                self.task_done = asyncio.Event()
                while not self.output_queue.empty():
                    await self.output_queue.get()
                
                # Start the run task in background to feed the output queue
                asyncio.create_task(self._run_and_signal_done(request=text_content))
                
                # Return the stream generator
                return self.stream()
            else:
                # Standard synchronous response
                return await self.run(request=text_content)
        except Exception as e:
            logger.error(f"Agent {self.name} error processing message: {str(e)}")
            return f"Error processing message: {str(e)}"
    
    async def _run_and_signal_done(self, request: Optional[str] = None):
        """
        Helper method to run the agent and signal when done for streaming purposes
        """
        try:
            await self.run(request=request)
        except Exception as e:
            logger.error(f"Error in streaming run: {str(e)}")
        finally:
            # Signal that the task is done
            self.task_done.set()
            
            # Reset state to IDLE but preserve chat history
            if hasattr(self, 'reset_state'):
                self.reset_state()
            elif self.state != AgentState.IDLE:
                logger.info(f"Resetting agent {self.name} state from {self.state} to IDLE")
                self.state = AgentState.IDLE
                self.current_step = 0

    async def __aenter__(self):
        """Async context manager entry"""
        logger.debug(f"Initializing agent '{self.name}' async context")
        
        # Initialize async resources 
        if hasattr(self, 'initialize') and callable(getattr(self, 'initialize')):
            try:
                await self.initialize()
                logger.info(f"Agent '{self.name}' initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize agent '{self.name}': {e}")
                raise RuntimeError(f"Agent initialization failed: {e}") from e
        
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with comprehensive cleanup"""
        cleanup_errors = []
        
        try:
            logger.debug(f"Cleaning up agent '{self.name}' (exception: {exc_type is not None})")
            
            # 1. Disconnect from external services
            if hasattr(self, 'disconnect') and callable(getattr(self, 'disconnect')):
                try:
                    await self.disconnect()
                    logger.debug("Agent disconnected from external services")
                except Exception as e:
                    cleanup_errors.append(f"Disconnect error: {e}")
            
            # 2. Save state/history if needed
            if hasattr(self, 'save_state') and callable(getattr(self, 'save_state')):
                try:
                    await self.save_state()
                    logger.debug("Agent state saved")
                except Exception as e:
                    cleanup_errors.append(f"State save error: {e}")
            
            # 3. Optimize memory before exit
            if hasattr(self, 'optimize_memory') and callable(getattr(self, 'optimize_memory')):
                try:
                    self.optimize_memory()
                    logger.debug("Memory optimized")
                except Exception as e:
                    cleanup_errors.append(f"Memory optimization error: {e}")
            
            # 4. Reset agent state
            try:
                self.state = AgentState.IDLE
                self.current_step = 0
                if hasattr(self, 'task_done'):
                    self.task_done.set()  # Signal completion
                logger.debug("Agent state reset")
            except Exception as e:
                cleanup_errors.append(f"State reset error: {e}")
            
            # 5. Clear sensitive data
            sensitive_attrs = ['_temp_data', '_cache', '_session_data']
            for attr in sensitive_attrs:
                if hasattr(self, attr):
                    try:
                        delattr(self, attr)
                    except Exception as e:
                        cleanup_errors.append(f"Attribute cleanup error ({attr}): {e}")
            
            # Log cleanup results
            if cleanup_errors:
                logger.warning(f"Agent '{self.name}' cleanup completed with {len(cleanup_errors)} errors: {'; '.join(cleanup_errors)}")
            else:
                logger.info(f"Agent '{self.name}' cleaned up successfully")
            
        except Exception as critical_error:
            logger.error(f"Critical error during agent '{self.name}' cleanup: {critical_error}")
            cleanup_errors.append(f"Critical cleanup error: {critical_error}")
        
        # Don't suppress original exceptions - only raise cleanup errors if no original exception
        if cleanup_errors and exc_type is None:
            raise RuntimeError(f"Cleanup failed: {'; '.join(cleanup_errors)}")
        
        # Return False to not suppress the original exception if there was one
        return False
