import json
import asyncio
import time
import uuid
from logging import getLogger
from typing import Any, List, Optional
import logging

from pydantic import AliasChoices, Field
from termcolor import colored

from spoon_ai.agents.react import ReActAgent
from spoon_ai.prompts.toolcall import \
    NEXT_STEP_PROMPT as TOOLCALL_NEXT_STEP_PROMPT
from spoon_ai.prompts.toolcall import SYSTEM_PROMPT as TOOLCALL_SYSTEM_PROMPT
from spoon_ai.schema import TOOL_CHOICE_TYPE, AgentState, ToolCall, ToolChoice, Message, Role
from spoon_ai.tools import ToolManager
from mcp.types import Tool as MCPTool

logging.getLogger("spoon_ai").setLevel(logging.INFO)

logger = getLogger("spoon_ai")

class ToolCallAgent(ReActAgent):

    name: str = "toolcall"
    description: str = "Useful when you need to call a tool"

    system_prompt: str = TOOLCALL_SYSTEM_PROMPT
    next_step_prompt: str = TOOLCALL_NEXT_STEP_PROMPT

    available_tools: ToolManager = Field(
        default_factory=lambda: ToolManager(tools=[]),
        validation_alias=AliasChoices("available_tools", "avaliable_tools"),
    )
    special_tool_names: List[str] = Field(default_factory=list)

    tool_choices: TOOL_CHOICE_TYPE = ToolChoice.AUTO # type: ignore

    tool_calls: List[ToolCall] = Field(default_factory=list)

    output_queue: asyncio.Queue = Field(default_factory=asyncio.Queue)

    # Track last tool error for higher-level fallbacks
    last_tool_error: Optional[str] = Field(default=None, exclude=True)

    # MCP Tools Caching
    mcp_tools_cache: Optional[List[MCPTool]] = Field(default=None, exclude=True)
    mcp_tools_cache_timestamp: Optional[float] = Field(default=None, exclude=True)
    mcp_tools_cache_ttl: float = Field(default=300.0, exclude=True)  # 5 minutes TTL

    async def _get_cached_mcp_tools(self) -> List[MCPTool]:
        """Get MCP tools with caching to avoid repeated server calls."""
        current_time = time.time()

        # Thread-safe cache check
        async with asyncio.Lock() if not hasattr(self, '_cache_lock') else asyncio.Lock():
            if not hasattr(self, '_cache_lock'):
                self._cache_lock = asyncio.Lock()

        async with self._cache_lock:
            # Check if cache is valid and not expired
            if (self.mcp_tools_cache is not None and
                self.mcp_tools_cache_timestamp is not None and
                current_time - self.mcp_tools_cache_timestamp < self.mcp_tools_cache_ttl):
                logger.info(f"♻️ {self.name} using cached MCP tools ({len(self.mcp_tools_cache)} tools)")
                return self.mcp_tools_cache.copy() # Return copy to prevent external modification

            # Cache expired or invalid - clean up and fetch fresh
            self._invalidate_mcp_cache()


            if hasattr(self, "list_mcp_tools"):
                try:
                    logger.info(f"🔄 {self.name} fetching MCP tools from server...")
                    mcp_tools = await self.list_mcp_tools()

                    # Validate and limit cache size (prevent memory bloat)
                    if isinstance(mcp_tools, list) and len(mcp_tools) <= 100:
                        self.mcp_tools_cache = mcp_tools
                        self.mcp_tools_cache_timestamp = current_time
                        logger.info(f"📋 {self.name} cached {len(mcp_tools)} MCP tools")
                        return mcp_tools.copy()
                    else:
                        logger.warning(f"⚠️ {self.name} received {len(mcp_tools) if isinstance(mcp_tools, list) else 'invalid'} tools - not caching")
                        return mcp_tools if isinstance(mcp_tools, list) else []

                except Exception as e:
                    logger.error(f"❌ {self.name} failed to fetch MCP tools: {e}")
                    # Return empty list on error rather than crashing
                    return []

        return []





    # Legacy compatibility alias for previous misspelling 'avaliable_tools'
    @property
    def avaliable_tools(self) -> ToolManager:  # type: ignore[override]
        return self.available_tools

    @avaliable_tools.setter
    def avaliable_tools(self, value: ToolManager) -> None:  # type: ignore[override]
        self.available_tools = value



    async def think(self) -> bool:
        if self.next_step_prompt:
            await self.add_message("user", self.next_step_prompt)

        # Use cached MCP tools to avoid repeated server calls
        mcp_tools = await self._get_cached_mcp_tools()

        def convert_mcp_tool(tool: MCPTool) -> dict:
            # Convert MCPTool to function call parameter format, using the actual
            # server tool name if ensure_parameters_loaded renamed it.
            params = getattr(tool, 'parameters', None) or getattr(tool, 'inputSchema', None) or {
                "type": "object",
                "properties": {},
                "required": []
            }
            return {
                "type": "function",
                "function": {
                    "name": getattr(tool, 'name', 'mcp_tool'),
                    "description": getattr(tool, 'description', 'MCP tool'),
                    "parameters": params
                }
            }

        all_tools = self.available_tools.to_params()
        mcp_tools_params = [convert_mcp_tool(tool) for tool in mcp_tools]
        unique_tools = {}
        for tool in all_tools + mcp_tools_params:
            tool_name = tool["function"]["name"]
            unique_tools[tool_name] = tool
        unique_tools_list = list(unique_tools.values())

        # Check if messages contain images or documents (multimodal content)
        # Images and documents require more processing time, so increase timeout
        has_images = False
        has_documents = False
        try:
            for msg in self.memory.messages:
                # Check for images using Message.has_images property
                if hasattr(msg, 'has_images') and msg.has_images:
                    has_images = True
                # Check for documents using Message.has_documents property
                if hasattr(msg, 'has_documents') and msg.has_documents:
                    has_documents = True
                # Fallback: check for data URLs in string content (for Data URL method)
                if hasattr(msg, 'content') and isinstance(msg.content, str):
                    if "data:image" in msg.content:
                        has_images = True
                    if "data:application/pdf" in msg.content or "application/pdf" in str(msg.content):
                        has_documents = True
                # Early exit if both found
                if has_images and has_documents:
                    break
        except Exception:
            pass  # If check fails, use default timeout

        # Bound LLM tool selection time to avoid step-level timeouts
        # Increase timeout for image/document processing (especially Data URLs and PDFs which can be slower)
        # Also account for slow proxy endpoints (e.g. cliproxy on free HuggingFace Spaces)
        base_timeout = max(90.0, getattr(self, '_default_timeout', 120.0) - 5.0)
        if has_images or has_documents:
            # Increase timeout for image/document processing
            # Documents (especially PDFs) can be large and require more processing time
            # Large PDFs (4MB+) may need up to 180 seconds (3 minutes) for processing
            if has_documents:
                llm_timeout = 180.0  # 3 minutes for large PDF processing
            elif has_images:
                llm_timeout = min(120.0, base_timeout * 2)  # 2 minutes for images
            else:
                llm_timeout = min(60.0, base_timeout * 2)
            
            content_type = "images and documents" if (has_images and has_documents) else ("images" if has_images else "documents")
            logger.debug(f"Detected {content_type}, increased timeout to {llm_timeout}s for processing")
        else:
            llm_timeout = base_timeout
        try:
            if hasattr(self, '_middleware_pipeline') and self._middleware_pipeline:
                response = await self._call_llm_with_middleware(
                    unique_tools_list,
                    llm_timeout
                )
            else:
                # Fallback: direct LLM call without middleware
                response = await asyncio.wait_for(
                    self.llm.ask_tool(
                        messages=self.memory.messages,
                        system_msg=self.system_prompt,
                        tools=unique_tools_list,
                        tool_choice=self.tool_choices,
                        output_queue=self.output_queue,
                    ),
                    timeout=llm_timeout,
                )
        except asyncio.TimeoutError:
            logger.error(f"{self.name} LLM tool selection timed out after {llm_timeout}s")
            # Gracefully continue without tools
            await self.add_message("assistant", "Tool selection timed out.")
            self.tool_calls = []
            return False

        self.tool_calls = response.tool_calls

        # Only terminate on finish_reason if there are NO tool calls
        # If there are tool calls, we should execute them regardless of finish_reason
        if not self.tool_calls and self._should_terminate_on_finish_reason(response):
            logger.info(f"🏁 {self.name} terminating due to finish_reason signals (no tool calls)")
            self.state = AgentState.FINISHED
            await self.add_message("assistant", response.content or "Task completed")
            # Emit content to output queue for streaming consumers
            if self.output_queue:
                self.output_queue.put_nowait({"content": response.content or "Task completed"})
            # Set a flag to indicate finish_reason termination and store the content
            self._finish_reason_terminated = True
            self._final_response_content = response.content or "Task completed"
            return False

        # Reduce log verbosity: only log length and presence
        logger.info(colored(f"🤔 {self.name}'s thoughts received (len={len(response.content) if response.content else 0})", "cyan"))
        tool_count = len(self.tool_calls) if self.tool_calls else 0
        if tool_count:
            logger.info(colored(f"🛠️ {self.name} selected {tool_count} tools", "green"))
        else:
            logger.info(colored(f"🛠️ {self.name} selected no tools", "yellow"))

        if self.output_queue:
            self.output_queue.put_nowait({"content": response.content})
            self.output_queue.put_nowait({"tool_calls": response.tool_calls})

        try:
            if self.tool_choices == ToolChoice.NONE:
                if response.tool_calls:
                    logger.warning(f"{self.name} selected {len(self.tool_calls)} tools, but tool_choice is NONE")
                    return False
                if response.content:
                    await self.add_message("assistant", response.content)
                    return True
                return False
            await self.add_message("assistant", response.content, tool_calls=self.tool_calls)
            if self.tool_choices == ToolChoice.REQUIRED and not self.tool_calls:
                return True
            if self.tool_choices == ToolChoice.AUTO and not self.tool_calls:
                return bool(response.content)
            return bool(self.tool_calls)
        except Exception as e:
            logger.error(f"{self.name} failed to think: {e}")
            import traceback
            logger.error(traceback.format_exc())
            await self.add_message("assistant", f"Error encountered while thinking: {e}")
            return False

    async def run(self, request: Optional[str] = None, timeout: Optional[float] = None) -> str:
        """

        This ensures:
        1. Thread-safe execution (no concurrent runs)
        2. Proper timeout handling
        3. Plan/Reflect/Finish phases are executed
        4. Middleware hooks are called correctly
        """
        timeout = timeout or self._default_timeout
        try:
            async with asyncio.timeout(1.0):
                async with self._run_lock:
                    if self.state != AgentState.IDLE:
                        raise RuntimeError(f"Agent {self.name} is not in the IDLE state")
                    self.state = AgentState.RUNNING
        except asyncio.TimeoutError:
            raise RuntimeError(f"Agent {self.name} is busy - another run() operation is in progress")

        if request is not None:
            await self.add_message("user", request)

        # Reset finish_reason termination flag
        self._finish_reason_terminated = False
        self._final_response_content = None

        results: List[str] = []
        runtime = None  # Will be set in try block

        try:
    

            run_id = uuid.uuid4()
            runtime = self._create_runtime_context(run_id)

            if self._middleware_pipeline:
                state_updates = self._middleware_pipeline.execute_before_agent(
                    self._agent_state,
                    runtime
                )
                if state_updates:
                    logger.debug(f"Agent {self.name} state updated by before_agent hooks: {list(state_updates.keys())}")

            if self.enable_plan_phase:
                await self._execute_plan_phase(runtime)

            # Main action loop
            while (
                self.current_step < self.max_steps and
                self.state == AgentState.RUNNING
            ):
                self.current_step += 1
                runtime.current_step = self.current_step
                logger.info(f"Agent {self.name} is running step {self.current_step}/{self.max_steps}")

                # For steps with tool calls, allow more time to avoid premature timeouts
                try:
                    step_timeout = self._default_timeout
                    has_mcp_tools = False
                    try:
                        if hasattr(self, 'available_tools') and hasattr(self.available_tools, 'tool_map'):
                            has_mcp_tools = any(hasattr(t, 'mcp_config') for t in self.available_tools.tool_map.values())
                        if not has_mcp_tools:
                            has_mcp_tools = bool(getattr(self, 'mcp_tools_cache', None))
                    except Exception:
                        pass
                    if has_mcp_tools:
                        step_timeout = max(step_timeout, 120.0)  # 2 minutes for MCP tools
                    if getattr(self, 'tool_calls', None):
                        step_timeout = max(step_timeout, 60.0)

                    # Check for multimodal content (images/documents need more processing time)
                    has_images = False
                    has_documents = False
                    try:
                        for msg in self.memory.messages:
                            if hasattr(msg, 'has_images') and msg.has_images:
                                has_images = True
                            if hasattr(msg, 'has_documents') and msg.has_documents:
                                has_documents = True
                            if hasattr(msg, 'content') and isinstance(msg.content, str):
                                if "data:image" in msg.content:
                                    has_images = True
                                if "data:application/pdf" in msg.content:
                                    has_documents = True
                            if has_images and has_documents:
                                break
                    except Exception:
                        pass

                    if has_documents:
                        step_timeout = max(step_timeout, 180.0)  # 3 minutes for large PDF
                    elif has_images:
                        step_timeout = max(step_timeout, 120.0)  # 2 minutes for images

                    # Check for subagent middleware - subagents need much more time
                    if hasattr(self, 'middleware') and self.middleware:
                        for mw in self.middleware:
                            if hasattr(mw, 'subagents') and mw.subagents:
                                max_subagent_steps = max((getattr(s, 'max_steps', 5) or 5 for s in mw.subagents), default=5)
                                estimated_time = max_subagent_steps * 120 * 2
                                step_timeout = max(step_timeout, estimated_time)
                                logger.info(f"Subagent detected: step_timeout={step_timeout}s")
                                break

                    step_result = await asyncio.wait_for(self.step(), timeout=step_timeout)
                    if await self.is_stuck():
                        await self.handle_stuck_state()
                except asyncio.TimeoutError:
                    logger.error(f"Step {self.current_step} timed out for agent {self.name}")
                    break

                # REFLECT PHASE: Token-based or step-based reflection (Deep Agent feature)
                if self.should_trigger_reflection():
                    await self._execute_reflect_phase(runtime, {
                        "current_step": self.current_step,
                        "step_result": step_result,
                        "results": results,
                        "token_count": self.estimate_token_count(),
                    })
                    self._on_reflection_complete()

                # Check if terminated by finish_reason
                if hasattr(self, '_finish_reason_terminated') and self._finish_reason_terminated:
                    # Return the LLM content directly without step formatting
                    final_content = getattr(self, '_final_response_content', step_result)
                    # Clean up flags
                    self._finish_reason_terminated = False
                    if hasattr(self, '_final_response_content'):
                        delattr(self, '_final_response_content')
                    return final_content

                results.append(f"Step {self.current_step}: {step_result}")
                logger.info(f"Step {self.current_step}: {step_result}")

            if self.current_step >= self.max_steps:
                results.append(f"Step {self.current_step}: Stuck in loop. Resetting state.")

            if self._middleware_pipeline:
                await self._execute_finish_phase(runtime, {"results": results})

            return "\n".join(results) if results else "No results"

        except Exception as e:
            logger.error(f"Error during agent run: {e}")
            raise
        finally:
            if runtime and self._middleware_pipeline:
                try:
                    # Update runtime with current messages
                    runtime.messages = self.memory.get_messages() if hasattr(self.memory, 'get_messages') else []
                    runtime.current_step = self.current_step

                    state_updates = self._middleware_pipeline.execute_after_agent(
                        self._agent_state,
                        runtime
                    )
                    if state_updates:
                        logger.debug(f"Agent {self.name} state updated by after_agent hooks: {list(state_updates.keys())}")
                except Exception as e:
                    logger.error(f"Error in after_agent hooks: {e}")

            # Always reset to IDLE state after run completes or fails
            if self.state != AgentState.IDLE:
                logger.info(f"Resetting agent {self.name} state from {self.state} to IDLE")
                self.state = AgentState.IDLE
                self.current_step = 0

    async def step(self) -> str:
        """Override the step method to handle finish_reason termination properly."""
        should_act = await self.think()
        if not should_act:
            if self.state == AgentState.FINISHED:
                # For finish_reason termination, return a simple message
                # The run() method will handle returning the actual content
                return "Task completed based on finish_reason signal"
            else:
                # Default behavior for other cases
                self.state = AgentState.FINISHED
                return "Thinking completed. No action needed. Task finished."

        return await self.act()

    async def act(self) -> str:
        if not self.tool_calls:
            if self.tool_choices == ToolChoice.REQUIRED:
                raise ValueError("No tools to call")
            return self.memory.messages[-1].content or "No response from assistant"

        results = []
        for tool_call in self.tool_calls:
            try:
                result = await self.execute_tool(tool_call)
                logger.info(f"Tool {tool_call.function.name} executed with result: {result}")
                # Flag error-like results so callers can decide on fallbacks
                if isinstance(result, str) and (
                    "not healthy" in result.lower() or "execution failed" in result.lower()
                ):
                    self.last_tool_error = result
            except Exception as e:
                # Ensure we always create a tool response, even on failure
                result = f"Error executing tool {tool_call.function.name}: {str(e)}"
                logger.error(f"Tool {tool_call.function.name} execution failed: {e}")
                self.last_tool_error = str(e)

            # Always add a tool message for each tool call to satisfy OpenAI API requirements
            await self.add_message("tool", result, tool_call_id=tool_call.id, tool_name=tool_call.function.name)
            results.append(result)
        return "\n\n".join(results)

    async def execute_tool(self, tool_call: ToolCall) -> str:
        """Execute tool with middleware wrapping.

        CRITICAL: This method now routes ALL tool executions through the middleware
        pipeline, enabling HITL approval, observability, and other middleware features.
        """
        def parse_tool_arguments(arguments):
            """Parse tool arguments using improved logic."""
            if isinstance(arguments, str):
                arguments = arguments.strip()
                if not arguments:
                    return {}
                try:
                    return json.loads(arguments)
                except json.JSONDecodeError:
                    print(f"JSON decode failed for arguments string: {arguments}")
                    return {}
            elif isinstance(arguments, dict):
                return arguments
            else:
                return {}

        # Parse arguments once
        kwargs = parse_tool_arguments(tool_call.function.arguments)

        if hasattr(self, '_middleware_pipeline') and self._middleware_pipeline:
            from spoon_ai.middleware.base import ToolCallRequest, ToolCallResult

            # Create runtime context
            runtime = self._create_runtime_context() if hasattr(self, '_create_runtime_context') else None

            # Create tool call request
            request = ToolCallRequest(
                tool_name=tool_call.function.name,
                arguments=kwargs,
                tool_call_id=tool_call.id,
                runtime=runtime,
                tool_call=tool_call
            )

            # Define base handler that does the actual tool execution
            async def base_handler(req: ToolCallRequest) -> ToolCallResult:
                try:
                    result = await self._execute_tool_direct(req.tool_name, req.arguments)
                    return ToolCallResult.from_string(result)
                except Exception as e:
                    return ToolCallResult.from_error(str(e))

            # Execute through middleware pipeline
            result = await self._middleware_pipeline.awrap_tool_call(request, base_handler)

            if result.error:
                self.last_tool_error = result.error
                raise Exception(result.error)

            return result.output

        else:
            # Fallback: direct execution without middleware
            return await self._execute_tool_direct(tool_call.function.name, kwargs)

    async def _execute_tool_direct(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """Direct tool execution without middleware wrapping.

        This is the actual execution logic, separated so it can be wrapped by middleware.
        """
        # Check if tool is in available_tools
        if tool_name not in self.available_tools.tool_map:
            # Handle MCP tools
            try:
                mcp_tools = [t for t in self.available_tools.tool_map.values() if hasattr(t, 'mcp_config')]
                # Direct name match to a specific MCPTool instance (post-rename)
                direct_match = next((t for t in mcp_tools if getattr(t, 'name', None) == tool_name), None)
                if direct_match is not None:
                    return await direct_match.execute(**arguments)

                # Otherwise, route the requested name to the first MCPTool's server
                if mcp_tools:
                    primary_mcp_tool = mcp_tools[0]
                    return await primary_mcp_tool.call_mcp_tool(tool_name, **arguments)
            except Exception as e:
                logger.warning(f"MCPTool direct execution failed, falling back: {e}")

            # Agent-level fallback if it implements MCP client methods
            if hasattr(self, "call_mcp_tool"):
                try:
                    actual_tool_name = self._map_mcp_tool_name(tool_name)
                    if not actual_tool_name:
                        return f"MCP tool '{tool_name}' not found. Available tools: {list(self.available_tools.tool_map.keys())}"

                    # If mapping resolves to a local tool, execute locally (do NOT route via MCP).
                    if actual_tool_name in self.available_tools.tool_map:
                        result = await self.available_tools.execute(name=actual_tool_name, tool_input=arguments)
                        observation = (
                            f"Observed output of cmd {actual_tool_name} execution: {result}"
                            if result
                            else f"cmd {actual_tool_name} execution without any output"
                        )
                        self._handle_special_tool(actual_tool_name, result)
                        return observation

                    result = await self.call_mcp_tool(actual_tool_name, **arguments)
                    return result
                except Exception as e:
                    return f"MCP tool execution failed: {str(e)}"

            # Nothing worked
            return f"MCP tool '{tool_name}' not found"

        # Execute standard tool
        try:
            result = await self.available_tools.execute(name=tool_name, tool_input=arguments)

            observation = (
                f"Observed output of cmd {tool_name} execution: {result}"
                if result
                else f"cmd {tool_name} execution without any output"
            )

            self._handle_special_tool(tool_name, result)
            return observation

        except Exception as e:
            print(f"❌ Tool execution error for {tool_name}: {e}")
            self.last_tool_error = str(e)
            raise

    def consume_last_tool_error(self) -> Optional[str]:
        err = getattr(self, "last_tool_error", None)
        self.last_tool_error = None
        return err


    def _handle_special_tool(self, name: str, result:Any, **kwargs):
        if not self._is_special_tool(name):
            return
        if self._should_finish_execution(name, result, **kwargs):
            self.state = AgentState.FINISHED
        return

    def _is_special_tool(self, name: str) -> bool:
        return name.lower() in [n.lower() for n in self.special_tool_names]

    def _should_finish_execution(self, name: str, result: Any, **kwargs) -> bool:
        return True

    def _should_terminate_on_finish_reason(self, response) -> bool:
        """Check if agent should terminate based on finish_reason signals."""
        # For Anthropic: native_finish_reason="end_turn" maps to finish_reason="stop"
        # For OpenAI: both finish_reason and native_finish_reason are "stop"
        finish_reason = getattr(response, 'finish_reason', None)
        native_finish_reason = getattr(response, 'native_finish_reason', None)

        if finish_reason == "stop":
            # Accept either "stop" (OpenAI) or "end_turn" (Anthropic) as valid termination signals
            return native_finish_reason in ["stop", "end_turn"]
        return False

    async def _call_llm_with_middleware(self, tools: list, timeout: float):
        """Call LLM through middleware pipeline for observability.

        """
        from spoon_ai.middleware.base import ModelRequest, ModelResponse, AgentPhase

        # Create runtime context
        runtime = self._create_runtime_context() if hasattr(self, '_create_runtime_context') else None

        # If tool_choices is an enum, extract its value; otherwise use as-is
        tool_choice = (
            self.tool_choices.value
            if hasattr(self.tool_choices, "value")
            else self.tool_choices
        )

        # Create model request
        request = ModelRequest(
            system_prompt=self.system_prompt,
            messages=self.memory.messages,
            tools=tools,
            tool_choice=tool_choice,
            runtime=runtime,
            phase=AgentPhase.THINK
        )

        # Define base handler that calls the actual LLM
        async def base_handler(req: ModelRequest) -> ModelResponse:
            # Call LLM directly
            llm_response = await asyncio.wait_for(
                self.llm.ask_tool(
                    messages=req.messages,
                    system_msg=req.system_prompt,
                    tools=req.tools,
                    tool_choice=req.tool_choice,
                    output_queue=self.output_queue,
                ),
                timeout=timeout,
            )
            # The response from ask_tool is already in the right format
            return llm_response

        # Execute through middleware pipeline
        response = await self._middleware_pipeline.awrap_model_call(request, base_handler)

        return response

    def _invalidate_mcp_cache(self):
        """Properly invalidate and clean up MCP tools cache."""
        self.mcp_tools_cache = None
        self.mcp_tools_cache_timestamp = None
        logger.debug(f"🧹 {self.name} invalidated MCP tools cache")

    def clear(self):
        self.memory.clear()
        self.tool_calls = []
        self.state = AgentState.IDLE
        self.current_step = 0

        #cache cleanup
        self._invalidate_mcp_cache()

        # Clean up lock if it exists
        if hasattr(self, '_cache_lock'):
            delattr(self, '_cache_lock')

        logger.debug(f"🧹 {self.name} fully cleared state and cache")
