import json
import asyncio
import time
from logging import getLogger
from typing import Any, List, Optional
import logging

from pydantic import Field
from termcolor import colored

from spoon_ai.agents.react import ReActAgent
from spoon_ai.prompts.toolcall import \
    NEXT_STEP_PROMPT as TOOLCALL_NEXT_STEP_PROMPT
from spoon_ai.prompts.toolcall import SYSTEM_PROMPT as TOOLCALL_SYSTEM_PROMPT
from spoon_ai.schema import TOOL_CHOICE_TYPE, AgentState, ToolCall, ToolChoice, Message, Role
from spoon_ai.tools import ToolManager
from mcp.types import Tool as MCPTool
from spoon_ai.tools.mcp_tool import MCPTool as SpoonMCPTool

logging.getLogger("spoon_ai").setLevel(logging.INFO)

logger = getLogger("spoon_ai")

class ToolCallAgent(ReActAgent):

    name: str = "toolcall"
    description: str = "Useful when you need to call a tool"

    system_prompt: str = TOOLCALL_SYSTEM_PROMPT
    next_step_prompt: str = TOOLCALL_NEXT_STEP_PROMPT

    avaliable_tools: ToolManager = Field(default_factory=ToolManager)
    special_tool_names: List[str] = Field(default_factory=list)

    tool_choices: TOOL_CHOICE_TYPE = ToolChoice.AUTO # type: ignore

    tool_calls: List[ToolCall] = Field(default_factory=list)

    output_queue: asyncio.Queue = Field(default_factory=asyncio.Queue)

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
    
    

    async def think(self) -> bool:
        if self.next_step_prompt:
            self.add_message("user", self.next_step_prompt)

        # Use cached MCP tools to avoid repeated server calls
        mcp_tools = await self._get_cached_mcp_tools()

        def convert_mcp_tool(tool: MCPTool) -> SpoonMCPTool:
            return SpoonMCPTool(
                name=tool.name,
                description=tool.description,
                parameters=tool.inputSchema,
            ).to_param()

        all_tools = self.avaliable_tools.to_params()
        mcp_tools_params = [convert_mcp_tool(tool) for tool in mcp_tools]
        unique_tools = {}
        for tool in all_tools + mcp_tools_params:
            tool_name = tool["function"]["name"]
            unique_tools[tool_name] = tool
        unique_tools_list = list(unique_tools.values())

        response = await self.llm.ask_tool(
            messages=self.memory.messages,
            system_msg=self.system_prompt,
            tools=unique_tools_list,
            tool_choice=self.tool_choices,
            output_queue=self.output_queue,
        )

        self.tool_calls = response.tool_calls

        # Only terminate on finish_reason if there are NO tool calls
        # If there are tool calls, we should execute them regardless of finish_reason
        if not self.tool_calls and self._should_terminate_on_finish_reason(response):
            logger.info(f"🏁 {self.name} terminating due to finish_reason signals (no tool calls)")
            self.state = AgentState.FINISHED
            self.add_message("assistant", response.content or "Task completed")
            # Set a flag to indicate finish_reason termination and store the content
            self._finish_reason_terminated = True
            self._final_response_content = response.content or "Task completed"
            return False

        logger.info(colored(f"🤔 {self.name}'s thoughts: {response.content}", "cyan"))
        tool_count = len(self.tool_calls) if self.tool_calls else 0
        logger.info(colored(f"🛠️ {self.name} selected {tool_count} tools: {self.tool_calls}", "green" if tool_count else "yellow"))

        if self.output_queue:
            self.output_queue.put_nowait({"content": response.content})
            self.output_queue.put_nowait({"tool_calls": response.tool_calls})

        try:
            if self.tool_choices == ToolChoice.NONE:
                if response.tool_calls:
                    logger.warning(f"{self.name} selected {len(self.tool_calls)} tools, but tool_choice is NONE")
                    return False
                if response.content:
                    self.add_message("assistant", response.content)
                    return True
                return False
            self.add_message("assistant", response.content, tool_calls=self.tool_calls)
            if self.tool_choices == ToolChoice.REQUIRED and not self.tool_calls:
                return True
            if self.tool_choices == ToolChoice.AUTO and not self.tool_calls:
                return bool(response.content)
            return bool(self.tool_calls)
        except Exception as e:
            logger.error(f"{self.name} failed to think: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self.add_message("assistant", f"Error encountered while thinking: {e}")
            return False

    async def run(self, request: Optional[str] = None) -> str:
        """Override run method to handle finish_reason termination specially."""
        if self.state != AgentState.IDLE:
            raise RuntimeError(f"Agent {self.name} is not in the IDLE state")

        self.state = AgentState.RUNNING

        if request is not None:
            self.memory.add_message(Message(role=Role.USER, content=request))

        # Reset finish_reason termination flag
        self._finish_reason_terminated = False
        self._final_response_content = None

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
            result = await self.execute_tool(tool_call)
            logger.info(f"Tool {tool_call.function.name} executed with result: {result}")
            self.add_message("tool", result, tool_call_id=tool_call.id, tool_name=tool_call.function.name)
            results.append(result)
        return "\n\n".join(results)

    async def execute_tool(self, tool_call: ToolCall) -> str:
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

        if tool_call.function.name not in self.avaliable_tools.tool_map:
            if not hasattr(self, "call_mcp_tool"):
                raise ValueError(f"Tool {tool_call.function.name} not found")

            kwargs = parse_tool_arguments(tool_call.function.arguments)
            result = await self.call_mcp_tool(tool_call.function.name, **kwargs)
            return result

        if not tool_call or not tool_call.function or not tool_call.function.name:
            return "Error: Invalid tool call"

        name = tool_call.function.name
        if name not in self.avaliable_tools.tool_map:
            return f"Error: Tool {name} not found"

        try:
            args = parse_tool_arguments(tool_call.function.arguments)
            result = await self.avaliable_tools.execute(name=name, tool_input=args)

            observation = (
                f"Observed output of cmd {name} execution: {result}"
                if result
                else f"cmd {name} execution without any output"
            )

            self._handle_special_tool(name, result)
            return observation

        except Exception as e:
            print(f"❌ Tool execution error for {name}: {e}")
            raise


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