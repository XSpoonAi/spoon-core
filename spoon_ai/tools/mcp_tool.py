
from typing import Union, Dict, Any, Optional, List
import asyncio
import os
import time
import logging

from fastmcp.client.transports import (PythonStdioTransport, SSETransport, WSTransport, NpxStdioTransport,
                                       UvxStdioTransport, StdioTransport, StreamableHttpTransport)
from fastmcp.client import Client as MCPClient
from pydantic import Field

from .base import BaseTool
from ..agents.mcp_client_mixin import MCPClientMixin

logger = logging.getLogger(__name__)

class MCPTool(BaseTool, MCPClientMixin):
    mcp_config: Dict[str, Any] = Field(default_factory=dict, description="MCP transport and tool configuration")

    model_config = {
        "arbitrary_types_allowed": True
    }

    def __init__(self,
                 name: str = "mcp_tool",
                 description: str = "MCP tool for calling external MCP servers",
                 parameters: dict = None,
                 mcp_config: Dict[str, Any] = None):

        if mcp_config is None:
            raise ValueError("`mcp_config` is required to initialize an MCPTool.")

        transport_obj = self._create_transport_from_config(mcp_config)

        BaseTool.__init__(
            self,
            name=name,
            description=description,
            parameters=parameters or {
                "type": "object",
                "properties": {
                    "tool_name": {
                        "type": "string",
                        "description": "Name of the MCP tool to call"
                    },
                    "arguments": {
                        "type": "object",
                        "description": "Arguments to pass to the MCP tool"
                    }
                },
                "required": ["tool_name"]
            },
            mcp_config=mcp_config
        )

        MCPClientMixin.__init__(self, transport_obj)

        self._parameters_loaded = False
        self._parameters_loading = False
        self._last_health_check = 0
        # Support legacy/alias keys from config
        self._health_check_interval = mcp_config.get('health_check_interval', 300)
        # Prefer explicit connection_timeout, fall back to generic timeout
        self._connection_timeout = mcp_config.get('connection_timeout', mcp_config.get('timeout', 30))
        # Also apply per-transport override if present
        if isinstance(self._connection_timeout, (int, float)) and self._connection_timeout <= 0:
            self._connection_timeout = 30
        self._max_retries = mcp_config.get('max_retries', 3)

        logger.info(f"Initialized MCP tool '{self.name}' with deferred parameter loading")

    def _create_transport_from_config(self, config: dict):
        """Create a transport object from a configuration dictionary."""
        url = config.get("url")
        command = config.get("command")

        if url:
            if url.startswith("ws://") or url.startswith("wss://"):
                logger.debug(f"Creating WSTransport for URL: {url}")
                return WSTransport(url)
            elif url.startswith("http://") or url.startswith("https://"):
                # Check if we should use HTTP or SSE transport
                transport_type = config.get("transport", "sse")
                if transport_type == "http":
                    logger.debug(f"Creating StreamableHttpTransport for URL: {url}")
                    headers = config.get("headers", {})
                    return StreamableHttpTransport(url=url, headers=headers)
                else:
                    logger.debug(f"Creating SSETransport for URL: {url}")
                    return SSETransport(url)
            else:
                raise ValueError(f"Unsupported URL scheme in: {url}")

        if command:
            logger.debug(f"Creating stdio-based transport for command: {command}")
            args = config.get("args", [])
            env = config.get("env", {})

            merged_env = os.environ.copy()
            merged_env.update(env)

            for key, value in env.items():
                os.environ[key] = value

            if command == "npx":
                if not args:
                    raise ValueError("No package specified for npx transport")
                package = args[0]
                return NpxStdioTransport(package=package, args=args[1:], env_vars=env)
            elif command == "uvx":
                if not args:
                    raise ValueError("No package specified for uvx transport")
                tool_name = args[0]
                tool_args = args[1:] if len(args) > 1 else None
                return UvxStdioTransport(tool_name=tool_name, tool_args=tool_args, env_vars=env)
            elif command in ["python", "python3"]:
                if not args:
                    raise ValueError("No script path specified for python transport")
                script_path = args[0]
                script_args = args[1:] if len(args) > 1 else None
                return PythonStdioTransport(script_path=script_path, args=script_args, env=merged_env)
            else:
                full_command = [command] + args
                return StdioTransport(command=full_command[0], args=full_command[1:], env=merged_env)

        raise ValueError("Invalid mcp_config: must contain either 'url' or 'command'.")

    async def _fetch_and_set_parameters(self):
        if self._parameters_loading:
            return

        self._parameters_loading = True
        try:
            if not await self._check_mcp_health():
                logger.warning(f"MCP server health check failed for tool '{self.name}'")
                return

            retry_count = 0
            while retry_count < self._max_retries:
                try:
                    async with self.get_session() as session:
                        tools = await asyncio.wait_for(session.list_tools(), timeout=self._connection_timeout)
                        if tools:
                            # Try exact name match first; fall back to first tool
                            target_tool = None
                            for tool in tools:
                                if getattr(tool, 'name', '') == self.name:
                                    target_tool = tool
                                    break
                            # Also try matching against _server_name (the original
                            # alias) so that the first call still resolves after
                            # a previous expand_server_tools renamed `self.name`.
                            if not target_tool and hasattr(self, '_server_name'):
                                for tool in tools:
                                    if getattr(tool, 'name', '') == self._server_name:
                                        target_tool = tool
                                        break

                            if not target_tool and tools:
                                target_tool = tools[0]

                            if target_tool:
                                input_schema = None
                                tool_description = None
                                actual_tool_name = getattr(target_tool, 'name', None)

                                if hasattr(target_tool, 'inputSchema'):
                                    input_schema = target_tool.inputSchema
                                    tool_description = getattr(target_tool, 'description', None)
                                elif hasattr(target_tool, 'dict'):
                                    tool_dict = target_tool.dict()
                                    input_schema = tool_dict.get('inputSchema')
                                    tool_description = tool_dict.get('description')
                                else:
                                    input_schema = getattr(target_tool, 'parameters', None)
                                    tool_description = getattr(target_tool, 'description', None)

                                # If the actual server tool name differs from our current name,
                                # update this tool's name so downstream exposure uses the real name.
                                if actual_tool_name and actual_tool_name != self.name:
                                    object.__setattr__(self, 'name', actual_tool_name)

                                if input_schema:
                                    self.parameters = input_schema
                                    logger.debug(f"Applied dynamic schema from MCP server for tool '{self.name}': {input_schema}")
                                else:
                                    logger.warning(f"No input schema found for MCP tool '{self.name}'")

                                if tool_description:
                                    self.description = tool_description
                                    logger.debug(f"Updated description for tool '{self.name}': {tool_description}")

                                self._parameters_loaded = True
                                logger.debug(f"Successfully configured parameters for tool '{self.name}' from MCP server.")
                                return
                            else:
                                logger.warning(f"No matching tool found for '{self.name}' on MCP server")
                                return
                        else:
                            logger.warning(f"No tools available from MCP server for '{self.name}'")
                            return

                except asyncio.TimeoutError:
                    retry_count += 1
                    if retry_count < self._max_retries:
                        wait_time = 2 ** retry_count
                        logger.warning(f"MCP connection timeout for '{self.name}', retrying in {wait_time}s (attempt {retry_count}/{self._max_retries})")
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(f"Max retries exceeded for MCP parameter fetch for '{self.name}'")
                        raise
                except Exception as e:
                    retry_count += 1
                    if retry_count < self._max_retries:
                        wait_time = 2 ** retry_count
                        logger.warning(f"MCP parameter fetch failed for '{self.name}': {e}, retrying in {wait_time}s (attempt {retry_count}/{self._max_retries})")
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(f"Failed to fetch parameters for tool '{self.name}' after {self._max_retries} retries: {e}")
                        raise

        except Exception as e:
            logger.error(f"Critical error fetching parameters for tool '{self.name}': {e}")
            raise
        finally:
            self._parameters_loading = False

    async def ensure_parameters_loaded(self):
        if self._parameters_loaded:
            return

        if not self.parameters or self.parameters == {
            "type": "object",
            "properties": {
                "tool_name": {"type": "string", "description": "Name of the MCP tool to call"},
                "arguments": {"type": "object", "description": "Arguments to pass to the MCP tool"}
            },
            "required": ["tool_name"]
        }:
            await self._fetch_and_set_parameters()

    async def execute(self, **kwargs) -> Any:
        actual_tool_name = self.name
        try:
            await self.ensure_parameters_loaded()

            if not await self._check_mcp_health():
                raise ConnectionError(f"MCP server for '{self.name}' is not healthy")

            # Remove tool_name from kwargs if it exists to avoid duplicate parameter
            final_args = {k: v for k, v in kwargs.items() if k != 'tool_name'}

            retry_count = 0
            last_exception = None

            while retry_count < self._max_retries:
                try:
                    result = await self.call_mcp_tool(actual_tool_name, **final_args)
                    return result
                except asyncio.TimeoutError as e:
                    last_exception = e
                    retry_count += 1
                    if retry_count < self._max_retries:
                        wait_time = 2 ** retry_count
                        logger.warning(f"MCP tool '{actual_tool_name}' timeout, retrying in {wait_time}s (attempt {retry_count}/{self._max_retries})")
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(f"MCP tool '{actual_tool_name}' failed after {self._max_retries} timeout retries")
                        raise
                except ConnectionError as e:
                    last_exception = e
                    retry_count += 1
                    if retry_count < self._max_retries:
                        wait_time = 2 ** retry_count
                        logger.warning(f"MCP connection error for '{actual_tool_name}', retrying in {wait_time}s (attempt {retry_count}/{self._max_retries})")
                        await asyncio.sleep(wait_time)
                        self._last_health_check = 0
                    else:
                        logger.error(f"MCP tool '{actual_tool_name}' failed after {self._max_retries} connection retries")
                        raise
                except Exception as e:
                    logger.error(f"MCP tool '{actual_tool_name}' execution failed: {e}")
                    raise

        except asyncio.CancelledError:
            logger.warning(f"MCP tool execution '{actual_tool_name}' was cancelled")
            raise
        except Exception as e:
            error_msg = f"Failed to execute MCP tool '{actual_tool_name}': {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    async def _check_mcp_health(self) -> bool:
        current_time = time.time()
        if current_time - self._last_health_check < self._health_check_interval:
            return True

        try:
            async with self.get_session() as session:
                tools = await asyncio.wait_for(session.list_tools(), timeout=min(10, max(5, int(self._connection_timeout))))
                self._last_health_check = current_time
                logger.debug(f"MCP health check passed for '{self.name}' - {len(tools) if tools else 0} tools available")
                return True
        except asyncio.TimeoutError:
            logger.warning(f"MCP health check timeout for '{self.name}'")
            return False
        except Exception as e:
            logger.warning(f"MCP health check failed for '{self.name}': {e}")
            return False

    async def call_mcp_tool(self, tool_name: str, **kwargs):
        """Override the mixin method to add tool-specific error handling."""
        try:
            async with self.get_session() as session:
                logger.debug(f"Calling MCP tool '{tool_name}' with args: {kwargs}")
                res = await asyncio.wait_for(session.call_tool(tool_name, arguments=kwargs), timeout=self._connection_timeout)
                if not res:
                    return ""

                # Handle different result formats
                if hasattr(res, 'content') and res.content:
                    # StreamableHttpTransport returns CallToolResult with content
                    for item in res.content:
                        if hasattr(item, 'text') and item.text is not None:
                            text = item.text
                            if "<coroutine object" in text and "at 0x" in text:
                                raise RuntimeError(f"MCP tool '{tool_name}' returned a coroutine object instead of executing it.")
                            return text
                        elif hasattr(item, 'json') and item.json is not None:
                            import json
                            return json.dumps(item.json, ensure_ascii=False, indent=2)
                elif hasattr(res, '__iter__'):
                    # SSE transport returns iterable result
                    for item in res:
                        if hasattr(item, 'text') and item.text is not None:
                            text = item.text
                            if "<coroutine object" in text and "at 0x" in text:
                                raise RuntimeError(f"MCP tool '{tool_name}' returned a coroutine object instead of executing it.")
                            return text
                        elif hasattr(item, 'json') and item.json is not None:
                            import json
                            return json.dumps(item.json, ensure_ascii=False, indent=2)

                # Fallback: convert to string
                if res:
                    result_str = str(res)
                    if "<coroutine object" in result_str and "at 0x" in result_str:
                        raise RuntimeError(f"MCP tool '{tool_name}' returned a coroutine object instead of executing it.")
                    return result_str
                return ""
        except asyncio.TimeoutError:
            logger.error(f"MCP tool '{tool_name}' call timed out after {self._connection_timeout}s")
            raise
        except asyncio.CancelledError:
            logger.warning(f"MCP tool '{tool_name}' call was cancelled")
            raise
        except Exception as e:
            logger.error(f"MCP tool '{tool_name}' call failed: {e}")
            raise RuntimeError(f"MCP tool '{tool_name}' execution failed: {str(e)}") from e

    async def expand_server_tools(self) -> List["MCPTool"]:
        """Expand this single MCPTool (one-per-server) into one MCPTool per
        real server tool.  Each returned tool shares the same MCP transport
        config and delegates execution to ``call_mcp_tool(real_name)``.

        If the server is unreachable or returns no tools, an empty list is
        returned (callers should keep the original proxy as fallback).

        Returns:
            List of MCPTool instances, one per discovered server tool.
        """
        server_tools = await self.list_available_tools()
        if not server_tools:
            return []

        expanded: List[MCPTool] = []
        server_name = self.name  # Original alias (e.g. "filesystem")

        for tool_info in server_tools:
            real_name = tool_info.get("name", "")
            if not real_name:
                continue
            desc = tool_info.get("description", f"MCP tool from {server_name}")
            schema = tool_info.get("inputSchema") or {
                "type": "object", "properties": {}, "required": []
            }

            child = MCPTool(
                name=real_name,
                description=desc,
                parameters=schema,
                mcp_config=self.mcp_config,
            )
            child._parameters_loaded = True  # Already resolved
            # Store original server alias so execute fallback can find it
            object.__setattr__(child, '_server_name', server_name)
            expanded.append(child)

        logger.info(
            f"Expanded MCP server '{server_name}' into {len(expanded)} tools: "
            f"{[t.name for t in expanded]}"
        )
        return expanded

    async def list_available_tools(self) -> list:
        """List available tools from the MCP server."""
        try:
            if not await self._check_mcp_health():
                raise ConnectionError(f"MCP server for '{self.name}' is not healthy")

            async with self.get_session() as session:
                tools = await asyncio.wait_for(session.list_tools(), timeout=self._connection_timeout)
                if not tools:
                    return []

                # Convert tools to a list of dictionaries
                tool_list = []
                for tool in tools:
                    tool_dict = {
                        "name": getattr(tool, 'name', ''),
                        "description": getattr(tool, 'description', ''),
                        "inputSchema": getattr(tool, 'inputSchema', None)
                    }
                    tool_list.append(tool_dict)

                return tool_list

        except Exception as e:
            logger.error(f"Failed to list available tools: {e}")
            return []
