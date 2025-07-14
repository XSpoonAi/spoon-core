from typing import Union, Dict, Any, Optional, List, AsyncIterator, AsyncContextManager
import asyncio
import uuid
from contextlib import asynccontextmanager

from fastmcp.client.transports import (FastMCPTransport, PythonStdioTransport,
                                       SSETransport, WSTransport)
from fastmcp.client import Client as MCPClient
import logging

logger = logging.getLogger(__name__)

# Monkey patch to fix FastMCP SSE server issue
def _patch_fastmcp_sse():
    """Patch FastMCP to handle the SSE server TypeError"""
    try:
        from fastmcp.server.server import FastMCP
        from mcp.server.sse import SseServerTransport
        from starlette.applications import Starlette
        from starlette.routing import Route, Mount
        
        original_sse_app = FastMCP.sse_app
        
        def patched_sse_app(self) -> Starlette:
            """Return a properly configured SSE server app."""
            sse = SseServerTransport(self.settings.message_path)
            
            async def handle_sse(request):
                async with sse.connect_sse(
                    request.scope,
                    request.receive,
                    request._send,
                ) as streams:
                    await self._mcp_server.run(
                        streams[0],
                        streams[1],
                        self._mcp_server.create_initialization_options(),
                    )
            
            # Create a proper ASGI app wrapper for handle_post_message
            async def message_app(scope, receive, send):
                await sse.handle_post_message(scope, receive, send)
            
            return Starlette(
                debug=self.settings.debug,
                routes=[
                    Route(self.settings.sse_path, endpoint=handle_sse),
                    Mount(self.settings.message_path, app=message_app),
                ],
            )
        
        FastMCP.sse_app = patched_sse_app
        logger.info("FastMCP SSE server patched successfully")
    except Exception as e:
        logger.debug(f"FastMCP patch not applied: {e}")

# Apply patch when module is imported
_patch_fastmcp_sse()

class MCPClientMixin:
    def __init__(self, mcp_transport: Union[str, WSTransport, SSETransport, PythonStdioTransport, FastMCPTransport]):
        self.mcp_transport = mcp_transport
        self._client = MCPClient(self.mcp_transport)
        self._last_sender = None
        self._last_topic = None
        self._last_message_id = None
        self.output_topic = None
        
        # We'll keep track of the active client session
        self._active_session = None
        self._session_lock = asyncio.Lock()
        # save the session for each task
        self._task_sessions = {}
    
    @asynccontextmanager
    async def get_session(self):
        """
        Get a session to the MCP server using the proper context manager pattern.
        This follows the official usage: async with Client(...) as client
        
        Yields:
            An active client session
        """
        # Generate a unique task ID using UUID
        task_id = id(asyncio.current_task())
        
        # Check if the current task already has a session
        if task_id in self._task_sessions:
            try:
                yield self._task_sessions[task_id]
            finally:
                pass
            return
        
        # If not, create a new session
        async with self._session_lock:
            # Create a new session for the current task
            session = await self._client.__aenter__()
            self._task_sessions[task_id] = session
            
            try:
                yield session
            finally:
                # Clean up the session
                try:
                    await self._client.__aexit__(None, None, None)
                except Exception as e:
                    logger.error(f"Error closing MCP session: {e}")
                finally:
                    if task_id in self._task_sessions:
                        del self._task_sessions[task_id]
    
    async def list_mcp_tools(self):
        """Get the list of available tools from the MCP server"""
        async with self.get_session() as session:
            return await session.list_tools()
    
    async def call_mcp_tool(self, tool_name: str, **kwargs):
        """Call a tool on the MCP server"""
        async with self.get_session() as session:
            res = await session.call_tool(tool_name, arguments=kwargs)
            if not res:
                return ""
                
            for item in res:
                if hasattr(item, 'text') and item.text is not None:
                    return item.text
            
            if res:
                return str(res[0])
                
            return ""
    
    async def send_mcp_message(self, recipient: str, message: Union[str, Dict[str, Any]], 
                              topic: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Send a message to the MCP system
        
        Args:
            recipient: Recipient ID
            message: Message content (string or dictionary)
            topic: Message topic
            metadata: Additional metadata
            
        Returns:
            bool: Whether the message was sent successfully
        """
        if isinstance(message, str):
            content = {
                "text": message,
                "source": "agent",
            }
            if metadata:
                content["metadata"] = metadata
        else:
            content = message
            
        try:
            async with self.get_session() as session:
                await session.send_message(
                    recipient=recipient,
                    message=content,
                    topic=topic or "general"
                )
            return True
        except Exception as e:
            logger.error(f"Failed to send MCP message: {str(e)}")
            return False
    
    async def reply_to_mcp(self, message: str, topic: Optional[str] = None, 
                         metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Reply to a previously received MCP message
        
        Args:
            message: Reply content
            topic: Reply topic, defaults to the topic of the previous message
            metadata: Additional metadata
            
        Returns:
            bool: Whether the reply was successful
        """
        if not hasattr(self, "_last_sender") or not self._last_sender:
            logger.warning(f"No previous sender to reply to")
            return False
            
        recipient = self._last_sender
        reply_topic = topic or self._last_topic or "general"
        
        reply_metadata = {
            "reply_to": self._last_message_id
        }
        
        if metadata:
            reply_metadata.update(metadata)
            
        return await self.send_mcp_message(
            recipient=recipient,
            message=message,
            topic=reply_topic,
            metadata=reply_metadata
        )
    
    async def process_mcp_message(self, content: Any, sender: str, message: Dict[str, Any]):
        """
        Process a message received from the MCP system (should be overridden by subclasses)
        
        Args:
            content: Message content
            sender: Sender ID
            message: Complete message
        """
        if isinstance(content, dict) and "text" in content:
            text_content = content["text"]
        elif isinstance(content, str):
            text_content = content
        else:
            text_content = str(content)
            
        self._last_sender = sender
        self._last_topic = message.get("topic", "general")
        self._last_message_id = message.get("id")
        
        # This method should be overridden by subclasses to handle messages
        logger.info(f"Received MCP message from {sender}: {text_content[:50]}{'...' if len(text_content) > 50 else ''}")
        
    async def connect(self):
        """
        Establish a connection to the MCP server (this just creates a session to verify connection)
        """
        try:
            async with self.get_session() as session:
                # Just verify we can connect by doing a simple operation
                await session.ping()
                logger.info("Successfully connected to MCP server")
        except Exception as e:
            logger.error(f"Failed to connect to MCP server: {str(e)}")
            raise
            
    async def cleanup(self):
        """
        Clean up resources (no explicit disconnection needed with context manager pattern)
        """
        logger.info("MCP client resources cleaned up")
        # Clean up all task sessions
        for task_id, session in list(self._task_sessions.items()):
            try:
                await self._client.__aexit__(None, None, None)
            except Exception as e:
                logger.error(f"Error closing MCP session during cleanup: {e}")
            finally:
                del self._task_sessions[task_id]
