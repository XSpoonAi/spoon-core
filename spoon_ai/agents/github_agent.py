"""
GitHub agent implementation example
Demonstrates how to create specific agents based on the new agent registration system
"""

import logging
from typing import List, Dict, Any, Optional
from .enhanced_base import EnhancedBaseAgent

logger = logging.getLogger(__name__)


class GitHubAgent(EnhancedBaseAgent):
    """
    GitHub integration agent supporting repository management, issue tracking, and other functions
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "GitHubAgent"
        self.description = "GitHub integration agent supporting repository management, issue tracking, Pull Request management, and other functions"
        
        # GitHub-specific configuration
        self.github_token = kwargs.get('github_token')
        self.default_repo = kwargs.get('default_repo')
        
        # Set system prompt
        self.system_prompt = """
        You are a GitHub agent specialized in handling GitHub-related operations.
        
        You can perform the following operations:
        - Get repository information
        - Manage issues (create, view, update)
        - Manage Pull Requests
        - View commit history
        - Manage branches
        
        Always provide accurate and useful GitHub-related information, and use available tools when needed.
        """
        
        self.next_step_prompt = """
        Based on the user's request, decide what GitHub operation to perform next.
        If you need more information, ask the user.
        """
    
    async def _initialize_agent(self) -> None:
        """
        Initialize GitHub agent specific logic
        """
        if hasattr(self, 'list_mcp_tools'):
            try:
                # Verify MCP connection and get available tools
                tools = await self.list_mcp_tools()
                github_tools = [tool for tool in tools if 'github' in tool.name.lower()]
                logger.info(f"GitHub agent initialization complete, available GitHub tools: {len(github_tools)}")
            except Exception as e:
                logger.warning(f"Failed to initialize GitHub MCP tools: {e}")
        
        # Verify GitHub configuration
        if self.github_token:
            logger.info("GitHub token configured")
        else:
            logger.warning("GitHub token not configured, some features may be limited")
    
    async def step(self) -> str:
        """
        Execute one step of the GitHub agent
        """
        messages = self.memory.get_messages()
        if not messages:
            return "No messages to process"
        
        last_message = messages[-1]
        if last_message.role.value != "user":
            return "Waiting for user input"
        
        user_content = last_message.content
        
        # Analyze user request to determine GitHub operations to execute
        try:
            # Check if it contains GitHub-related keywords
            github_keywords = ['repo', 'repository', 'issue', 'pull request', 'pr', 'commit', 'branch']
            is_github_request = any(keyword in user_content.lower() for keyword in github_keywords)
            
            if is_github_request:
                # If has MCP tools, try to use GitHub-related tools
                if hasattr(self, 'list_mcp_tools'):
                    return await self._handle_github_request(user_content)
                else:
                    return await self._handle_github_request_without_mcp(user_content)
            else:
                # Normal conversation handling
                response = await self.llm.achat(
                    messages=[{"role": "user", "content": user_content}],
                    system_prompt=self.system_prompt
                )
                
                self.add_message("assistant", response)
                return response
                
        except Exception as e:
            error_msg = f"Error processing GitHub request: {str(e)}"
            logger.error(error_msg)
            self.add_message("assistant", error_msg)
            return error_msg
    
    async def _handle_github_request(self, user_content: str) -> str:
        """
        Handle GitHub requests using MCP tools
        """
        try:
            # Here you can call corresponding GitHub MCP tools based on user requests
            # Example: Get repository information
            if 'repo info' in user_content.lower() or 'repository info' in user_content.lower():
                if self.default_repo:
                    result = await self.call_mcp_tool('get_repository', repo=self.default_repo)
                    response = f"Repository information:\n{result}"
                else:
                    response = "Please specify the repository name to query"
            
            # Example: List issues
            elif 'list issues' in user_content.lower() or 'show issues' in user_content.lower():
                if self.default_repo:
                    result = await self.call_mcp_tool('list_issues', repo=self.default_repo)
                    response = f"Issue list:\n{result}"
                else:
                    response = "Please specify the repository name to query issues for"
            
            else:
                # Use LLM to handle other requests
                response = await self.llm.achat(
                    messages=[{"role": "user", "content": user_content}],
                    system_prompt=self.system_prompt + "\n\nNote: You can use GitHub MCP tools to handle related requests."
                )
            
            self.add_message("assistant", response)
            return response
            
        except Exception as e:
            error_msg = f"Failed to handle request using GitHub MCP tools: {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    async def _handle_github_request_without_mcp(self, user_content: str) -> str:
        """
        Handle GitHub requests without MCP tools
        """
        response = await self.llm.achat(
            messages=[{"role": "user", "content": user_content}],
            system_prompt=self.system_prompt + "\n\nNote: Currently not connected to GitHub MCP service, can only provide general GitHub-related advice."
        )
        
        self.add_message("assistant", response)
        return response
    
    async def get_repository_info(self, repo_name: str) -> Dict[str, Any]:
        """
        Get repository information
        """
        if hasattr(self, 'call_mcp_tool'):
            try:
                result = await self.call_mcp_tool('get_repository', repo=repo_name)
                return {"success": True, "data": result}
            except Exception as e:
                return {"success": False, "error": str(e)}
        else:
            return {"success": False, "error": "MCP tools not available"}
    
    async def list_issues(self, repo_name: str, state: str = "open") -> Dict[str, Any]:
        """
        List repository issues
        """
        if hasattr(self, 'call_mcp_tool'):
            try:
                result = await self.call_mcp_tool('list_issues', repo=repo_name, state=state)
                return {"success": True, "data": result}
            except Exception as e:
                return {"success": False, "error": str(e)}
        else:
            return {"success": False, "error": "MCP tools not available"}
    
    async def create_issue(self, repo_name: str, title: str, body: str = "") -> Dict[str, Any]:
        """
        Create new issue
        """
        if hasattr(self, 'call_mcp_tool'):
            try:
                result = await self.call_mcp_tool('create_issue', 
                                                repo=repo_name, 
                                                title=title, 
                                                body=body)
                return {"success": True, "data": result}
            except Exception as e:
                return {"success": False, "error": str(e)}
        else:
            return {"success": False, "error": "MCP tools not available"} 