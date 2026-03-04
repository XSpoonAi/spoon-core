"""
Integration tests for ChatBot with LLM Manager architecture.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from spoon_ai.chat import ChatBot
from spoon_ai.schema import Message, LLMResponse, ToolCall, Function
from spoon_ai.llm.manager import LLMManager
from spoon_ai.llm.interface import LLMResponse as ManagerLLMResponse


class TestChatBotIntegration:
    """Test ChatBot integration with LLM Manager."""
    
    @pytest.fixture
    def mock_llm_manager(self):
        """Create a mock LLM manager."""
        manager = Mock(spec=LLMManager)
        manager.chat = AsyncMock()
        manager.chat_with_tools = AsyncMock()
        return manager
    
    @pytest.fixture
    def chatbot_with_manager(self, mock_llm_manager):
        """Create ChatBot instance with mocked LLM manager."""
        with patch('spoon_ai.chat.get_llm_manager', return_value=mock_llm_manager):
            return ChatBot(use_llm_manager=True, llm_provider="openai")
    
    @pytest.fixture
    def chatbot_legacy(self):
        """Create ChatBot instance with legacy mode."""
        return ChatBot(use_llm_manager=False)
    
    @pytest.mark.asyncio
    async def test_ask_with_manager(self, chatbot_with_manager, mock_llm_manager):
        """Test ask method using LLM manager."""
        # Setup mock response
        mock_response = ManagerLLMResponse(
            content="Hello, how can I help you?",
            provider="openai",
            model="gpt-4.1",
            finish_reason="stop",
            native_finish_reason="stop"
        )
        mock_llm_manager.chat.return_value = mock_response
        
        # Test the ask method
        messages = [{"role": "user", "content": "Hello"}]
        result = await chatbot_with_manager.ask(messages)
        
        # Verify the result
        assert result == "Hello, how can I help you?"
        
        # Verify manager was called correctly
        mock_llm_manager.chat.assert_called_once()
        call_args = mock_llm_manager.chat.call_args
        assert call_args[1]['provider'] == "openai"
        assert len(call_args[1]['messages']) == 1
        assert call_args[1]['messages'][0].content == "Hello"
    
    @pytest.mark.asyncio
    async def test_ask_with_system_message(self, chatbot_with_manager, mock_llm_manager):
        """Test ask method with system message using LLM manager."""
        # Setup mock response
        mock_response = ManagerLLMResponse(
            content="I understand the context.",
            provider="openai",
            model="gpt-4.1",
            finish_reason="stop",
            native_finish_reason="stop"
        )
        mock_llm_manager.chat.return_value = mock_response
        
        # Test with system message
        messages = [{"role": "user", "content": "Hello"}]
        system_msg = "You are a helpful assistant."
        result = await chatbot_with_manager.ask(messages, system_msg=system_msg)
        
        # Verify the result
        assert result == "I understand the context."
        
        # Verify manager was called with system message
        call_args = mock_llm_manager.chat.call_args
        assert len(call_args[1]['messages']) == 2
        assert call_args[1]['messages'][0].role == "system"
        assert call_args[1]['messages'][0].content == system_msg
    
    @pytest.mark.asyncio
    async def test_ask_tool_with_manager(self, chatbot_with_manager, mock_llm_manager):
        """Test ask_tool method using LLM manager."""
        # Setup mock response with tool calls
        mock_tool_call = ToolCall(
            id="call_123",
            type="function",
            function=Function(
                name="get_weather",
                arguments='{"location": "New York"}'
            )
        )
        
        mock_response = ManagerLLMResponse(
            content="I'll check the weather for you.",
            provider="openai",
            model="gpt-4.1",
            finish_reason="tool_calls",
            native_finish_reason="tool_calls",
            tool_calls=[mock_tool_call]
        )
        mock_llm_manager.chat_with_tools.return_value = mock_response
        
        # Test the ask_tool method
        messages = [{"role": "user", "content": "What's the weather in New York?"}]
        tools = [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"}
                    }
                }
            }
        }]
        
        output_queue = asyncio.Queue()
        result = await chatbot_with_manager.ask_tool(
            messages,
            tools=tools,
            output_queue=output_queue,
        )
        
        # Verify the result
        assert result.content == "I'll check the weather for you."
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "get_weather"
        
        # Verify manager was called correctly
        mock_llm_manager.chat_with_tools.assert_called_once()
        call_args = mock_llm_manager.chat_with_tools.call_args
        assert call_args[1]['provider'] == "openai"
        assert call_args[1]['tools'] == tools
        assert call_args[1]['output_queue'] is output_queue
    
    @pytest.mark.asyncio
    async def test_backward_compatibility_flag(self, mock_llm_manager):
        """Test that use_llm_manager=False uses legacy mode."""
        # Create ChatBot with legacy mode
        chatbot = ChatBot(use_llm_manager=False)
        
        # Verify it doesn't use the manager
        assert not chatbot.use_llm_manager
        assert not hasattr(chatbot, 'llm_manager')
    
    @pytest.mark.asyncio
    async def test_message_format_conversion(self, chatbot_with_manager, mock_llm_manager):
        """Test that different message formats are properly converted."""
        # Setup mock response
        mock_response = ManagerLLMResponse(
            content="Converted successfully",
            provider="openai",
            model="gpt-4.1",
            finish_reason="stop",
            native_finish_reason="stop"
        )
        mock_llm_manager.chat.return_value = mock_response
        
        # Test with mixed message formats
        messages = [
            {"role": "user", "content": "Dict message"},
            Message(role="assistant", content="Message object")
        ]
        
        result = await chatbot_with_manager.ask(messages)
        
        # Verify conversion worked
        assert result == "Converted successfully"
        
        # Verify all messages were converted to Message objects
        call_args = mock_llm_manager.chat.call_args
        for msg in call_args[1]['messages']:
            assert isinstance(msg, Message)
    
    @pytest.mark.asyncio
    async def test_error_handling(self, chatbot_with_manager, mock_llm_manager):
        """Test error handling in manager mode."""
        # Setup mock to raise an exception
        mock_llm_manager.chat.side_effect = Exception("Provider error")
        
        # Test that exception is propagated
        messages = [{"role": "user", "content": "Hello"}]
        
        with pytest.raises(Exception, match="Provider error"):
            await chatbot_with_manager.ask(messages)
    
    def test_initialization_with_manager(self):
        """Test ChatBot initialization with LLM manager."""
        with patch('spoon_ai.chat.get_llm_manager') as mock_get_manager:
            mock_manager = Mock()
            mock_get_manager.return_value = mock_manager
            
            chatbot = ChatBot(
                use_llm_manager=True,
                model_name="gpt-4.1",
                llm_provider="openai"
            )
            
            assert chatbot.use_llm_manager
            assert chatbot.llm_manager == mock_manager
            assert chatbot.model_name == "gpt-4.1"
            assert chatbot.llm_provider == "openai"
    
    def test_initialization_legacy_mode(self):
        """Test ChatBot initialization in legacy mode."""
        # This test might need mocking depending on environment setup
        with patch('spoon_ai.utils.config_manager.ConfigManager'):
            chatbot = ChatBot(use_llm_manager=False)
            
            assert not chatbot.use_llm_manager
            assert not hasattr(chatbot, 'llm_manager')


class TestAgentIntegration:
    """Test agent integration with new LLM architecture."""
    
    @pytest.fixture
    def mock_chatbot(self):
        """Create a mock ChatBot."""
        chatbot = Mock(spec=ChatBot)
        chatbot.ask = AsyncMock(return_value="Agent response")
        chatbot.ask_tool = AsyncMock()
        return chatbot
    
    @pytest.mark.asyncio
    async def test_agent_with_new_chatbot(self, mock_chatbot):
        """Test that agents work with the new ChatBot architecture."""
        from spoon_ai.agents.base import BaseAgent
        from spoon_ai.schema import AgentState
        
        # Create a simple test agent
        class TestAgent(BaseAgent):
            async def step(self) -> str:
                messages = self.memory.get_messages()
                response = await self.llm.ask(messages)
                self.add_message("assistant", response)
                self.state = AgentState.FINISHED
                return response
        
        # Create agent with mocked ChatBot
        agent = TestAgent(
            name="test_agent",
            llm=mock_chatbot
        )
        
        # Test agent run
        result = await agent.run("Test request")
        
        # Verify agent used the ChatBot
        mock_chatbot.ask.assert_called_once()
        assert "Agent response" in result


class TestPerformanceOptimization:
    """Test performance optimizations and caching."""
    
    @pytest.mark.asyncio
    async def test_response_caching(self):
        """Test that responses can be cached for performance."""
        # This would test caching mechanisms if implemented
        pass
    
    @pytest.mark.asyncio
    async def test_connection_pooling(self):
        """Test connection pooling for better performance."""
        # This would test connection pooling if implemented
        pass


if __name__ == "__main__":
    pytest.main([__file__])
