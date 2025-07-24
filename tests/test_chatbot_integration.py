<<<<<<< HEAD
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
        
        result = await chatbot_with_manager.ask_tool(messages, tools=tools)
        
        # Verify the result
        assert result.content == "I'll check the weather for you."
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "get_weather"
        
        # Verify manager was called correctly
        mock_llm_manager.chat_with_tools.assert_called_once()
        call_args = mock_llm_manager.chat_with_tools.call_args
        assert call_args[1]['provider'] == "openai"
        assert call_args[1]['tools'] == tools
    
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
=======
#!/usr/bin/env python3
"""
Integration tests for the refactored ChatBot class using LLMFactory.

Tests the full integration: ChatBot -> LLMFactory -> Providers
"""
import asyncio
import os
import sys
import pytest

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spoon_ai.chat import ChatBot
from spoon_ai.utils.config_manager import ConfigManager

class TestChatBotIntegration:
    """Test ChatBot integration with LLMFactory system"""
    
    @pytest.mark.asyncio
    async def test_chatbot_factory_integration(self):
        """Test that ChatBot uses LLMFactory correctly"""
        chatbot = ChatBot(llm_provider="anthropic")
        
        # Verify ChatBot is using factory-created provider
        assert hasattr(chatbot, 'llm'), "ChatBot should have llm provider"
        assert chatbot.llm_provider == "anthropic"
        assert chatbot.model_name == "claude-sonnet-4-20250514"
        assert hasattr(chatbot.llm, 'config'), "Provider should have config"
        
    @pytest.mark.asyncio
    async def test_chatbot_provider_creation(self):
        """Test ChatBot creates different providers correctly"""
        providers_to_test = ["openai", "anthropic", "deepseek", "gemini"]
        
        for provider_name in providers_to_test:
            try:
                chatbot = ChatBot(llm_provider=provider_name)
                assert chatbot.llm_provider == provider_name
                assert hasattr(chatbot, 'llm')
                assert hasattr(chatbot.llm, 'config')
                print(f"✅ {provider_name} provider created successfully")
            except ValueError as e:
                # Expected for missing API keys
                print(f"⚠️  {provider_name} provider failed (expected): {str(e)}")
                assert "API key" in str(e)
    
    @pytest.mark.asyncio 
    async def test_chatbot_config_integration(self):
        """Test ChatBot integrates with ConfigManager properly"""
        config_manager = ConfigManager()
        
        # Test provider auto-detection
        chatbot = ChatBot()  # No explicit provider
        
        # Should have selected a provider based on config/env
        assert chatbot.llm_provider is not None
        assert chatbot.model_name is not None
        print(f"Auto-detected provider: {chatbot.llm_provider}")
        print(f"Auto-detected model: {chatbot.model_name}")
        
    @pytest.mark.asyncio
    async def test_chatbot_memory_system(self):
        """Test that ChatBot memory system works with new architecture"""
        chatbot = ChatBot(llm_provider="anthropic")
        
        # Verify memory system is initialized
        assert hasattr(chatbot, 'memory')
        assert hasattr(chatbot.memory, 'messages')
        
        # Test memory operations
        initial_count = len(chatbot.memory.messages)
        
        # Add a message to memory (simulated)
        from spoon_ai.schema import Message
        test_message = Message(role="user", content="test message")
        chatbot.memory.add_message(test_message)
        
        assert len(chatbot.memory.messages) == initial_count + 1
        assert chatbot.memory.messages[-1].content == "test message"
        
    @pytest.mark.asyncio
    async def test_chatbot_cache_metrics(self):
        """Test cache metrics integration works"""
        chatbot = ChatBot(llm_provider="anthropic")
        
        # Should have cache metrics (either from provider or local)
        metrics = chatbot.get_cache_metrics()
        assert isinstance(metrics, dict)
        assert "cache_creation_input_tokens" in metrics
        assert "cache_read_input_tokens" in metrics
        assert "total_input_tokens" in metrics
        
        # Test cache_metrics property
        assert hasattr(chatbot, 'cache_metrics')
        assert isinstance(chatbot.cache_metrics, dict)
        
    @pytest.mark.asyncio
    async def test_chatbot_backward_compatibility(self):
        """Test that ChatBot maintains backward compatibility"""
        # Test old-style constructor parameters
        chatbot = ChatBot(
            llm_provider="anthropic",
            model_name="claude-sonnet-4-20250514", 
            enable_prompt_cache=True
        )
        
        # Should have expected attributes for backward compatibility
        assert chatbot.llm_provider == "anthropic"
        assert chatbot.model_name == "claude-sonnet-4-20250514"
        assert chatbot.enable_prompt_cache == True
        
        # Should have old method signatures
        assert hasattr(chatbot, 'ask')
        assert hasattr(chatbot, 'ask_tool')
        assert hasattr(chatbot, 'get_cache_metrics')
        
    @pytest.mark.asyncio
    async def test_chatbot_error_handling(self):
        """Test ChatBot error handling with invalid configurations"""
        # Test invalid provider
        with pytest.raises(ValueError) as exc_info:
            ChatBot(llm_provider="invalid_provider")
        assert "does not exist" in str(exc_info.value)
        
        # Test provider instantiation with missing API key
        # (Temporarily remove env API keys for this test)
        original_anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
        if original_anthropic_key:
            del os.environ["ANTHROPIC_API_KEY"]
        
        try:
            with pytest.raises(ValueError) as exc_info:
                ChatBot(llm_provider="anthropic")
            assert "API key not found" in str(exc_info.value)
        finally:
            # Restore original key
            if original_anthropic_key:
                os.environ["ANTHROPIC_API_KEY"] = original_anthropic_key

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
>>>>>>> 777212d (Refactor: Integrate ChatBot with LLMFactory for modular provider architecture)
