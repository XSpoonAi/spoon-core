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