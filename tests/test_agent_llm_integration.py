"""
Integration tests for agents with the new LLM architecture.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from spoon_ai.agents.toolcall import ToolCallAgent
from spoon_ai.agents.spoon_react import SpoonReactAI
from spoon_ai.chat import ChatBot
from spoon_ai.schema import Message, LLMResponse, ToolCall, Function, AgentState
from spoon_ai.llm.interface import LLMResponse as ManagerLLMResponse
from spoon_ai.tools import ToolManager


class TestAgentLLMIntegration:
    """Test agent integration with new LLM architecture."""
    
    @pytest.fixture
    def mock_chatbot_manager(self):
        """Create a mock ChatBot using LLM manager."""
        chatbot = Mock(spec=ChatBot)
        chatbot.use_llm_manager = True
        chatbot.ask = AsyncMock()
        chatbot.ask_tool = AsyncMock()
        return chatbot
    
    @pytest.fixture
    def mock_chatbot_legacy(self):
        """Create a mock ChatBot using legacy mode."""
        chatbot = Mock(spec=ChatBot)
        chatbot.use_llm_manager = False
        chatbot.ask = AsyncMock()
        chatbot.ask_tool = AsyncMock()
        return chatbot
    
    @pytest.fixture
    def tool_manager(self):
        """Create a basic tool manager."""
        return ToolManager([])
    
    @pytest.mark.asyncio
    async def test_toolcall_agent_with_manager(self, mock_chatbot_manager, tool_manager):
        """Test ToolCallAgent with LLM manager architecture."""
        # Setup mock responses
        mock_response = LLMResponse(
            content="I'll help you with that task.",
            tool_calls=[],
            finish_reason="stop",
            native_finish_reason="stop"
        )
        mock_chatbot_manager.ask_tool.return_value = mock_response
        
        # Create agent
        agent = ToolCallAgent(
            name="test_agent",
            llm=mock_chatbot_manager,
            available_tools=tool_manager
        )
        
        # Test agent run
        result = await agent.run("Test request")
        
        # Verify agent used the ChatBot
        mock_chatbot_manager.ask_tool.assert_called()
        assert "I'll help you with that task." in result
    
    @pytest.mark.asyncio
    async def test_toolcall_agent_with_tools(self, mock_chatbot_manager, tool_manager):
        """Test ToolCallAgent with tool calls using LLM manager."""
        # Setup mock response with tool calls
        mock_tool_call = ToolCall(
            id="call_123",
            type="function",
            function=Function(
                name="test_tool",
                arguments='{"param": "value"}'
            )
        )
        
        mock_response = LLMResponse(
            content="I'll use a tool to help.",
            tool_calls=[mock_tool_call],
            finish_reason="tool_calls",
            native_finish_reason="tool_calls"
        )
        mock_chatbot_manager.ask_tool.return_value = mock_response
        
        # Mock tool execution
        tool_manager.execute = AsyncMock(return_value="Tool executed successfully")
        tool_manager.tool_map = {"test_tool": Mock()}
        
        # Create agent
        agent = ToolCallAgent(
            name="test_agent",
            llm=mock_chatbot_manager,
            available_tools=tool_manager
        )
        
        # Test agent run
        result = await agent.run("Use a tool")
        
        # Verify tool was executed
        tool_manager.execute.assert_called_once_with(
            name="test_tool",
            tool_input={"param": "value"}
        )
        assert "Tool executed successfully" in result
    
    @pytest.mark.asyncio
    async def test_spoon_react_ai_initialization(self):
        """Test SpoonReactAI initialization with new architecture."""
        with patch('spoon_ai.agents.spoon_react.create_configured_chatbot') as mock_create:
            mock_chatbot = Mock(spec=ChatBot)
            mock_chatbot.use_llm_manager = True
            mock_create.return_value = mock_chatbot
            
            # Create SpoonReactAI agent
            agent = SpoonReactAI(name="spoon_agent")
            
            # Verify it uses the new architecture
            assert agent.llm.use_llm_manager is True
    
    @pytest.mark.asyncio
    async def test_spoon_react_ai_fallback_to_legacy(self):
        """Test SpoonReactAI fallback to legacy mode on initialization failure."""
        with patch('spoon_ai.agents.spoon_react.create_configured_chatbot') as mock_create:
            # First call fails (new architecture), second succeeds (legacy)
            mock_create.side_effect = [
                Exception("LLM manager initialization failed"),
                Mock(spec=ChatBot, use_llm_manager=False)
            ]
            
            # This should not raise an exception due to fallback
            agent = SpoonReactAI(name="spoon_agent")
            
            # Verify it fell back to legacy mode
            assert agent.llm.use_llm_manager is False
    
    @pytest.mark.asyncio
    async def test_agent_backward_compatibility(self, mock_chatbot_legacy, tool_manager):
        """Test that agents work with legacy ChatBot mode."""
        # Setup mock responses for legacy mode
        mock_response = LLMResponse(
            content="Legacy response",
            tool_calls=[],
            finish_reason="stop",
            native_finish_reason="stop"
        )
        mock_chatbot_legacy.ask_tool.return_value = mock_response
        
        # Create agent with legacy ChatBot
        agent = ToolCallAgent(
            name="legacy_agent",
            llm=mock_chatbot_legacy,
            available_tools=tool_manager
        )
        
        # Test agent run
        result = await agent.run("Test request")
        
        # Verify agent worked with legacy mode
        mock_chatbot_legacy.ask_tool.assert_called()
        assert "Legacy response" in result
    
    @pytest.mark.asyncio
    async def test_agent_error_handling_with_manager(self, mock_chatbot_manager, tool_manager):
        """Test agent error handling with LLM manager."""
        # Setup mock to raise an exception
        mock_chatbot_manager.ask_tool.side_effect = Exception("LLM manager error")
        
        # Create agent
        agent = ToolCallAgent(
            name="error_agent",
            llm=mock_chatbot_manager,
            available_tools=tool_manager
        )
        
        # Test that error is handled appropriately
        with pytest.raises(Exception, match="LLM manager error"):
            await agent.run("Test request")
    
    @pytest.mark.asyncio
    async def test_agent_streaming_support(self, mock_chatbot_manager, tool_manager):
        """Test agent streaming support with new architecture."""
        # Setup streaming mock
        async def mock_stream():
            yield "Streaming"
            yield " response"
            yield " chunk"
        
        mock_chatbot_manager.ask_tool.return_value = LLMResponse(
            content="Streaming response chunk",
            tool_calls=[],
            finish_reason="stop",
            native_finish_reason="stop"
        )
        
        # Create agent
        agent = ToolCallAgent(
            name="streaming_agent",
            llm=mock_chatbot_manager,
            available_tools=tool_manager
        )
        
        # Test streaming functionality
        result = await agent.run("Stream response")
        
        # Verify streaming worked
        assert "Streaming response chunk" in result

    @pytest.mark.asyncio
    async def test_streamed_tool_response_does_not_enqueue_full_content_twice(self, mock_chatbot_manager, tool_manager):
        """When provider already streamed chunks to output_queue, agent should not append full content again."""
        streamed_response = ManagerLLMResponse(
            content="already streamed full text",
            provider="openai",
            model="gpt-4.1",
            finish_reason="stop",
            native_finish_reason="stop",
            tool_calls=[],
            metadata={"streamed_content": True, "stream_chunk_count": 5},
        )
        mock_chatbot_manager.ask_tool.return_value = streamed_response

        agent = ToolCallAgent(
            name="streaming_agent",
            llm=mock_chatbot_manager,
            available_tools=tool_manager,
        )
        mock_queue = Mock()
        agent.output_queue = mock_queue
        await agent.add_message("user", "hello")

        should_continue = await agent.think()

        assert should_continue is False
        put_calls = mock_queue.put_nowait.call_args_list
        assert all(call.args != ({"content": "already streamed full text"},) for call in put_calls)

    @pytest.mark.asyncio
    async def test_toolcall_agent_labels_non_streamed_pre_tool_content_with_think_phase(self, mock_chatbot_manager, tool_manager):
        mock_tool_call = ToolCall(
            id="call_123",
            type="function",
            function=Function(
                name="test_tool",
                arguments='{"param":"value"}',
            ),
        )
        mock_chatbot_manager.ask_tool.return_value = ManagerLLMResponse(
            content="First I will inspect the workspace.",
            provider="openai",
            model="gpt-4.1",
            finish_reason="tool_calls",
            native_finish_reason="tool_calls",
            tool_calls=[mock_tool_call],
            metadata={"streamed_content": False},
        )

        agent = ToolCallAgent(
            name="streaming_agent",
            llm=mock_chatbot_manager,
            available_tools=tool_manager,
        )
        mock_queue = Mock()
        agent.output_queue = mock_queue
        await agent.add_message("user", "hello")

        should_continue = await agent.think()

        assert should_continue is True
        put_calls = mock_queue.put_nowait.call_args_list
        assert put_calls[0].args[0] == {
            "type": "content",
            "delta": "First I will inspect the workspace.",
            "content": "First I will inspect the workspace.",
            "metadata": {
                "phase": "think",
                "source": "toolcall_agent",
            },
        }
        assert put_calls[1].args[0] == {"tool_calls": [mock_tool_call]}
    
    @pytest.mark.asyncio
    async def test_agent_memory_consistency(self, mock_chatbot_manager, tool_manager):
        """Test that agent memory works consistently with new architecture."""
        # Setup mock responses
        responses = [
            LLMResponse(content="First response", tool_calls=[], finish_reason="stop", native_finish_reason="stop"),
            LLMResponse(content="Second response", tool_calls=[], finish_reason="stop", native_finish_reason="stop")
        ]
        mock_chatbot_manager.ask_tool.side_effect = responses
        
        # Create agent
        agent = ToolCallAgent(
            name="memory_agent",
            llm=mock_chatbot_manager,
            available_tools=tool_manager
        )
        
        # Run agent multiple times
        await agent.run("First request")
        await agent.run("Second request")
        
        # Verify memory contains both interactions
        messages = agent.memory.get_messages()
        user_messages = [msg for msg in messages if msg.role == "user"]
        assistant_messages = [msg for msg in messages if msg.role == "assistant"]
        
        assert len(user_messages) == 2
        assert len(assistant_messages) == 2
        assert user_messages[0].content == "First request"
        assert user_messages[1].content == "Second request"
    
    @pytest.mark.asyncio
    async def test_agent_state_management(self, mock_chatbot_manager, tool_manager):
        """Test agent state management with new architecture."""
        # Setup mock response
        mock_response = LLMResponse(
            content="Task completed",
            tool_calls=[],
            finish_reason="stop",
            native_finish_reason="stop"
        )
        mock_chatbot_manager.ask_tool.return_value = mock_response
        
        # Create agent
        agent = ToolCallAgent(
            name="state_agent",
            llm=mock_chatbot_manager,
            available_tools=tool_manager
        )
        
        # Verify initial state
        assert agent.state == AgentState.IDLE
        
        # Run agent
        await agent.run("Test request")
        
        # Verify state returns to IDLE after completion
        assert agent.state == AgentState.IDLE
    
    @pytest.mark.asyncio
    async def test_agent_performance_with_manager(self, mock_chatbot_manager, tool_manager):
        """Test agent performance with LLM manager architecture."""
        import time
        
        # Setup mock response
        mock_response = LLMResponse(
            content="Performance test response",
            tool_calls=[],
            finish_reason="stop",
            native_finish_reason="stop"
        )
        mock_chatbot_manager.ask_tool.return_value = mock_response
        
        # Create agent
        agent = ToolCallAgent(
            name="perf_agent",
            llm=mock_chatbot_manager,
            available_tools=tool_manager
        )
        
        # Measure execution time
        start_time = time.time()
        await agent.run("Performance test")
        end_time = time.time()
        
        # Verify reasonable performance (should complete quickly with mocks)
        execution_time = end_time - start_time
        assert execution_time < 1.0  # Should complete in less than 1 second
    
    def test_agent_configuration_compatibility(self):
        """Test that agent configuration works with both architectures."""
        # Test with manager architecture
        with patch('spoon_ai.agents.spoon_react.create_configured_chatbot') as mock_create:
            mock_chatbot = Mock(spec=ChatBot)
            mock_chatbot.use_llm_manager = True
            mock_create.return_value = mock_chatbot
            
            agent_manager = SpoonReactAI(
                name="manager_agent",
                max_steps=5,
                system_prompt="Custom system prompt"
            )
            
            assert agent_manager.max_steps == 5
            assert agent_manager.system_prompt == "Custom system prompt"
            assert agent_manager.llm.use_llm_manager is True
        
        # Test with legacy architecture
        with patch('spoon_ai.agents.spoon_react.create_configured_chatbot') as mock_create:
            mock_chatbot = Mock(spec=ChatBot)
            mock_chatbot.use_llm_manager = False
            mock_create.return_value = mock_chatbot
            
            agent_legacy = SpoonReactAI(
                name="legacy_agent",
                max_steps=3,
                system_prompt="Legacy system prompt"
            )
            
            assert agent_legacy.max_steps == 3
            assert agent_legacy.system_prompt == "Legacy system prompt"
            assert agent_legacy.llm.use_llm_manager is False


class TestAgentMigrationCompatibility:
    """Test migration compatibility between old and new architectures."""
    
    @pytest.mark.asyncio
    async def test_existing_agent_code_compatibility(self):
        """Test that existing agent code works without modification."""
        # This test ensures that existing agent implementations
        # continue to work without any code changes
        
        with patch('spoon_ai.agents.spoon_react.create_configured_chatbot') as mock_create:
            mock_chatbot = Mock(spec=ChatBot)
            mock_chatbot.use_llm_manager = True
            mock_chatbot.ask_tool = AsyncMock(return_value=LLMResponse(
                content="Compatibility test",
                tool_calls=[],
                finish_reason="stop",
                native_finish_reason="stop"
            ))
            mock_create.return_value = mock_chatbot
            
            # Create agent using existing pattern
            agent = SpoonReactAI(name="compat_agent")
            
            # Run using existing pattern
            result = await agent.run("Test compatibility")
            
            # Verify it works
            assert "Compatibility test" in result
    
    def test_api_surface_compatibility(self):
        """Test that the API surface remains compatible."""
        # Test ChatBot API compatibility
        chatbot = ChatBot(use_llm_manager=False)
        
        # These methods should exist and be callable
        assert hasattr(chatbot, 'ask')
        assert hasattr(chatbot, 'ask_tool')
        assert callable(chatbot.ask)
        assert callable(chatbot.ask_tool)
        
        # Test agent API compatibility
        with patch('spoon_ai.agents.spoon_react.create_configured_chatbot'):
            agent = SpoonReactAI(name="api_test")
            
            # These methods should exist and be callable
            assert hasattr(agent, 'run')
            assert hasattr(agent, 'add_message')
            assert callable(agent.run)
            assert callable(agent.add_message)


if __name__ == "__main__":
    pytest.main([__file__])
