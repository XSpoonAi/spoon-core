"""
Integration tests for agents with the new LLM architecture.
"""

import inspect
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from spoon_ai.agents.toolcall import ToolCallAgent
from spoon_ai.agents.spoon_react import SpoonReactAI
from spoon_ai.agents.spoon_react_skill import SpoonReactSkill
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
            available_tools=tool_manager,
        )
        
        # Test agent run
        result = await agent.run("Test request")
        
        # Verify agent used the ChatBot
        mock_chatbot_manager.ask_tool.assert_called()
        assert "I'll help you with that task." in result

    @pytest.mark.asyncio
    async def test_toolcall_agent_terminates_on_stop_even_when_native_reason_is_completed(
        self,
        mock_chatbot_manager,
        tool_manager,
    ):
        """Responses API may use native status values like 'completed' for terminal answers."""
        mock_chatbot_manager.ask_tool.return_value = LLMResponse(
            content="FINAL_OK",
            tool_calls=[],
            finish_reason="stop",
            native_finish_reason="completed",
        )

        agent = ToolCallAgent(
            name="test_agent",
            llm=mock_chatbot_manager,
            available_tools=tool_manager,
            max_steps=1,
        )

        result = await agent.run("Reply with exactly: FINAL_OK")

        assert result == "FINAL_OK"
        assert mock_chatbot_manager.ask_tool.await_count == 1

    @pytest.mark.asyncio
    async def test_toolcall_agent_forwards_thinking_flag_to_llm(self, mock_chatbot_manager, tool_manager):
        mock_chatbot_manager.ask_tool.return_value = LLMResponse(
            content="I'll help you with that task.",
            tool_calls=[],
            finish_reason="stop",
            native_finish_reason="stop",
        )

        agent = ToolCallAgent(
            name="test_agent",
            llm=mock_chatbot_manager,
            available_tools=tool_manager,
        )

        await agent.run("Test request", thinking=True)

        assert mock_chatbot_manager.ask_tool.await_args.kwargs["thinking"] is True

    @pytest.mark.asyncio
    async def test_toolcall_agent_forwards_reasoning_effort_to_llm(self, mock_chatbot_manager, tool_manager):
        mock_chatbot_manager.ask_tool.return_value = LLMResponse(
            content="I'll help you with that task.",
            tool_calls=[],
            finish_reason="stop",
            native_finish_reason="stop",
        )

        agent = ToolCallAgent(
            name="test_agent",
            llm=mock_chatbot_manager,
            available_tools=tool_manager,
        )

        await agent.run("Test request", thinking=True, reasoning_effort="high")

        assert mock_chatbot_manager.ask_tool.await_args.kwargs["thinking"] is True
        assert mock_chatbot_manager.ask_tool.await_args.kwargs["reasoning_effort"] == "high"

    @pytest.mark.asyncio
    async def test_toolcall_agent_treats_openai_responses_completed_as_terminal(
        self,
        mock_chatbot_manager,
        tool_manager,
    ):
        mock_chatbot_manager.ask_tool.return_value = LLMResponse(
            content="Done.",
            tool_calls=[],
            finish_reason="stop",
            native_finish_reason="completed",
        )

        agent = ToolCallAgent(
            name="test_agent",
            llm=mock_chatbot_manager,
            available_tools=tool_manager,
            max_steps=3,
        )

        result = await agent.run("Test request")

        assert result == "Done."
        assert mock_chatbot_manager.ask_tool.await_count == 1

    @pytest.mark.asyncio
    async def test_toolcall_agent_omits_disabled_thinking_flag_for_llm(self, mock_chatbot_manager, tool_manager):
        mock_chatbot_manager.ask_tool.return_value = LLMResponse(
            content="I'll help you with that task.",
            tool_calls=[],
            finish_reason="stop",
            native_finish_reason="stop",
        )

        agent = ToolCallAgent(
            name="test_agent",
            llm=mock_chatbot_manager,
            available_tools=tool_manager,
        )

        await agent.run("Test request")

        assert "thinking" not in mock_chatbot_manager.ask_tool.await_args.kwargs

    @pytest.mark.asyncio
    async def test_toolcall_agent_does_not_append_next_step_prompt_after_user_message(self, mock_chatbot_manager, tool_manager):
        mock_chatbot_manager.ask_tool.return_value = LLMResponse(
            content="I'll help you with that task.",
            tool_calls=[],
            finish_reason="stop",
            native_finish_reason="stop",
        )

        agent = ToolCallAgent(
            name="test_agent",
            llm=mock_chatbot_manager,
            available_tools=tool_manager,
            next_step_prompt="Think step-by-step before choosing tools.",
        )
        await agent.add_message("user", "Test request")

        await agent.think(thinking=True)

        messages = mock_chatbot_manager.ask_tool.await_args.kwargs["messages"]
        user_messages = [msg for msg in messages if getattr(msg, "role", None) == "user"]

        assert len(user_messages) == 1
        assert getattr(user_messages[0], "content", None) == "Test request"
    
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
        mock_chatbot_manager.ask.return_value = "Tool executed successfully"
        
        # Mock tool execution
        tool_manager.execute = AsyncMock(return_value="Tool executed successfully")
        tool_manager.tool_map = {"test_tool": Mock()}
        
        # Create agent
        agent = ToolCallAgent(
            name="test_agent",
            llm=mock_chatbot_manager,
            available_tools=tool_manager,
            max_steps=1,
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
    async def test_toolcall_agent_streams_text_with_tool_calls_as_thinking(self, mock_chatbot_manager, tool_manager):
        mock_tool_call = ToolCall(
            id="call_123",
            type="function",
            function=Function(
                name="test_tool",
                arguments='{"param": "value"}',
            ),
        )
        mock_chatbot_manager.ask_tool.return_value = LLMResponse(
            content="Need to inspect the file first.",
            tool_calls=[mock_tool_call],
            finish_reason="tool_calls",
            native_finish_reason="tool_calls",
        )

        agent = ToolCallAgent(
            name="test_agent",
            llm=mock_chatbot_manager,
            available_tools=tool_manager,
        )
        await agent.add_message("user", "Use a tool")

        should_act = await agent.think()

        assert should_act is True
        pre_tool_event = agent.output_queue.get_nowait()
        assert pre_tool_event["type"] == "thinking"
        assert pre_tool_event["delta"] == "Need to inspect the file first."
        assert pre_tool_event["metadata"]["phase"] == "pre_tool"
        tool_event = agent.output_queue.get_nowait()
        assert tool_event["tool_calls"] == [mock_tool_call]

    @pytest.mark.asyncio
    async def test_toolcall_agent_preserves_tool_call_metadata_in_memory(self, mock_chatbot_manager, tool_manager):
        agent = ToolCallAgent(
            name="test_agent",
            llm=mock_chatbot_manager,
            available_tools=tool_manager,
        )

        await agent.add_message(
            "assistant",
            "I'll use a tool.",
            tool_calls=[
                ToolCall(
                    id="call_sig",
                    type="function",
                    function=Function(
                        name="test_tool",
                        arguments='{"param":"value"}',
                    ),
                    metadata={"thought_signature": "c2lnLTEyMw=="},
                )
            ],
        )

        stored = agent.memory.messages[-1].tool_calls[0]
        assert stored.metadata == {"thought_signature": "c2lnLTEyMw=="}
    
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

    def test_spoon_react_run_signatures_accept_reasoning_kwargs(self):
        assert "thinking" in inspect.signature(SpoonReactAI.run).parameters
        assert "thinking" in inspect.signature(SpoonReactSkill.run).parameters
        assert "reasoning_effort" in inspect.signature(SpoonReactAI.run).parameters
        assert "reasoning_effort" in inspect.signature(SpoonReactSkill.run).parameters
    
    @pytest.mark.asyncio
    async def test_spoon_react_ai_accepts_legacy_llm_instance(self):
        """Test SpoonReactAI can still be constructed with a legacy ChatBot."""
        mock_chatbot = Mock(spec=ChatBot)
        mock_chatbot.use_llm_manager = False

        agent = SpoonReactAI(name="spoon_agent", llm=mock_chatbot)

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
    async def test_toolcall_agent_emits_thinking_for_non_streamed_pre_tool_content(self, mock_chatbot_manager, tool_manager):
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
            "type": "thinking",
            "delta": "First I will inspect the workspace.",
            "content": "First I will inspect the workspace.",
            "metadata": {
                "phase": "pre_tool",
                "source": "toolcall_agent",
            },
        }
        assert put_calls[1].args[0] == {"tool_calls": [mock_tool_call]}

    @pytest.mark.asyncio
    async def test_toolcall_agent_uses_final_tool_free_summary_after_budget_exhaustion(self, mock_chatbot_manager, tool_manager):
        mock_tool_call = ToolCall(
            id="call_123",
            type="function",
            function=Function(
                name="test_tool",
                arguments='{"param":"value"}',
            ),
        )
        mock_chatbot_manager.ask_tool.side_effect = [
            ManagerLLMResponse(
                content="First I will use a tool.",
                provider="openai",
                model="gpt-4.1",
                finish_reason="tool_calls",
                native_finish_reason="tool_calls",
                tool_calls=[mock_tool_call],
                metadata={"streamed_content": False},
            ),
            ManagerLLMResponse(
                content="Final summary after tool execution.",
                provider="openai",
                model="gpt-4.1",
                finish_reason="stop",
                native_finish_reason="stop",
                tool_calls=[],
            ),
        ]
        mock_chatbot_manager.ask.return_value = "Final summary after tool execution."

        tool_manager.execute = AsyncMock(return_value="Tool executed successfully")
        tool_manager.tool_map = {"test_tool": Mock()}

        agent = ToolCallAgent(
            name="streaming_agent",
            llm=mock_chatbot_manager,
            available_tools=tool_manager,
            max_steps=1,
        )

        result = await agent.run("Use a tool and summarize")

        assert result == "Final summary after tool execution."
        assert mock_chatbot_manager.ask_tool.await_count == 1
        mock_chatbot_manager.ask.assert_awaited_once()
        final_messages = mock_chatbot_manager.ask.await_args.kwargs["messages"]
        assert any(
            "Follow the latest user's requested output format exactly." in msg.content
            for msg in final_messages
            if isinstance(getattr(msg, "content", None), str)
        )
        assert any(
            "Do not replace it with a recap or progress summary" in msg.content
            for msg in final_messages
            if isinstance(getattr(msg, "content", None), str)
        )
    
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
        mock_chatbot = Mock(spec=ChatBot)
        mock_chatbot.use_llm_manager = True

        with patch('spoon_ai.agents.spoon_react.create_configured_chatbot'):
            agent_manager = SpoonReactAI(
                name="manager_agent",
                max_steps=5,
                system_prompt="Custom system prompt",
                llm=mock_chatbot,
            )
            
            assert agent_manager.max_steps == 5
            assert agent_manager.system_prompt == "Custom system prompt"
            assert agent_manager.llm.use_llm_manager is True
        
        # Test with legacy architecture
        mock_chatbot = Mock(spec=ChatBot)
        mock_chatbot.use_llm_manager = False

        with patch('spoon_ai.agents.spoon_react.create_configured_chatbot'):
            agent_legacy = SpoonReactAI(
                name="legacy_agent",
                max_steps=3,
                system_prompt="Legacy system prompt",
                llm=mock_chatbot,
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
        
        mock_chatbot = Mock(spec=ChatBot)
        mock_chatbot.use_llm_manager = True
        mock_chatbot.ask_tool = AsyncMock(return_value=LLMResponse(
            content="Compatibility test",
            tool_calls=[],
            finish_reason="stop",
            native_finish_reason="stop"
        ))

        with patch('spoon_ai.agents.spoon_react.create_configured_chatbot'):
            
            # Create agent using existing pattern
            agent = SpoonReactAI(name="compat_agent", llm=mock_chatbot)
            
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
