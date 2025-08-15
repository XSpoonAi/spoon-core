import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from spoon_ai.agents.custom_agent import CustomAgent
from spoon_ai.schema import AgentState

@pytest.mark.asyncio
class TestAsyncContextManagement:
    """Test async context management for agents"""
    
    async def test_async_context_manager_success(self):
        """Test successful async context management"""
        mock_llm = Mock()
        
        async with CustomAgent(name="test_agent", llm=mock_llm) as agent:
            assert agent.name == "test_agent"
            assert agent.state == AgentState.IDLE
            
            # Simulate some work
            result = "test completed"
            
        # Agent should be cleaned up
        assert agent.state == AgentState.IDLE
    
    async def test_async_context_manager_with_exception(self):
        """Test cleanup happens even when exception occurs"""
        mock_llm = Mock()
        
        with pytest.raises(ValueError, match="test error"):
            async with CustomAgent(name="test_agent", llm=mock_llm) as agent:
                raise ValueError("test error")
        
        # Agent should still be cleaned up
        assert agent.state == AgentState.IDLE
    
    async def test_initialization_during_context_entry(self):
        """Test that initialize is called during context entry if it exists"""
        mock_llm = Mock()
        
        class MockAgentWithInit(CustomAgent):
            async def initialize(self):
                self._initialized = True
        
        async with MockAgentWithInit(name="test_agent", llm=mock_llm) as agent:
            assert hasattr(agent, '_initialized')
            assert agent._initialized is True
    
    async def test_disconnect_during_context_exit(self):
        """Test that disconnect is called during cleanup"""
        mock_llm = Mock()
        
        class MockAgentWithDisconnect(CustomAgent):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.disconnected = False
            
            async def disconnect(self):
                self.disconnected = True
        
        agent_ref = None
        async with MockAgentWithDisconnect(name="test_agent", llm=mock_llm) as agent:
            agent_ref = agent
            assert not agent.disconnected
        
        assert agent_ref.disconnected is True
    
    @patch('spoon_ai.agents.base.logger')
    async def test_cleanup_error_handling(self, mock_logger):
        """Test that cleanup errors are handled gracefully"""
        mock_llm = Mock()
        
        class MockAgentWithFailingCleanup(CustomAgent):
            async def disconnect(self):
                raise RuntimeError("Disconnect failed")
        
        # Should not raise exception due to cleanup failure
        async with MockAgentWithFailingCleanup(name="test_agent", llm=mock_llm) as agent:
            pass
        
        # Should have logged the cleanup error
        mock_logger.warning.assert_called()
        warning_call = mock_logger.warning.call_args[0][0]
        assert "cleanup completed with" in warning_call
        assert "Disconnect failed" in warning_call
    
    async def test_memory_optimization_during_cleanup(self):
        """Test memory optimization is called during cleanup"""
        mock_llm = Mock()
        
        class MockAgentWithMemoryOpt(CustomAgent):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.memory_optimized = False
            
            def optimize_memory(self):
                self.memory_optimized = True
        
        agent_ref = None
        async with MockAgentWithMemoryOpt(name="test_agent", llm=mock_llm) as agent:
            agent_ref = agent
        
        assert agent_ref.memory_optimized is True