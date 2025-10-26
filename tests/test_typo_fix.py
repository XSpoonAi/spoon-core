import pytest
import warnings
from spoon_ai.agents.custom_agent import CustomAgent
from spoon_ai.tools import ToolManager
from unittest.mock import Mock

class TestTypoFix:
    """Test the available_tools typo fix and backward compatibility"""
    
    def test_available_tools_field_exists(self):
        """Test that available_tools field works correctly"""
        agent = CustomAgent(name="test", llm=Mock())
        assert hasattr(agent, 'available_tools')
        assert isinstance(agent.available_tools, ToolManager)
    
    def test_backward_compatibility_property(self):
        """Test that avaliable_tools still works with deprecation warning"""
        agent = CustomAgent(name="test", llm=Mock())
        
        # Should raise deprecation warning
        with pytest.warns(DeprecationWarning, match="avaliable_tools is deprecated"):
            tools = agent.avaliable_tools
        
        assert tools is agent.available_tools
    
    def test_backward_compatibility_setter(self):
        """Test that avaliable_tools setter works with deprecation warning"""
        agent = CustomAgent(name="test", llm=Mock())
        new_tools = ToolManager([])
        
        # Should raise deprecation warning
        with pytest.warns(DeprecationWarning, match="avaliable_tools is deprecated"):
            agent.avaliable_tools = new_tools
        
        assert agent.available_tools is new_tools
    
    def test_no_regression_in_functionality(self):
        """Test that core functionality still works"""
        agent = CustomAgent(name="test", llm=Mock())
        
        # Test that tool operations work
        assert agent.list_tools() == []
        assert isinstance(agent.get_tool_info(), dict)