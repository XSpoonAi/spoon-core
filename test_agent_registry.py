#!/usr/bin/env python3
"""
Simple agent registry system test script
Verifies that basic functionality is working correctly
"""

import asyncio
import sys
import logging
from pathlib import Path

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

async def test_agent_registry():
    """
    Test basic functionality of the agent registry system
    """
    print("🧪 Agent Registry System Basic Functionality Test")
    print("=" * 40)
    
    try:
        # Import necessary modules
        from spoon_ai.agents import (
            AgentRegistry, 
            AgentInterface, 
            EnhancedBaseAgent, 
            StandardAgent
        )
        print("✅ Module import successful")
        
        # Test 1: Create empty registry
        print("\n📋 Test 1: Create Registry")
        registry = AgentRegistry()
        print("✅ Registry creation successful")
        
        # Test 2: Manually add standard agent
        print("\n📋 Test 2: Add Standard Agent")
        simple_config = {
            'name': 'test_agent',
            'class': 'spoon_ai.agents.enhanced_base.StandardAgent',
            'description': 'Test agent',
            'system_prompt': 'You are a test AI assistant.',
            'max_steps': 2
        }
        
        # Simulate configuration
        registry.config = {'agents': [simple_config]}
        
        # Load agents
        await registry.load_agents()
        print("✅ Agent loading successful")
        
        # Test 3: Get agent
        print("\n📋 Test 3: Get Agent")
        agent = registry.get_agent('test_agent')
        if agent:
            print(f"✅ Agent retrieval successful: {agent.name}")
            
            # Test if agent implements interface
            if isinstance(agent, AgentInterface):
                print("✅ Agent implements AgentInterface")
            else:
                print("❌ Agent does not implement AgentInterface")
                
            # Test agent methods
            if hasattr(agent, 'initialize') and hasattr(agent, 'list_tools'):
                print("✅ Agent has required methods")
            else:
                print("❌ Agent missing required methods")
        else:
            print("❌ Agent retrieval failed")
            return False
        
        # Test 4: Agent functionality
        print("\n📋 Test 4: Agent Functionality Test")
        
        # Check initialization status
        if hasattr(agent, 'is_initialized'):
            initialized = agent.is_initialized()
            print(f"Agent initialization status: {initialized}")
        
        # Get agent capabilities
        if hasattr(agent, 'get_capabilities'):
            capabilities = await agent.get_capabilities()
            print(f"Agent capabilities: {capabilities}")
        
        # Health check
        if hasattr(agent, 'health_check'):
            health = await agent.health_check()
            print(f"Health status: {health['status']}")
        
        # List tools
        if hasattr(agent, 'list_tools'):
            tools = await agent.list_tools()
            print(f"Available tools count: {len(tools)}")
        
        print("✅ Agent functionality test completed")
        
        # Test 5: Cleanup
        print("\n📋 Test 5: Resource Cleanup")
        await registry.cleanup()
        print("✅ Cleanup completed")
        
        print("\n🎉 All tests passed!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please ensure you run this script from the project root directory")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_config_loading():
    """
    Test configuration file loading functionality
    """
    print("\n🔧 Configuration File Loading Test")
    print("-" * 30)
    
    try:
        from spoon_ai.agents import AgentRegistry
        
        # Check if configuration files exist
        config_files = [
            Path("config/agents.yaml"),
            Path("config/agents.json")
        ]
        
        for config_file in config_files:
            if config_file.exists():
                print(f"✅ Configuration file exists: {config_file}")
                
                try:
                    registry = AgentRegistry(config_file)
                    print(f"✅ Configuration file loaded successfully: {config_file}")
                    
                    # Check configuration content
                    if 'agents' in registry.config:
                        agent_count = len(registry.config['agents'])
                        print(f"  - Configured {agent_count} agents")
                    else:
                        print("  - No agent definitions in configuration")
                        
                except Exception as e:
                    print(f"❌ Configuration file loading failed {config_file}: {e}")
            else:
                print(f"⚠️  Configuration file does not exist: {config_file}")
                
    except Exception as e:
        print(f"❌ Configuration loading test failed: {e}")

def main():
    """
    Main function
    """
    print("🤖 Spoon-AI Agent Registry System Test")
    print("=" * 50)
    
    try:
        # Run async tests
        success = asyncio.run(test_agent_registry())
        asyncio.run(test_config_loading())
        
        if success:
            print("\n🎯 Test Result: Success")
            print("Agent registry system basic functionality is working properly!")
        else:
            print("\n❌ Test Result: Failed")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error during testing: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 