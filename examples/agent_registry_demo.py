"""
Agent Registry System Usage Example
Demonstrates how to use the new agent registry mechanism to dynamically load and manage agents
"""

import asyncio
import logging
from pathlib import Path
from spoon_ai.agents.registry import AgentRegistry
from spoon_ai.chat import ChatBot

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def demo_agent_registry():
    """
    Demonstrate basic functionality of the agent registry system
    """
    print("🤖 Agent Registry System Demo")
    print("=" * 50)
    
    # 1. Create agent registry and load configuration
    print("\n📋 Step 1: Load agent configuration")
    config_path = Path("config/agents.yaml")
    
    try:
        registry = AgentRegistry(config_path)
        print(f"✅ Successfully loaded config file: {config_path}")
    except FileNotFoundError:
        print(f"⚠️  Config file not found: {config_path}")
        print("Creating registry with default configuration...")
        registry = AgentRegistry()
        
        # Manually add a simple agent configuration
        simple_config = {
            'agents': [
                {
                    'name': 'chat',
                    'class': 'spoon_ai.agents.enhanced_base.StandardAgent',
                    'description': 'Standard chat agent',
                    'system_prompt': 'You are a helpful AI assistant.',
                    'max_steps': 3
                }
            ]
        }
        registry.config = simple_config
    
    # 2. Load agents
    print("\n🔧 Step 2: Load agents")
    try:
        await registry.load_agents()
        print("✅ Agents loaded successfully")
    except Exception as e:
        print(f"❌ Agent loading failed: {e}")
        return
    
    # 3. List loaded agents
    print("\n📋 Step 3: List loaded agents")
    agent_names = registry.list_agents()
    print(f"Number of loaded agents: {len(agent_names)}")
    for name in agent_names:
        agent = registry.get_agent(name)
        print(f"  - {name}: {agent.description if hasattr(agent, 'description') else 'No description'}")
    
    # 4. Test agent functionality
    print("\n🧪 Step 4: Test agent functionality")
    
    # Get first agent for testing
    if agent_names:
        test_agent_name = agent_names[0]
        test_agent = registry.get_agent(test_agent_name)
        
        print(f"Testing agent: {test_agent_name}")
        
        # Check if agent is initialized
        if hasattr(test_agent, 'is_initialized') and test_agent.is_initialized():
            print("✅ Agent is initialized")
        else:
            print("⚠️  Agent is not initialized")
        
        # Get agent capabilities
        if hasattr(test_agent, 'get_capabilities'):
            capabilities = await test_agent.get_capabilities()
            print(f"Agent capabilities: {capabilities}")
        
        # Health check
        if hasattr(test_agent, 'health_check'):
            health = await test_agent.health_check()
            print(f"Health status: {health}")
        
        # List available tools
        if hasattr(test_agent, 'list_tools'):
            tools = await test_agent.list_tools()
            print(f"Number of available tools: {len(tools)}")
            for tool in tools[:3]:  # Only show first 3 tools
                print(f"  - {tool.get('name', 'Unknown')}: {tool.get('description', 'No description')}")
        
        # Test chat functionality
        print(f"\n💬 Testing conversation with {test_agent_name}:")
        try:
            response = await test_agent.run("Hello, please introduce yourself.")
            print(f"Agent response: {response}")
        except Exception as e:
            print(f"Conversation test failed: {e}")
    
    # 5. Dynamically add agent
    print("\n➕ Step 5: Dynamically add agent")
    
    new_agent_config = {
        'name': 'dynamic_agent',
        'class': 'spoon_ai.agents.enhanced_base.StandardAgent',
        'description': 'Dynamically added agent',
        'system_prompt': 'You are a dynamically created AI assistant, specifically designed to demonstrate dynamic agent functionality.',
        'max_steps': 2
    }
    
    try:
        new_agent = await registry.load_agent(new_agent_config)
        print(f"✅ Successfully added dynamic agent: {new_agent.name}")
        
        # Test new agent
        response = await new_agent.run("How were you created?")
        print(f"New agent response: {response}")
        
    except Exception as e:
        print(f"❌ Dynamic agent addition failed: {e}")
    
    # 6. Agent management
    print("\n🗂️  Step 6: Agent management")
    
    # List agents again
    updated_agent_names = registry.list_agents()
    print(f"Current number of agents: {len(updated_agent_names)}")
    
    # Remove an agent
    if 'dynamic_agent' in updated_agent_names:
        success = await registry.remove_agent('dynamic_agent')
        if success:
            print("✅ Successfully removed dynamic agent")
        else:
            print("❌ Failed to remove dynamic agent")
    
    # 7. Cleanup
    print("\n🧹 Step 7: Cleanup resources")
    await registry.cleanup()
    print("✅ Cleanup completed")
    
    print("\n🎉 Agent registry system demo completed!")


async def demo_plugin_system():
    """
    Demonstrate plugin system functionality (requires plugin directory)
    """
    print("\n🔌 Plugin System Demo")
    print("=" * 30)
    
    # Create plugin directory example
    plugin_dir = Path("plugins")
    if not plugin_dir.exists():
        plugin_dir.mkdir()
        print(f"Created plugin directory: {plugin_dir}")
        
        # Create example plugin
        plugin_file = plugin_dir / "example_plugin.py"
        plugin_content = '''
"""
Example Plugin Agent
"""
from spoon_ai.agents.enhanced_base import EnhancedBaseAgent

class ExamplePluginAgent(EnhancedBaseAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "PluginAgent"
        self.description = "Example agent from plugin"
    
    async def step(self) -> str:
        return "This is a response from the plugin agent!"
'''
        
        plugin_file.write_text(plugin_content, encoding='utf-8')
        print(f"Created example plugin: {plugin_file}")
    
    # Configure registry with plugins
    config_with_plugins = {
        'plugin_directories': ['./plugins'],
        'agents': [
            {
                'name': 'plugin_agent',
                'class': 'example_plugin.ExamplePluginAgent',
                'description': 'Agent loaded from plugin'
            }
        ]
    }
    
    registry = AgentRegistry()
    registry.config = config_with_plugins
    registry.load_plugin_directories(['./plugins'])
    
    try:
        await registry.load_agents()
        plugin_agent = registry.get_agent('plugin_agent')
        if plugin_agent:
            print("✅ Successfully loaded plugin agent")
            response = await plugin_agent.run("Test plugin agent")
            print(f"Plugin agent response: {response}")
        else:
            print("❌ Plugin agent loading failed")
    except Exception as e:
        print(f"Plugin demo error: {e}")


async def main():
    """
    Main function, run all demos
    """
    await demo_agent_registry()
    await demo_plugin_system()


if __name__ == "__main__":
    asyncio.run(main()) 