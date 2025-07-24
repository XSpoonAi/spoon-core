#!/usr/bin/env python3
"""
Simple test script for Anthropic prompt caching functionality
"""
import asyncio
import os
import sys
import logging

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spoon_ai.chat import ChatBot

# Enable logging to see debug messages
logging.basicConfig(level=logging.INFO)

async def test_prompt_caching():
    """Test prompt caching with Anthropic provider"""
    
    # Check if Anthropic API key is available via config or environment
    from spoon_ai.utils.config_manager import ConfigManager
    config_manager = ConfigManager()
    api_key = config_manager.get_api_key("anthropic")
    
    if not api_key:
        print("Anthropic API key not found in config or environment, skipping test")
        return
    
    # Create ChatBot with Anthropic provider
    chatbot = ChatBot(
        llm_provider="anthropic",
        model_name="claude-sonnet-4-20250514",
        enable_prompt_cache=True,
        base_url=None,  # Ensure we use native Anthropic API
        api_key=api_key  # Pass API key explicitly
    )
    
    # Test with a long system message (over 4000 characters to ensure 1024+ tokens for caching)
    long_system_msg = """You are an expert AI assistant with deep knowledge across multiple domains including science, technology, history, literature, and philosophy. Your responses should be accurate, well-reasoned, and helpful. You should provide detailed explanations when appropriate and cite relevant information. Always strive to be informative while maintaining clarity and accessibility in your communication. When dealing with complex topics, break them down into digestible parts and use examples where helpful. Your goal is to assist users in understanding complex concepts and solving problems effectively while maintaining accuracy and depth in your responses. You have access to vast knowledge spanning mathematics, physics, chemistry, biology, computer science, engineering, economics, psychology, sociology, political science, linguistics, anthropology, archaeology, geography, environmental science, and many other fields. When answering questions, you should draw upon this knowledge to provide comprehensive and nuanced responses that consider multiple perspectives and potential implications. You should also be aware of the limitations of your knowledge and acknowledge when information may be uncertain or when you should recommend consulting additional sources.""" * 5
    
    print(f"System message length: {len(long_system_msg)} characters")
    print(f"Caching enabled: {chatbot.enable_prompt_cache}")
    print(f"Provider: {chatbot.llm_provider}")
    print(f"Model: {chatbot.model_name}")
    print(f"Base URL: {chatbot.base_url}")
    
    # Test with ask() method first (system message caching)
    print("\n=== First request with ask() (system cache creation) ===")
    response1 = await chatbot.ask(
        messages=[{"role": "user", "content": "What is artificial intelligence?"}],
        system_msg=long_system_msg
    )
    print(f"Response 1 length: {len(response1)}")
    print(f"Cache metrics after ask() request: {chatbot.cache_metrics}")
    
    # Define tools for testing tool caching (in OpenAI format that ask_tool expects)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "The city and state"}
                    },
                    "required": ["location"]
                }
            }
        },
        {
            "type": "function", 
            "function": {
                "name": "calculate",
                "description": "Perform mathematical calculations",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string", "description": "Math expression"}
                    },
                    "required": ["expression"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "search_web",
                "description": "Search the web for information", 
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"}
                    },
                    "required": ["query"]
                }
            }
        }
    ]

    # Test with ask_tool() method (system + tools caching)
    print("\n=== Second request with ask_tool() (system + tools cache creation) ===")
    response2 = await chatbot.ask_tool(
        messages=[{"role": "user", "content": "Explain machine learning briefly."}],
        system_msg=long_system_msg,
        tools=tools,
        tool_choice="auto"
    )
    print(f"Response 2 content length: {len(response2.content)}")
    print(f"Response 2 finish reason: {response2.finish_reason}")
    print(f"Cache metrics after ask_tool(): {chatbot.cache_metrics}")

    # Third request with same system message and tools - should use cache for both
    print("\n=== Third request with ask_tool() (system + tools cache read) ===")
    response3 = await chatbot.ask_tool(
        messages=[{"role": "user", "content": "What are the latest developments in AI?"}],
        system_msg=long_system_msg,
        tools=tools,
        tool_choice="auto"
    )
    print(f"Response 3 content length: {len(response3.content)}")
    print(f"Response 3 finish reason: {response3.finish_reason}")
    print(f"Final cache metrics: {chatbot.cache_metrics}")
    
    # Print cache efficiency
    total_tokens = chatbot.cache_metrics["total_input_tokens"]
    cache_read_tokens = chatbot.cache_metrics["cache_read_input_tokens"]
    cache_creation_tokens = chatbot.cache_metrics["cache_creation_input_tokens"]
    
    if total_tokens > 0:
        cache_efficiency = (cache_read_tokens / (total_tokens + cache_read_tokens + cache_creation_tokens)) * 100
        print(f"\nCache efficiency: {cache_efficiency:.1f}%")
        print(f"Tokens saved through caching: {cache_read_tokens}")

if __name__ == "__main__":
    asyncio.run(test_prompt_caching())