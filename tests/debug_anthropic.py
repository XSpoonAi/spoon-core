"""
Simple test to debug the Anthropic provider issue
"""
import asyncio
import os
import logging
from spoon_ai.llm.factory import LLMFactory
from spoon_ai.schema import Message

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Import providers to register them
import spoon_ai.llm.providers

async def test_anthropic_directly():
    """Test Anthropic provider directly to debug the issue"""
    print("Testing Anthropic provider directly...")
    
    try:
        # Create provider
        provider = LLMFactory.create("anthropic")
        print(f"✅ Provider created: {type(provider)}")
        print(f"Model: {provider.config.model}")
        
        # Create test message
        test_message = Message(role="user", content="Hello")
        print(f"✅ Test message created: {test_message}")
        
        # Let's check what happens in _format_system_message_with_cache
        print("Testing _format_system_message_with_cache...")
        result1 = provider._format_system_message_with_cache(None)
        print(f"None -> {result1}")
        result2 = provider._format_system_message_with_cache("")  
        print(f"Empty string -> {result2}")
        result3 = provider._format_system_message_with_cache("test")
        print(f"'test' -> {result3}")
        
        # Test the chat method with detailed debugging
        print("Calling chat method...")
        print("Input messages:", [test_message])
        print("System messages:", None)
        
        # Let's manually call the chat method parts to see where it fails
        system_formatted = None
        if None:  # system_msgs
            system_content = " ".join([msg.content for msg in None])
            system_formatted = provider._format_system_message_with_cache(system_content)
        
        print("System formatted:", system_formatted)
        
        # Format messages for Anthropic API (exclude system messages)
        anthropic_messages = []
        for message in [test_message]:
            if message.role != "system":  # Anthropic handles system separately
                anthropic_messages.append({
                    "role": message.role,
                    "content": message.content
                })
        
        print("Anthropic messages:", anthropic_messages)
        
        response = await provider.chat(messages=[test_message])
        print(f"✅ Response received: {response}")
        print(f"Content: {response.content}")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_anthropic_directly())