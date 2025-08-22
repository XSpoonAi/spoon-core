import asyncio
import datetime
import json
import logging
import os
import shlex
import sys
import traceback
from pathlib import Path
from typing import Callable, Dict, List, Any

from prompt_toolkit import PromptSession, print_formatted_text
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.formatted_text import HTML as PromptHTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style

from spoon_ai.agents import SpoonReactAI, SpoonReactMCP
from spoon_ai.agents.my_agents import CUSTOM_AGENTS
from spoon_ai.retrieval.document_loader import DocumentLoader
from spoon_ai.schema import Message, Role
from spoon_ai.trade.aggregator import Aggregator
from spoon_ai.config.manager import ConfigManager


# Create a log filter to filter out log messages containing specific keywords
class KeywordFilter(logging.Filter):
    def __init__(self, keywords):
        super().__init__()
        self.keywords = keywords

    def filter(self, record):
        # If the log message contains any keywords, don't display this message
        if record.getMessage():
            for keyword in self.keywords:
                if keyword.lower() in record.getMessage().lower():
                    return False
        return True

# Set up basic logging configuration
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("cli")

# Add keyword filter for startup noise reduction
keyword_filter = KeywordFilter([
    "telemetry",
    "anonymized",
    "n_results",
    "updating n_results",
    "number of requested results",
    "elements in index",
    "loaded configuration from",
    "found configuration file",
    "using provider from config file",
    "created provider instance",
    "configured provider",
    "llm manager initialized",
    "default provider",
    "fallback chain",
    "loaded environment variables",
    "using fallback chain from config file",
    "chatbot initialized"
])
logger.addFilter(keyword_filter)

# Also apply filter to root logger
root_logger = logging.getLogger()
root_logger.addFilter(keyword_filter)

# Disable logs from third-party libraries by setting them to ERROR level or higher
logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("requests").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("chromadb").setLevel(logging.ERROR)
logging.getLogger("chroma").setLevel(logging.ERROR)
logging.getLogger("langchain").setLevel(logging.ERROR)
logging.getLogger("anthropic").setLevel(logging.ERROR)
logging.getLogger("google").setLevel(logging.ERROR)
logging.getLogger("fastmcp").setLevel(logging.ERROR)

# Disable verbose logs from internal modules during startup
logging.getLogger("spoon_ai.config").setLevel(logging.WARNING)
logging.getLogger("spoon_ai.llm").setLevel(logging.WARNING)
logging.getLogger("spoon_ai.utils.config_manager").setLevel(logging.WARNING)
from spoon_ai.schema import AgentState
from spoon_ai.social_media.telegram import TelegramClient

# handler = logging.StreamHandler()
# handler.setFormatter(ColoredFormatter())
# logger.addHandler(handler)

DEBUG = False
def cli_debug_log(message):
    if DEBUG:
        logger.info(f"CLI DEBUG: {message}\n")

class SpoonCommand:
    name: str
    description: str
    handler: Callable
    aliases: List[str] = []
    def __init__(self, name: str, description: str, handler: Callable, aliases: List[str] = []):
        self.name = name
        self.description = description
        self.handler = handler
        self.aliases = aliases

class SpoonAICLI:
    def __init__(self):
        self.agents = {}
        self.current_agent = None
        self.config_dir = Path(__file__).resolve().parents[1]
        self.commands: Dict[str, SpoonCommand] = {}
        self.config_manager = ConfigManager()

        # Get blockchain configuration from environment variables
        config_data = {}

        # Get RPC_URL from config.json or environment
        rpc_url = config_data.get("RPC_URL") or os.getenv("RPC_URL")

        # Get CHAIN_ID from config.json or environment
        chain_id_str = config_data.get("CHAIN_ID") or os.getenv("CHAIN_ID", "1")
        chain_id = int(chain_id_str) if chain_id_str and str(chain_id_str).strip() else 1

        # Get SCAN_URL from config.json or environment
        scan_url = config_data.get("SCAN_URL") or os.getenv("SCAN_URL", "https://etherscan.io")



        self.aggregator = Aggregator(
            rpc_url=rpc_url,
            chain_id=chain_id,
            scan_url=scan_url
        )
        self._should_exit = False
        self._init_commands()
        self._set_prompt_toolkit()


    def _init_commands(self):

        # Help Command
        self.add_command(SpoonCommand(
            name="help",
            description="Show help information",
            handler=self._help,
            aliases=["h", "?"]
        ))

        # Exit Command
        self.add_command(SpoonCommand(
            name="exit",
            description="Exit the CLI",
            handler=self._exit,
            aliases=["quit", "q"]
        ))

        # Load Agent Command
        self.add_command(SpoonCommand(
            name="load-agent",
            description="Load an agent by name",
            handler=self._handle_load_agent,
            aliases=["load"]
        ))

        # List Agents Command
        self.add_command(SpoonCommand(
            name="list-agents",
            description="List all available agents",
            handler=self._handle_list_agents,
            aliases=["agents"]
        ))

        # Config Command
        self.add_command(SpoonCommand(
            name="config",
            description="Configure settings like API keys",
            handler=self._handle_config,
            aliases=["cfg", "settings"]
        ))

        # Reload Config Command
        self.add_command(SpoonCommand(
            name="reload-config",
            description="Reload configuration for current agent",
            handler=self._handle_reload_config,
            aliases=["reload"]
        ))

        # Action Command
        self.add_command(SpoonCommand(
            name="action",
            description="Perform an action with the current agent",
            handler=self._handle_action,
            aliases=["a"]
        ))

        # Chat History Commands
        self.add_command(SpoonCommand(
            name="new-chat",
            description="Start a new chat (clear history)",
            handler=self._handle_new_chat,
            aliases=["new"]
        ))

        self.add_command(SpoonCommand(
            name="list-chats",
            description="List available chat histories",
            handler=self._handle_list_chats,
            aliases=["chats"]
        ))

        self.add_command(SpoonCommand(
            name="load-chat",
            description="Load a specific chat history",
            handler=self._handle_load_chat
        ))

        # Transfer Command
        self.add_command(SpoonCommand(
            name="transfer",
            description="Transfer tokens to an address",
            handler=self._handle_transfer,
            aliases=["send"]
        ))

        # LLM Status Command
        self.add_command(SpoonCommand(
            name="llm-status",
            description="Show LLM provider status and configuration",
            handler=self._handle_llm_status,
            aliases=["llm", "providers"]
        ))

        # Swap Command
        self.add_command(SpoonCommand(
            name="swap",
            description="Swap tokens using aggregator",
            handler=self._handle_swap
        ))

        # Token Info Commands
        self.add_command(SpoonCommand(
            name="token-info",
            description="Get token information by address",
            handler=self._handle_token_info_by_address,
            aliases=["token"]
        ))

        self.add_command(SpoonCommand(
            name="token-by-symbol",
            description="Get token information by symbol",
            handler=self._handle_token_info_by_symbol,
            aliases=["symbol"]
        ))

        # Load Documents Command
        self.add_command(SpoonCommand(
            name="load-docs",
            description="Load documents from a directory into the current agent",
            handler=self._handle_load_docs,
            aliases=["docs"]
        ))

        # Telegram Command
        self.add_command(SpoonCommand(
            name="telegram",
            description="Start the Telegram client",
            handler=self._handle_telegram_run,
            aliases=["tg"]
        ))

        # Toolkit Commands
        self.add_command(SpoonCommand(
            name="list-toolkit-categories",
            description="List all available toolkit categories",
            handler=self._handle_list_toolkit_categories,
            aliases=["toolkit-categories", "categories"]
        ))

        self.add_command(SpoonCommand(
            name="list-toolkit-tools",
            description="List tools in a specific category",
            handler=self._handle_list_toolkit_tools,
            aliases=["toolkit-tools"]
        ))

        self.add_command(SpoonCommand(
            name="load-toolkit-tools",
            description="Load toolkit tools from specified categories",
            handler=self._handle_load_toolkit_tools,
            aliases=["load-tools"]
        ))

        # System Info Command
        self.add_command(SpoonCommand(
            name="system-info",
            description="Display system information, environment status, and health checks",
            handler=self._handle_system_info,
            aliases=["sysinfo", "status", "info"]
        ))

        # Configuration Migration Commands
        self.add_command(SpoonCommand(
            name="migrate-config",
            description="Migrate legacy configuration to new unified format",
            handler=self._handle_migrate_config,
            aliases=["migrate"]
        ))

        self.add_command(SpoonCommand(
            name="check-config",
            description="Check if configuration needs migration",
            handler=self._handle_check_config,
            aliases=["check-migration"]
        ))

        self.add_command(SpoonCommand(
            name="validate-config",
            description="Validate current configuration and check for issues",
            handler=self._handle_validate_config,
            aliases=["validate"]
        ))

    def add_command(self, command: SpoonCommand):
        # Store primary command
        self.commands[command.name] = command
        # Store all aliases pointing to the same command
        for alias in command.aliases:
            self.commands[alias] = command

    def _help(self, input_list: List[str]):
        if len(input_list) <= 1:
            # show all available commands
            logger.info("Available commands:")
            for command in self.commands.values():
                logger.info(f"  {command.name}: {command.description}")
        else:
            # show help for a specific command
            command_name = input_list[1]
            if command_name in self.commands:
                logger.info(f"Help for {command_name}:")
                logger.info(self.commands[command_name].description)
            else:
                logger.error(f"Command {command_name} not found")

    def _get_prompt(self):
        agent_part = f"({self.current_agent.name})" if self.current_agent else "(no agent)"
        return f"Spoon AI {agent_part} > "

    async def _handle_load_agent(self, input_list: List[str]):
        if not input_list:
            logger.error("Missing agent name. Usage: load-agent <agent_name>")
            return

        name = input_list[0]

        # Get available agents to validate the name
        available_agents = self._get_available_agents()

        # Check if agent exists (by name or alias)
        agent_found = False
        if name in available_agents:
            agent_found = True
        else:
            # Check aliases
            for agent_name, config in available_agents.items():
                if name in config.get("aliases", []):
                    agent_found = True
                    break

        if not agent_found:
            logger.error(f"Agent '{name}' not found. Use 'list-agents' to see available agents.")
            return

        # Run the async load_agent method
        await self._load_agent(name)

    def _get_available_agents(self) -> Dict[str, Dict[str, Any]]:
        """Get available agents from unified configuration"""
        # Built-in agent definitions
        builtin_agents = {
            "react": {
                "class": "SpoonReactAI",
                "aliases": ["spoon_react"],
                "description": "A smart AI agent for blockchain operations"
            },
            "spoon_react_mcp": {
                "class": "SpoonReactMCP",
                "aliases": [],
                "description": "SpoonReact agent with MCP protocol support"
            }
        }

        # Get agents from unified configuration
        try:
            config = self.config_manager.load_config()
            config_agents = {}
            for name, agent_config in config.agents.items():
                config_agents[name] = {
                    "class": agent_config.class_name,
                    "aliases": agent_config.aliases,
                    "description": agent_config.description or f"Agent: {name}",
                    "config": agent_config.config
                }
        except Exception as e:
            logger.warning(f"Failed to load agent configuration: {e}")
            config_agents = {}

        # Merge built-in and config agents (config takes precedence)
        available_agents = {**builtin_agents, **config_agents}

        return available_agents

    async def _load_agent(self, name: str):
        """Load agent by name using unified tool configuration system"""
        try:
            # Load configuration using unified system
            config = self.config_manager.load_config()

            # Get agent configuration
            agent_config = self.config_manager.get_agent_config(name)

            # Create agent instance based on class
            agent_class = agent_config.class_name
            agent_instance_config = agent_config.config

            if agent_class == "SpoonReactAI":
                agent_instance = SpoonReactAI(**agent_instance_config)
            elif agent_class == "SpoonReactMCP":
                # For MCP agents, the unified system handles MCP server lifecycle
                agent_instance = SpoonReactMCP(**agent_instance_config)
            elif agent_class in CUSTOM_AGENTS:
                # Handle custom agents
                custom_agent_class = CUSTOM_AGENTS[agent_class]
                agent_instance = custom_agent_class(**agent_instance_config)
            else:
                logger.error(f"Unknown agent class: {agent_class}")
                logger.error(f"Available custom agents: {list(CUSTOM_AGENTS.keys())}")
                return

            # Load tools using the unified system
            tools = await self.config_manager.load_agent_tools(name)

            # Set the agent's name to the configuration name
            if hasattr(agent_instance, 'name'):
                agent_instance.name = name

            # Add tools to agent's tool manager
            if hasattr(agent_instance, 'avaliable_tools') and tools:
                agent_instance.avaliable_tools.add_tools(*tools)
                logger.info(f"Added {len(tools)} tools to agent {name}")

            # Store agent and set as current
            self.agents[name] = agent_instance
            self.current_agent = agent_instance

            logger.info(f"Successfully loaded agent: {name} with {len(tools)} tools")

        except Exception as e:
            logger.error(f"Failed to load agent {name}: {e}")
            logger.debug(f"Agent loading error details: {e}", exc_info=True)







    def _handle_list_agents(self, input_list: List[str]):
        """List all available agents from configuration and built-in definitions"""
        available_agents = self._get_available_agents()

        logger.info("Available agents:")
        for name, config in available_agents.items():
            aliases = config.get("aliases", [])
            alias_str = f" (aliases: {', '.join(aliases)})" if aliases else ""
            description = config.get("description", "No description")
            logger.info(f"  {name}{alias_str}: {description}")

        # Also show currently loaded agents (avoid duplicates)
        if self.agents:
            logger.info("\nCurrently loaded agents:")
            loaded_agents = {}
            for agent_key, agent in self.agents.items():
                # Use agent name as key to avoid duplicates
                if agent.name not in loaded_agents:
                    loaded_agents[agent.name] = agent

            for agent in loaded_agents.values():
                logger.info(f"  {agent.name}: {agent.description}")

    async def _load_default_agent(self):
        """Load the default agent from configuration"""
        default_agent = self.config_manager.get("default_agent", "react")
        logger.info(f"Loading default agent: {default_agent}")
        await self._load_agent(default_agent)

    def _set_prompt_toolkit(self):
        self.style = Style.from_dict({
            'prompt': 'ansicyan bold',
            'command': 'ansigreen',
            'error': 'ansired bold',
            'success': 'ansigreen bold',
            'warning': 'ansiyellow',
        })

        # Collect all command names and aliases
        all_names = set()
        for cmd in self.commands.values():
            all_names.add(cmd.name)
            all_names.update(cmd.aliases)

        self.completer = WordCompleter(
            list(all_names),
            ignore_case=True,
        )
        history_file = self.config_dir / "history.txt"
        history_file.touch(exist_ok=True)
        self.session = PromptSession(
            style=self.style,
            completer=self.completer,
            history=FileHistory(history_file),
        )

    async def _handle_input(self, input_text: str):
        try:
            input_list = shlex.split(input_text)
            command_name = input_list[0]
            command = self.commands.get(command_name)
            if command:
                if asyncio.iscoroutinefunction(command.handler):
                    await command.handler(input_list[1:] if len(input_list) > 1 else [])
                else:
                    command.handler(input_list[1:] if len(input_list) > 1 else [])
            else:
                logger.error(f"Command {command_name} not found")
        except Exception as e:
            logger.error(f"Error: {e}")
            logger.error(traceback.format_exc())

    async def _handle_action(self, input_list: List[str]):
        if not self.current_agent:
            logger.error("No agent loaded")
            return

        if len(input_list) < 1:
            logger.error("Usage: action <action_name> [action_args]")
            return

        action_name = input_list[0]
        action_args = input_list[1:] if len(input_list) > 1 else []
        try:
            if action_name == "list_mcp_tools":
                print(await self.current_agent.list_mcp_tools())
                return

            if action_name == "chat":
                try:
                    if action_args:
                        # If arguments provided, use the old behavior
                        # Check if current agent is SpoonReactAI
                        from spoon_ai.agents.spoon_react import SpoonReactAI
                        if isinstance(self.current_agent, SpoonReactAI):
                            # For SpoonReactAI agents, use run method
                            res = await self.current_agent.run(action_args[0])
                        else:
                            # For other agents, use perform_action method
                            res = self.current_agent.perform_action(action_name, action_args)
                        logger.info(res)
                    else:
                        # Start interactive chat mode
                        await self._start_interactive_chat()
                except Exception as e:
                    logger.error(f"Error during action: {e}")
                    logger.error(traceback.format_exc())
            elif action_name == "react":
                await self._start_interactive_react()
            elif action_name == "new":
                self._handle_new_chat([])
            elif action_name == "list":
                self._handle_list_chats([])
            elif action_name == "load":
                if len(action_args) != 1:
                    logger.error("Usage: action load <agent_name>")
                    return
                self._handle_load_chat(action_args)
            else:
                if hasattr(self.current_agent, "perform_action") and callable(getattr(self.current_agent, "perform_action", None)):
                    self.current_agent.perform_action(action_name, action_args)
                else:
                    logger.warning(f"command '{action_name}' is invalid, the current Agent does not support custom actions")
        except Exception as e:
            logger.error(f"Error during action '{action_name}': {e}")
            logger.debug(traceback.format_exc())

    async def _start_interactive_chat(self):
        """Start an interactive chat session with the current agent."""
        # Initialize chat history if not exists
        if not hasattr(self.current_agent, 'chat_history'):
            self.current_agent.chat_history = {
                'metadata': {
                    'agent_name': self.current_agent.name,
                    'created_at': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'updated_at': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                },
                'messages': []
            }

        # Create a new prompt session for chat
        chat_style = Style.from_dict({
            'agent': 'ansicyan bold',
            'user': 'ansigreen',
            'system': 'ansigray',
            'header': 'ansiyellow bold',
            'thinking': 'ansiyellow',
            'info': 'ansiblue',
        })

        # Create a chat log file
        chat_log_dir = Path('chat_logs')
        chat_log_dir.mkdir(exist_ok=True)
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        chat_log_file = chat_log_dir / f"chat_{self.current_agent.name}_{timestamp}.txt"

        # Display welcome message
        logger.info("="*80)
        logger.info(f"Starting chat with {self.current_agent.name}")
        logger.info("📝 Type your message and press Enter to send.")
        logger.info("🔄 Press Ctrl+C or Ctrl+D to exit chat mode and return to main CLI.")
        logger.info(f"📋 Chat log will be saved to: {chat_log_file}")
        logger.info("="*80 + "\n")

        # Function to save chat to log file
        def save_chat_to_log():
            with open(chat_log_file, 'w') as f:
                f.write(f"Chat session with {self.current_agent.name}\n")
                f.write(f"Started at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

                # Get message list
                chat_messages = []
                if isinstance(self.current_agent.chat_history, dict) and 'messages' in self.current_agent.chat_history:
                    chat_messages = self.current_agent.chat_history.get('messages', [])
                elif isinstance(self.current_agent.chat_history, list):
                    chat_messages = self.current_agent.chat_history

                for entry in chat_messages:
                    if entry['role'] == 'user':
                        f.write(f"You: {entry['content']}\n\n")
                    else:
                        f.write(f"{self.current_agent.name}: {entry['content']}\n\n")

                f.write(f"\nChat ended at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

            self.current_agent.save_chat_history()

        # Display chat history
        chat_messages = []
        if isinstance(self.current_agent.chat_history, dict) and 'messages' in self.current_agent.chat_history:
            chat_messages = self.current_agent.chat_history.get('messages', [])
        elif isinstance(self.current_agent.chat_history, list):
            chat_messages = self.current_agent.chat_history

        if chat_messages:
            print_formatted_text(PromptHTML("<header>Chat History:</header>"), style=chat_style)
            for entry in chat_messages:
                if entry['role'] == 'user':
                    print_formatted_text(PromptHTML(f"<user>You:</user> {entry['content']}"), style=chat_style)
                else:
                    print_formatted_text(PromptHTML(f"<agent>{self.current_agent.name}:</agent> {entry['content']}"), style=chat_style)
            logger.info("\n" + "-"*50 + "\n")

        # Check if current agent is SpoonReactAI
        from spoon_ai.agents.spoon_react import SpoonReactAI
        is_react_agent = isinstance(self.current_agent, SpoonReactAI)

        # Start chat loop
        try:
            while True:
                try:
                    # Get user input
                    user_message = await self.session.prompt_async(
                        PromptHTML("<user>You</user> > "),
                        style=self.style,
                    )

                    user_message = user_message.strip()
                    if not user_message:
                        continue

                    # Add to history
                    if isinstance(self.current_agent.chat_history, dict) and 'messages' in self.current_agent.chat_history:
                        self.current_agent.chat_history['messages'].append({
                            'role': 'user',
                            'content': user_message
                        })
                    else:
                        self.current_agent.chat_history.append({
                            'role': 'user',
                            'content': user_message
                        })

                    # Get response from agent
                    print_formatted_text(PromptHTML(f"<thinking>{self.current_agent.name} is thinking...</thinking>"), style=chat_style)

                    # Use streaming response if available
                    try:
                        # Define a simpler streaming function
                        async def stream_response():
                            cli_debug_log("Starting stream_response function")

                            # Display agent name
                            print_formatted_text(
                                PromptHTML(f"<agent>{self.current_agent.name}:</agent>"),
                                style=chat_style,
                                end=" "
                            )

                            # Collect the full response
                            full_response = ""
                            chunk_count = 0
                            agent_name_prefix = f"{self.current_agent.name}: "
                            agent_name_prefix_lower = agent_name_prefix.lower()

                            # Stream the response
                            cli_debug_log("Starting to stream response")
                            try:
                                if hasattr(self.current_agent, 'astream_chat_response'):
                                    async for chunk in self.current_agent.astream_chat_response(user_message):
                                        if chunk:
                                            chunk_count += 1
                                            cli_debug_log(f"Received chunk #{chunk_count}: {chunk[:20]}...")

                                            # Check if first chunk starts with agent name and remove it
                                            if chunk_count == 1:
                                                # Check for exact match
                                                if chunk.startswith(agent_name_prefix):
                                                    chunk = chunk[len(agent_name_prefix):]
                                                    cli_debug_log(f"Removed agent name prefix from chunk")
                                                # Check for case-insensitive match
                                                elif chunk.lower().startswith(agent_name_prefix_lower):
                                                    chunk = chunk[len(agent_name_prefix):]
                                                    cli_debug_log(f"Removed case-insensitive agent name prefix from chunk")

                                            full_response += chunk
                                            print(chunk, end="", flush=True)
                                else:
                                    # For SpoonReactAI which doesn't have astream_chat_response
                                    if is_react_agent:
                                        full_response = await self.current_agent.run(user_message)
                                        print(full_response)
                                        chunk_count = 1

                                cli_debug_log(f"Finished streaming, received {chunk_count} chunks")

                                # Ensure a new line after the response
                                if chunk_count > 0:
                                    print()
                            except Exception as e:
                                cli_debug_log(f"Error during streaming iteration: {e}")
                                print(f"\nError during streaming: {e}")
                                if not full_response:
                                    cli_debug_log("No response received, using non-streaming")
                                    print("Using non-streaming response...")
                                    if is_react_agent:
                                        full_response = await self.current_agent.run(user_message)
                                    else:
                                        full_response = self.current_agent._generate_response(user_message)
                                    print(full_response)

                            return full_response

                        # Run the streaming function
                        cli_debug_log("Running stream_response function")
                        response = await stream_response()
                        cli_debug_log(f"Stream response completed, got response of length {len(response)}")

                        # Add to history if we got a response
                        if response:
                            cli_debug_log("Adding streaming response to chat history")
                            if isinstance(self.current_agent.chat_history, dict) and 'messages' in self.current_agent.chat_history:
                                self.current_agent.chat_history['messages'].append({
                                    'role': 'assistant',
                                    'content': response
                                })
                            else:
                                self.current_agent.chat_history.append({
                                    'role': 'assistant',
                                    'content': response
                                })
                        else:
                            # Fallback if streaming returned empty
                            cli_debug_log("Streaming returned empty response, falling back to non-streaming")
                            logger.info("Streaming returned empty response, falling back to non-streaming...")
                            if is_react_agent:
                                response = await self.current_agent.run(user_message)
                            else:
                                response = self.current_agent._generate_response(user_message)
                            cli_debug_log(f"Got non-streaming response of length {len(response)}")

                            # Check if response starts with agent name and remove it
                            agent_name_prefix = f"{self.current_agent.name}: "
                            if response.startswith(agent_name_prefix):
                                response = response[len(agent_name_prefix):]
                                cli_debug_log(f"Removed agent name prefix from non-streaming response")

                            # Add to history
                            if isinstance(self.current_agent.chat_history, dict) and 'messages' in self.current_agent.chat_history:
                                self.current_agent.chat_history['messages'].append({
                                    'role': 'assistant',
                                    'content': response
                                })
                            else:
                                self.current_agent.chat_history.append({
                                    'role': 'assistant',
                                    'content': response
                                })

                            # Display response
                            print_formatted_text(PromptHTML(f"<agent>{self.current_agent.name}:</agent> {response}"), style=chat_style)

                        # Reset agent state to IDLE after response is processed
                        cli_debug_log("Resetting agent state to IDLE")
                        if hasattr(self.current_agent, 'reset_state'):
                            self.current_agent.reset_state()
                        elif hasattr(self.current_agent, 'state'):
                            from spoon_ai.schema import AgentState
                            self.current_agent.state = AgentState.IDLE
                            self.current_agent.current_step = 0

                    except Exception as e:
                        # Fallback to non-streaming if streaming not available or failed
                        cli_debug_log(f"Streaming failed with error: {e}")
                        logger.info(f"Streaming failed: {e}. Using non-streaming response...")
                        if is_react_agent:
                            response = await self.current_agent.run(user_message)
                        else:
                            response = self.current_agent._generate_response(user_message)
                        cli_debug_log(f"Got non-streaming response of length {len(response)}")

                        # Check if response starts with agent name and remove it
                        agent_name_prefix = f"{self.current_agent.name}: "
                        if response.startswith(agent_name_prefix):
                            response = response[len(agent_name_prefix):]
                            cli_debug_log(f"Removed agent name prefix from non-streaming response")

                        # Add to history
                        if isinstance(self.current_agent.chat_history, dict) and 'messages' in self.current_agent.chat_history:
                            self.current_agent.chat_history['messages'].append({
                                'role': 'assistant',
                                'content': response
                            })
                        else:
                            self.current_agent.chat_history.append({
                                'role': 'assistant',
                                'content': response
                            })

                        # Display response
                        print_formatted_text(PromptHTML(f"<agent>{self.current_agent.name}:</agent> {response}"), style=chat_style)

                        # Reset agent state to IDLE after response is processed
                        cli_debug_log("Resetting agent state to IDLE")
                        if hasattr(self.current_agent, 'reset_state'):
                            self.current_agent.reset_state()
                        elif hasattr(self.current_agent, 'state'):
                            from spoon_ai.schema import AgentState
                            self.current_agent.state = AgentState.IDLE
                            self.current_agent.current_step = 0

                except (KeyboardInterrupt, EOFError):
                    logger.info("\nExiting chat mode...")
                    break
        finally:
            # Save chat log when exiting
            save_chat_to_log()
            print_formatted_text(
                PromptHTML(f"<info>Chat log saved to: {chat_log_file}</info>"),
                style=chat_style
            )
            print("="*50 + "\n")

    async def _start_interactive_react(self):
        """Start an interactive react session with the current agent."""
        # Check if current agent is a ReActAgent
        from spoon_ai.agents.react import ReActAgent
        if not isinstance(self.current_agent, ReActAgent):
            logger.warning(f"Current agent {self.current_agent.name} is not a ReActAgent. Switching to chat mode.")
            await self._start_interactive_chat()
            return

        # Create a new prompt session for react
        react_style = Style.from_dict({
            'agent': 'ansicyan bold',
            'user': 'ansigreen',
            'system': 'ansigray',
            'header': 'ansiyellow bold',
            'thinking': 'ansiyellow',
            'info': 'ansiblue',
        })

        # Create a react log file
        react_log_dir = Path('react_logs')
        react_log_dir.mkdir(exist_ok=True)
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        react_log_file = react_log_dir / f"react_{self.current_agent.name}_{timestamp}.txt"

        # Display welcome message
        logger.info("="*80)
        logger.info(f"Starting react session with {self.current_agent.name}")
        logger.info("📝 Type your message and press Enter to send.")
        logger.info("🔄 Press Ctrl+C or Ctrl+D to exit react mode and return to main CLI.")
        logger.info(f"📋 React log will be saved to: {react_log_file}")
        logger.info("⚠️ Note: This session will not save chat history.")
        logger.info("="*80 + "\n")

        # Function to save react to log file
        def save_react_to_log():
            with open(react_log_file, 'w') as f:
                f.write(f"React session with {self.current_agent.name}\n")
                f.write(f"Started at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

                # Get message list
                react_messages = []
                if hasattr(self.current_agent, 'memory') and hasattr(self.current_agent.memory, 'messages'):
                    react_messages = self.current_agent.memory.messages

                for message in react_messages:
                    if message.role == Role.USER:
                        f.write(f"You: {message.content}\n\n")
                    elif message.role == Role.ASSISTANT:
                        f.write(f"{self.current_agent.name}: {message.content}\n\n")
                    elif message.role == Role.TOOL:
                        f.write(f"Tool: {message.content}\n\n")

                f.write(f"\nReact session ended at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        # Start react loop
        try:
            while True:
                try:
                    # Get user input
                    user_message = await self.session.prompt_async(
                        PromptHTML("<user>You</user> > "),
                        style=self.style,
                    )

                    user_message = user_message.strip()

                    if not user_message:
                        continue

                    # Add to memory
                    # self.current_agent.add_message("user", user_message)

                    # Get response from agent
                    print_formatted_text(PromptHTML(f"<thinking>{self.current_agent.name} is thinking...</thinking>"), style=react_style)

                    # Run the ReAct agent's step method
                    result = await self.current_agent.run(user_message)

                    # Display the result
                    print_formatted_text(PromptHTML(f"<agent>{self.current_agent.name}:</agent> {result}"), style=react_style)

                    # Reset the agent state
                    if hasattr(self.current_agent, 'reset_state'):
                        self.current_agent.reset_state()
                    else:
                        self.current_agent.state = AgentState.IDLE
                        self.current_agent.current_step = 0

                except (KeyboardInterrupt, EOFError):
                    logger.info("\nExiting react mode...")
                    break
        finally:
            # Save react log when exiting
            save_react_to_log()
            print_formatted_text(
                PromptHTML(f"<info>React log saved to: {react_log_file}</info>"),
                style=react_style
            )
            print("="*50 + "\n")

    async def run(self):
        await self._load_default_agent()
        self._should_exit = False

        while not self._should_exit:
            try:
                input_text = await self.session.prompt_async(
                    self._get_prompt(),
                    style=self.style,
                )
                input_text = input_text.strip()
                if not input_text:
                    continue
                await self._handle_input(input_text)
            except KeyboardInterrupt:
                continue
            except EOFError:
                self._should_exit = True
                # Perform cleanup before exit
                try:
                    await self.config_manager.cleanup()
                except Exception as e:
                    logger.warning(f"Error during cleanup: {e}")



    def _exit(self, input_list: List[str]):
        logger.info("Exiting Spoon AI")
        self._should_exit = True

    def _handle_new_chat(self, input_list: List[str]):
        if not self.current_agent:
            logger.error("No agent loaded")
            return

        # Reset chat history with metadata
        self.current_agent.chat_history = {
            'metadata': {
                'agent_name': self.current_agent.name,
                'created_at': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'updated_at': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            },
            'messages': []
        }

        logger.info(f"Started new chat with {self.current_agent.name} (chat history cleared)")

    def _handle_list_chats(self, input_list: List[str]):
        chat_logs_dir = Path('chat_logs')
        if not chat_logs_dir.exists():
            logger.info("No chat histories found")
            return

        chat_files = list(chat_logs_dir.glob('*_history.json'))
        if not chat_files:
            logger.info("No chat histories found")
            return

        logger.info("Available chat histories:")
        for chat_file in chat_files:
            agent_name = chat_file.stem.replace('_history', '')
            try:
                with open(chat_file, 'r', encoding='utf-8') as f:
                    history = json.load(f)
                    msg_count = len(history)
                    if msg_count > 0:
                        first_date = "Unknown"
                        last_date = "Unknown"

                        if 'metadata' in history and 'created_at' in history['metadata']:
                            first_date = history['metadata']['created_at']
                            last_date = history['metadata'].get('updated_at', first_date)
                        else:
                            file_time = datetime.datetime.fromtimestamp(chat_file.stat().st_mtime)
                            last_date = file_time.strftime('%Y-%m-%d')

                        logger.info(f"  {agent_name}: {msg_count} messages ({first_date} - {last_date})")
                    else:
                        logger.info(f"  {agent_name}: Empty chat history")
            except Exception as e:
                logger.info(f"  {agent_name}: Error reading history - {e}")

    def _handle_load_chat(self, input_list: List[str]):
        if not self.current_agent:
            logger.error("No agent loaded")
            return

        if len(input_list) != 1:
            logger.error("Usage: load-chat <agent_name>")
            return

        agent_name = input_list[0]
        chat_file = Path('chat_logs') / f'{agent_name}_history.json'

        if not chat_file.exists():
            logger.error(f"Chat history for {agent_name} not found")
            return

        try:
            with open(chat_file, 'r', encoding='utf-8') as f:
                history = json.load(f)

            self.current_agent.save_chat_history()

            self.current_agent.chat_history = history
            logger.info(f"Loaded chat history from {agent_name} ({len(history)} messages)")
        except Exception as e:
            logger.error(f"Error loading chat history: {e}")

    def _handle_config(self, input_list: List[str]):
        """Handle configuration command"""
        if not input_list:
            self._show_config()
            return

        if input_list[0] == "help":
            self._show_config_help()
            return

        if len(input_list) == 1:
            key = input_list[0]
            value = self.config_manager.get(key)
            if value is not None:
                logger.info(f"{key}: {value}")
            else:
                logger.info(f"Configuration item '{key}' does not exist")
            return

        if len(input_list) >= 2:
            key = input_list[0]
            value = " ".join(input_list[1:])

            if key.startswith("api_keys.") or key == "api_key":
                provider = key.split(".")[-1] if "." in key else input_list[1]
                if key == "api_key":
                    if len(input_list) < 3:
                        logger.info("Usage: config api_key <provider> <key>")
                        return
                    provider = input_list[1]
                    value = " ".join(input_list[2:])
                self.config_manager.set_api_key(provider, value)
                logger.info(f"Set {provider} API key")
            else:
                self.config_manager.set(key, value)
                logger.info(f"Set {key} = {value}")

    def _show_config(self):
        """Show all configuration"""
        config = self.config_manager.list_config()
        logger.info("Current configuration:")

        # Handle API keys, don't show full keys
        if "api_keys" in config:
            logger.info("API Keys:")
            for provider, key in config["api_keys"].items():
                masked_key = "Not set" if not key else f"{key[:4]}...{key[-4:]}" if len(key) > 8 else "Set"
                logger.info(f"  {provider}: {masked_key}")

        # Show other configuration
        for key, value in config.items():
            if key != "api_keys":
                logger.info(f"{key}: {value}")

    def _show_config_help(self):
        """Show configuration command help"""
        logger.info("Configuration command usage:")
        logger.info("  config                 - Show all configuration")
        logger.info("  config <key>           - Show specific configuration item")
        logger.info("  config <key> <value>   - Set configuration item")
        logger.info("  config api_key <provider> <key> - Set API key")
        logger.info("  config help            - Show this help")
        logger.info("")
        logger.info("Examples:")
        logger.info("  config api_key openai sk-xxxx")
        logger.info("  config api_key anthropic sk-ant-xxxx")
        logger.info("  config default_agent my_agent")


    def _handle_llm_status(self, input_list: List[str]):
        """Handle LLM status command to show provider configuration and availability"""
        try:
            from spoon_ai.llm.config import ConfigurationManager

            config_manager = ConfigurationManager()

            logger.info("=" * 60)
            logger.info("LLM PROVIDER STATUS")
            logger.info("=" * 60)

            # Show current default provider
            default_provider = config_manager.get_default_provider()
            logger.info(f"Current Default Provider: {default_provider}")

            # Show available providers by priority
            available_providers = config_manager.get_available_providers_by_priority()
            if available_providers:
                logger.info(f"Available Providers (by priority): {', '.join(available_providers)}")
            else:
                logger.warning("No providers with valid API keys found!")

            # Show detailed provider information
            logger.info("\nProvider Details:")
            provider_info = config_manager.get_provider_info()

            for provider, info in provider_info.items():
                status_icon = "Available" if info['available'] else "Not Available"
                logger.info(f"  {provider.upper()}: {status_icon}")

                if info['available']:
                    logger.info(f"    Model: {info['model']}")
                    if info.get('base_url'):
                        logger.info(f"    Base URL: {info['base_url']}")
                    logger.info(f"    Configured via: {info['configured_via']}")
                else:
                    if 'error' in info:
                        logger.info(f"    Error: {info['error']}")

            # Show current agent's LLM configuration
            if self.current_agent and hasattr(self.current_agent, 'llm'):
                logger.info(f"\nCurrent Agent LLM:")
                llm = self.current_agent.llm
                if hasattr(llm, 'use_llm_manager'):
                    architecture = "New LLM Manager" if llm.use_llm_manager else "Legacy"
                    logger.info(f"    Architecture: {architecture}")
                if hasattr(llm, 'llm_provider'):
                    logger.info(f"    Provider: {llm.llm_provider}")
                if hasattr(llm, 'model_name'):
                    logger.info(f"    Model: {llm.model_name}")

            # Show configuration tips
            logger.info("\nConfiguration Tips:")
            logger.info("  • Set DEFAULT_LLM_PROVIDER environment variable to override default selection")
            logger.info("  • Configure API keys in .env file or environment variables")
            logger.info("  • Provider priority: anthropic > openai > gemini")
            logger.info("  • Use 'config' command to manage other settings")

            logger.info("=" * 60)

        except Exception as e:
            logger.error(f"Error getting LLM status: {e}")
            import traceback
            logger.debug(traceback.format_exc())

    def _handle_reload_config(self, input_list: List[str]):
        """Reload configuration"""
        if not self.current_agent:
            logger.info("No agent loaded, please load an agent first")
            return

        try:
            # Clean up current configuration
            asyncio.run(self.config_manager.cleanup())

            # Reload configuration manager
            self.config_manager = ConfigManager()

            # Get current agent name
            agent_name = self.current_agent.name

            # Reload the agent with new configuration
            asyncio.run(self._load_agent(agent_name))

            logger.info(f"Reloaded configuration for agent: {agent_name}")

        except Exception as e:
            logger.error(f"Failed to reload configuration: {e}")
            logger.debug(f"Config reload error details: {e}", exc_info=True)

    def _handle_transfer(self, input_list: List[str]):
        """
        Handle the transfer command
        Usage: transfer <to_address> <amount> [token_address]
        """

        if len(input_list) < 2:
            print("Usage: transfer <to_address> <amount> [token_address]")
            return

        to_address = input_list[0]

        try:
            amount = float(input_list[1])
        except ValueError:
            print("Amount must be a number")
            return

        token_address = None
        if len(input_list) >= 3:
            token_address = input_list[2]

        # Initialize aggregator
        try:

            # Get account address from private key
            private_key = os.getenv("PRIVATE_KEY")
            if not private_key:
                print("PRIVATE_KEY is not set in environment variables")
                return

            account = self.aggregator._web3.eth.account.from_key(private_key)
            account_address = account.address

            # Confirm transaction details with user
            token_name = "Native Token" if not token_address else token_address
            print(f"\nTransaction Confirmation:")
            print(f"From: {account_address}")
            print(f"To: {to_address}")
            print(f"Amount: {amount} {token_name}")

            confirm = input("Confirm transaction? (y/n): ")
            if confirm.lower() != 'y':
                print("Transaction cancelled")
                return

            # Execute transfer
            tx_hash = self.aggregator.transfer(to_address, amount, token_address)
            print(f"Transaction sent! Transaction hash: {tx_hash}")

        except Exception as e:
            print(f"Transfer failed: {str(e)}")

    def _handle_swap(self, input_list: List[str]):
        """
        Handle the swap command
        Usage: swap <token_in> <token_out> <amount> [slippage]
        """

        if len(input_list) < 3:
            print("Usage: swap <token_in> <token_out> <amount> [slippage]")
            return

        token_in = input_list[0]
        token_out = input_list[1]

        try:
            amount = float(input_list[2])
        except ValueError:
            print("Amount must be a number")
            return

        slippage = 0.5  # Default slippage
        if len(input_list) >= 4:
            try:
                slippage = float(input_list[3])
            except ValueError:
                print("Slippage must be a number")
                return

        # Initialize aggregator
        try:

            # Get account address from private key
            private_key = os.getenv("PRIVATE_KEY")
            if not private_key:
                print("PRIVATE_KEY is not set in environment variables")
                return

            account = self.aggregator._web3.eth.account.from_key(private_key)
            account_address = account.address

            # Get current balance
            current_balance = self.aggregator.get_balance(
                token_address=None if token_in.lower() == self.aggregator.get_native_token_address().lower() else token_in
            )

            # Confirm transaction details with user
            print(f"\nSwap Confirmation:")
            print(f"Account Address: {account_address}")
            print(f"Swap: {amount} {token_in}")
            print(f"Receive: {token_out}")
            print(f"Slippage: {slippage}%")
            print(f"Current Balance: {current_balance} {token_in}")

            confirm = input("Confirm swap? (y/n): ")
            if confirm.lower() != 'y':
                print("Swap cancelled")
                return

            # Execute swap
            result = self.aggregator.swap(token_in, token_out, amount, slippage)
            print(result)

        except Exception as e:
            print(f"Swap failed: {str(e)}")

    def _handle_token_info_by_address(self, input_list: List[str]):
        """
        Handle the token-info command
        Usage: token-info <token_address>
        """
        if not input_list:
            print("Usage: token-info <token_address>")
            return

        token_address = input_list[0]

        try:
            token_info = self.aggregator.get_token_info_by_address(token_address)
            if token_info:
                print("\nToken Information:")
                print(f"Name: {token_info.get('name')}")
                print(f"Symbol: {token_info.get('symbol')}")
                print(f"Address: {token_info.get('address')}")
                print(f"Decimals: {token_info.get('decimals')}")
                print(f"Total Supply: {token_info.get('totalSupply')}")
                print(f"Network: {token_info.get('network')}")
                print(f"Chain ID: {token_info.get('chainId')}")

                # Print additional information if available
                if 'price_usd' in token_info and token_info['price_usd']:
                    print(f"Price (USD): ${token_info['price_usd']:.6f}")
                if 'market_cap' in token_info and token_info['market_cap']:
                    print(f"Market Cap (USD): ${token_info['market_cap']:,.2f}")
                if 'image' in token_info and token_info['image']:
                    print(f"Image URL: {token_info['image']}")
            else:
                print(f"No information found for token address: {token_address}")
        except Exception as e:
            print(f"Error getting token information: {str(e)}")

    def _handle_token_info_by_symbol(self, input_list: List[str]):
        """
        Handle the token-by-symbol command
        Usage: token-by-symbol <symbol>
        """
        if not input_list:
            print("Usage: token-by-symbol <symbol>")
            return

        symbol = input_list[0]

        try:
            token_info = self.aggregator.get_token_info_by_symbol(symbol)
            if token_info:
                print("\nToken Information:")
                print(f"Name: {token_info.get('name')}")
                print(f"Symbol: {token_info.get('symbol')}")
                print(f"Address: {token_info.get('address')}")
                print(f"Decimals: {token_info.get('decimals')}")
                print(f"Total Supply: {token_info.get('totalSupply')}")
                print(f"Network: {token_info.get('network')}")
                print(f"Chain ID: {token_info.get('chainId')}")

                # Print additional information if available
                if 'price_usd' in token_info and token_info['price_usd']:
                    print(f"Price (USD): ${token_info['price_usd']:.6f}")
                if 'market_cap' in token_info and token_info['market_cap']:
                    print(f"Market Cap (USD): ${token_info['market_cap']:,.2f}")
                if 'image' in token_info and token_info['image']:
                    print(f"Image URL: {token_info['image']}")
            else:
                print(f"No token found with symbol: {symbol} on network: {self.aggregator.network}")
        except Exception as e:
            print(f"Error getting token information: {str(e)}")

    def _handle_load_docs(self, input_list: List[str]):
        """Handle the load-docs command"""
        if not self.current_agent:
            print("No agent loaded. Please load an agent first.")
            return

        if len(input_list) < 1:
            print("Usage: load-docs <path> [glob_pattern]")
            print("\nThe path can be either a directory or a specific file.")
            print("\nSupported file types (auto-detected):")
            print("  - Text files (*.txt)")
            print("  - PDF files (*.pdf)")
            print("  - CSV files (*.csv)")
            print("  - JSON files (*.json)")
            print("  - HTML files (*.html, *.htm)")
            print("\nExamples:")
            print("  load-docs /path/to/documents")
            print("  load-docs /path/to/documents \"**/*.txt\"")
            print("  load-docs /path/to/documents \"**/*.{txt,pdf,md}\"")
            print("  load-docs /path/to/specific_file.pdf")
            print("\nIf a directory is provided without a glob pattern, the system will automatically detect and load all supported file types.")
            return

        path = input_list[0]
        glob_pattern = input_list[1] if len(input_list) > 1 else None

        try:
            loader = DocumentLoader()
            print(f"Loading documents from {path}...")
            documents = loader.load_directory(path, glob_pattern)
            print(f"Loaded {len(documents)} document chunks.")

            print("Adding documents to agent...")
            self.current_agent.add_documents(documents)
            print(f"Successfully added {len(documents)} document chunks to agent {self.current_agent.name}.")
            print("You can now ask questions about these documents.")
        except Exception as e:
            print(f"Error loading documents: {e}")

    def _handle_delete_docs(self, input_list: List[str]):
        """Handle the delete-docs command"""
        if not self.current_agent and len(self.agents) == 0:
            ("No agent loaded. Please load an agent first.")
            return

        if len(input_list) >= 1:
            print("Usage: delete-docs <agent_name>")
            return

        if len(input_list) == 1:
            agent_name = input_list[0]
            if agent_name in self.agents:
                self.agents[agent_name].delete_documents()
            else:
                print(f"Agent {agent_name} not found")
        elif len(input_list) == 0:
            self.current_agent.delete_documents()

    async def _handle_telegram_run(self, input_list: List[str]):
        telegram = TelegramClient(self.agents["react"])
        asyncio.create_task(telegram.run())
        print_formatted_text(PromptHTML("<green>Telegram client started</green>"))

    def _handle_list_toolkit_categories(self, input_list: List[str]):
        """List all available toolkit categories"""
        try:
            categories = ToolkitConfig.get_all_categories()
            logger.info("Available toolkit categories:")
            for category in categories:
                tools = ToolkitConfig.get_tools_by_category(category)
                logger.info(f"  {category}: {len(tools)} tools")

            logger.info("\nUse 'list-toolkit-tools <category>' to see tools in a specific category")
            logger.info("Use 'load-toolkit-tools <category1> <category2> ...' to load tools from categories")

        except Exception as e:
            logger.error(f"Failed to list toolkit categories: {e}")

    def _handle_list_toolkit_tools(self, input_list: List[str]):
        """List tools in a specific category"""
        try:
            if len(input_list) < 2:
                logger.error("Usage: list-toolkit-tools <category>")
                logger.info("Available categories: " + ", ".join(ToolkitConfig.get_all_categories()))
                return

            category = input_list[1]
            tools = ToolkitConfig.get_tools_by_category(category)

            if not tools:
                logger.error(f"Unknown category: {category}")
                logger.info("Available categories: " + ", ".join(ToolkitConfig.get_all_categories()))
                return

            logger.info(f"Tools in '{category}' category:")
            for tool in tools:
                logger.info(f"  - {tool}")

            # Show which tools require configuration
            config_tools = ToolkitConfig.get_tools_requiring_config()
            category_config_tools = [tool for tool in tools if tool in config_tools]
            if category_config_tools:
                logger.info(f"\nTools requiring configuration: {', '.join(category_config_tools)}")

        except Exception as e:
            logger.error(f"Failed to list toolkit tools: {e}")

    def _handle_load_toolkit_tools(self, input_list: List[str]):
        """Load toolkit tools from specified categories"""
        try:
            if len(input_list) < 2:
                logger.error("Usage: load-toolkit-tools <category1> [category2] ...")
                logger.info("Available categories: " + ", ".join(ToolkitConfig.get_all_categories()))
                return

            if not self.current_agent:
                logger.error("No agent loaded. Use 'load-agent <name>' first.")
                return

            categories = input_list[1:]
            available_categories = ToolkitConfig.get_all_categories()

            # Validate categories
            invalid_categories = [cat for cat in categories if cat not in available_categories]
            if invalid_categories:
                logger.error(f"Invalid categories: {', '.join(invalid_categories)}")
                logger.info("Available categories: " + ", ".join(available_categories))
                return

            # Load toolkit tools
            tool_manager = get_all_toolkit_tools(categories=categories)
            add_all_toolkit_tools_to_manager(self.current_agent.avaliable_tools, categories=categories)

            logger.info(f"✅ Successfully loaded {len(tool_manager.tools)} toolkit tools from categories: {', '.join(categories)}")

        except Exception as e:
            logger.error(f"Error loading toolkit tools: {e}")

    def _handle_system_info(self, input_list: List[str]):
        """Display comprehensive system information and health checks"""
        import platform
        import sys
        from datetime import datetime

        print_formatted_text(PromptHTML("<ansiblue><b>🔍 SpoonAI System Information</b></ansiblue>"))
        print_formatted_text(PromptHTML("<ansiwhite>═══════════════════════════════════════</ansiwhite>"))

        # System Information
        print_formatted_text(PromptHTML("<ansiyellow><b>📊 System Details:</b></ansiyellow>"))
        print_formatted_text(f"  Platform: {platform.system()} {platform.release()}")
        print_formatted_text(f"  Python Version: {sys.version}")
        print_formatted_text(f"  Architecture: {platform.machine()}")
        print_formatted_text(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()

        # Environment Variables Status
        print_formatted_text(PromptHTML("<ansiyellow><b>🔑 Environment Variables:</b></ansiyellow>"))
        env_vars = [
            ("OPENAI_API_KEY", "OpenAI API"),
            ("ANTHROPIC_API_KEY", "Anthropic API"),
            ("DEEPSEEK_API_KEY", "DeepSeek API"),
            ("TAVILY_API_KEY", "Tavily Search API"),
            ("SECRET_KEY", "JWT Secret Key"),
            ("TELEGRAM_BOT_TOKEN", "Telegram Bot"),
            ("GITHUB_TOKEN", "GitHub API"),
            ("PRIVATE_KEY", "Wallet Private Key"),
            ("RPC_URL", "Blockchain RPC"),
            ("DATABASE_URL", "Database"),
            ("REDIS_HOST", "Redis Host"),
        ]

        for var_name, description in env_vars:
            value = os.getenv(var_name)
            if value:
                # Don't show actual secret values, just indicate they're set
                if "KEY" in var_name or "TOKEN" in var_name or "SECRET" in var_name:
                    masked_value = f"{'*' * min(len(value), 8)}... (length: {len(value)})"
                    status = f"<ansigreen>✓ Set</ansigreen> - {masked_value}"
                else:
                    status = f"<ansigreen>✓ Set</ansigreen> - {value}"
            else:
                status = "<ansired>✗ Not set</ansired>"
            print_formatted_text(PromptHTML(f"  {description:20} {status}"))
        print()

        # Configuration Status
        print_formatted_text(PromptHTML("<ansiyellow><b>⚙️  Configuration Status:</b></ansiyellow>"))
        config_file = Path("config.json")
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    config_data = json.load(f)
                print_formatted_text(PromptHTML("  <ansigreen>✓ config.json found</ansigreen>"))

                # Check API keys in config
                api_keys = config_data.get("api_keys", {})
                if api_keys:
                    print_formatted_text(f"  API Keys in config: {len(api_keys)}")
                    for provider, key in api_keys.items():
                        if key and not self.config_manager._is_placeholder_value(key):
                            print_formatted_text(PromptHTML(f"    <ansigreen>✓ {provider}</ansigreen>"))
                        else:
                            print_formatted_text(PromptHTML(f"    <ansired>✗ {provider} (placeholder/empty)</ansired>"))
                else:
                    print_formatted_text(PromptHTML("  <ansiyellow>! No API keys in config</ansiyellow>"))

            except Exception as e:
                print_formatted_text(PromptHTML(f"  <ansired>✗ Error reading config.json: {e}</ansired>"))
        else:
            print_formatted_text(PromptHTML("  <ansired>✗ config.json not found</ansired>"))
        print()

        # Current Agent Status
        print_formatted_text(PromptHTML("<ansiyellow><b>🤖 Agent Status:</b></ansiyellow>"))
        if self.current_agent:
            print_formatted_text(PromptHTML(f"  <ansigreen>✓ Active agent: {self.current_agent.name}</ansigreen>"))
            print_formatted_text(f"  Agent type: {type(self.current_agent).__name__}")
            if hasattr(self.current_agent, 'avaliable_tools'):
                tool_count = len(self.current_agent.avaliable_tools.tools)
                print_formatted_text(f"  Available tools: {tool_count}")
            if hasattr(self.current_agent, 'llm') and self.current_agent.llm:
                llm_info = f"{getattr(self.current_agent.llm, 'llm_provider', 'unknown')}"
                model_name = getattr(self.current_agent.llm, 'model_name', 'unknown')
                print_formatted_text(f"  LLM Provider: {llm_info}")
                print_formatted_text(f"  Model: {model_name}")
        else:
            print_formatted_text(PromptHTML("  <ansired>✗ No agent loaded</ansired>"))
        print()

        # Available Commands
        print_formatted_text(PromptHTML("<ansiyellow><b>📝 CLI Commands:</b></ansiyellow>"))
        print_formatted_text(f"  Total commands: {len(self.commands)}")
        print_formatted_text("  Categories:")

        # Group commands by category
        categories = {
            "Core": ["help", "exit", "system-info"],
            "Agent": ["load-agent", "list-agents", "action", "reload-config"],
            "Chat": ["new-chat", "list-chats", "load-chat"],
            "Config": ["config"],
            "Crypto": ["transfer", "swap", "token-info", "token-by-symbol"],
            "Tools": ["load-docs", "list-toolkit-categories", "list-toolkit-tools", "load-toolkit-tools"],
            "Social": ["telegram"],
        }

        for category, cmd_list in categories.items():
            available_cmds = [cmd for cmd in cmd_list if cmd in self.commands]
            if available_cmds:
                print_formatted_text(f"    {category}: {len(available_cmds)} commands")
        print()

        # Health Check Summary
        print_formatted_text(PromptHTML("<ansiyellow><b>🏥 Health Check Summary:</b></ansiyellow>"))

        health_score = 0
        total_checks = 5

        # Check 1: At least one API key is set
        has_api_key = any(os.getenv(key) for key in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "DEEPSEEK_API_KEY"])
        if has_api_key:
            health_score += 1
            print_formatted_text(PromptHTML("  <ansigreen>✓ LLM API key configured</ansigreen>"))
        else:
            print_formatted_text(PromptHTML("  <ansired>✗ No LLM API key found</ansired>"))

        # Check 2: SECRET_KEY is set
        if os.getenv("SECRET_KEY"):
            health_score += 1
            print_formatted_text(PromptHTML("  <ansigreen>✓ Security key configured</ansigreen>"))
        else:
            print_formatted_text(PromptHTML("  <ansired>✗ SECRET_KEY not set (security risk)</ansired>"))

        # Check 3: Config file exists
        if config_file.exists():
            health_score += 1
            print_formatted_text(PromptHTML("  <ansigreen>✓ Configuration file present</ansigreen>"))
        else:
            print_formatted_text(PromptHTML("  <ansired>✗ No configuration file</ansired>"))

        # Check 4: Agent is loaded
        if self.current_agent:
            health_score += 1
            print_formatted_text(PromptHTML("  <ansigreen>✓ Agent is loaded and ready</ansigreen>"))
        else:
            print_formatted_text(PromptHTML("  <ansired>✗ No agent loaded</ansired>"))

        # Check 5: Dependencies check (basic)
        try:
            import spoon_ai
            health_score += 1
            print_formatted_text(PromptHTML("  <ansigreen>✓ Core dependencies available</ansigreen>"))
        except ImportError:
            print_formatted_text(PromptHTML("  <ansired>✗ Core dependencies missing</ansired>"))

        # Overall health score
        health_percentage = (health_score / total_checks) * 100
        if health_percentage >= 80:
            health_color = "ansigreen"
            health_status = "Excellent"
        elif health_percentage >= 60:
            health_color = "ansiyellow"
            health_status = "Good"
        else:
            health_color = "ansired"
            health_status = "Poor"

        print()
        print_formatted_text(PromptHTML(f"  <{health_color}><b>Overall Health: {health_status} ({health_score}/{total_checks} checks passed)</b></{health_color}>"))

        if health_score < total_checks:
            print()
            print_formatted_text(PromptHTML("<ansiyellow><b>💡 Recommendations:</b></ansiyellow>"))
            if not has_api_key:
                print_formatted_text("  • Set up at least one LLM API key (OPENAI_API_KEY, ANTHROPIC_API_KEY, or DEEPSEEK_API_KEY)")
            if not os.getenv("SECRET_KEY"):
                print_formatted_text("  • Set SECRET_KEY for security: python -c \"import secrets; print(secrets.token_urlsafe(32))\"")
            if not config_file.exists():
                print_formatted_text("  • Run 'config' command to set up initial configuration")
            if not self.current_agent:
                print_formatted_text("  • Load an agent with 'load-agent <name>' to start using SpoonAI")

        print_formatted_text(PromptHTML("<ansiwhite>═══════════════════════════════════════</ansiwhite>"))
    def _handle_migrate_config(self, input_list: List[str]):
        """Handle configuration migration command"""
        from spoon_ai.config.migrate_config import migrate_config, interactive_migration

        print_formatted_text(PromptHTML("<ansiblue><b>🔄 Configuration Migration</b></ansiblue>"))
        print_formatted_text(PromptHTML("<ansiwhite>═══════════════════════════════════════</ansiwhite>"))

        # Parse arguments
        config_file = "config.json"
        dry_run = False
        interactive = False

        i = 0
        while i < len(input_list):
            arg = input_list[i]
            if arg in ["-f", "--file"]:
                if i + 1 < len(input_list):
                    config_file = input_list[i + 1]
                    i += 1
                else:
                    logger.error("Missing file path after -f/--file")
                    return
            elif arg in ["--dry-run", "-d"]:
                dry_run = True
            elif arg in ["--interactive", "-i"]:
                interactive = True
            elif arg in ["-h", "--help"]:
                self._show_migrate_help()
                return
            i += 1

        try:
            if interactive or not input_list:
                # Run interactive migration
                success = interactive_migration()
            else:
                # Run direct migration
                success = migrate_config(
                    input_file=config_file,
                    output_file=None,
                    backup=True,
                    validate=True,
                    dry_run=dry_run
                )

            if success:
                print_formatted_text(PromptHTML("<ansigreen>✅ Migration completed successfully!</ansigreen>"))
                if not dry_run:
                    print_formatted_text("💡 Tip: Reload your agent to use the new configuration")
            else:
                print_formatted_text(PromptHTML("<ansired>❌ Migration failed. Check the error messages above.</ansired>"))

        except Exception as e:
            logger.error(f"Migration error: {e}")
            print_formatted_text(PromptHTML(f"<ansired>❌ Migration failed: {e}</ansired>"))

    def _handle_check_config(self, input_list: List[str]):
        """Handle configuration check command"""
        print_formatted_text(PromptHTML("<ansiblue><b>🔍 Configuration Check</b></ansiblue>"))
        print_formatted_text(PromptHTML("<ansiwhite>═══════════════════════════════════════</ansiwhite>"))

        config_file = "config.json"
        if input_list and not input_list[0].startswith("-"):
            config_file = input_list[0]

        try:
            if not Path(config_file).exists():
                print_formatted_text(PromptHTML(f"<ansired>❌ Configuration file not found: {config_file}</ansired>"))
                return

            with open(config_file, 'r') as f:
                config_data = json.load(f)

            # Check if migration is needed
            manager = ConfigManager(config_file)
            is_legacy = manager._detect_legacy_config(config_data)

            if is_legacy:
                print_formatted_text(PromptHTML("<ansiyellow>⚠️  Legacy configuration format detected</ansiyellow>"))

                # Show what needs migration
                legacy_sections = []
                if "tool_sets" in config_data:
                    legacy_sections.append(f"tool_sets ({len(config_data['tool_sets'])} items)")
                if "mcp_servers" in config_data:
                    legacy_sections.append(f"mcp_servers ({len(config_data['mcp_servers'])} items)")

                if legacy_sections:
                    print_formatted_text(f"📋 Legacy sections found: {', '.join(legacy_sections)}")

                agents_count = len(config_data.get("agents", {}))
                print_formatted_text(f"👥 Agents to migrate: {agents_count}")

                print_formatted_text(PromptHTML("<ansiyellow>💡 Run 'migrate-config' to upgrade to the new format</ansiyellow>"))
            else:
                print_formatted_text(PromptHTML("<ansigreen>✅ Configuration is already in the new unified format</ansigreen>"))

                # Show current configuration summary
                agents = config_data.get("agents", {})
                if agents:
                    print_formatted_text(f"👥 Agents configured: {len(agents)}")

                    total_tools = 0
                    for agent_name, agent_config in agents.items():
                        tools = agent_config.get("tools", [])
                        total_tools += len(tools)
                        if tools:
                            print_formatted_text(f"  • {agent_name}: {len(tools)} tools")

                    print_formatted_text(f"🔧 Total tools: {total_tools}")
                else:
                    print_formatted_text(PromptHTML("<ansiyellow>⚠️  No agents configured</ansiyellow>"))

        except json.JSONDecodeError as e:
            print_formatted_text(PromptHTML(f"<ansired>❌ Invalid JSON in configuration file: {e}</ansired>"))
        except Exception as e:
            print_formatted_text(PromptHTML(f"<ansired>❌ Error checking configuration: {e}</ansired>"))

    def _handle_validate_config(self, input_list: List[str]):
        """Handle configuration validation command"""
        from spoon_ai.config.migrate_config import validate_environment_variables, check_mcp_server_availability

        print_formatted_text(PromptHTML("<ansiblue><b>✅ Configuration Validation</b></ansiblue>"))
        print_formatted_text(PromptHTML("<ansiwhite>═══════════════════════════════════════</ansiwhite>"))

        config_file = "config.json"
        check_env = "--check-env" in input_list or "-e" in input_list
        check_servers = "--check-servers" in input_list or "-s" in input_list

        if input_list and not input_list[0].startswith("-"):
            config_file = input_list[0]

        try:
            if not Path(config_file).exists():
                print_formatted_text(PromptHTML(f"<ansired>❌ Configuration file not found: {config_file}</ansired>"))
                return

            # Load and validate configuration
            manager = ConfigManager(config_file)
            config = manager.load_config()

            print_formatted_text(PromptHTML("<ansigreen>✅ Configuration loaded successfully</ansigreen>"))

            # Run validation
            issues = manager.validate_configuration()

            if issues:
                print_formatted_text(PromptHTML("<ansired>❌ Configuration validation issues found:</ansired>"))
                for issue in issues:
                    print_formatted_text(f"  • {issue}")
            else:
                print_formatted_text(PromptHTML("<ansigreen>✅ Configuration validation passed</ansigreen>"))

            # Check environment variables if requested
            if check_env or not input_list:
                print_formatted_text(PromptHTML("\n<ansiyellow><b>🔑 Environment Variables Check:</b></ansiyellow>"))

                with open(config_file, 'r') as f:
                    config_data = json.load(f)

                missing_vars = validate_environment_variables(config_data)

                if missing_vars:
                    print_formatted_text(PromptHTML("<ansired>❌ Missing environment variables:</ansired>"))
                    for var in missing_vars:
                        print_formatted_text(f"  • {var}")
                else:
                    print_formatted_text(PromptHTML("<ansigreen>✅ All required environment variables are configured</ansigreen>"))

            # Check MCP server availability if requested
            if check_servers or not input_list:
                print_formatted_text(PromptHTML("\n<ansiyellow><b>🖥️  MCP Server Availability Check:</b></ansiyellow>"))

                with open(config_file, 'r') as f:
                    config_data = json.load(f)

                unavailable = check_mcp_server_availability(config_data)

                if unavailable:
                    print_formatted_text(PromptHTML("<ansired>❌ Unavailable MCP server commands:</ansired>"))
                    for server in unavailable:
                        print_formatted_text(f"  • {server}")
                    print_formatted_text(PromptHTML("\n<ansiyellow>💡 Installation suggestions:</ansiyellow>"))
                    print_formatted_text("  • For npx: npm install -g npm")
                    print_formatted_text("  • For uvx: pip install uv")
                else:
                    print_formatted_text(PromptHTML("<ansigreen>✅ All MCP server commands are available</ansigreen>"))

            # Summary
            print_formatted_text(PromptHTML("\n<ansiyellow><b>📊 Validation Summary:</b></ansiyellow>"))

            total_issues = len(issues)
            if check_env or not input_list:
                total_issues += len(missing_vars) if 'missing_vars' in locals() else 0
            if check_servers or not input_list:
                total_issues += len(unavailable) if 'unavailable' in locals() else 0

            if total_issues == 0:
                print_formatted_text(PromptHTML("<ansigreen>🎉 Configuration is fully valid and ready to use!</ansigreen>"))
            else:
                print_formatted_text(PromptHTML(f"<ansiyellow>⚠️  Found {total_issues} issues that need attention</ansiyellow>"))

        except Exception as e:
            print_formatted_text(PromptHTML(f"<ansired>❌ Validation failed: {e}</ansired>"))

    def _show_migrate_help(self):
        """Show help for migration command"""
        print_formatted_text(PromptHTML("<ansiblue><b>🔄 Configuration Migration Help</b></ansiblue>"))
        print_formatted_text(PromptHTML("<ansiwhite>═══════════════════════════════════════</ansiwhite>"))
        print_formatted_text("Usage:")
        print_formatted_text("  migrate-config                    # Interactive migration")
        print_formatted_text("  migrate-config -f config.json     # Migrate specific file")
        print_formatted_text("  migrate-config --dry-run          # Preview changes without applying")
        print_formatted_text("  migrate-config --interactive      # Force interactive mode")
        print_formatted_text("")
        print_formatted_text("Options:")
        print_formatted_text("  -f, --file FILE     Configuration file to migrate (default: config.json)")
        print_formatted_text("  -d, --dry-run       Show what would be changed without making changes")
        print_formatted_text("  -i, --interactive   Run in interactive mode")
        print_formatted_text("  -h, --help          Show this help message")
        print_formatted_text("")
        print_formatted_text("Examples:")
        print_formatted_text("  migrate-config                           # Interactive migration")
        print_formatted_text("  migrate-config -f my_config.json         # Migrate specific file")
        print_formatted_text("  migrate-config --dry-run                 # Preview migration")