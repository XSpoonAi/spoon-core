import asyncio
import datetime
import json
import logging
import os
import shlex
import sys
import traceback
from pathlib import Path
from typing import Callable, Dict, List

from prompt_toolkit import PromptSession, print_formatted_text
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.formatted_text import HTML as PromptHTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style

from spoon_ai.agents import SpoonReactAI, SpoonReactMCP
from spoon_ai.retrieval.document_loader import DocumentLoader
from spoon_ai.schema import Message, Role
from spoon_ai.trade.aggregator import Aggregator
from spoon_ai.utils.config_manager import ConfigManager
from spoon_ai.tools.crypto_tools import get_crypto_tools, add_crypto_tools_to_manager, CryptoToolsConfig

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

# Add keyword filter
keyword_filter = KeywordFilter([
    "telemetry",
    "anonymized",
    "n_results",
    "updating n_results",
    "number of requested results",
    "elements in index"
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
        self.aggregator = Aggregator(rpc_url=os.getenv("RPC_URL"), chain_id=int(os.getenv("CHAIN_ID", 1)), scan_url=os.getenv("SCAN_URL", "https://etherscan.io"))
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
            handler=self.,
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

    def add_command(self, command: SpoonCommand):
        self.commands[command.name] = command

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

    def _handle_load_agent(self, input_list: List[str]):
    if not input_list:
        logger.error("Missing agent name. Usage: load-agent <agent_name>")
        return

    name = input_list[0]
    if name not in ["react", "spoon_react_mcp"]:
        logger.error(f"Invalid agent name: '{name}'. Use 'list-agents' to see available agents.")
        return

    self._load_agent(name)

    def  _load_agent(self, name: str):
        if name == "react":
            self.agents[name] = SpoonReactAI()
            self.current_agent = self.agents[name]
            # Add crypto tools to the agent
            self._add_crypto_tools_to_agent(self.current_agent)
            logger.info(f"Loaded agent: {self.current_agent.name}")
        elif name == "spoon_react_mcp":
            self.agents[name] = SpoonReactMCP()
            self.current_agent = self.agents[name]
            # Add crypto tools to the agent
            self._add_crypto_tools_to_agent(self.current_agent)
            logger.info(f"Loaded agent: {self.current_agent.name}")
        else:
            logger.error(f"Agent {name} not found")

    def _add_crypto_tools_to_agent(self, agent):
        """Add crypto tools to the agent's tool manager."""
        try:
            # Add crypto tools to the agent's tool manager and get the tools for logging
            updated_manager = add_crypto_tools_to_manager(agent.avaliable_tools)

            # Get the crypto tools from the updated manager to avoid duplicate loading
            crypto_tools = [tool for tool in updated_manager.tools if hasattr(tool, 'name') and
                          any(tool.name == crypto_name for crypto_name in [
                              "get_token_price", "get_24h_stats", "get_kline_data",
                              "price_threshold_alert", "lp_range_check", "monitor_sudden_price_increase",
                              "lending_rate_monitor", "crypto_market_monitor", "predict_price", "token_holders",
                              "crypto_powerdata_cex", "crypto_powerdata_dex",
                              "crypto_powerdata_indicators", "crypto_powerdata_price"
                          ])]

            if crypto_tools:
                logger.info(f"Successfully added {len(crypto_tools)} crypto tools to agent")

                # Log available crypto tools
                tool_names = [tool.name for tool in crypto_tools]
                logger.info(f"Available crypto tools: {', '.join(tool_names)}")
            else:
                logger.warning("No crypto tools were loaded")

        except Exception as e:
            logger.error(f"Failed to add crypto tools to agent: {e}")
            logger.debug(f"Crypto tools integration error details: {e}", exc_info=True)

    def _handle_list_agents(self, input_list: List[str]):
        logger.info("Available agents:")
        for agent in self.agents.values():
            logger.info(f"  {agent.name}: {agent.description}")

    def _load_default_agent(self):
        # self._load_agent("spoon_react_mcp")
        self._load_agent("react")

    def _set_prompt_toolkit(self):
        self.style = Style.from_dict({
            'prompt': 'ansicyan bold',
            'command': 'ansigreen',
            'error': 'ansired bold',
            'success': 'ansigreen bold',
            'warning': 'ansiyellow',
        })

        self.completer = WordCompleter(
            list(self.commands.keys()),
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
        self._load_default_agent()
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

    def _handle_reload_config(self, input_list: List[str]):
        """Reload configuration"""
        if not self.current_agent:
            logger.info("No agent loaded, please load an agent first")
            return

        self.current_agent.reload_config()
        logger.info(f"Reloaded configuration for agent '{self.current_agent.name}'")

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
