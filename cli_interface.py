"""
Command Line Interface for the SolidLight framework.
Minimal implementation with basic command processing and error handling.
"""
import logging
import asyncio
import sys
import signal
import json
import time
import contextlib
from typing import Dict, List, Optional, Any, Set, Tuple

from .orchestrator import QuantumSynapseOrchestrator

logger = logging.getLogger(__name__)

class ExecutionInterface:
    """
    Minimal CLI interface for the SolidLight framework that processes commands
    and manages interaction with the orchestrator.
    """
    def __init__(self, 
                 orchestrator: Optional[QuantumSynapseOrchestrator] = None,
                 verbose_errors: bool = False,
                 command_timeout: int = 30,
                 show_examples: bool = True):
        """
        Initialize the execution interface
        
        Args:
            orchestrator: Optional pre-configured orchestrator instance
            verbose_errors: Show detailed error messages
            command_timeout: Timeout for command execution in seconds
            show_examples: Include examples in help command output
        """
        # Create orchestrator if not provided
        self.orchestrator = orchestrator
        self.verbose_errors = verbose_errors
        self.command_timeout = command_timeout
        self.show_examples = show_examples
        self.last_error = None
        self.recent_commands = []  # Track recent commands for contextual examples
        
        # Define command handlers
        self.command_map = {
            "chat": self._handle_chat,
            "search": self._handle_search,
            "save": self._handle_save,
            "help": self._handle_help,
        }
        
        logger.info("ExecutionInterface initialized")

    async def execute_command(self, command: str) -> Dict[str, str]:
        """
        Parse and execute a command from the user
        
        Args:
            command: The user's command string
            
        Returns:
            Dictionary with status and response message
        """
        if not command:
            return {'status': 'error', 'message': 'Empty command'}

        start_time = time.time()
        command_name = command.split()[0] if command.split() else "unknown"
            
        # Set default return value for timeout cases
        result = {
            'status': 'error', 
            'message': f'Command timed out after {self.command_timeout} seconds'
        }
        
        try:
            # Parse command with error handling
            try:
                command_tokens = command.strip().split()
                cmd_name = command_tokens[0].lower() if command_tokens else ""
                args = " ".join(command_tokens[1:]) if len(command_tokens) > 1 else ""
            except Exception as e:
                logger.error(f"Error parsing command: {e}")
                return {
                    'status': 'error',
                    'message': f"Invalid command format: {str(e) if self.verbose_errors else 'Could not parse command'}"
                }

            # Check if orchestrator is initialized
            if not self.orchestrator and cmd_name != "help":
                return {
                    'status': 'error',
                    'message': 'System not initialized. Please restart or contact support.'
                }

            # Execute the command with timeout
            try:
                # Use asyncio.wait_for to implement timeout
                if cmd_name in self.command_map:
                    result = await asyncio.wait_for(
                        self.command_map[cmd_name](args),
                        timeout=self.command_timeout
                    )
                    # Store command in recent commands if successful
                    if result.get('status') == 'success' and cmd_name not in ['help', 'save']:
                        self.recent_commands.append((cmd_name, args))
                        # Keep only last 5 commands
                        self.recent_commands = self.recent_commands[-5:]
                else:
                    # Default to treating unknown commands as chat input
                    logger.info(f"Unknown command '{cmd_name}', treating as chat input.")
                    result = await asyncio.wait_for(
                        self._handle_chat(command),
                        timeout=self.command_timeout
                    )
                    # Store as chat command
                    self.recent_commands.append(('chat', command))
                    self.recent_commands = self.recent_commands[-5:]
            except asyncio.TimeoutError:
                elapsed_time = time.time() - start_time
                logger.error(f"Command '{cmd_name}' timed out after {elapsed_time:.2f} seconds (limit: {self.command_timeout}s)")
                return {
                    'status': 'error',
                    'message': f"Command '{cmd_name}' timed out after {elapsed_time:.2f} seconds. Please try again with a simpler request."
                }
                
        except Exception as e:
            self.last_error = e
            error_msg = str(e) if self.verbose_errors else "An error occurred executing the command"
            logger.error(f"Error executing command: {e}", exc_info=True)
            return {
                'status': 'error',
                'message': f"Error: {error_msg}"
            }
        
        # Log command execution time
        elapsed_time = time.time() - start_time
        logger.info(f"Command '{command_name}' executed in {elapsed_time:.2f} seconds")
            
        return result

    async def _handle_chat(self, args: str) -> Dict[str, str]:
        """
        Handle the chat command
        
        Args:
            args: The user's message
            
        Returns:
            Dictionary with status and response
        """
        if not args.strip():
            return {
                'status': 'error',
                'message': 'Please provide a message to chat with.'
            }
        
        if not self.orchestrator:
            return {
                'status': 'error',
                'message': 'Orchestrator not initialized'
            }
            
        start_time = time.time()
        try:
            response = await self.orchestrator.chat(args)
            elapsed_time = time.time() - start_time
            if elapsed_time > (self.command_timeout * 0.8):  # If used more than 80% of timeout
                logger.warning(f"Chat command took {elapsed_time:.2f}s, close to timeout of {self.command_timeout}s")
                
            return {
                'status': 'success',
                'message': response
            }
        except Exception as e:
            logger.error(f"Error in chat: {e}")
            return {
                'status': 'error',
                'message': f"Failed to process chat: {str(e) if self.verbose_errors else 'An error occurred'}"
            }

    async def _handle_search(self, args: str) -> Dict[str, str]:
        """
        Handle the search command
        
        Args:
            args: Search query and optional parameters
            
        Returns:
            Dictionary with search results
        """
        if not args.strip():
            return {
                'status': 'error',
                'message': 'Please provide a search query.'
            }
            
        if not self.orchestrator:
            return {
                'status': 'error',
                'message': 'Orchestrator not initialized'
            }
            
        # Extract top_k parameter if provided
        top_k = 5
        query = args
        try:
            if "--top_k" in args:
                parts = args.split()
                for i, part in enumerate(parts):
                    if part == "--top_k" and i + 1 < len(parts):
                        try:
                            top_k = int(parts[i + 1])
                            if top_k < 1 or top_k > 100:
                                return {
                                    'status': 'error',
                                    'message': "Invalid top_k value. Must be between 1 and 100."
                                }
                            query = " ".join([p for p in parts if p != "--top_k" and p != parts[i + 1]])
                        except ValueError:
                            return {
                                'status': 'error',
                                'message': "Invalid top_k value. Must be an integer."
                            }
        except Exception as e:
            logger.error(f"Error parsing search parameters: {e}")
            return {
                'status': 'error',
                'message': f"Error parsing search parameters: {str(e) if self.verbose_errors else 'Invalid parameters'}"
            }

        # Perform semantic search
        start_time = time.time()
        try:
            with contextlib.AsyncExitStack() as stack:
                # Create a task for semantic search with timeout
                search_task = asyncio.create_task(self.orchestrator.memory.semantic_search(query, top_k=top_k))
                # Wait for the task to complete with timeout
                results = await asyncio.wait_for(search_task, timeout=self.command_timeout * 0.9)  # 90% of total timeout
        except asyncio.TimeoutError:
            # Cancel the task if it times out
            search_task.cancel()
            elapsed = time.time() - start_time
            logger.error(f"Search operation timed out after {elapsed:.2f}s")
            return {
                'status': 'error',
                'message': f"Search operation timed out. Try with a simpler query or fewer results (--top_k)."
            }
        except Exception as e:
            logger.error(f"Error during search: {e}")
            return {
                'status': 'error',
                'message': f"Search failed: {str(e) if self.verbose_errors else 'An error occurred'}"
            }
        
        # Format results
        try:
            if results:
                result_text = f"Search results for: {query}\n\n"
                for i, result in enumerate(results):
                    result_text += f"{i+1}. {result.get('text', 'No text')[:100]}... "
                    result_text += f"(Score: {result.get('score', 0):.2f})\n\n"
            else:
                result_text = f"No results found for: {query}"

            return {
                'status': 'success',
                'message': result_text,
                'results': results
            }
        except Exception as e:
            logger.error(f"Error formatting search results: {e}")
            return {
                'status': 'error',
                'message': "Error formatting search results"
            }

    async def _handle_save(self, args: str) -> Dict[str, str]:
        """
        Handle the save command
        
        Args:
            args: Unused arguments
            
        Returns:
            Dictionary with status and message
        """
        if not self.orchestrator:
            return {
                'status': 'error',
                'message': 'Orchestrator not initialized'
            }
            
        start_time = time.time()
        try:
            save_task = asyncio.create_task(self.orchestrator.save_state())
            success = await asyncio.wait_for(save_task, timeout=self.command_timeout * 0.9)
            elapsed_time = time.time() - start_time
            
            return {
                'status': 'success' if success else 'error',
                'message': f"Save complete in {elapsed_time:.2f}s" if success else "Save failed"
            }
        except asyncio.TimeoutError:
            logger.error(f"Save operation timed out after {self.command_timeout * 0.9}s")
            return {
                'status': 'error',
                'message': f"Save operation timed out. The system may be under heavy load."
            }
        except Exception as e:
            logger.error(f"Error saving state: {e}")
            return {
                'status': 'error',
                'message': f"Save failed: {str(e) if self.verbose_errors else 'An error occurred'}"
            }

    async def _handle_help(self, args: str) -> Dict[str, str]:
        """
        Handle the help command
        
        Args:
            args: Optional command to get help for
            
        Returns:
            Dictionary with help text
        """
        try:
            help_text = "SolidLight Framework Commands:\n\n"
            
            # Get contextual examples from recent commands
            contextual_examples = self._get_contextual_examples()
            
            if args.strip():
                # Get help for specific command
                cmd = args.strip().lower()
                if cmd == "chat":
                    help_text = "chat <message> - Chat with the assistant\n\n"
                    if self.show_examples:
                        help_text += "Examples:\n"
                        # Add contextual examples first if available
                        chat_examples = [ex for cmd, ex in contextual_examples if cmd == 'chat']
                        if chat_examples:
                            for example in chat_examples[:2]:  # Show at most 2 contextual examples
                                help_text += f"  chat {example}\n"
                        else:
                            help_text += "  chat What is Redis?\n"
                            help_text += "  chat Tell me about vector databases\n"
                elif cmd == "search":
                    help_text = "search <query> [--top_k N] - Search for information\n\n"
                    help_text += "Parameters:\n"
                    help_text += "  --top_k N : Return N results (default: 5, max: 100)\n\n"
                    if self.show_examples:
                        help_text += "Examples:\n"
                        # Add contextual examples first if available
                        search_examples = [ex for cmd, ex in contextual_examples if cmd == 'search']
                        if search_examples:
                            for example in search_examples[:2]:  # Show at most 2 contextual examples
                                help_text += f"  search {example}\n"
                        else:
                            help_text += "  search Redis vector database\n"
                            help_text += "  search Python programming --top_k 10\n"
                elif cmd == "save":
                    help_text = "save - Save current state to Redis\n\n"
                    help_text += "This command persists the current context, preferences, and memory connections.\n"
                    help_text += "It is automatically called when you exit the application.\n"
                    if self.show_examples:
                        help_text += "Example:\n"
                        help_text += "  save\n"
                elif cmd == "help":
                    help_text = "help [command] - Display help information\n\n"
                    help_text += "Get general help or specific help for a command.\n"
                    if self.show_examples:
                        help_text += "Examples:\n"
                        help_text += "  help\n"
                        help_text += "  help search\n"
                else:
                    help_text = f"Unknown command: {cmd}\n\n"
                    help_text += "Type 'help' to see available commands."
            else:
                # General help
                help_text += "chat <message> - Chat with the assistant\n"
                help_text += "search <query> [--top_k N] - Search for information\n"
                help_text += "save - Save current state\n"
                help_text += "help [command] - Display this help message or help for a specific command\n\n"
                
                if self.show_examples:
                    help_text += "Examples:\n"
                    
                    # Show contextual examples if available
                    if contextual_examples:
                        help_text += "Recent Commands:\n"
                        for cmd_type, example in contextual_examples[:3]:  # Show top 3 recent commands
                            help_text += f"  {cmd_type} {example}\n"
                        help_text += "\n"
                    
                    # Always show these standard examples
                    help_text += "Standard Commands:\n"
                    help_text += "  chat What is Redis?\n"
                    help_text += "  search Python programming --top_k 10\n"
                    help_text += "  save\n"
                    help_text += "  help search\n\n"
                
                help_text += "For more detailed help, type 'help <command>'"
        except Exception as e:
            logger.error(f"Error generating help: {e}")
            help_text = "Error displaying help information. Please try again."
            
        return {
            'status': 'success',
            'message': help_text
        }

    def _get_contextual_examples(self) -> List[Tuple[str, str]]:
        """
        Get contextual examples based on recent user commands
        
        Returns:
            List of (command_type, argument) tuples
        """
        # Filter out commands that were just "help" or "save" with no arguments
        return [(cmd, arg) for cmd, arg in self.recent_commands 
                if arg and cmd in ['chat', 'search']] 
