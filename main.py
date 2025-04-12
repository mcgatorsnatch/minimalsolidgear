"""
Main entry point for the SolidLight framework.
Initializes components and provides a simple CLI interface.
"""
import logging
import asyncio
import sys
import signal
import os
import torch
import argparse
import time
from pathlib import Path

from .cli_interface import ExecutionInterface
from .orchestrator import QuantumSynapseOrchestrator

logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="SolidLight Framework - A minimal static model adaptation framework")
    
    # Redis configuration
    parser.add_argument("--redis-host", type=str, default="localhost", 
                      help="Redis server hostname (default: localhost)")
    parser.add_argument("--redis-port", type=int, default=6379, 
                      help="Redis server port (default: 6379)")
    parser.add_argument("--redis-timeout", type=float, default=2.0,
                      help="Redis connection timeout in seconds (default: 2.0)")
    parser.add_argument("--redis-retry", type=int, default=3,
                      help="Redis connection retry attempts (default: 3)")
    
    # Model configuration
    parser.add_argument("--use-gpu", action="store_true", default=True,
                      help="Enable GPU acceleration if available (default: True)")
    parser.add_argument("--model-inference", action="store_true", default=False,
                      help="Enable actual Mistral 7B inference (default: False)")
    parser.add_argument("--embedding-model", type=str, default="all-MiniLM-L6-v2",
                      help="Embedding model to use (default: all-MiniLM-L6-v2)")
    
    # Feature flags
    parser.add_argument("--enhanced-tagging", action="store_true", default=False,
                      help="Enable enhanced NLP-based tagging (requires nltk)")
    parser.add_argument("--preference-analysis", action="store_true", default=False,
                      help="Enable memory-based preference analysis")
    parser.add_argument("--verbose-errors", action="store_true", default=False,
                      help="Show detailed error messages")
    parser.add_argument("--show-examples", action="store_true", default=True,
                      help="Include examples in help command output")
    
    # Security
    parser.add_argument("--encryption-key", type=str, default=None,
                      help="Path to PGP encryption key file")
    parser.add_argument("--alignment-json", type=str, default="alignment.json",
                      help="Path to alignment rules JSON file (default: alignment.json)")
    
    # Performance
    parser.add_argument("--command-timeout", type=int, default=30,
                      help="Command execution timeout in seconds (default: 30)")
    parser.add_argument("--max-chat-history", type=int, default=10,
                      help="Maximum number of chat turns to keep (default: 10)")
    
    return parser.parse_args()

def setup_logging():
    """Configure the logging system"""
    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )

def load_encryption_key(key_path):
    """
    Load and validate PGP encryption key
    
    Args:
        key_path: Path to the encryption key file
        
    Returns:
        Key content if valid, None otherwise
    """
    try:
        if not key_path or not os.path.exists(key_path):
            if key_path:
                logger.error(f"Encryption key file not found: {key_path}")
            return None
            
        # Validate that this is a PGP key file
        with open(key_path, 'r') as f:
            key_content = f.read()
            
        if not key_content.startswith("-----BEGIN PGP") or "KEY" not in key_content:
            logger.error(f"File doesn't appear to be a valid PGP key: {key_path}")
            return None
            
        logger.info(f"Loaded encryption key from {key_path}")
        return key_content
    except Exception as e:
        logger.error(f"Failed to load encryption key: {e}")
        return None

def test_redis_connection(host, port, timeout=2.0, retry_attempts=3, retry_delay=1.0):
    """
    Test the Redis connection with retries
    
    Args:
        host: Redis hostname
        port: Redis port
        timeout: Connection timeout
        retry_attempts: Number of retry attempts
        retry_delay: Delay between retries
        
    Returns:
        True if connection succeeded, False otherwise
    """
    try:
        import redis
        
        for attempt in range(retry_attempts):
            try:
                start_time = time.time()
                redis_client = redis.Redis(
                    host=host, 
                    port=port,
                    socket_timeout=timeout,
                    socket_connect_timeout=timeout
                )
                redis_client.ping()
                elapsed = time.time() - start_time
                logger.info(f"Redis connection test successful at {host}:{port} in {elapsed:.2f}s")
                return True
            except redis.ConnectionError as e:
                if attempt < retry_attempts - 1:
                    logger.warning(f"Redis connection attempt {attempt+1}/{retry_attempts} failed: {e}")
                    time.sleep(retry_delay)
                else:
                    logger.error(f"Redis connection test failed after {retry_attempts} attempts: {e}")
                    return False
            except Exception as e:
                logger.error(f"Unexpected error testing Redis connection: {e}")
                return False
    except ImportError:
        logger.error("Redis package not installed. Please install it with: pip install redis")
        return False
        
    return False

def main():
    """Main entry point for the SolidLight framework"""
    # Configure logging
    setup_logging()
    
    # Parse command-line arguments
    args = parse_args()
    
    # Check for GPU
    use_gpu = args.use_gpu and torch.cuda.is_available()
    logger.info(f"GPU acceleration {'enabled' if use_gpu else 'disabled'}")

    if use_gpu and torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        logger.info(f"GPU detected: {device_name}")
        if any(gpu_type in device_name for gpu_type in ["3090", "4090"]):
            logger.info(f"Optimizing for {device_name} (24 GB VRAM)")
            torch.cuda.set_per_process_memory_fraction(0.9)

    try:
        # Get encryption key if specified
        encryption_key = load_encryption_key(args.encryption_key)
        
        # Check if alignment file exists
        alignment_path = Path(args.alignment_json)
        if not alignment_path.exists():
            logger.warning(f"Alignment file not found: {args.alignment_json}")
            logger.warning("Default alignment rules will be used")
            
        # Check if Redis is available using our custom test function
        if not test_redis_connection(
            args.redis_host, 
            args.redis_port, 
            args.redis_timeout, 
            args.redis_retry
        ):
            logger.error("Redis is required for SolidLight to function properly.")
            logger.error("Please ensure Redis is running, or specify the correct host/port.")
            sys.exit(1)
        
        # Create the main components
        logger.info("Initializing SolidLight components...")
        
        try:
            # Create orchestrator with all parameters
            orchestrator = QuantumSynapseOrchestrator(
                use_gpu=use_gpu,
                encryption_key=encryption_key,
                model_name=args.embedding_model,
                alignment_json_path=args.alignment_json,
                enable_model_inference=args.model_inference,
                redis_host=args.redis_host,
                redis_port=args.redis_port,
                enable_enhanced_tagging=args.enhanced_tagging,
                preference_analysis_enabled=args.preference_analysis,
                max_chat_history=args.max_chat_history
            )
            
            # Create interface
            interface = ExecutionInterface(
                orchestrator=orchestrator,
                verbose_errors=args.verbose_errors,
                command_timeout=args.command_timeout,
                show_examples=args.show_examples
            )
            
            # Verify that memory system and Redis are functioning
            memory_status = "Ready" if orchestrator.memory else "Not initialized"
            logger.info(f"Memory system: {memory_status}")
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            sys.exit(1)

        # Set up signal handlers for graceful shutdown
        def signal_handler(sig, frame):
            logger.info(f"Received signal {sig}, initiating shutdown")
            # Save state before exiting if possible
            try:
                if orchestrator:
                    loop = asyncio.get_event_loop()
                    loop.run_until_complete(orchestrator.save_state())
                    logger.info("Saved state before shutdown")
            except Exception as e:
                logger.error(f"Failed to save state before shutdown: {e}")
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Main command loop
        logger.info("SolidLight Framework ready for commands")
        loop = asyncio.get_event_loop()

        # Display welcome message
        print("\n" + "="*50)
        print("Welcome to SolidLight Framework")
        print("Type 'help' for available commands or 'exit' to quit")
        print("="*50 + "\n")

        # Command loop
        while True:
            try:
                prompt = "> "
                user_input = input(prompt).strip()

                # Handle exit commands
                if not user_input:
                    continue

                if user_input.lower() in ["exit", "quit"]:
                    logger.info("Exit command received")
                    # Save state before exiting
                    try:
                        if orchestrator:
                            print("Saving state before exit...")
                            save_task = asyncio.create_task(orchestrator.save_state())
                            # Add a timeout for the save operation
                            save_timeout = min(args.command_timeout, 10)  # Max 10s for exit save
                            try:
                                result = loop.run_until_complete(asyncio.wait_for(save_task, save_timeout))
                                if result:
                                    print("State saved successfully")
                                else:
                                    print("State could not be saved completely")
                            except asyncio.TimeoutError:
                                print("Save operation timed out during exit")
                                save_task.cancel()
                    except Exception as e:
                        logger.error(f"Failed to save state on exit: {e}")
                        if args.verbose_errors:
                            print(f"Failed to save state: {e}")
                        else:
                            print("Failed to save state on exit")
                    break

                # Execute the command
                try:
                    result = loop.run_until_complete(interface.execute_command(user_input))
                    if 'message' in result:
                        print(result['message'])
                except Exception as e:
                    logger.error(f"Unhandled error in command execution: {e}")
                    if args.verbose_errors:
                        print(f"An unexpected error occurred: {e}")
                    else:
                        print(f"An unexpected error occurred. Please try again.")

            except KeyboardInterrupt:
                logger.info("KeyboardInterrupt received, exiting")
                break
            except EOFError:
                logger.info("EOF received, exiting")
                break
            except Exception as e:
                logger.error(f"Error in command loop: {e}", exc_info=True)
                if args.verbose_errors:
                    print(f"Error: {str(e)}")
                else:
                    print("An error occurred in the command loop. Please try again.")

        print("\nThank you for using SolidLight Framework\n")

    except Exception as e:
        logger.error(f"Critical error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main() 
