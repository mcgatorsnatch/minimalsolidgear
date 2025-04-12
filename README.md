# SolidLight Framework
[(recursive_alignment.md)](https://github.com/mcgatorsnatch/minimalsolidgear/blob/main/recursive_alignment.md)
A lightweight implementation inspired by the Solid Framework, focusing on memory persistence, alignment checking, and adaptability.

## Features

- **Persistent Memory System**: Redis-backed vector memory with semantic search
- **Episodic Memory Buffer**: Temporary storage for recent interactions
- **Automatic Tagging**: Basic keyword extraction or enhanced NLP-based tagging
- **Zettelkasten Organization**: Memory items are interconnected through tags
- **Dual-Time Scale Adaptation**: Fast and slow adaptation cycles
- **Alignment Checking**: Content filtering based on rules
- **Command-Line Interface**: Simple, intuitive command set
- **Optional Mistral 7B Integration**: For local inference capabilities

## Requirements

- Redis 6.2+ with vector search support (RediSearch module)
- Python 3.9+
- Optional: CUDA-capable GPU with 24+ GB VRAM for model inference

## Installation

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Ensure Redis with RediSearch is running:
   ```
   docker run -p 6379:6379 -d redis/redis-stack
   ```

## Usage

### Command Line Interface

Run the framework with default settings:
```
python -m minimalsolid.main
```

With custom configuration:
```
python -m minimalsolid.main --redis-host localhost --redis-port 6379 --enhanced-tagging --model-inference
```

### Available Commands

- `chat <message>`: Chat with the framework
- `search <query> [--top_k <number>]`: Search memory for relevant items
- `save`: Save current state
- `help [command]`: Display help for commands
- `exit`: Exit the framework

### Command Line Options

| Option | Description |
|--------|-------------|
| `--redis-host` | Redis server hostname (default: localhost) |
| `--redis-port` | Redis server port (default: 6379) |
| `--use-gpu` | Use GPU if available (default: True) |
| `--model-inference` | Enable actual Mistral 7B inference (default: False) |
| `--embedding-model` | Embedding model to use (default: all-MiniLM-L6-v2) |
| `--enhanced-tagging` | Use enhanced NLP-based tagging (default: False) |
| `--preference-analysis` | Enable memory-based preference analysis (default: False) |
| `--verbose-errors` | Show detailed error messages (default: False) |
| `--show-examples` | Show examples in help text (default: False) |
| `--encryption-key` | Path to PGP encryption key (default: None) |
| `--alignment-json` | Path to alignment rules JSON (default: alignment.json) |
| `--command-timeout` | Command execution timeout in seconds (default: 30) |
| `--max-chat-history` | Maximum number of chat history items (default: 10) |

## Programmatic Usage

```python
import asyncio
from minimalsolid import QuantumMemory, QuantumSynapseOrchestrator

async def main():
    # Initialize memory system
    memory = QuantumMemory(
        use_gpu=True,
        redis_host="localhost",
        redis_port=6379,
        enable_enhanced_tagging=True
    )
    
    # Initialize orchestrator
    orchestrator = QuantumSynapseOrchestrator(
        use_gpu=True,
        redis_host="localhost",
        redis_port=6379,
        enable_enhanced_tagging=True,
        enable_model_inference=False
    )
    
    # Chat with orchestrator
    response = await orchestrator.chat("What is the best programming language for AI development?")
    print(f"Response: {response}")
    
    # Save state
    await orchestrator.save_state()

if __name__ == "__main__":
    asyncio.run(main())
```

## Error Handling

The framework implements comprehensive error handling throughout:

- Redis connection issues are handled with retries
- Memory operations have proper error handling and fallbacks
- Command-line interface provides clear error messages
- Timeout handling for long-running operations
- Graceful shutdown with state persistence

## Feature Flags

The framework uses feature flags to enable/disable optional functionality:

- `enable_enhanced_tagging`: Use NLP-based tagging (requires NLTK)
- `enable_model_inference`: Use actual Mistral 7B model (requires transformers)
- `preference_analysis_enabled`: Enable memory-based user preference analysis
- `verbose_errors`: Show detailed error messages in the CLI
- `show_examples`: Show examples in help text

## Development

### Testing

Run tests with:
```
pytest tests/
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## License

MIT License 
