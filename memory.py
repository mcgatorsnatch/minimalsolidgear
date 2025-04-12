"""
Minimal Redis-based memory system for the SolidLight framework.
Provides semantic search, episodic buffer, and memory consolidation.
"""
import redis
import numpy as np
import time
import json
import torch
import pgpy
import os
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from redis.commands.search.field import VectorField, TagField, NumericField, TextField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query
from typing import List, Dict, Optional, Any, Tuple, Union, Set
import logging
import asyncio

# Try to import optional NLP components for enhanced tagging
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    ENHANCED_TAGGING_AVAILABLE = True
except ImportError:
    ENHANCED_TAGGING_AVAILABLE = False

# Add transformer tokenizer import for accurate token counting
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

class QuantumMemory:
    """
    Manages memory storage, retrieval, and semantic search with Redis persistence.
    Implements a minimal episodic buffer for recent interactions with relevance-based
    transfer to long-term memory, and simplified Zettelkasten tagging for interconnections.
    """
    def __init__(self, 
                 model_name: str = 'all-MiniLM-L6-v2', 
                 use_gpu: bool = True, 
                 encryption_key: Optional[str] = None,
                 redis_host: str = 'localhost',
                 redis_port: int = 6379,
                 enable_enhanced_tagging: bool = False,
                 max_reconnect_attempts: int = 3,
                 reconnect_delay: int = 2,
                 max_retry_queue_size: int = 50):
        """
        Initialize the memory system with Redis vector storage.
        
        Args:
            model_name: Name of the embedding model to use
            use_gpu: Whether to use GPU acceleration if available
            encryption_key: Key for encrypting memory entries with PGP
            redis_host: Redis server hostname
            redis_port: Redis server port
            enable_enhanced_tagging: Use NLP techniques for better tagging (requires nltk)
            max_reconnect_attempts: Maximum Redis reconnection attempts
            reconnect_delay: Seconds to wait between reconnection attempts
            max_retry_queue_size: Maximum size of the operation retry queue
        """
        self.model_name = model_name
        self.use_gpu = use_gpu
        self.encryption_key = encryption_key
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.max_reconnect_attempts = max_reconnect_attempts
        self.reconnect_delay = reconnect_delay
        self.enable_enhanced_tagging = enable_enhanced_tagging and ENHANCED_TAGGING_AVAILABLE
        self.max_retry_queue_size = max_retry_queue_size

        # Memory storage tracking - initialized after Redis connection
        self.max_episodic_buffer_size = 10
        self.max_buffer_tokens = 128
        
        # Memory tracking variables (now maintained in Redis)
        self.next_id = 0  # Will be read from Redis
        self.connection_weights = {}  # For static co-occurrence counts

        # Failed operations queue for retry
        self.retry_queue = []
        self.is_retrying = False

        # Initialize tokenizer for accurate token counting
        try:
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
            logger.info("Initialized GPT-2 tokenizer for accurate token counting")
        except Exception as e:
            logger.warning(f"Failed to initialize tokenizer: {e}. Falling back to basic tokenization.")
            self.tokenizer = None

        # Initialize NLTK resources if enhanced tagging is enabled
        if self.enable_enhanced_tagging:
            try:
                # Check if the nltk resources already exist before attempting to download
                nltk_data_path = nltk.data.path[0]
                if not os.path.exists(os.path.join(nltk_data_path, 'tokenizers', 'punkt')):
                    logger.info("Downloading NLTK punkt tokenizer...")
                    nltk.download('punkt', quiet=True)
                if not os.path.exists(os.path.join(nltk_data_path, 'corpora', 'stopwords')):
                    logger.info("Downloading NLTK stopwords...")
                    nltk.download('stopwords', quiet=True)
                logger.info("Enhanced tagging enabled with NLTK resources")
            except Exception as e:
                logger.warning(f"Failed to download NLTK resources: {e}. Falling back to basic tagging.")
                self.enable_enhanced_tagging = False

        # Initialize Redis client with connection retry
        self._initialize_redis_client()
        
        # Initialize embedding model
        try:
            self.embed_model = SentenceTransformer(model_name)
            if use_gpu and torch.cuda.is_available():
                self.embed_model = self.embed_model.to('cuda')
            self.device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
            logger.info(f"QuantumMemory initialized with {model_name} on {self.device}, using Redis vector storage")
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            raise
            
        # Load connection weights from Redis if they exist
        self._load_connection_weights()
        
        # Ensure encryption consistency with a short delay to allow async operations
        # This is done in a separate task to avoid blocking initialization
        asyncio.create_task(self._ensure_encryption_on_init())

    def _initialize_redis_client(self) -> None:
        """Initialize Redis client with connection retry"""
        for attempt in range(self.max_reconnect_attempts):
            try:
                self.redis_client = redis.Redis(
                    host=self.redis_host, 
                    port=self.redis_port, 
                    decode_responses=True,
                    socket_timeout=5.0,
                    socket_connect_timeout=5.0
                )
                # Test connection
                self.redis_client.ping()
                self.index_name = "memory_index"
                self.embedding_dim = 384  # For 'all-MiniLM-L6-v2'
                self._init_redis_index()
                
                # Initialize memory tracking data structures in Redis
                self._init_memory_tracking()
                
                logger.info(f"Successfully connected to Redis at {self.redis_host}:{self.redis_port}")
                return
            except redis.ConnectionError as e:
                logger.warning(f"Redis connection attempt {attempt+1}/{self.max_reconnect_attempts} failed: {e}")
                if attempt < self.max_reconnect_attempts - 1:
                    time.sleep(self.reconnect_delay)
                else:
                    logger.error(f"Failed to connect to Redis after {self.max_reconnect_attempts} attempts")
                    raise

    def _init_redis_index(self) -> None:
        """Initialize the Redis vector search index for memory storage"""
        try:
            # Define schema with essential fields that need to be searchable
            # Note: When using encryption, 'embedding' field is still stored separately
            # for vector search to work, while other fields are in the encrypted 'data' blob
            schema = (
                VectorField("embedding", "HNSW", {
                    "TYPE": "FLOAT32",
                    "DIM": self.embedding_dim,
                    "DISTANCE_METRIC": "COSINE"
                }),
                # Fields below may be empty when using encryption, but kept for schema consistency
                TextField("text", weight=5.0),
                TagField("tags", separator=","),
                NumericField("timestamp"),
                NumericField("score"),
                TextField("task"),
                TagField("links", separator=",")
            )
            
            # Create or update the search index
            try:
                self.redis_client.ft(self.index_name).create_index(
                    schema,
                    definition=IndexDefinition(prefix=["memory:"], index_type=IndexType.HASH)
                )
                logger.info(f"Created Redis index: {self.index_name}")
            except redis.ResponseError as e:
                if "Index already exists" not in str(e):
                    logger.error(f"Failed to create Redis index: {e}")
                    raise
                logger.info(f"Redis index {self.index_name} already exists")
                
            # Check if we need to retroactively ensure vector search consistency
            if self.encryption_key:
                # Sample a few memory keys to check if they have both 'data' and 'embedding' fields
                memory_keys = self.redis_client.keys("memory:*")[:5]
                for key in memory_keys:
                    if not self.redis_client.hexists(key, "embedding"):
                        logger.warning(f"Memory {key} missing embedding field, vector search may not work correctly")
                
        except Exception as e:
            logger.error(f"Unexpected error creating Redis index: {e}")
            raise

    def _init_memory_tracking(self) -> None:
        """Initialize memory tracking data structures in Redis if they don't already exist"""
        try:
            # Initialize next_id counter if it doesn't exist
            if not self.redis_client.exists("memory:next_id"):
                self.redis_client.set("memory:next_id", 0)
            
            # Load next_id from Redis
            self.next_id = int(self.redis_client.get("memory:next_id") or 0)
            
            # Initialize episodic buffer from Redis
            self._load_episodic_buffer()
            
            logger.info(f"Initialized memory tracking. Next memory ID: {self.next_id}")
        except Exception as e:
            logger.error(f"Error initializing memory tracking: {e}")
            raise
            
    def _load_episodic_buffer(self) -> None:
        """Load episodic buffer state from Redis"""
        try:
            # Initialize the episodic buffer and token count
            self.episodic_buffer = []
            self.episodic_buffer_token_count = 0
            
            # Check if Redis has saved episodic buffer entries
            buffer_size = self.redis_client.llen("memory:episodic_buffer")
            
            if buffer_size > 0:
                # Check if the buffer is encrypted
                is_encrypted = self.redis_client.get("memory:episodic_buffer:encrypted") == "1"
                
                # Get all entries from Redis list
                buffer_entries = self.redis_client.lrange("memory:episodic_buffer", 0, -1)
                
                # Handle decryption if needed
                if is_encrypted and self.encryption_key:
                    try:
                        # Decrypt entries
                        decrypted_entries = []
                        for encrypted_entry in buffer_entries:
                            if encrypted_entry.startswith("-----BEGIN PGP MESSAGE-----"):
                                enc_message = pgpy.PGPMessage.from_blob(encrypted_entry)
                                key, _ = pgpy.PGPKey.from_blob(self.encryption_key)
                                decrypted_message = key.decrypt(enc_message)
                                decrypted_entries.append(decrypted_message.message)
                            else:
                                # Handle entries that might not be encrypted
                                logger.warning("Found unencrypted entry in encrypted buffer")
                                decrypted_entries.append(encrypted_entry)
                        buffer_entries = decrypted_entries
                    except Exception as e:
                        logger.error(f"Error decrypting episodic buffer: {e}. Buffer may be inaccessible.")
                        buffer_entries = []
                
                # Add each entry to the local buffer and count tokens
                for entry in buffer_entries:
                    self.episodic_buffer.append(entry)
                    
                    # Count tokens using the proper tokenizer
                    if self.tokenizer:
                        tokens = len(self.tokenizer.encode(entry))
                    else:
                        tokens = len(entry.split())
                        
                    self.episodic_buffer_token_count += tokens
                
                # Get the token count directly from Redis (as a backup)
                stored_token_count = self.redis_client.get("memory:episodic_buffer_tokens")
                if stored_token_count is not None:
                    # If there's a significant difference, use the stored count
                    stored_count = int(stored_token_count)
                    if abs(stored_count - self.episodic_buffer_token_count) > 10:
                        logger.warning(f"Token count discrepancy: calculated {self.episodic_buffer_token_count} vs stored {stored_count}")
                        self.episodic_buffer_token_count = stored_count
                
                logger.info(f"Loaded episodic buffer from Redis: {len(self.episodic_buffer)} entries, {self.episodic_buffer_token_count} tokens")
            else:
                logger.info("No saved episodic buffer entries found in Redis")
                
        except Exception as e:
            logger.error(f"Error loading episodic buffer from Redis: {e}")
            # Initialize with empty buffer
            self.episodic_buffer = []
            self.episodic_buffer_token_count = 0
            
    def _save_episodic_buffer(self) -> None:
        """Save episodic buffer state to Redis"""
        try:
            # Clear existing buffer in Redis
            self.redis_client.delete("memory:episodic_buffer")
            
            # Apply encryption if needed
            if self.encryption_key and self.episodic_buffer:
                try:
                    # Encrypt the buffer entries
                    encrypted_entries = []
                    for entry in self.episodic_buffer:
                        message = pgpy.PGPMessage.new(entry)
                        key, _ = pgpy.PGPKey.from_blob(self.encryption_key)
                        encrypted_data = str(key.encrypt(message))
                        encrypted_entries.append(encrypted_data)
                    
                    # Save encrypted entries
                    if encrypted_entries:
                        self.redis_client.rpush("memory:episodic_buffer", *encrypted_entries)
                    # Mark that the buffer is encrypted
                    self.redis_client.set("memory:episodic_buffer:encrypted", "1")
                except Exception as e:
                    logger.error(f"Error encrypting episodic buffer: {e}. Falling back to unencrypted storage.")
                    # Fallback to unencrypted if encryption fails
                    if self.episodic_buffer:
                        self.redis_client.rpush("memory:episodic_buffer", *self.episodic_buffer)
                    self.redis_client.set("memory:episodic_buffer:encrypted", "0")
            else:
                # Save entries to Redis list without encryption
                if self.episodic_buffer:
                    self.redis_client.rpush("memory:episodic_buffer", *self.episodic_buffer)
                self.redis_client.set("memory:episodic_buffer:encrypted", "0")
            
            # Save token count
            self.redis_client.set("memory:episodic_buffer_tokens", self.episodic_buffer_token_count)
            
            logger.debug(f"Saved episodic buffer to Redis: {len(self.episodic_buffer)} entries, {self.episodic_buffer_token_count} tokens")
        except Exception as e:
            logger.error(f"Error saving episodic buffer to Redis: {e}")

    def _get_memory_ids(self) -> List[int]:
        """Get list of all memory IDs from Redis"""
        try:
            # Get all memory IDs from Redis sorted set
            id_strings = self.redis_client.smembers("memory:ids")
            if not id_strings:
                return []
            
            # Convert to integers and sort
            return sorted([int(id_str) for id_str in id_strings])
        except Exception as e:
            logger.error(f"Error getting memory IDs: {e}")
            return []

    def _add_memory_id(self, memory_id: int) -> None:
        """Add a memory ID to the tracking set in Redis"""
        try:
            self.redis_client.sadd("memory:ids", memory_id)
            self.redis_client.set("memory:next_id", memory_id + 1)
            self.next_id = memory_id + 1
        except Exception as e:
            logger.error(f"Error adding memory ID {memory_id} to tracking: {e}")

    def _get_memory_tags(self, memory_id: int) -> Set[str]:
        """Get tags for a specific memory from Redis"""
        try:
            tags_str = self.redis_client.smembers(f"memory:{memory_id}:tags")
            return set(tags_str)
        except Exception as e:
            logger.error(f"Error getting tags for memory {memory_id}: {e}")
            return set()

    def _set_memory_tags(self, memory_id: int, tags: List[str]) -> None:
        """Set tags for a specific memory in Redis"""
        try:
            key = f"memory:{memory_id}:tags"
            # Clear existing tags
            self.redis_client.delete(key)
            # Add new tags if any
            if tags:
                self.redis_client.sadd(key, *tags)
        except Exception as e:
            logger.error(f"Error setting tags for memory {memory_id}: {e}")

    def _add_tag_to_memory(self, tag: str, memory_id: int) -> None:
        """Associate a tag with a memory in Redis"""
        try:
            # Add tag to memory's tag set
            self.redis_client.sadd(f"memory:{memory_id}:tags", tag)
            # Add memory to tag's memory set
            self.redis_client.sadd(f"tag:{tag}:memories", memory_id)
        except Exception as e:
            logger.error(f"Error adding tag {tag} to memory {memory_id}: {e}")

    def _get_memories_by_tag(self, tag: str) -> Set[int]:
        """Get all memory IDs associated with a specific tag"""
        try:
            memory_ids = self.redis_client.smembers(f"tag:{tag}:memories")
            return {int(mid) for mid in memory_ids}
        except Exception as e:
            logger.error(f"Error getting memories for tag {tag}: {e}")
            return set()

    def _get_all_tags(self) -> Set[str]:
        """Get all unique tags in the system"""
        try:
            # Scan for all tag:*:memories keys
            cursor = 0
            all_tags = set()
            
            while True:
                cursor, keys = self.redis_client.scan(cursor, match="tag:*:memories")
                # Extract tag names from keys (format is tag:{tagname}:memories)
                for key in keys:
                    parts = key.split(":")
                    if len(parts) == 3:
                        all_tags.add(parts[1])
                
                if cursor == 0:
                    break
                    
            return all_tags
        except Exception as e:
            logger.error(f"Error getting all tags: {e}")
            return set()

    async def _get_embedding(self, text: str) -> np.ndarray:
        """Generate an embedding vector for the given text"""
        try:
            return self.embed_model.encode(text, convert_to_numpy=True)
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise

    async def add_memory(self, text: str, task: str, score: float) -> int:
        """
        Add a new memory to Redis with embeddings and auto-tagging
        
        Args:
            text: The text content to store
            task: The task/context of the memory
            score: Initial importance score (0-1)
            
        Returns:
            Memory ID if successful, -1 otherwise
        """
        if not text or score < 0.0:
            logger.warning("Invalid memory: empty text or negative score")
            return -1

        try:
            embedding = await self._get_embedding(text)
            
            # Get next memory ID from Redis
            self.next_id = int(self.redis_client.get("memory:next_id") or 0)
            memory_id = self.next_id
            memory_key = f"memory:{memory_id}"
            timestamp = time.time()

            tags = await self._generate_auto_tags(text)
            links = await self._link_related_items(text, embedding, memory_id)

            memory_data = {
                "text": text,
                "task": task,
                "timestamp": timestamp,
                "score": score,
                "embedding": embedding.tobytes(),
                "tags": ",".join(tags),
                "links": ",".join(map(str, links))
            }
            
            # Store in Redis with appropriate encryption
            try:
                if self.encryption_key:
                    # Encrypt the JSON data but store embedding separately for search
                    memory_json = {
                        "text": text,
                        "task": task,
                        "timestamp": timestamp,
                        "score": score,
                        "tags": tags,
                        "links": links
                    }
                    message = pgpy.PGPMessage.new(json.dumps(memory_json))
                    key, _ = pgpy.PGPKey.from_blob(self.encryption_key)
                    encrypted_data = key.encrypt(message)
                    
                    # Store the encrypted data and separate embedding for search
                    self.redis_client.hset(memory_key, "data", str(encrypted_data))
                    self.redis_client.hset(memory_key, "embedding", embedding.tobytes())
                else:
                    # Store all fields without encryption
                    self.redis_client.hset(memory_key, mapping=memory_data)
            except redis.ConnectionError as e:
                logger.error(f"Redis connection error during memory storage: {e}")
                # Try to reconnect
                try:
                    self._initialize_redis_client()
                    # Retry once after reconnection
                    if self.encryption_key:
                        memory_json = {
                            "text": text,
                            "task": task,
                            "timestamp": timestamp,
                            "score": score,
                            "tags": tags,
                            "links": links
                        }
                        message = pgpy.PGPMessage.new(json.dumps(memory_json))
                        key, _ = pgpy.PGPKey.from_blob(self.encryption_key)
                        encrypted_data = key.encrypt(message)
                        
                        # Store the encrypted data and separate embedding for search
                        self.redis_client.hset(memory_key, "data", str(encrypted_data))
                        self.redis_client.hset(memory_key, "embedding", embedding.tobytes())
                    else:
                        self.redis_client.hset(memory_key, mapping=memory_data)
                except Exception:
                    # Queue for retry if still failing
                    self._queue_failed_operation(
                        operation_type="add_memory",
                        text=text,
                        task=task,
                        score=score
                    )
                    return -1
            except Exception as e:
                logger.error(f"Failed to store memory in Redis: {e}")
                return -1

            # Update tracking data structures in Redis
            try:
                self._add_memory_id(memory_id)
                
                # Store tags in Redis
                self._set_memory_tags(memory_id, tags)
                
                # Update tag to memories mapping
                for tag in tags:
                    self._add_tag_to_memory(tag, memory_id)
            except redis.ConnectionError as e:
                logger.error(f"Redis connection error during tag updates: {e}")
                # Queue tag operations for retry
                for tag in tags:
                    self._queue_failed_operation(
                        operation_type="add_tag",
                        memory_id=memory_id,
                        tag=tag
                    )
            
            # Update connection weights for linked items
            for linked_id in links:
                key = (min(memory_id, linked_id), max(memory_id, linked_id))
                self.connection_weights[key] = self.connection_weights.get(key, 0) + 1

            logger.info(f"Added memory {memory_id}: {text[:50]}...")
            return memory_id
        except Exception as e:
            logger.error(f"Failed to add memory: {e}")
            return -1

    async def semantic_search(self, query: str, top_k: int = 5, threshold: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Perform semantic search on stored memories using vector similarity
        
        Args:
            query: The search query text
            top_k: Maximum number of results to return
            threshold: Minimum similarity score threshold
            
        Returns:
            List of matching memory items with scores
        """
        if not query:
            logger.warning("Empty search query")
            return []

        try:
            query_embedding = await self._get_embedding(query)
            query_vec = query_embedding.tobytes()

            # Construct Redis vector search query
            redis_query = Query(f"*=>[KNN {top_k} @embedding $vec AS score]").sort_by("score", asc=False).return_fields("text", "task", "timestamp", "score", "tags", "links", "data").dialect(2)
            
            try:
                result = self.redis_client.ft(self.index_name).search(redis_query, query_params={"vec": query_vec})
            except redis.ConnectionError as e:
                logger.error(f"Redis connection error during search: {e}")
                # Try to reconnect
                self._initialize_redis_client()
                # Retry once after reconnection
                result = self.redis_client.ft(self.index_name).search(redis_query, query_params={"vec": query_vec})
            except Exception as e:
                logger.error(f"Error executing Redis search: {e}")
                return []

            results = []
            for doc in result.docs:
                try:
                    memory_id = int(doc.id.split(":")[1])
                    
                    # Handle data consistently with encryption
                    if self.encryption_key:
                        # If "data" field exists, decrypt it
                        if hasattr(doc, "data") and doc.data:
                            enc_message = pgpy.PGPMessage.from_blob(doc.data)
                            key, _ = pgpy.PGPKey.from_blob(self.encryption_key)
                            decrypted_message = key.decrypt(enc_message)
                            memory_data = json.loads(decrypted_message.message)
                            memory_data["id"] = memory_id
                        # Otherwise encrypt all relevant fields for consistency
                        else:
                            # Build data dict from individual fields
                            memory_data = {
                                "id": memory_id,
                                "text": getattr(doc, "text", ""),
                                "task": getattr(doc, "task", ""),
                                "timestamp": float(getattr(doc, "timestamp", 0)),
                                "score": float(getattr(doc, "score", 0)),
                                "tags": getattr(doc, "tags", "").split(",") if getattr(doc, "tags", "") else [],
                                "links": [int(link) for link in getattr(doc, "links", "").split(",")] if getattr(doc, "links", "") else []
                            }
                            
                            # Encrypt for future storage
                            try:
                                message = pgpy.PGPMessage.new(json.dumps(memory_data))
                                key, _ = pgpy.PGPKey.from_blob(self.encryption_key)
                                encrypted_data = key.encrypt(message)
                                # Store encrypted data for next time
                                self.redis_client.hset(f"memory:{memory_id}", "data", str(encrypted_data))
                                logger.debug(f"Retroactively encrypted memory data for ID: {memory_id}")
                            except Exception as e:
                                logger.warning(f"Failed to retroactively encrypt memory {memory_id}: {e}")
                    else:
                        # No encryption, just build the data dictionary
                        memory_data = {
                            "id": memory_id,
                            "text": getattr(doc, "text", ""),
                            "task": getattr(doc, "task", ""),
                            "timestamp": float(getattr(doc, "timestamp", 0)),
                            "score": float(getattr(doc, "score", 0)),
                            "tags": getattr(doc, "tags", "").split(",") if getattr(doc, "tags", "") else [],
                            "links": [int(link) for link in getattr(doc, "links", "").split(",")] if getattr(doc, "links", "") else []
                        }

                    # Filter by threshold if provided
                    if threshold is not None and memory_data.get("score", 0) < threshold:
                        continue

                    results.append(memory_data)
                except Exception as e:
                    logger.error(f"Error processing search result for document {doc.id}: {e}")
                    continue

            return results
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            return []

    async def add_to_episodic_buffer(self, user_input: str, response: str) -> None:
        """
        Add a user-assistant interaction to the episodic buffer
        
        Args:
            user_input: User message
            response: Assistant response
            
        The buffer automatically processes entries when full
        """
        try:
            entry = f"User: {user_input}\nAssistant: {response}"
            
            # Use proper tokenizer instead of simple split()
            if self.tokenizer:
                tokens = len(self.tokenizer.encode(entry))
            else:
                # Fallback to basic estimation if tokenizer failed to load
                tokens = len(entry.split())
            
            # Process buffer if it would exceed capacity
            if self.episodic_buffer_token_count + tokens > self.max_buffer_tokens or len(self.episodic_buffer) >= self.max_episodic_buffer_size:
                await self._process_episodic_buffer()
                
            self.episodic_buffer.append(entry)
            self.episodic_buffer_token_count += tokens
            
            # Persist episodic buffer state to Redis
            self._save_episodic_buffer()
        except Exception as e:
            logger.error(f"Error adding to episodic buffer: {e}")

    async def _process_episodic_buffer(self) -> None:
        """Transfer episodic buffer contents to long-term memory"""
        try:
            success_count = 0
            for entry in self.episodic_buffer:
                try:
                    memory_id = await self.add_memory(entry, "conversation", 1.0)
                    if memory_id >= 0:
                        success_count += 1
                except Exception as e:
                    logger.error(f"Error transferring entry to long-term memory: {e}")
                    # Queue the failed entry for retry
                    self._queue_failed_operation(
                        operation_type="add_memory",
                        text=entry,
                        task="conversation",
                        score=1.0
                    )
            
            # Clear buffer only if all entries were processed or added to retry queue
            self.episodic_buffer = []
            self.episodic_buffer_token_count = 0
            
            # Update persisted buffer state in Redis
            self._save_episodic_buffer()
            
            logger.info(f"Processed episodic buffer: {success_count} entries added to long-term memory")
        except redis.ConnectionError as e:
            logger.error(f"Redis connection error during episodic buffer processing: {e}")
            # Queue the entire operation for retry
            self._queue_failed_operation(operation_type="process_episodic")
        except Exception as e:
            logger.error(f"Error processing episodic buffer: {e}")
            
    async def get_episodic_buffer_contents(self) -> List[str]:
        """
        Get the current contents of the episodic buffer
        
        Returns:
            List of entries in the episodic buffer
        """
        return self.episodic_buffer.copy()
    
    async def get_episodic_buffer_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the episodic buffer
        
        Returns:
            Dictionary with buffer statistics
        """
        return {
            "size": len(self.episodic_buffer),
            "token_count": self.episodic_buffer_token_count,
            "max_size": self.max_episodic_buffer_size,
            "max_tokens": self.max_buffer_tokens
        }
        
    async def clear_episodic_buffer(self) -> None:
        """Clear the episodic buffer without transferring to long-term memory"""
        self.episodic_buffer = []
        self.episodic_buffer_token_count = 0
        self._save_episodic_buffer()
        logger.info("Episodic buffer cleared")

    async def _generate_auto_tags(self, text: str) -> List[str]:
        """
        Generate automatic tags for memory organization
        
        Args:
            text: Text to extract tags from
            
        Returns:
            List of extracted keyword tags
        """
        try:
            # Use enhanced NLP-based tagging if enabled
            if self.enable_enhanced_tagging:
                return await self._generate_nltk_tags(text)
            
            # Use proper tokenizer for the basic fallback
            if self.tokenizer:
                # Tokenize the text
                tokenized = self.tokenizer.tokenize(text.lower())
                # Keep words longer than 3 chars, limit to top 3
                return [word for word in tokenized if len(word) > 3 and not word.startswith('##')][:3]
            else:
                # Simple keyword extraction fallback if tokenizer not available
                words = text.lower().split()
                # Keep words longer than 3 chars, limit to top 3
                return [word for word in words if len(word) > 3][:3]
        except Exception as e:
            logger.error(f"Error generating tags: {e}")
            return []

    async def _generate_nltk_tags(self, text: str) -> List[str]:
        """
        Generate tags using NLTK for better NLP-based extraction
        
        Args:
            text: Text to extract tags from
            
        Returns:
            List of extracted keyword tags
        """
        try:
            # Tokenize and lowercase
            tokens = word_tokenize(text.lower())
            
            # Remove stopwords and punctuation
            stop_words = set(stopwords.words('english'))
            words = [word for word in tokens if word.isalnum() and word not in stop_words and len(word) > 3]
            
            # Count word frequencies
            word_counts = {}
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
                
            # Sort by frequency and return top 5
            sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
            return [word for word, count in sorted_words[:5]]
        except Exception as e:
            # Provide more specific error message based on the error type
            if "Resource 'corpora/stopwords' not found" in str(e):
                logger.error(f"NLTK stopwords resource missing: {e}. Make sure NLTK data is downloaded.")
            elif "Resource 'tokenizers/punkt' not found" in str(e):
                logger.error(f"NLTK punkt tokenizer missing: {e}. Make sure NLTK data is downloaded.")
            else:
                logger.error(f"Error in NLTK tagging: {e}.")
                
            logger.warning("Falling back to basic tagging.")
            words = text.lower().split()
            return [word for word in words if len(word) > 3][:3]

    async def _link_related_items(self, text: str, embedding: np.ndarray, memory_id: int) -> List[int]:
        """
        Find and link semantically related memory items
        
        Args:
            text: Memory text 
            embedding: Vector embedding of text
            memory_id: ID of the current memory entry
            
        Returns:
            List of linked memory IDs
        """
        try:
            # Only link if we have other memories
            if not self._get_memory_ids() or len(self._get_memory_ids()) < 2:
                return []
                
            # Get similar items from Redis
            results = await self.semantic_search(text, top_k=3, threshold=0.7)
            related_ids = []
            
            for item in results:
                related_id = item.get("id", -1)
                # Don't link to self
                if related_id != memory_id and related_id >= 0:
                    related_ids.append(related_id)
                    
            # Generate tokens from text for co-occurrence tracking
            if self.tokenizer:
                tokens = self.tokenizer.tokenize(text.lower())
                # Filter out special tokens and partial wordpieces
                words = [word for word in tokens if len(word) > 3 and not word.startswith('##')]
            else:
                # Fallback to basic tokenization
                words = text.lower().split()
                
            # Update co-occurrence counts
            for word in words:
                # Add word occurrence to related items
                for related_id in related_ids:
                    # This is a simplified version of Hebbian learning without weight updates
                    # Just counting co-occurrences for the static model
                    key = (min(memory_id, related_id), max(memory_id, related_id))
                    self.connection_weights[key] = self.connection_weights.get(key, 0) + 1
                    
            return related_ids
        except Exception as e:
            logger.error(f"Error linking items: {e}")
            return []

    async def reinforce_memory_schemas(self) -> None:
        """
        Static memory consolidation to reinforce schemas over time
        This simplifies the dynamic Hebbian learning by using static increments
        """
        try:
            # Static consolidation: increment co-occurrence counts
            for (id1, id2), weight in list(self.connection_weights.items()):
                self.connection_weights[(id1, id2)] = weight + 0.1  # Static increment
                
            # Save connection weights to Redis for persistence
            self._save_connection_weights()
        except Exception as e:
            logger.error(f"Error reinforcing memory schemas: {e}")

    def _save_connection_weights(self) -> None:
        """Save connection weights to Redis for persistence"""
        try:
            weights_data = {str(k): v for k, v in self.connection_weights.items()}
            weights_json = json.dumps(weights_data)
            
            # Apply encryption if encryption key is provided
            if self.encryption_key:
                try:
                    message = pgpy.PGPMessage.new(weights_json)
                    key, _ = pgpy.PGPKey.from_blob(self.encryption_key)
                    encrypted_data = key.encrypt(message)
                    self.redis_client.set("memory:connection_weights", str(encrypted_data))
                except Exception as e:
                    logger.error(f"Error encrypting connection weights: {e}")
                    # Fall back to unencrypted storage
                    self.redis_client.set("memory:connection_weights", weights_json)
            else:
                self.redis_client.set("memory:connection_weights", weights_json)
                
            logger.info("Connection weights saved to Redis")
        except Exception as e:
            logger.error(f"Error saving connection weights: {e}")

    def _load_connection_weights(self) -> None:
        """Load connection weights from Redis"""
        try:
            weights_data = self.redis_client.get("memory:connection_weights")
            if not weights_data:
                logger.info("No connection weights found in Redis")
                return
                
            # Decrypt data if encryption is enabled
            if self.encryption_key and weights_data.startswith("-----BEGIN PGP MESSAGE-----"):
                try:
                    enc_message = pgpy.PGPMessage.from_blob(weights_data)
                    key, _ = pgpy.PGPKey.from_blob(self.encryption_key)
                    decrypted_message = key.decrypt(enc_message)
                    weights_json = decrypted_message.message
                except Exception as e:
                    logger.error(f"Error decrypting connection weights: {e}")
                    logger.warning("Using stored connection weights as-is, may be encrypted or corrupted")
                    weights_json = weights_data
            else:
                weights_json = weights_data
            
            # Parse the JSON data
            weights_dict = json.loads(weights_json)
            
            # Convert string keys back to tuples
            self.connection_weights = {
                tuple(map(int, k.strip('()').split(','))): float(v) 
                for k, v in weights_dict.items()
            }
            logger.info(f"Loaded {len(self.connection_weights)} connection weights from Redis")
            
        except Exception as e:
            logger.error(f"Error loading connection weights: {e}")
            # Initialize with empty dict in case of error
            self.connection_weights = {}
            
    async def decay_memory_importance(self, decay_factor: float = 0.85) -> None:
        """
        Apply temporal decay to memory importance scores
        
        Args:
            decay_factor: Factor to multiply scores by (α=0.85)
        """
        try:
            memory_ids = self._get_memory_ids()
            for memory_id in memory_ids:
                memory_key = f"memory:{memory_id}"
                try:
                    score = float(self.redis_client.hget(memory_key, "score") or 0.0)
                    new_score = score * decay_factor
                    self.redis_client.hset(memory_key, "score", new_score)
                except redis.ConnectionError:
                    # Try to reconnect
                    self._initialize_redis_client()
                    # Retry once
                    score = float(self.redis_client.hget(memory_key, "score") or 0.0)
                    new_score = score * decay_factor
                    self.redis_client.hset(memory_key, "score", new_score)
                except Exception as e:
                    logger.error(f"Error decaying memory {memory_id}: {e}")
                    continue
            logger.info(f"Applied memory decay with factor {decay_factor}")
        except Exception as e:
            logger.error(f"Error in memory decay process: {e}")
    
    async def ensure_encryption_consistency(self) -> Tuple[int, int]:
        """
        Verifies and fixes encryption consistency in existing memory data.
        
        This method will:
        1. Check all memory entries for encryption consistency
        2. Fix entries that don't match the current encryption state
        3. Return counts of checked and fixed entries
        
        Returns:
            Tuple of (checked_count, fixed_count)
        """
        memory_ids = self._get_memory_ids()
        if not memory_ids:
            logger.info("No memories to check for encryption consistency")
            return 0, 0
            
        checked = 0
        fixed = 0
        
        try:
            for memory_id in memory_ids:
                memory_key = f"memory:{memory_id}"
                checked += 1
                
                try:
                    # Get all fields for this memory
                    memory_data = self.redis_client.hgetall(memory_key)
                    
                    # Check encryption consistency
                    if self.encryption_key:
                        # Should have encrypted data
                        if "data" not in memory_data or not memory_data["data"].startswith("-----BEGIN PGP MESSAGE-----"):
                            # Need to encrypt this entry
                            logger.info(f"Converting memory {memory_id} to use encryption")
                            
                            # Build JSON data from individual fields
                            memory_json = {
                                "text": memory_data.get("text", ""),
                                "task": memory_data.get("task", ""),
                                "timestamp": float(memory_data.get("timestamp", 0)),
                                "score": float(memory_data.get("score", 0)),
                                "tags": memory_data.get("tags", "").split(",") if memory_data.get("tags") else [],
                                "links": [int(link) for link in memory_data.get("links", "").split(",")] 
                                         if memory_data.get("links") else []
                            }
                            
                            # Encrypt the data
                            message = pgpy.PGPMessage.new(json.dumps(memory_json))
                            key, _ = pgpy.PGPKey.from_blob(self.encryption_key)
                            encrypted_data = key.encrypt(message)
                            
                            # Store encrypted data and make sure embedding is preserved separately
                            self.redis_client.hset(memory_key, "data", str(encrypted_data))
                            # Ensure the embedding vector is kept separate for search if it exists
                            if "embedding" in memory_data:
                                # Keep the embedding as is for vector search to work
                                pass
                            fixed += 1
                    else:
                        # Should not have encrypted data
                        if "data" in memory_data and memory_data["data"].startswith("-----BEGIN PGP MESSAGE-----"):
                            # Need to decrypt and store individual fields
                            logger.info(f"Converting memory {memory_id} to remove encryption")
                            
                            # Decrypt the data
                            enc_message = pgpy.PGPMessage.from_blob(memory_data["data"])
                            key, _ = pgpy.PGPKey.from_blob(self.encryption_key)
                            decrypted_message = key.decrypt(enc_message)
                            decrypted_data = json.loads(decrypted_message.message)
                            
                            # Store individual fields
                            fields_to_set = {
                                "text": decrypted_data.get("text", ""),
                                "task": decrypted_data.get("task", ""),
                                "timestamp": str(decrypted_data.get("timestamp", 0)),
                                "score": str(decrypted_data.get("score", 0)),
                                "tags": ",".join(decrypted_data.get("tags", [])),
                                "links": ",".join(map(str, decrypted_data.get("links", [])))
                            }
                            
                            # Store unencrypted data
                            self.redis_client.hset(memory_key, mapping=fields_to_set)
                            # Remove encrypted data
                            self.redis_client.hdel(memory_key, "data")
                            fixed += 1
                
                except Exception as e:
                    logger.error(f"Error checking encryption for memory {memory_id}: {e}")
                    continue
            
            logger.info(f"Encryption consistency check: checked {checked} memories, fixed {fixed}")
            return checked, fixed
            
        except Exception as e:
            logger.error(f"Error ensuring encryption consistency: {e}")
            return checked, fixed

    async def _ensure_encryption_on_init(self) -> None:
        """Initialize and verify encryption consistency"""
        try:
            # Short delay to allow other initialization to complete
            await asyncio.sleep(0.5)
            
            # Migrate from in-memory to Redis-based storage if necessary
            await self._migrate_to_redis_storage()
            
            # Verify encryption consistency for all stored memories
            checked, fixed = await self.ensure_encryption_consistency()
            if fixed > 0:
                logger.info(f"Fixed encryption consistency for {fixed} memories during initialization")
                
            # Ensure episodic buffer persistence
            if not self.redis_client.exists("memory:episodic_buffer") and self.episodic_buffer:
                self._save_episodic_buffer()
                logger.info("Initialized episodic buffer persistence")
                
            # Load and process any pending retry operations
            await self._load_retry_queue()
            if self.retry_queue:
                logger.info(f"Found {len(self.retry_queue)} failed operations to retry")
                await self._process_retry_queue()
                
        except Exception as e:
            logger.error(f"Error during initialization checks: {e}")
            
    async def _migrate_to_redis_storage(self) -> None:
        """
        Check for legacy in-memory data structures and migrate to Redis if found.
        This is a transitional method to help migrate from older versions of the code
        that used in-memory structures.
        """
        try:
            # Check for any lingering in-memory structures from older versions
            # These would be set if loading from a pickle or similar mechanism
            if hasattr(self, '_legacy_memory_ids') and self._legacy_memory_ids:
                logger.info(f"Migrating {len(self._legacy_memory_ids)} memories from in-memory to Redis")
                
                # Migrate memory IDs
                for memory_id in self._legacy_memory_ids:
                    self._add_memory_id(memory_id)
                
                # Migrate memory tags
                if hasattr(self, '_legacy_memory_tags'):
                    for memory_id, tags in self._legacy_memory_tags.items():
                        self._set_memory_tags(memory_id, list(tags))
                
                # Migrate tag to memories mapping
                if hasattr(self, '_legacy_tag_to_memories'):
                    for tag, memory_ids in self._legacy_tag_to_memories.items():
                        for memory_id in memory_ids:
                            self._add_tag_to_memory(tag, memory_id)
                
                logger.info("Migration to Redis storage complete")
                
                # Clean up legacy attributes to free memory
                if hasattr(self, '_legacy_memory_ids'):
                    del self._legacy_memory_ids
                if hasattr(self, '_legacy_memory_tags'):
                    del self._legacy_memory_tags
                if hasattr(self, '_legacy_tag_to_memories'):
                    del self._legacy_tag_to_memories
        except Exception as e:
            logger.error(f"Error migrating to Redis storage: {e}")

    async def get_tag_statistics(self) -> Dict[str, Any]:
        """Get statistics about tags and memory distribution"""
        try:
            # Get all tags
            all_tags = self._get_all_tags()
            
            # Count memories for each tag
            tag_counts = {}
            for tag in all_tags:
                memories = self._get_memories_by_tag(tag)
                tag_counts[tag] = len(memories)
            
            # Sort tags by count (descending)
            top_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            
            # Calculate statistics
            total_memories = len(self._get_memory_ids())
            total_tags = len(all_tags)
            avg_tags_per_memory = sum(len(self._get_memory_tags(mid)) for mid in self._get_memory_ids()) / max(total_memories, 1)
            
            return {
                "total_memories": total_memories,
                "total_tags": total_tags,
                "avg_tags_per_memory": avg_tags_per_memory,
                "top_tags": dict(top_tags)
            }
        except Exception as e:
            logger.error(f"Error getting tag statistics: {e}")
            return {
                "error": str(e),
                "total_memories": 0,
                "total_tags": 0,
                "avg_tags_per_memory": 0,
                "top_tags": {}
            }

    async def search_by_tag(self, tag: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for memories with a specific tag
        
        Args:
            tag: Tag to search for
            limit: Maximum number of results to return
            
        Returns:
            List of memory dictionaries
        """
        try:
            memory_ids = self._get_memories_by_tag(tag)
            
            results = []
            for memory_id in sorted(memory_ids, reverse=True)[:limit]:
                memory_key = f"memory:{memory_id}"
                
                try:
                    # Try to get memory data
                    if self.encryption_key:
                        # Handle encrypted data
                        encrypted_data = self.redis_client.hget(memory_key, "data")
                        if encrypted_data:
                            enc_message = pgpy.PGPMessage.from_blob(encrypted_data)
                            key, _ = pgpy.PGPKey.from_blob(self.encryption_key)
                            decrypted_message = key.decrypt(enc_message)
                            memory_data = json.loads(decrypted_message.message)
                            memory_data["id"] = memory_id
                            results.append(memory_data)
                    else:
                        # Handle unencrypted data
                        memory_data = self.redis_client.hgetall(memory_key)
                        if memory_data:
                            memory_data["id"] = memory_id
                            # Convert numeric fields
                            if "timestamp" in memory_data:
                                memory_data["timestamp"] = float(memory_data["timestamp"])
                            if "score" in memory_data:
                                memory_data["score"] = float(memory_data["score"])
                            # Convert tag and link strings to lists
                            if "tags" in memory_data:
                                memory_data["tags"] = memory_data["tags"].split(",") if memory_data["tags"] else []
                            if "links" in memory_data:
                                memory_data["links"] = [int(link) for link in memory_data["links"].split(",")] if memory_data["links"] else []
                            results.append(memory_data)
                except Exception as e:
                    logger.error(f"Error retrieving memory {memory_id}: {e}")
                    continue
                    
            return results
        except Exception as e:
            logger.error(f"Error searching by tag: {e}")
            return []

    async def get_memory_by_id(self, memory_id: int) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific memory by ID
        
        Args:
            memory_id: ID of the memory to retrieve
            
        Returns:
            Memory dictionary if found, None otherwise
        """
        try:
            memory_key = f"memory:{memory_id}"
            
            # Check if memory exists
            if not self.redis_client.exists(memory_key):
                return None
                
            # Get memory data
            if self.encryption_key:
                # Handle encrypted data
                encrypted_data = self.redis_client.hget(memory_key, "data")
                if encrypted_data:
                    enc_message = pgpy.PGPMessage.from_blob(encrypted_data)
                    key, _ = pgpy.PGPKey.from_blob(self.encryption_key)
                    decrypted_message = key.decrypt(enc_message)
                    memory_data = json.loads(decrypted_message.message)
                    memory_data["id"] = memory_id
                    return memory_data
                else:
                    return None
            else:
                # Handle unencrypted data
                memory_data = self.redis_client.hgetall(memory_key)
                if memory_data:
                    memory_data["id"] = memory_id
                    # Convert numeric fields
                    if "timestamp" in memory_data:
                        memory_data["timestamp"] = float(memory_data["timestamp"])
                    if "score" in memory_data:
                        memory_data["score"] = float(memory_data["score"])
                    # Convert tag and link strings to lists
                    if "tags" in memory_data:
                        memory_data["tags"] = memory_data["tags"].split(",") if memory_data["tags"] else []
                    if "links" in memory_data:
                        memory_data["links"] = [int(link) for link in memory_data["links"].split(",")] if memory_data["links"] else []
                    return memory_data
                else:
                    return None
        except Exception as e:
            logger.error(f"Error retrieving memory {memory_id}: {e}")
            return None 

    async def delete_memory(self, memory_id: int) -> bool:
        """
        Delete a memory and all its associated data
        
        Args:
            memory_id: ID of the memory to delete
            
        Returns:
            True if deleted, False otherwise
        """
        try:
            memory_key = f"memory:{memory_id}"
            
            # Check if memory exists
            if not self.redis_client.exists(memory_key):
                logger.warning(f"Cannot delete non-existent memory {memory_id}")
                return False
                
            # Get tags before deleting memory
            tags = self._get_memory_tags(memory_id)
            
            # Delete memory data
            self.redis_client.delete(memory_key)
            
            # Remove from memory IDs set
            self.redis_client.srem("memory:ids", memory_id)
            
            # Remove from tag mappings
            for tag in tags:
                self.redis_client.srem(f"tag:{tag}:memories", memory_id)
            
            # Delete memory tags set
            self.redis_client.delete(f"memory:{memory_id}:tags")
            
            # Remove from connection weights
            for key in list(self.connection_weights.keys()):
                if memory_id in key:
                    del self.connection_weights[key]
            
            logger.info(f"Deleted memory {memory_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting memory {memory_id}: {e}")
            return False
            
    async def prune_memories(self, keep_top_n: int = 1000, min_score: float = 0.2) -> int:
        """
        Prune memories to keep only the top N highest-scored ones above minimum score
        
        Args:
            keep_top_n: Number of high-scoring memories to keep
            min_score: Minimum score threshold
            
        Returns:
            Number of memories pruned
        """
        try:
            memory_ids = self._get_memory_ids()
            if not memory_ids or len(memory_ids) <= keep_top_n:
                return 0
                
            # Get scores for all memories
            memories_with_scores = []
            for memory_id in memory_ids:
                try:
                    memory_key = f"memory:{memory_id}"
                    if self.encryption_key:
                        # Get score from encrypted data
                        encrypted_data = self.redis_client.hget(memory_key, "data")
                        if encrypted_data:
                            enc_message = pgpy.PGPMessage.from_blob(encrypted_data)
                            key, _ = pgpy.PGPKey.from_blob(self.encryption_key)
                            decrypted_message = key.decrypt(enc_message)
                            memory_data = json.loads(decrypted_message.message)
                            score = memory_data.get("score", 0.0)
                        else:
                            score = 0.0
                    else:
                        # Get score directly
                        score = float(self.redis_client.hget(memory_key, "score") or 0.0)
                    
                    memories_with_scores.append((memory_id, score))
                except Exception as e:
                    logger.error(f"Error getting score for memory {memory_id}: {e}")
                    memories_with_scores.append((memory_id, 0.0))
            
            # Sort by score (descending)
            sorted_memories = sorted(memories_with_scores, key=lambda x: x[1], reverse=True)
            
            # Keep top N and those above minimum score
            keep_ids = set(memory_id for memory_id, score in sorted_memories[:keep_top_n])
            keep_ids.update(memory_id for memory_id, score in sorted_memories if score >= min_score)
            
            # Delete memories not in keep_ids
            delete_ids = [memory_id for memory_id in memory_ids if memory_id not in keep_ids]
            deleted_count = 0
            
            for memory_id in delete_ids:
                if await self.delete_memory(memory_id):
                    deleted_count += 1
            
            logger.info(f"Pruned {deleted_count} memories, kept {len(keep_ids)}")
            return deleted_count
        except Exception as e:
            logger.error(f"Error pruning memories: {e}")
            return 0
            
    async def optimize_tag_storage(self) -> int:
        """
        Clean up empty tag sets and orphaned tags
        
        Returns:
            Number of orphaned tags removed
        """
        try:
            # Get all tags
            all_tags = self._get_all_tags()
            removed = 0
            
            for tag in all_tags:
                # Check if tag has any memories
                memories = self._get_memories_by_tag(tag)
                if not memories:
                    # Remove empty tag
                    self.redis_client.delete(f"tag:{tag}:memories")
                    removed += 1
            
            logger.info(f"Removed {removed} orphaned tags")
            return removed
        except Exception as e:
            logger.error(f"Error optimizing tag storage: {e}")
            return 0
            
    async def backup_memory_metadata(self, backup_file: str) -> bool:
        """
        Export memory metadata to a JSON file (does not include vector embeddings)
        
        Args:
            backup_file: Path to backup file
            
        Returns:
            True if backup was successful, False otherwise
        """
        try:
            memory_ids = self._get_memory_ids()
            backup_data = {
                "version": "1.0",
                "timestamp": time.time(),
                "memory_count": len(memory_ids),
                "memories": {}
            }
            
            # Export memory metadata for each memory
            for memory_id in memory_ids:
                memory_data = await self.get_memory_by_id(memory_id)
                if memory_data:
                    # Don't include embeddings in backup (they're large)
                    if "embedding" in memory_data:
                        del memory_data["embedding"]
                    backup_data["memories"][str(memory_id)] = memory_data
            
            # Write backup to file
            with open(backup_file, 'w') as f:
                json.dump(backup_data, f, indent=2)
                
            logger.info(f"Backed up {len(memory_ids)} memories to {backup_file}")
            return True
        except Exception as e:
            logger.error(f"Error backing up memory metadata: {e}")
            return False 

    async def _load_retry_queue(self) -> None:
        """Load pending retry operations from Redis"""
        try:
            # Check if Redis has saved retry operations
            retry_queue_size = self.redis_client.llen("memory:retry_queue")
            if retry_queue_size > 0:
                # Load serialized operations
                serialized_ops = self.redis_client.lrange("memory:retry_queue", 0, -1)
                
                # Deserialize operations
                for serialized_op in serialized_ops:
                    try:
                        operation = json.loads(serialized_op)
                        if self._is_valid_operation(operation):
                            self.retry_queue.append(operation)
                    except Exception as e:
                        logger.error(f"Error deserializing retry operation: {e}")
                
                logger.info(f"Loaded {len(self.retry_queue)} retry operations from Redis")
            else:
                logger.debug("No retry operations found in Redis")
        except Exception as e:
            logger.error(f"Error loading retry queue: {e}")
            self.retry_queue = []
            
    def _is_valid_operation(self, operation: Dict) -> bool:
        """Check if an operation dictionary is valid for retry"""
        required_fields = ["operation_type", "timestamp"]
        if not all(field in operation for field in required_fields):
            return False
            
        valid_types = ["add_memory", "add_tag", "update_memory", "delete_memory", "process_episodic"]
        if operation["operation_type"] not in valid_types:
            return False
            
        return True
            
    def _save_retry_queue(self) -> None:
        """Save pending retry operations to Redis"""
        try:
            # Clear existing retry queue in Redis
            self.redis_client.delete("memory:retry_queue")
            
            # Save serialized operations
            if self.retry_queue:
                # Limit queue size
                limited_queue = self.retry_queue[:self.max_retry_queue_size]
                
                # Serialize operations
                serialized_ops = [json.dumps(op) for op in limited_queue]
                
                # Save to Redis
                self.redis_client.rpush("memory:retry_queue", *serialized_ops)
                
                logger.debug(f"Saved {len(limited_queue)} retry operations to Redis")
        except Exception as e:
            logger.error(f"Error saving retry queue to Redis: {e}")
            
    async def _process_retry_queue(self, max_retries: int = 10) -> int:
        """
        Process pending retry operations
        
        Args:
            max_retries: Maximum number of operations to retry in one batch
            
        Returns:
            Number of successfully retried operations
        """
        if self.is_retrying or not self.retry_queue:
            return 0
            
        self.is_retrying = True
        successful_retries = 0
        
        try:
            # Process up to max_retries operations
            retry_count = min(max_retries, len(self.retry_queue))
            
            for _ in range(retry_count):
                if not self.retry_queue:
                    break
                    
                # Get the oldest operation
                operation = self.retry_queue.pop(0)
                
                # Skip very old operations (older than 24 hours)
                if time.time() - operation.get("timestamp", 0) > 86400:
                    logger.warning(f"Skipping stale operation: {operation['operation_type']}")
                    continue
                
                # Process based on operation type
                success = False
                try:
                    if operation["operation_type"] == "add_memory":
                        memory_id = await self.add_memory(
                            operation["text"], 
                            operation.get("task", "unknown"),
                            operation.get("score", 0.5)
                        )
                        success = memory_id >= 0
                        
                    elif operation["operation_type"] == "add_tag":
                        if "memory_id" in operation and "tag" in operation:
                            self._add_tag_to_memory(operation["tag"], operation["memory_id"])
                            success = True
                            
                    elif operation["operation_type"] == "update_memory":
                        if "memory_id" in operation and "data" in operation:
                            memory_key = f"memory:{operation['memory_id']}"
                            self.redis_client.hset(memory_key, mapping=operation["data"])
                            success = True
                            
                    elif operation["operation_type"] == "delete_memory":
                        if "memory_id" in operation:
                            success = await self.delete_memory(operation["memory_id"])
                            
                    elif operation["operation_type"] == "process_episodic":
                        await self._process_episodic_buffer()
                        success = True
                
                except Exception as e:
                    logger.error(f"Error retrying operation {operation['operation_type']}: {e}")
                    # Re-queue if the operation is still failing due to Redis issues
                    if isinstance(e, redis.RedisError):
                        self.retry_queue.append(operation)
                    
                if success:
                    successful_retries += 1
                    
            # Save updated retry queue
            self._save_retry_queue()
            
            if successful_retries > 0:
                logger.info(f"Successfully retried {successful_retries}/{retry_count} operations")
                
            return successful_retries
            
        except Exception as e:
            logger.error(f"Error processing retry queue: {e}")
            return successful_retries
        finally:
            self.is_retrying = False
            
    def _queue_failed_operation(self, operation_type: str, **kwargs) -> None:
        """
        Queue a failed operation for later retry
        
        Args:
            operation_type: Type of operation that failed
            **kwargs: Operation-specific parameters
        """
        try:
            # Create operation record
            operation = {
                "operation_type": operation_type,
                "timestamp": time.time(),
                **kwargs
            }
            
            # Add to queue
            self.retry_queue.append(operation)
            
            # Trim queue if needed
            if len(self.retry_queue) > self.max_retry_queue_size:
                # Keep newest operations
                self.retry_queue = self.retry_queue[-self.max_retry_queue_size:]
                
            # Save queue to Redis
            self._save_retry_queue()
            
            logger.debug(f"Queued failed {operation_type} operation for retry")
        except Exception as e:
            logger.error(f"Error queueing failed operation: {e}") 
