"""
Orchestrator module for the SolidLight framework.
Implements a minimal chat loop with alignment, drift detection, and dual-time scale adaptation.
"""
import pgpy
import asyncio
import torch
import json
import time
import logging
import os
import re
from typing import Dict, List, Optional, Any, Tuple, Union
from pathlib import Path

from .memory import QuantumMemory

# Conditionally import the Mistral 7B components
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    HAS_MISTRAL_DEPS = True
except ImportError:
    HAS_MISTRAL_DEPS = False

# Try to import sentiment analysis for preference detection
try:
    from nltk.sentiment import SentimentIntensityAnalyzer
    HAS_SENTIMENT_ANALYSIS = True
except ImportError:
    HAS_SENTIMENT_ANALYSIS = False

logger = logging.getLogger(__name__)

class QuantumSynapseOrchestrator:
    """
    Orchestrates the chat interaction loop, memory management, and alignment checks.
    Simplified version with minimal components and no weight updates.
    """
    def __init__(self, 
                 use_gpu: bool = True, 
                 max_chat_history: int = 10, 
                 encryption_key: Optional[str] = None,
                 model_name: str = 'all-MiniLM-L6-v2',
                 alignment_json_path: str = "alignment.json",
                 enable_model_inference: bool = False,
                 redis_host: str = "localhost",
                 redis_port: int = 6379,
                 enable_enhanced_tagging: bool = False,
                 preference_analysis_enabled: bool = False):
        """
        Initialize the orchestrator with memory management and alignment checks
        
        Args:
            use_gpu: Whether to use GPU acceleration if available
            max_chat_history: Maximum number of chat turns to keep in memory
            encryption_key: Key for memory encryption with PGP
            model_name: Name of the embedding model for memory
            alignment_json_path: Path to PGP-encrypted alignment rules
            enable_model_inference: Whether to enable actual LLM inference
            redis_host: Redis server hostname
            redis_port: Redis server port
            enable_enhanced_tagging: Use NLP for better memory tagging
            preference_analysis_enabled: Enable memory-based preference analysis
        """
        self.use_gpu = use_gpu
        self.max_chat_history = max_chat_history
        self.chat_history = []
        self.chat_context_turns = min(5, max_chat_history)
        self.enable_model_inference = enable_model_inference and HAS_MISTRAL_DEPS
        self.preference_analysis_enabled = preference_analysis_enabled
        
        # Initialize memory system
        try:
            self.memory = QuantumMemory(
                model_name=model_name, 
                use_gpu=use_gpu, 
                encryption_key=encryption_key,
                redis_host=redis_host,
                redis_port=redis_port,
                enable_enhanced_tagging=enable_enhanced_tagging
            )
        except Exception as e:
            logger.error(f"Failed to initialize memory system: {e}")
            raise
        
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        self.rejection_history = []
        self.model_name = model_name
        self.user_preferences = {"style": "concise", "tone": "friendly"}
        
        # Initialize sentiment analyzer if available
        self.sentiment_analyzer = None
        if HAS_SENTIMENT_ANALYSIS and preference_analysis_enabled:
            try:
                self.sentiment_analyzer = SentimentIntensityAnalyzer()
                logger.info("Sentiment analysis enabled for preference detection")
            except Exception as e:
                logger.warning(f"Failed to initialize sentiment analyzer: {e}")

        # Load alignment JSON with PGP encryption
        self.alignment_json_path = alignment_json_path
        self.alignment_key = encryption_key
        self.alignment_rules = self._load_alignment_json()
        
        # Track adaptation timestamps
        self.last_fast_adapt_time = time.time()
        self.last_slow_adapt_time = time.time()
        self.fast_adaptation_interval = 30  # 30 seconds
        self.slow_adaptation_interval = 86400  # 24 hours (α=0.85 decay)
        
        # Initialize LLM (Mistral 7B) if enabled
        self.model = None
        self.tokenizer = None
        if self.enable_model_inference:
            try:
                self._init_model()
            except Exception as e:
                logger.error(f"Failed to initialize Mistral model: {e}")
                logger.warning("Falling back to simulated responses")
                self.enable_model_inference = False
        
        logger.info(f"QuantumSynapseOrchestrator initialized on {self.device}" +
                   (f" with Mistral 7B" if self.enable_model_inference else " with simulated responses"))

    def _load_alignment_json(self) -> Dict[str, Any]:
        """
        Load and decrypt alignment rules from JSON
        
        Returns:
            Dictionary of alignment rules
        """
        try:
            # Default alignment rules if file not found or decryption fails
            default_rules = {
                "blocked_keywords": [
                    "harmful",
                    "illegal",
                    "unethical"
                ],
                "response_templates": {
                    "blocked_request": "I cannot assist with that request as it appears to involve potentially harmful activities."
                }
            }
            
            alignment_path = Path(self.alignment_json_path)
            if not alignment_path.exists():
                logger.warning(f"Alignment file {self.alignment_json_path} not found, using defaults")
                return default_rules
                
            with open(self.alignment_json_path, 'r') as f:
                if self.alignment_key:
                    try:
                        # Decrypt the alignment file with PGP
                        encrypted_data = f.read()
                        enc_message = pgpy.PGPMessage.from_blob(encrypted_data)
                        key, _ = pgpy.PGPKey.from_blob(self.alignment_key)
                        decrypted_message = key.decrypt(enc_message)
                        return json.loads(decrypted_message.message)
                    except pgpy.errors.PGPError as e:
                        logger.error(f"PGP error decrypting alignment file: {e}")
                        return default_rules
                    except Exception as e:
                        logger.error(f"Failed to decrypt alignment rules: {e}")
                        return default_rules
                else:
                    # If no encryption key, treat as plain JSON
                    try:
                        return json.load(f)
                    except json.JSONDecodeError as e:
                        logger.error(f"Invalid JSON in alignment file: {e}")
                        return default_rules
        except Exception as e:
            logger.error(f"Failed to load alignment JSON: {e}")
            # Return default rules as fallback
            return default_rules
    
    def _init_model(self) -> None:
        """Initialize the Mistral 7B language model with 4-bit quantization"""
        if not HAS_MISTRAL_DEPS:
            logger.error("Cannot initialize model: transformers or bitsandbytes not installed")
            self.enable_model_inference = False
            return
        
        try:
            # Set up 4-bit quantization config
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16
            )
            
            # Load model and tokenizer
            model_id = "mistralai/Mistral-7B-v0.1"
            logger.info(f"Loading {model_id}...")
            
            # Load tokenizer first with error handling
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            except Exception as e:
                logger.error(f"Failed to load tokenizer: {e}")
                self.enable_model_inference = False
                return
                
            # Then load model with error handling for CUDA OOM
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    device_map="auto",
                    quantization_config=quantization_config
                )
            except torch.cuda.OutOfMemoryError:
                logger.error("CUDA out of memory when loading model. Trying with CPU fallback.")
                try:
                    # Fallback to CPU with 8-bit quantization
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_id,
                        device_map="cpu",
                        load_in_8bit=True
                    )
                    logger.warning(f"Loaded {model_id} on CPU with 8-bit quantization (slower)")
                except Exception as e2:
                    logger.error(f"Failed to load model on CPU as well: {e2}")
                    self.enable_model_inference = False
                    return
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                self.enable_model_inference = False
                return
                
            # Verify model loaded successfully
            if self.model is None or self.tokenizer is None:
                logger.error("Model or tokenizer initialization failed")
                self.enable_model_inference = False
                return
                
            logger.info(f"Loaded Mistral 7B model with 4-bit quantization")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.enable_model_inference = False

    async def chat(self, user_input: str) -> str:
        """
        Process user input through the minimal chat loop:
        Data Ingestion → Feature Selection → Prediction → Alignment Check → Memory Update
        
        Args:
            user_input: User message text
            
        Returns:
            Assistant response text
        """
        try:
            # Check for empty input
            if not user_input or user_input.strip() == "":
                return "I didn't receive any input. How can I help you?"

            # Check if alignment rules loaded successfully
            if not self.alignment_rules:
                logger.warning("Alignment rules not loaded, using defaults")
                self.alignment_rules = {
                    "blocked_keywords": [
                        "harmful",
                        "illegal",
                        "unethical"
                    ],
                    "response_templates": {
                        "blocked_request": "I cannot assist with that request as it appears to involve potentially harmful activities."
                    }
                }

            # 1. Data Ingestion
            # Get previous conversation context
            context = await self._get_chat_context()
            
            # 2. Feature Selection
            # Analyze user preferences based on history
            if self.preference_analysis_enabled:
                preferences = await self._analyze_user_preferences_from_memory()
            else:
                preferences = await self._analyze_user_preferences()
            
            # Retrieve relevant memories using semantic search
            try:
                relevant_memories = await self.memory.semantic_search(user_input, top_k=3, threshold=0.7)
            except Exception as e:
                logger.error(f"Error retrieving relevant memories: {e}")
                relevant_memories = []

            # Build context with memory and preferences
            memory_context = "Relevant information:\n" + "\n".join(f"- {m['text']}" for m in relevant_memories) if relevant_memories else ""
            personalization_context = "User preferences:\n" + "\n".join(f"- {k}: {v}" for k, v in preferences.items()) if preferences else ""
            full_context = f"{context}\n{memory_context}\n{personalization_context}\nUser: {user_input}\nAssistant:"

            # 3. Alignment Check
            if not await self._check_alignment(user_input):
                self.rejection_history.append(True)
                blocked_message = self.alignment_rules.get("response_templates", {}).get(
                    "blocked_request", 
                    "I cannot respond to that request as it doesn't align with ethical guidelines."
                )
                return blocked_message

            # Track that message passed alignment check
            self.rejection_history.append(False)
            
            # 4. Prediction (Generate response)
            try:
                response = await self._generate_response(full_context)
            except Exception as e:
                logger.error(f"Error generating response: {e}")
                response = "I apologize, but I encountered an error generating a response."
            
            # 5. Memory Update
            self.chat_history.append({"role": "user", "content": user_input})
            self.chat_history.append({"role": "assistant", "content": response})

            # Maintain max history size
            while len(self.chat_history) > self.max_chat_history:
                self.chat_history.pop(0)

            # Add interaction to episodic buffer
            try:
                await self.memory.add_to_episodic_buffer(user_input, response)
            except Exception as e:
                logger.error(f"Error adding to episodic buffer: {e}")
            
            # Fast adaptation cycle (every interaction)
            await self._apply_fast_adaptation()
            
            # Slow adaptation cycle (check if it's time)
            current_time = time.time()
            if current_time - self.last_slow_adapt_time >= self.slow_adaptation_interval:
                await self._apply_slow_adaptation()
                self.last_slow_adapt_time = current_time

            return response
        except Exception as e:
            logger.error(f"Error in chat: {e}")
            return "I apologize, but I encountered an error processing your request."

    async def _get_chat_context(self) -> str:
        """
        Build conversation context from recent chat history
        
        Returns:
            Formatted string with recent conversation turns
        """
        try:
            context = []
            # Use the configured number of context turns
            start = max(0, len(self.chat_history) - 2 * self.chat_context_turns)
            for i in range(start, len(self.chat_history), 2):
                if i + 1 < len(self.chat_history):
                    user_msg = self.chat_history[i]["content"]
                    assistant_msg = self.chat_history[i + 1]["content"]
                    context.append(f"User: {user_msg}\nAssistant: {assistant_msg}")
            return "\n".join(context)
        except Exception as e:
            logger.error(f"Error building chat context: {e}")
            return ""

    async def _analyze_user_preferences(self) -> Dict[str, str]:
        """
        Basic static implementation of user preference analysis
        
        Returns:
            Dictionary of inferred user preferences
        """
        return self.user_preferences.copy()

    async def _analyze_user_preferences_from_memory(self) -> Dict[str, str]:
        """
        Analyze past interactions for user preferences using memory
        
        Returns:
            Dictionary of inferred user preferences
        """
        preferences = self.user_preferences.copy()
        
        try:
            # Search with different, more targeted queries to better capture preferences
            preference_queries = [
                "user preferences communication style",
                "user likes detailed explanations", 
                "user prefers brief responses",
                "user tone formal informal",
                "user technical expertise level"
            ]
            
            all_memories = []
            # Run multiple searches with different queries
            for query in preference_queries:
                memories = await self.memory.semantic_search(
                    query, 
                    top_k=3, 
                    threshold=0.6
                )
                all_memories.extend(memories)
            
            # Remove duplicates (if any)
            unique_memories = []
            seen_ids = set()
            for memory in all_memories:
                memory_id = memory.get("id")
                if memory_id not in seen_ids:
                    seen_ids.add(memory_id)
                    unique_memories.append(memory)
            
            if unique_memories:
                # Style preference detection
                style_keywords = {
                    "concise": ["brief", "concise", "short", "quick", "minimal"],
                    "detailed": ["detailed", "thorough", "comprehensive", "complete", "in-depth"],
                    "technical": ["technical", "advanced", "expert", "specialized", "precise"]
                }
                
                tone_keywords = {
                    "friendly": ["friendly", "casual", "informal", "conversational", "approachable"],
                    "professional": ["professional", "formal", "business", "objective", "serious"],
                    "enthusiastic": ["enthusiastic", "excited", "energetic", "passionate", "lively"]
                }
                
                # Count keyword occurrences with regex for better matching
                style_counts = {k: 0 for k in style_keywords}
                tone_counts = {k: 0 for k in tone_keywords}
                
                for memory in unique_memories:
                    text = memory["text"].lower()
                    
                    # Use sentiment analysis if available
                    if self.sentiment_analyzer:
                        sentiment = self.sentiment_analyzer.polarity_scores(text)
                        if sentiment['compound'] > 0.5:  # Very positive
                            tone_counts['enthusiastic'] += 1
                        elif sentiment['compound'] > 0.2:  # Positive
                            tone_counts['friendly'] += 1
                        elif sentiment['compound'] < -0.2:  # Negative/critical
                            tone_counts['professional'] += 1
                    
                    # Check for explicit preference statements
                    if re.search(r"(i|user) (want|prefer|like|need)s? (more|less|better)? ?(\w+)", text):
                        preference_match = re.search(r"(i|user) (want|prefer|like|need)s? (more|less|better)? ?(\w+)", text)
                        if preference_match:
                            preference_word = preference_match.group(4).lower()
                            # Check if the word relates to a style or tone
                            for style, keywords in style_keywords.items():
                                if preference_word in keywords or any(kw in preference_word for kw in keywords):
                                    style_counts[style] += 2  # Give higher weight to explicit preferences
                                    
                            for tone, keywords in tone_keywords.items():
                                if preference_word in keywords or any(kw in preference_word for kw in keywords):
                                    tone_counts[tone] += 2  # Give higher weight to explicit preferences
                    
                    # Regular keyword counting
                    for style, keywords in style_keywords.items():
                        for keyword in keywords:
                            if re.search(r"\b" + keyword + r"\b", text):
                                style_counts[style] += 1
                                
                    for tone, keywords in tone_keywords.items():
                        for keyword in keywords:
                            if re.search(r"\b" + keyword + r"\b", text):
                                tone_counts[tone] += 1
                
                # Update preferences if strong signals found
                if any(count > 1 for count in style_counts.values()):
                    preferences["style"] = max(style_counts.items(), key=lambda x: x[1])[0]
                    
                if any(count > 1 for count in tone_counts.values()):
                    preferences["tone"] = max(tone_counts.items(), key=lambda x: x[1])[0]
                
                # Detect technical level
                technical_indicators = ["code", "program", "technical", "developer", "engineer"]
                technical_count = sum(1 for indicator in technical_indicators if any(
                    re.search(r"\b" + indicator + r"\b", memory["text"].lower()) 
                    for memory in unique_memories
                ))
                
                if technical_count > 2:
                    preferences["technical_level"] = "advanced"
                elif technical_count > 0:
                    preferences["technical_level"] = "intermediate"
                else:
                    preferences["technical_level"] = "beginner"
                
                logger.info(f"Analyzed user preferences from {len(unique_memories)} memories: {preferences}")
            
            return preferences
        except Exception as e:
            logger.error(f"Error analyzing user preferences from memory: {e}")
            return preferences

    async def _check_alignment(self, text: str) -> bool:
        """
        Check user input against alignment rules
        
        Args:
            text: Text to check for alignment
            
        Returns:
            True if text aligns with rules, False otherwise
        """
        try:
            # Check against blocked keywords
            text_lower = text.lower()
            
            # First check blocked keywords
            for keyword in self.alignment_rules.get("blocked_keywords", []):
                if keyword.lower() in text_lower:
                    logger.warning(f"Alignment check failed: blocked keyword '{keyword}' detected")
                    return False
                    
            # Then check blocked topics
            for topic in self.alignment_rules.get("blocked_topics", []):
                if topic.lower() in text_lower:
                    logger.warning(f"Alignment check failed: blocked topic '{topic}' detected")
                    return False
            
            # Check sensitivity levels if available
            for level_name, level_data in self.alignment_rules.get("sensitivity_levels", {}).items():
                threshold = level_data.get("threshold", 0.8)
                action = level_data.get("action", "blocked_request")
                
                # Simple sensitivity check based on keyword count
                keywords = self.alignment_rules.get("blocked_keywords", [])
                topic_words = " ".join(self.alignment_rules.get("blocked_topics", [])).split()
                all_sensitive_words = keywords + topic_words
                
                word_count = sum(1 for word in all_sensitive_words if word.lower() in text_lower)
                total_words = len(text_lower.split())
                
                if total_words > 0 and word_count / total_words > threshold:
                    logger.warning(f"Alignment check failed: {level_name} sensitivity triggered")
                    return False
                    
            return True
        except Exception as e:
            logger.error(f"Error in alignment check: {e}")
            # Default to allowing the message if the check errors
            return True

    async def _generate_response(self, full_context: str) -> str:
        """
        Generate a response using the language model
        
        Args:
            full_context: The full context for generation
            
        Returns:
            Generated text response
        """
        # Use actual Mistral 7B inference if enabled
        if self.enable_model_inference and self.model is not None and self.tokenizer is not None:
            try:
                # Prepare input for model
                inputs = self.tokenizer(full_context, return_tensors="pt").to(self.device)
                
                # Generate response
                with torch.no_grad():
                    outputs = self.model.generate(
                        inputs.input_ids, 
                        max_new_tokens=256,
                        temperature=0.7,
                        top_p=0.9,
                        repetition_penalty=1.1
                    )
                
                # Decode and return response
                response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                return response
            except torch.cuda.OutOfMemoryError:
                logger.error("CUDA out of memory during generation. Response will be simulated instead.")
                return f"I apologize, but I'm experiencing resource constraints. Here's a simulated response about your query on {full_context.split('User: ')[-1][:30]}..."
            except Exception as e:
                logger.error(f"Error generating response with Mistral: {e}")
                return "I apologize, but I encountered an error generating a response."
        
        # Adjust simulated response based on user preferences
        response_style = self.user_preferences.get("style", "concise")
        response_tone = self.user_preferences.get("tone", "friendly")
        
        # Format simulated response based on style and tone
        base_response = f"This is a simulated response. In a real implementation, Mistral 7B would generate text here based on your query about {full_context.split('User: ')[-1][:30]}..."
        
        if response_style == "concise":
            formatted_response = f"Here's a brief answer to your question. {base_response}"
        elif response_style == "detailed":
            formatted_response = f"Let me provide a comprehensive explanation. {base_response} I would expand on this with additional details and examples to give you a thorough understanding."
        elif response_style == "technical":
            formatted_response = f"From a technical perspective, {base_response} This would include specific technical terminology and precise explanations."
        else:
            formatted_response = base_response
            
        return formatted_response

    async def _apply_fast_adaptation(self) -> None:
        """
        Fast adaptation cycle that runs after each interaction
        Adjusts context window based on rejection rate, processes episodic buffer
        """
        try:
            # Monitor rejection rate over a window of 10 interactions
            if len(self.rejection_history) > 10:
                self.rejection_history = self.rejection_history[-10:]
                
            rejection_window = self.rejection_history[-10:]
            rejection_rate = sum(1 for r in rejection_window if r) / len(rejection_window) if rejection_window else 0.0

            # If rejection rate is high, decrease context window
            if rejection_rate > 0.25:
                logger.info("Drift detected: Adjusting context window size")
                self.chat_context_turns = max(2, self.chat_context_turns - 1)
                
            # Process episodic buffer if needed
            if len(self.memory.episodic_buffer) > 0:
                await self.memory._process_episodic_buffer()
                
            self.last_fast_adapt_time = time.time()
        except Exception as e:
            logger.error(f"Error in fast adaptation: {e}")

    async def _apply_slow_adaptation(self) -> None:
        """
        Slow adaptation cycle that runs daily
        Applies memory decay and reinforces memory schemas
        """
        try:
            # Apply memory decay (α=0.85)
            await self.memory.decay_memory_importance(decay_factor=0.85)
            
            # Reinforce memory schemas
            await self.memory.reinforce_memory_schemas()
            
            logger.info("Applied slow adaptation cycle (24h)")
            self.last_slow_adapt_time = time.time()
        except Exception as e:
            logger.error(f"Error in slow adaptation: {e}")

    async def save_state(self) -> bool:
        """
        Save the current state to Redis
        
        Returns:
            True if save was successful
        """
        try:
            # Save state metadata to Redis
            state_data = {
                "timestamp": time.time(),
                "chat_context_turns": self.chat_context_turns,
                "rejection_history_length": len(self.rejection_history),
                "preferences": self.user_preferences
            }
            
            self.memory.redis_client.set("orchestrator:state", json.dumps(state_data))
            logger.info("State saved to Redis")
            return True
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
            return False 
