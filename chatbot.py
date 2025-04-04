import os
import time
import json
import asyncio
import tiktoken
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable
from dotenv import load_dotenv
from openai import OpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("chatbot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("YouTubeChatbot")

# Load environment variables
load_dotenv()

# API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Constants
INDEX_NAME = "youtube-transcript-mountaindog1"
CREATOR_NAME = "MountainDog1"  # This could be dynamically loaded
CHATBOT_NAME = f"{CREATOR_NAME} Assistant"
DAILY_BUDGET = 1.0  # $1 per day budget limit

class UsageTracker:
    """Track API usage and costs"""
    
    def __init__(self, log_file: str = "api_usage.json"):
        self.log_file = log_file
        self.usage_log = self._load_usage_log()
        
        # DeepSeek pricing (as of documentation)
        self.pricing = {
            "deepseek-chat": {
                "input": 0.14,  # per million tokens
                "output": 0.28   # per million tokens
            },
            "deepseek-reasoner": {
                "input": 0.20,   # per million tokens
                "output": 0.40   # per million tokens
            }
        }
        
        # Initialize tokenizer for estimating token counts
        self.tokenizer = tiktoken.get_encoding("cl100k_base")  # Similar to GPT-4 tokenizer
    
    def _load_usage_log(self) -> Dict:
        """Load the usage log from file or create a new one"""
        if os.path.exists(self.log_file):
            try:
                with open(self.log_file, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                logger.warning(f"Error loading usage log. Creating new log.")
        
        # Initialize new log structure
        return {
            "total_tokens": {
                "input": 0,
                "output": 0
            },
            "total_cost": 0.0,
            "requests": []
        }
    
    def _save_usage_log(self):
        """Save the usage log to file"""
        with open(self.log_file, 'w') as f:
            json.dump(self.usage_log, f, indent=2)
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate the number of tokens in a text string"""
        return len(self.tokenizer.encode(text))
    
    def log_request(self, 
                    model: str, 
                    input_text: str, 
                    output_text: str, 
                    actual_token_counts: Optional[Dict[str, int]] = None):
        """Log an API request with token counts and costs"""
        # Use actual token counts from API if provided, otherwise estimate
        if actual_token_counts:
            input_tokens = actual_token_counts.get("input", 0)
            output_tokens = actual_token_counts.get("output", 0)
        else:
            input_tokens = self.estimate_tokens(input_text)
            output_tokens = self.estimate_tokens(output_text)
        
        # Calculate cost
        model_pricing = self.pricing.get(model, self.pricing["deepseek-chat"])
        input_cost = (input_tokens / 1_000_000) * model_pricing["input"]
        output_cost = (output_tokens / 1_000_000) * model_pricing["output"]
        total_cost = input_cost + output_cost
        
        # Create log entry
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "tokens": {
                "input": input_tokens,
                "output": output_tokens,
                "total": input_tokens + output_tokens
            },
            "cost": {
                "input": input_cost,
                "output": output_cost,
                "total": total_cost
            }
        }
        
        # Update usage totals
        self.usage_log["total_tokens"]["input"] += input_tokens
        self.usage_log["total_tokens"]["output"] += output_tokens
        self.usage_log["total_cost"] += total_cost
        self.usage_log["requests"].append(log_entry)
        
        # Save updated log
        self._save_usage_log()
        
        return {
            "tokens": input_tokens + output_tokens,
            "cost": total_cost
        }
    
    def get_usage_summary(self) -> Dict:
        """Get a summary of API usage and costs"""
        total_requests = len(self.usage_log["requests"])
        total_tokens = self.usage_log["total_tokens"]["input"] + self.usage_log["total_tokens"]["output"]
        
        # Get usage for the last 24 hours
        current_time = datetime.now()
        last_24h_requests = []
        
        for request in self.usage_log["requests"]:
            request_time = datetime.fromisoformat(request["timestamp"])
            time_diff = (current_time - request_time).total_seconds()
            if time_diff <= 86400:  # 24 hours in seconds
                last_24h_requests.append(request)
        
        last_24h_tokens = sum(r["tokens"]["total"] for r in last_24h_requests)
        last_24h_cost = sum(r["cost"]["total"] for r in last_24h_requests)
        
        return {
            "total_requests": total_requests,
            "total_tokens": total_tokens,
            "total_cost": self.usage_log["total_cost"],
            "last_24h": {
                "requests": len(last_24h_requests),
                "tokens": last_24h_tokens,
                "cost": last_24h_cost
            }
        }


class YouTubeTranscriptChatbot:
    """Complete chatbot implementation with DeepSeek integration"""
    
    def __init__(self):
        self.client = OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url="https://api.deepseek.com/v1"
        )
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        self.embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
        self.chat_history = []
        self.max_retry_attempts = 3
        self.retry_delay = 2  # seconds
        
        # Initialize usage tracking
        self.usage_tracker = UsageTracker()
        
        # Initialize cache
        self.cache = {}
        self.query_embedding_cache = {}  # Cache for query embeddings
        self.cache_ttl = 3600  # Cache TTL in seconds (1 hour)
        self.cache_timestamps = {}  # When items were added to cache
        
        # Connect to the retriever
        self.retriever = self._initialize_retriever()
        
        logger.info(f"Initialized {CHATBOT_NAME} with DeepSeek API")
    
    def _initialize_retriever(self):
        """Set up connection to Pinecone vector database"""
        try:
            logger.info(f"Connecting to Pinecone index: {INDEX_NAME}")
            
            # Connect to existing Pinecone index
            vector_store = PineconeVectorStore(
                index_name=INDEX_NAME,
                embedding=self.embeddings,
                text_key="text"
            )
            
            # Create retriever with MMR
            retriever = vector_store.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": 6,
                    "fetch_k": 15,
                    "lambda_mult": 0.7
                }
            )
            
            return retriever
        
        except Exception as e:
            logger.error(f"Error initializing retriever: {e}")
            raise RuntimeError(f"Failed to connect to vector database: {e}")
    
    async def get_embedding(self, text: str) -> List[float]:
        """Get embedding with retry logic"""
        for attempt in range(self.max_retry_attempts):
            try:
                return self.embeddings.embed_query(text)
            except Exception as e:
                if attempt < self.max_retry_attempts - 1:
                    wait_time = self.retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(f"Embedding error: {e}. Retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"Failed to generate embedding after {self.max_retry_attempts} attempts")
                    raise RuntimeError(f"Failed to generate embedding: {e}")
    
    async def retrieve_context(self, query: str) -> List[Dict]:
        """Retrieve relevant context from vector store with caching"""
        try:
            # Check for normalized query to improve cache hits
            normalized_query = query.lower().strip()
            current_time = time.time()
            
            # Clean expired cache entries
            expired_keys = [k for k, t in self.cache_timestamps.items() 
                          if current_time - t > self.cache_ttl]
            for k in expired_keys:
                if k in self.query_embedding_cache:
                    del self.query_embedding_cache[k]
                if k in self.cache_timestamps:
                    del self.cache_timestamps[k]
                    
            # Use cached embedding if available
            if normalized_query in self.query_embedding_cache:
                logger.info(f"Using cached embedding for query: {normalized_query[:30]}...")
                documents = self.retriever.get_relevant_documents(query)
            else:
                # Generate new embedding
                documents = self.retriever.get_relevant_documents(query)
                # Cache for future use
                self.query_embedding_cache[normalized_query] = True
                self.cache_timestamps[normalized_query] = current_time
            
            context_docs = []
            for doc in documents:
                context_docs.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata
                })
            
            logger.info(f"Retrieved {len(context_docs)} context documents")
            return context_docs
        
        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            return []
    
    def optimize_context(self, context_docs: List[Dict], query: str) -> List[Dict]:
        """Optimize context to reduce token usage"""
        if not context_docs:
            return []
        
        # Estimate current token usage
        total_context_tokens = sum(
            self.usage_tracker.estimate_tokens(doc["content"]) 
            for doc in context_docs
        )
        
        # If context is already small, return as is
        if total_context_tokens < 2000:
            return context_docs
        
        # Simple relevance scoring - count word overlap
        query_words = set(query.lower().split())
        
        # Score each document
        scored_docs = []
        for doc in context_docs:
            doc_words = set(doc["content"].lower().split())
            overlap = len(query_words.intersection(doc_words))
            scored_docs.append((doc, overlap))
        
        # Sort by relevance score
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Take the most relevant documents until we reach a reasonable token count
        optimized_docs = []
        current_tokens = 0
        target_tokens = 2000  # Reduced target for faster processing
        
        for doc, score in scored_docs:
            doc_tokens = self.usage_tracker.estimate_tokens(doc["content"])
            
            if current_tokens + doc_tokens <= target_tokens:
                optimized_docs.append(doc)
                current_tokens += doc_tokens
            else:
                # If we're not at 50% capacity yet, add a truncated version
                if current_tokens < target_tokens * 0.5:
                    truncated_content = doc["content"][:500] + "..."  # Simple truncation
                    truncated_doc = {**doc}
                    truncated_doc["content"] = truncated_content
                    optimized_docs.append(truncated_doc)
                break
        
        logger.info(f"Optimized context from {len(context_docs)} to {len(optimized_docs)} documents")
        return optimized_docs
    
    def check_cache(self, query: str) -> Optional[str]:
        """Check if we have a cached response for this query"""
        # Normalize query for cache lookup
        normalized_query = query.lower().strip()
        
        if normalized_query in self.cache:
            logger.info("Cache hit - returning cached response")
            return self.cache[normalized_query]
        
        return None
    
    def update_cache(self, query: str, response: str):
        """Update the cache with a new response"""
        # Normalize query for cache storage
        normalized_query = query.lower().strip()
        
        # Store in cache
        self.cache[normalized_query] = response
        self.cache_timestamps[normalized_query] = time.time()
        
        # Limit cache size to prevent memory issues
        if len(self.cache) > 1000:
            # Remove oldest entry (simple approach)
            oldest_key = min(self.cache_timestamps.items(), key=lambda x: x[1])[0]
            if oldest_key in self.cache:
                del self.cache[oldest_key]
            if oldest_key in self.cache_timestamps:
                del self.cache_timestamps[oldest_key]
    
    def select_model(self, query: str, context_length: int) -> str:
        """Select the appropriate model based on query and budget"""
        # Check if we're approaching the budget limit
        usage = self.usage_tracker.get_usage_summary()
        
        if usage["last_24h"]["cost"] > DAILY_BUDGET * 0.8:
            # If at 80% of budget, use the more economical model
            logger.info("Near budget limit - using standard model")
            return "deepseek-chat"
        
        # Analyze query complexity
        complexity_indicators = [
            "why", "how", "explain", "analyze", "compare", 
            "difference", "recommend", "best", "worst"
        ]
        
        # Check for complexity indicators in the query
        complex_query = any(indicator in query.lower() for indicator in complexity_indicators)
        
        # If query is complex and context is large, use the reasoning model
        if complex_query and context_length > 1000:
            logger.info("Using reasoning model for complex query")
            return "deepseek-reasoner"
        
        # Default to the standard model
        return "deepseek-chat"
    
    async def get_streaming_response_with_timeout(self, 
                                               query: str,
                                               callback: Optional[Callable[[str], None]] = None,
                                               timeout_seconds: int = 90) -> str:
        """Get a streaming response with a timeout to prevent server hanging"""
        try:
            # Start timing
            start_time = time.time()
            logger.info(f"Starting response generation with {timeout_seconds}s timeout")
            
            # Create a task for the original function
            response_task = asyncio.create_task(
                self.get_streaming_response(query, callback)
            )
            
            # Wait for the response with a timeout
            response = await asyncio.wait_for(response_task, timeout=timeout_seconds)
            
            # Log completion time
            elapsed = time.time() - start_time
            logger.info(f"Response generated in {elapsed:.2f} seconds")
            
            return response
            
        except asyncio.TimeoutError:
            # Handle timeout gracefully
            logger.warning(f"Response timed out after {timeout_seconds} seconds")
            return "I'm sorry, but it took too long to generate a response. Could you try a simpler question or try again later?"
            
        except Exception as e:
            logger.error(f"Error in get_streaming_response_with_timeout: {str(e)}")
            return f"I encountered an error while generating a response: {str(e)}"
    
    async def get_streaming_response(self, 
                                     query: str, 
                                     callback: Optional[Callable[[str], None]] = None) -> str:
        """Get a streaming response from DeepSeek API"""
        # Check cache first
        cached_response = self.check_cache(query)
        if cached_response:
            if callback:
                callback(cached_response)
            return cached_response
        
        # Retrieve context
        context_docs = await self.retrieve_context(query)
        
        if not context_docs:
            return "I'm having trouble retrieving information from my knowledge base. Please try again."
        
        # Optimize context to reduce token usage
        optimized_context = self.optimize_context(context_docs, query)
        
        # Format context for the prompt
        context_text = "\n\n".join([
            f"From {doc['metadata']['title']}:\n{doc['content']}" 
            for doc in optimized_context
        ])
        
        # Format chat history
        history_text = ""
        if self.chat_history:
            history_pairs = []
            for i in range(0, len(self.chat_history), 2):
                if i+1 < len(self.chat_history):
                    q = self.chat_history[i]
                    a = self.chat_history[i+1]
                    history_pairs.append(f"Human: {q}\nAssistant: {a}")
            history_text = "\n".join(history_pairs)
        
        # Select model based on query complexity and budget
        model = self.select_model(query, len(context_text))
        
        # System message
        system_message = f"""
        You are an AI assistant specialized in {CREATOR_NAME}'s fitness and bodybuilding knowledge. 
        Answer the question based ONLY on the following context:
        
        {context_text}
        
        If you don't know the answer based on the context, just say "I don't have enough information about that in my knowledge base." Don't make up answers.
        
        Keep your answers concise and to the point.
        """
        
        # All messages combined for token estimation
        all_text = system_message + history_text + query
        estimated_input_tokens = self.usage_tracker.estimate_tokens(all_text)
        
        # Check daily budget
        usage = self.usage_tracker.get_usage_summary()
        if usage["last_24h"]["cost"] > DAILY_BUDGET:
            logger.warning("Daily budget exceeded - returning error message")
            return "I've reached my daily usage limit. Please try again tomorrow."
        
        try:
            # Set up messages for the API call
            messages = [
                {"role": "system", "content": system_message}
            ]
            
            # Add chat history if it exists
            if history_text:
                messages.append({"role": "user", "content": history_text})
            
            # Add current query
            messages.append({"role": "user", "content": query})
            
            # Make API call with streaming enabled
            response_stream = await asyncio.to_thread(
                lambda: self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.3,
                    max_tokens=500,        # Reduced tokens
                    stream=True,
                    timeout=60             # Explicit timeout for the API call
                )
            )
            
            # Process the streaming response
            full_response = ""
            for chunk in response_stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_response += content
                    
                    # Call the callback if provided
                    if callback:
                        callback(content)
            
            # Get usage data if available
            usage_data = getattr(response_stream, "usage", None)
            if usage_data:
                actual_tokens = {
                    "input": usage_data.prompt_tokens,
                    "output": usage_data.completion_tokens
                }
            else:
                actual_tokens = None
            
            # Log the request
            self.usage_tracker.log_request(
                model=model,
                input_text=all_text,
                output_text=full_response,
                actual_token_counts=actual_tokens
            )
            
            # Format with source citations
            formatted_response = self.format_response_with_citations(full_response, optimized_context)
            
            # Add to chat history
            self.chat_history.append(query)
            self.chat_history.append(formatted_response)
            
            # Limit history length to prevent context window issues
            if len(self.chat_history) > 8:  # Keep last 4 exchanges
                self.chat_history = self.chat_history[-8:]
            
            # Cache the response
            self.update_cache(query, formatted_response)
            
            return formatted_response
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error in DeepSeek API call: {error_msg}")
            
            if "rate limit" in error_msg.lower():
                return "I'm receiving too many requests right now. Please try again in a moment."
            elif "context length" in error_msg.lower():
                return "Your question and context are too long for me to process. Could you ask a shorter question?"
            else:
                return "I encountered an error processing your request. Please try again."
    
    def format_response_with_citations(self, response: str, context_docs: List[Dict]) -> str:
        """Add source citations to the response"""
        if not context_docs:
            return response
        
        citations = "\n\nSources:"
        seen_titles = set()
        
        for doc in context_docs[:3]:  # Limit to first 3 sources
            title = doc['metadata'].get('title', 'Unknown Video')
            url = doc['metadata'].get('url', '')
            
            # Skip duplicates
            if title in seen_titles:
                continue
            seen_titles.add(title)
            
            citations += f"\n- [{title}]({url})"
        
        return response + citations
    
    def clear_history(self):
        """Clear chat history"""
        self.chat_history = []
        logger.info("Chat history cleared")
    
    def get_usage_report(self):
        """Get a usage report"""
        usage = self.usage_tracker.get_usage_summary()
        
        report = f"""
        Usage Report:
        
        Total Requests: {usage['total_requests']}
        Total Tokens: {usage['total_tokens']:,}
        Total Cost: ${usage['total_cost']:.4f}
        
        Last 24 Hours:
        - Requests: {usage['last_24h']['requests']}
        - Tokens: {usage['last_24h']['tokens']:,}
        - Cost: ${usage['last_24h']['cost']:.4f}
        
        Daily Budget: ${DAILY_BUDGET:.2f}
        Remaining Budget: ${max(0, DAILY_BUDGET - usage['last_24h']['cost']):.2f}
        """
        
        return report


# Simple terminal-based streaming chat interface
async def terminal_chat():
    chatbot = YouTubeTranscriptChatbot()
    print(f"\n{CHATBOT_NAME} is ready!")
    print("Commands:")
    print("- Type 'exit' to end the conversation")
    print("- Type 'clear' to reset chat history")
    print("- Type 'usage' to see API usage stats")
    
    while True:
        user_input = input("\nYou: ")
        
        if user_input.lower() in ["exit", "quit"]:
            print("Thank you for using the chatbot!")
            break
        
        if user_input.lower() == "clear":
            chatbot.clear_history()
            print("Chat history cleared.")
            continue
        
        if user_input.lower() == "usage":
            print(chatbot.get_usage_report())
            continue
        
        if not user_input.strip():
            continue
        
        print(f"\n{CHATBOT_NAME}: ", end="", flush=True)
        
        # Define callback to print response chunks as they arrive
        def print_chunk(chunk):
            print(chunk, end="", flush=True)
        
        # Get streaming response with timeout protection
        await chatbot.get_streaming_response_with_timeout(user_input, print_chunk)
        print()  # Add newline after response


if __name__ == "__main__":
    asyncio.run(terminal_chat())