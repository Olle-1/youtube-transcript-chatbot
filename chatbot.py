# chatbot.py (Corrected Version)
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
# Removed PineconeVectorStore, OpenAIEmbeddings, Pinecone imports as they are handled by dependencies now
# from langchain_pinecone import PineconeVectorStore
# from langchain_openai import OpenAIEmbeddings
# from pinecone import Pinecone
from langchain.vectorstores.base import VectorStoreRetriever # Added for type hint
from langchain_core.embeddings import Embeddings # Added for type hint

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
# PINECONE_API_KEY = os.getenv("PINECONE_API_KEY") # Removed, handled by tenant config

# Constants
# INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "youtube-transcript-mountaindog1") # Removed, handled by tenant config
CREATOR_NAME = os.getenv("CREATOR_NAME", "MountainDog1") # Use env var or default
CHATBOT_NAME = f"{CREATOR_NAME} Assistant"
DAILY_BUDGET = float(os.getenv("DAILY_BUDGET", 1.0)) # Use env var or default

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
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")  # Similar to GPT-4 tokenizer
        except Exception as e:
            logger.error(f"Failed to load tiktoken tokenizer: {e}. Falling back to simple split.")
            self.tokenizer = None # Fallback

    def _load_usage_log(self) -> Dict:
        """Load the usage log from file or create a new one"""
        if os.path.exists(self.log_file):
            try:
                with open(self.log_file, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                logger.warning(f"Error loading usage log {self.log_file}. Creating new log.")
        # Initialize new log structure
        return {
            "total_tokens": {"input": 0, "output": 0},
            "total_cost": 0.0,
            "requests": []
        }

    def _save_usage_log(self):
        """Save the usage log to file"""
        try:
            with open(self.log_file, 'w') as f:
                json.dump(self.usage_log, f, indent=2)
        except IOError as e:
            logger.error(f"Failed to save usage log to {self.log_file}: {e}")

    def estimate_tokens(self, text: str) -> int:
        """Estimate the number of tokens in a text string"""
        if self.tokenizer:
            try:
                return len(self.tokenizer.encode(text))
            except Exception as e:
                logger.warning(f"Tiktoken encoding failed: {e}. Using simple split.")
                return len(text.split()) # Simple fallback
        else:
            return len(text.split()) # Simple fallback

    def log_request(self,
                    model: str,
                    input_text: str,
                    output_text: str,
                    actual_token_counts: Optional[Dict[str, int]] = None):
        """Log an API request with token counts and costs"""
        if actual_token_counts:
            input_tokens = actual_token_counts.get("input", 0)
            output_tokens = actual_token_counts.get("output", 0)
        else:
            input_tokens = self.estimate_tokens(input_text)
            output_tokens = self.estimate_tokens(output_text)

        # Calculate cost
        model_pricing = self.pricing.get(model, self.pricing["deepseek-chat"]) # Default to chat if model unknown
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
        total_requests = len(self.usage_log.get("requests", []))
        total_input_tokens = self.usage_log.get("total_tokens", {}).get("input", 0)
        total_output_tokens = self.usage_log.get("total_tokens", {}).get("output", 0)
        total_tokens = total_input_tokens + total_output_tokens
        total_cost = self.usage_log.get("total_cost", 0.0)

        # Get usage for the last 24 hours
        current_time = datetime.now()
        last_24h_requests = []
        last_24h_cost = 0.0
        last_24h_tokens = 0

        for request in self.usage_log.get("requests", []):
            try:
                request_time = datetime.fromisoformat(request["timestamp"])
                time_diff = (current_time - request_time).total_seconds()
                if time_diff <= 86400:  # 24 hours in seconds
                    last_24h_requests.append(request)
                    last_24h_cost += request.get("cost", {}).get("total", 0.0)
                    last_24h_tokens += request.get("tokens", {}).get("total", 0)
            except (ValueError, TypeError) as e:
                 logger.warning(f"Skipping invalid request log entry during summary: {e} - {request}")


        return {
            "total_requests": total_requests,
            "total_tokens": total_tokens,
            "total_cost": total_cost,
            "last_24h": {
                "requests": len(last_24h_requests),
                "tokens": last_24h_tokens,
                "cost": last_24h_cost
            }
        }


class YouTubeTranscriptChatbot:
    """Complete chatbot implementation with DeepSeek integration"""

    def __init__(self):
        if not DEEPSEEK_API_KEY:
            raise ValueError("DEEPSEEK_API_KEY environment variable not set")
        # Removed PINECONE_API_KEY check, tenant config handles this
        # Removed OPENAI_API_KEY check, dependency handles this
        # if not PINECONE_API_KEY:
        #     raise ValueError("PINECONE_API_KEY environment variable not set")
        # if not OPENAI_API_KEY:
        #     raise ValueError("OPENAI_API_KEY environment variable not set for embeddings")

        self.client = OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url="https://api.deepseek.com/v1"
        )
        # self.pc = Pinecone(api_key=PINECONE_API_KEY) # Removed, use dependency
        # self.embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY) # Removed, use dependency
        # self.chat_history = [] # Removed, history will be passed per request
        self.max_retry_attempts = 3
        self.retry_delay = 2  # seconds

        # Initialize usage tracking
        self.usage_tracker = UsageTracker()

        # Initialize cache (Query embedding cache only for now)
        # self.cache = {} # Full response cache disabled due to history complexity
        self.query_embedding_cache = {}
        self.cache_ttl = 3600  # Cache TTL in seconds (1 hour)
        self.cache_timestamps = {}

        # Connect to the retriever - Removed, retriever is now injected per request
        # self.retriever = self._initialize_retriever()

        logger.info(f"Initialized {CHATBOT_NAME} with DeepSeek API (Retriever/Embeddings are injected per request)")

    # Removed _initialize_retriever method, dependency handles this
    async def get_embedding(self, text: str, embeddings: Embeddings) -> List[float]: # Added embeddings dependency
        """Get embedding with retry logic"""
        for attempt in range(self.max_retry_attempts):
            try:
                # Run synchronous embedding function in a separate thread
                return await asyncio.to_thread(embeddings.embed_query, text) # Use injected embeddings
            except Exception as e:
                if attempt < self.max_retry_attempts - 1:
                    wait_time = self.retry_delay * (2 ** attempt)
                    logger.warning(f"Embedding error: {e}. Retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"Failed to generate embedding after {self.max_retry_attempts} attempts: {e}", exc_info=True)
                    raise RuntimeError(f"Failed to generate embedding: {e}")

    async def retrieve_context(self, query: str, retriever: VectorStoreRetriever, embeddings: Embeddings) -> List[Dict]: # Added retriever and embeddings dependencies
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

            # Use cached embedding if available (Note: This doesn't cache the retrieval itself)
            # Retrieval caching is complex with MMR, so we only cache the query embedding step
            if normalized_query not in self.query_embedding_cache:
                 # Generate and cache embedding if not present
                 # Run synchronous embedding in thread pool
                 # Use the injected embeddings object now
                 _ = await self.get_embedding(query, embeddings) # Pass embeddings
                 self.query_embedding_cache[normalized_query] = True # Mark as generated
                 self.cache_timestamps[normalized_query] = current_time
                 logger.info(f"Generated and cached embedding for query: {normalized_query[:30]}...")
            else:
                 logger.info(f"Using cached embedding status for query: {normalized_query[:30]}...")


            # Perform retrieval using Langchain retriever in thread pool
            # Use the injected retriever object now
            documents = await asyncio.to_thread(retriever.get_relevant_documents, query) # Use injected retriever

            context_docs = []
            for doc in documents:
                # Ensure metadata exists and has expected keys
                metadata = doc.metadata or {}
                context_docs.append({
                    "content": doc.page_content or "",
                    "metadata": {
                        "title": metadata.get("title", "Unknown Source"),
                        "url": metadata.get("url", "#")
                        # Add other relevant metadata fields if available
                    }
                })

            logger.info(f"Retrieved {len(context_docs)} context documents for query: {query[:50]}...")
            return context_docs

        except Exception as e:
            logger.error(f"Error retrieving context: {e}", exc_info=True)
            return [] # Return empty list on error

    def optimize_context(self, context_docs: List[Dict], query: str) -> List[Dict]:
        """Optimize context to reduce token usage (simple relevance scoring)."""
        if not context_docs:
            return []

        # Estimate current token usage
        total_context_tokens = sum(
            self.usage_tracker.estimate_tokens(doc.get("content", ""))
            for doc in context_docs
        )

        # Target token count for context
        target_tokens = 3000 # Adjust as needed based on model limits and desired performance

        # If context is already reasonably small, return as is
        if total_context_tokens < target_tokens * 1.2: # Allow slightly over target
            return context_docs

        # Simple relevance scoring - count word overlap with query
        query_words = set(query.lower().split())

        scored_docs = []
        for i, doc in enumerate(context_docs):
            doc_content = doc.get("content", "").lower()
            doc_words = set(doc_content.split())
            overlap = len(query_words.intersection(doc_words))
            # Add a small bonus for earlier retrieved docs (assuming retriever has some relevance ordering)
            score = overlap + (len(context_docs) - i) * 0.1
            scored_docs.append((doc, score))

        # Sort by relevance score (descending)
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        # Take the most relevant documents until we reach the target token count
        optimized_docs = []
        current_tokens = 0

        for doc, score in scored_docs:
            doc_content = doc.get("content", "")
            doc_tokens = self.usage_tracker.estimate_tokens(doc_content)

            if current_tokens + doc_tokens <= target_tokens:
                optimized_docs.append(doc)
                current_tokens += doc_tokens
            else:
                # Optional: Add a truncated version if still far below target
                if current_tokens < target_tokens * 0.5:
                    remaining_tokens = target_tokens - current_tokens
                    # Estimate chars per token (very rough)
                    chars_per_token = 4
                    estimated_chars = int(remaining_tokens * chars_per_token) # Ensure int
                    truncated_content = doc_content[:estimated_chars] + "..."
                    truncated_doc = {**doc, "content": truncated_content}
                    optimized_docs.append(truncated_doc)
                    current_tokens += self.usage_tracker.estimate_tokens(truncated_content)
                break # Stop adding docs once target is reached or exceeded

        logger.info(f"Optimized context from {len(context_docs)} to {len(optimized_docs)} documents ({current_tokens} tokens)")
        return optimized_docs

    # Caching is disabled for now due to history complexity
    # def check_cache(self, query: str) -> Optional[str]: ...
    # def update_cache(self, query: str, response: str): ...

    def select_model(self, query: str, context_length: int) -> str:
        """Select the appropriate model based on query and budget"""
        usage = self.usage_tracker.get_usage_summary()

        if usage["last_24h"]["cost"] > DAILY_BUDGET * 0.9: # Check if near 90% budget
            logger.warning("Near daily budget limit - using most economical model (deepseek-chat)")
            return "deepseek-chat"

        # Analyze query complexity (simple keyword check)
        complexity_indicators = [
            "why", "how", "explain", "analyze", "compare",
            "difference", "recommend", "best", "worst", "summarize"
        ]
        complex_query = any(indicator in query.lower() for indicator in complexity_indicators)

        # Use reasoning model for complex queries or very large contexts
        # Adjust threshold as needed
        if complex_query or context_length > 4000:
            logger.info("Using reasoning model (deepseek-reasoner) for complex query or large context")
            return "deepseek-reasoner"

        # Default to the standard chat model
        return "deepseek-chat"

    async def get_streaming_response(self,
                                 query: str,
                                 retriever: VectorStoreRetriever, # Added retriever dependency
                                 embeddings: Embeddings, # Added embeddings dependency
                                 history: List[Dict[str, str]] = [], # History passed from caller
                                 tenant_prompt_template: Optional[str] = None, # Added tenant prompt template
                                 callback: Optional[Callable[[Any], None]] = None) -> str:
        """
        Get a streaming response from DeepSeek API, incorporating chat history.
        The callback function will receive chunks of the response content.
        Returns the full, raw response content string.
        """
        # 1. Retrieve context
        context_docs = await self.retrieve_context(query, retriever, embeddings) # Pass dependencies
        if not context_docs:
            # Handle case where no context is found - maybe a direct response?
            logger.warning(f"No context retrieved for query: {query[:50]}...")
            # For now, return a specific message, could try direct LLM call later
            no_context_message = "I couldn't find specific information about that in the knowledge base."
            if callback:
                callback(no_context_message)
            return no_context_message # Return immediately

        # 2. Optimize context
        optimized_context = self.optimize_context(context_docs, query)

        # 3. Format prompt with system message, history, and context
        context_text = "\n\n".join([
            # Include source URL in context text if available
            f"Source: {doc['metadata'].get('url', 'N/A')}\nContent: {doc.get('content', '')}"
            for doc in optimized_context
        ])

        # Determine the prompt template to use
        prompt_template = None
        if tenant_prompt_template and tenant_prompt_template.strip():
            logger.info("Using tenant-specific system prompt template.")
            prompt_template = tenant_prompt_template
        else:
            logger.info("Tenant prompt not set, loading default from config/system_prompt.txt.")
            try:
                with open("config/system_prompt.txt", "r", encoding="utf-8") as f:
                    prompt_template = f.read()
            except FileNotFoundError:
                logger.error("Default system prompt file 'config/system_prompt.txt' not found. Using basic fallback.")
                # Basic fallback if file is missing
                prompt_template = "You are a helpful AI assistant for {CREATOR_NAME}. Answer based on the context:\nContext:\n---\n{context_text}\n---"

        # Format the chosen prompt template
        try:
            # Attempt to format with both placeholders, tenant prompts might not have CREATOR_NAME
            system_message = prompt_template.format(CREATOR_NAME=CREATOR_NAME, context_text=context_text)
        except KeyError as e:
            logger.warning(f"Placeholder {e} missing in the chosen prompt template. Attempting format with only context_text.")
            try:
                 # Fallback: Try formatting only with context_text if CREATOR_NAME caused error
                 system_message = prompt_template.format(context_text=context_text)
            except KeyError as e2:
                 logger.error(f"Placeholder {e2} also missing in the chosen prompt template. Using template as is with context appended.")
                 # Final fallback: Use the template string directly and append context separately
                 system_message = f"{prompt_template}\n\nContext:\n---\n{context_text}\n---"
        except Exception as format_exc:
             logger.error(f"Unexpected error formatting prompt template: {format_exc}. Using basic fallback.")
             system_message = f"You are a helpful AI assistant. Answer based on the context:\nContext:\n---\n{context_text}\n---"

        messages = [{"role": "system", "content": system_message}]

        # Add chat history (passed as argument)
        if history:
             # Limit history length (e.g., last 5 exchanges = 10 messages)
             max_history_messages = 10
             limited_history = history[-max_history_messages:]
             # Ensure format is correct before extending
             formatted_history = [{"role": msg.get("role"), "content": msg.get("content")}
                                  for msg in limited_history if msg.get("role") and msg.get("content")]
             messages.extend(formatted_history)

        # Add current query as the last user message
        messages.append({"role": "user", "content": query})

        # 4. Estimate tokens and check budget
        all_text_for_estimation = system_message + " ".join([msg["content"] for msg in messages if msg.get("role") != "system"])
        estimated_input_tokens = self.usage_tracker.estimate_tokens(all_text_for_estimation)
        logger.info(f"Estimated input tokens: {estimated_input_tokens}")

        usage = self.usage_tracker.get_usage_summary()
        if usage["last_24h"]["cost"] > DAILY_BUDGET:
            logger.warning("Daily budget exceeded - returning error message")
            budget_exceeded_msg = "I've reached my daily usage limit. Please try again tomorrow."
            if callback:
                 callback(budget_exceeded_msg)
            return budget_exceeded_msg # Return immediately

        # 5. Select model and make API call
        try:
            model = self.select_model(query, len(context_text)) # Use context_text length for selection heuristic
            logger.info(f"Using model: {model}")

            # Run synchronous SDK call in thread pool
            response_stream = await asyncio.to_thread(
                lambda: self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.3, # Lower temperature for more factual responses
                    max_tokens=1500, # Adjust based on expected response length vs cost
                    stream=True
                )
            )

            full_response = ""
            usage_info = {}

            # 6. Process the stream
            # Note: Iterating over the sync stream iterator directly might block.
            # Consider wrapping the iteration in asyncio.to_thread if issues arise.
            for chunk in response_stream: # This might block if client doesn't handle async iteration well
                # Check for content
                content_chunk = chunk.choices[0].delta.content if chunk.choices else None
                if content_chunk:
                    full_response += content_chunk
                    if callback:
                        callback(content_chunk) # Stream raw content chunk

                # Check for usage data (might only appear in the last chunk)
                if hasattr(chunk, 'usage') and chunk.usage:
                     usage_info = {
                         "input": chunk.usage.prompt_tokens,
                         "output": chunk.usage.completion_tokens
                     }

            # 7. Log request
            request_log = self.usage_tracker.log_request(
                model=model,
                input_text=all_text_for_estimation, # Use estimated text
                output_text=full_response,
                actual_token_counts=usage_info if usage_info else None
            )
            logger.info(f"Logged request: {request_log['tokens']} tokens, cost ${request_log['cost']:.6f}")

            # 8. Return the raw, complete response string
            # Citation formatting and saving is handled by the caller (app.py)
            return full_response

        except Exception as e:
            logger.error(f"Error during DeepSeek API call: {e}", exc_info=True)
            # Propagate error so the calling endpoint can handle it
            raise e # Re-raise the exception

    # Removed format_response_with_citations - caller handles source parsing/saving
    # def format_response_with_citations(...)

    # Removed clear_history as history is managed per session externally

    def get_usage_report(self):
        """Get a usage report"""
        summary = self.usage_tracker.get_usage_summary()
        report = f"""
Usage Report ({datetime.now().isoformat()}):
-----------------------------------------
Total Requests: {summary['total_requests']}
Total Tokens:   {summary['total_tokens']}
Total Cost:     ${summary['total_cost']:.4f}

Last 24 Hours:
  Requests: {summary['last_24h']['requests']}
  Tokens:   {summary['last_24h']['tokens']}
  Cost:     ${summary['last_24h']['cost']:.4f}
  Budget Remaining: ${max(0, DAILY_BUDGET - summary['last_24h']['cost']):.4f}
-----------------------------------------
"""
        return report


# Example for direct testing (optional)
async def terminal_chat():
    print(f"Starting {CHATBOT_NAME} terminal chat...")
    chatbot = YouTubeTranscriptChatbot()
    session_history = [] # Simple list of dicts for terminal example

    while True:
        try:
            query = await asyncio.to_thread(input, "You: ") # Run input in thread
        except EOFError:
            break # Handle Ctrl+D

        if query.lower() in ["quit", "exit"]:
            break
        if query.lower() == "/usage":
            print(chatbot.get_usage_report())
            continue

        print("Assistant: ", end="", flush=True)
        full_res = ""
        try:
            # Define a simple callback for terminal printing
            def print_chunk(chunk):
                print(chunk, end="", flush=True)

            # Call the main streaming function with history
            full_res = await chatbot.get_streaming_response(query, session_history, print_chunk)
            print() # Newline after streaming finishes

            # Update terminal history
            session_history.append({"role": "user", "content": query})
            # Basic parsing to remove potential source string before adding to history
            response_parts = full_res.split("(Source:")
            clean_response = response_parts[0].strip()
            session_history.append({"role": "assistant", "content": clean_response})

            # Limit history
            if len(session_history) > 10:
                session_history = session_history[-10:]

        except Exception as e:
            print(f"\nError: {e}")

    print("\nExiting chat.")

if __name__ == "__main__":
    # Run the terminal chat example
    try:
        asyncio.run(terminal_chat())
    except KeyboardInterrupt:
        print("\nExiting...")