import os
import json
import asyncio
import re
import time
import logging
from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv

# Import chatbot (the complete Module 4 implementation)
from chatbot import YouTubeTranscriptChatbot

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("AppAPI")

# Load environment variables
load_dotenv()

# Initialize FastAPI
app = FastAPI(
    title="YouTube Creator Chatbot",
    description="Chat with your favorite creator's content",
    version="1.0.0"
)

# Define allowed origins for CORS
allowed_origins = [
    "https://velvety-lollipop-8fbc93.netlify.app",  # Your Netlify domain
    "https://liftingchat.com",
    "https://www.liftingchat.com",
    "http://localhost:3000",  # For local development
    "http://localhost:8000",  # For local development
    "*"  # Allow all origins during development - remove this in production
]

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    max_age=86400,  # Cache preflight requests for 24 hours
)

# Request models
class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    # For future multi-tenant: creator_id: Optional[str] = None

# In-memory chatbot instance - will be replaced with dynamic loading for multi-tenant
chatbot_instance = YouTubeTranscriptChatbot()

# Helper function for future multi-tenant setup
def get_chatbot(creator_id: str = "mountaindog1"):
    """
    For now returns a single chatbot, but designed to be extended for multi-tenant.
    In the future, this will dynamically load the correct chatbot based on creator_id.
    """
    # Future code will get configuration based on creator_id
    return chatbot_instance

# API endpoint root
@app.get("/")
async def api_root():
    return {"status": "online", "message": "YouTube Creator Chatbot API is running"}

@app.post("/chat")
async def chat(request: ChatRequest):
    """Standard chat endpoint with full response"""
    logger.info(f"Received chat request with query: {request.query[:20]}...")
    logger.info(f"Starting response generation with 90s timeout")
    start_time = time.time()
    
    chatbot = get_chatbot()  # Will use creator_id in the future
    
    try:
        # Use asyncio.timeout to enforce a timeout limit
        async with asyncio.timeout(90):
            response = await chatbot.get_streaming_response(request.query)
        
        elapsed = time.time() - start_time
        logger.info(f"Response generated in {elapsed:.2f} seconds")
        
        # Extract sources from response if present
        sources = []
        response_parts = response.split("Sources:")
        if len(response_parts) > 1:
            main_response = response_parts[0].strip()
            sources_text = response_parts[1].strip()
            
            # Parse markdown links
            link_pattern = r'\[([^\]]+)\]\(([^)]+)\)'
            matches = re.findall(link_pattern, sources_text)
            
            for title, url in matches:
                sources.append({"title": title, "url": url})
        else:
            main_response = response
        
        logger.info(f"Successfully generated response for query: {request.query[:20]}...")
        return {
            "response": main_response,
            "sources": sources
        }
    
    except asyncio.TimeoutError:
        logger.error(f"Timeout after {time.time() - start_time:.2f}s for query: {request.query[:20]}...")
        raise HTTPException(status_code=504, detail="Request timed out. Please try a simpler question.")
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Streaming chat endpoint for real-time responses"""
    # Get the appropriate chatbot instance
    chatbot = get_chatbot()  # Will use creator_id in the future
    
    async def generate():
        """Generate streaming response chunks"""
        # Set proper headers for SSE
        yield "retry: 1000\n\n"  # Reconnection time in milliseconds
        
        # Create queue for passing chunks between tasks
        queue = asyncio.Queue()
        
        # Callback function that receives chunks from the chatbot
        def handle_chunk(chunk):
            """Callback to receive chunks from the streaming response"""
            queue.put_nowait(chunk)
        
        # Background task to process the request
        async def process():
            """Process the streaming request"""
            try:
                # Set a timeout for the entire request
                # This is important to prevent hanging connections
                async with asyncio.timeout(90):  # 90 second timeout
                    await chatbot.get_streaming_response(request.query, handle_chunk)
                    # Signal completion
                    queue.put_nowait(None)
            except asyncio.TimeoutError:
                # Handle timeout case
                error_message = "Request timed out after 90 seconds"
                error_data = json.dumps({"content": error_message})
                await queue.put(f"data: {error_data}\n\n")
                queue.put_nowait(None)
            except Exception as e:
                # Handle other errors
                error_message = str(e).replace('\n', ' ')
                error_data = json.dumps({"content": f"Error: {error_message}"})
                await queue.put(f"data: {error_data}\n\n")
                queue.put_nowait(None)
        
        # Start the processing task
        asyncio.create_task(process())
        
        # Stream chunks back to the client
        while True:
            try:
                # Wait for the next chunk
                chunk = await queue.get()
                
                # None signals the end of the stream
                if chunk is None:
                    break
                
                # Format chunk as a server-sent event
                if isinstance(chunk, str):
                    # Simple string content
                    data = json.dumps({"content": chunk})
                    yield f"data: {data}\n\n"
                else:
                    # Already formatted data
                    yield f"data: {json.dumps(chunk)}\n\n"
                
                # Small delay to prevent overwhelming the connection
                await asyncio.sleep(0.01)
                
            except Exception as e:
                # Log any errors that occur during streaming
                logger.error(f"Error during streaming: {e}")
                error_data = json.dumps({"content": "Error during streaming"})
                yield f"data: {error_data}\n\n"
                break
    
    # Return a streaming response with appropriate headers
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "X-Accel-Buffering": "no"  # Prevents proxies from buffering the response
        }
    )

@app.post("/chat/clear")
async def clear_chat(request: ChatRequest):
    """Clear chat history for a session"""
    try:
        chatbot = get_chatbot()
        chatbot.clear_history()
        logger.info(f"Cleared chat history for session: {request.session_id}")
        return {"status": "success", "message": "Chat history cleared"}
    except Exception as e:
        logger.error(f"Error clearing chat history: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to clear chat history: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    return {"status": "healthy"}

# Future endpoints for multi-tenant:
# - /creators (list available creators)
# - /auth (authentication)
# - /subscribe (payment processing)

if __name__ == "__main__":
    import uvicorn
    # Use 8080 as the default port to match DigitalOcean
    uvicorn.run(
        "app:app", 
        host="0.0.0.0", 
        port=int(os.getenv("PORT", 8080)),  # Changed default from 8000 to 8080
        timeout_keep_alive=120,  # Increase from default 5 seconds
        timeout_notify=60,       # Increase notification timeout
        log_level="info"
    )