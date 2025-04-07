import os
import json
import asyncio
import re
from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv
import logging

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
    """Standard chat endpoint with full response and timeout protection"""
    chatbot = get_chatbot()  # Will use creator_id in the future
    logger.info(f"Received chat request with query: {request.query[:30]}...")
    
    try:
        # Use the timeout-protected version
        response = await chatbot.get_streaming_response_with_timeout(
            request.query, 
            timeout_seconds=90  # 90 second timeout
        )
        
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
        
        logger.info(f"Successfully generated response for query: {request.query[:30]}...")
        return {
            "response": main_response,
            "sources": sources
        }
    
    except asyncio.TimeoutError:
        logger.warning(f"Request timed out for query: {request.query[:30]}...")
        raise HTTPException(
            status_code=504,
            detail="Request timed out. Please try a simpler question."
        )
    except Exception as e:
        # Better error handling
        error_msg = str(e)
        logger.error(f"Error in chat endpoint: {error_msg}")
        
        if "timeout" in error_msg.lower():
            raise HTTPException(
                status_code=504,
                detail="Request timed out. Please try a simpler question."
            )
        else:
            raise HTTPException(
                status_code=500, 
                detail=f"An error occurred: {error_msg}"
            )

# Replace the existing chat_stream function in app.py

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Streaming chat endpoint for real-time responses with no timeout"""
    chatbot = get_chatbot()
    logger.info(f"Received streaming chat request with query: {request.query[:30]}...")
    
    async def generate():
        queue = asyncio.Queue()
        completion_event = asyncio.Event()
        
        def handle_chunk(chunk):
            queue.put_nowait(chunk)
        
        async def process():
            try:
                # No timeout here - let it run as long as needed
                await chatbot.get_streaming_response(request.query, handle_chunk)
                queue.put_nowait(None)  # Signal completion
                completion_event.set()
            except Exception as e:
                error_msg = str(e)
                logger.error(f"Error in streaming process: {error_msg}")
                await queue.put(f"Error: {error_msg}")
                queue.put_nowait(None)
        
        # Start the task
        asyncio.create_task(process())
        
        # Yield chunks as they arrive - no timeout on the generator
        while True:
            try:
                chunk = await queue.get()
                if chunk is None:
                    break
                yield f"data: {json.dumps({'content': chunk})}\n\n"
            except Exception as e:
                logger.error(f"Error in stream generator: {str(e)}")
                # Fixed version that doesn't use backslashes in nested f-strings
                error_message = "Error during streaming: " + str(e)
                yield f"data: {json.dumps({'content': error_message})}\n\n"
                break
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
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