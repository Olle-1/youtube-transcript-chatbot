import os
import json
import asyncio
import re
import time
import logging
from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Optional, List, Dict
from dotenv import load_dotenv

# Import chatbot
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
logger = logging.getLogger("LiftingChat")

# Load environment variables
load_dotenv()

# Initialize FastAPI
app = FastAPI(
    title="YouTube Creator Chatbot",
    description="Chat with your favorite creator's content",
    version="1.0.0"
)

# Enhanced CORS middleware with explicit origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://velvety-lollipop-8fbc93.netlify.app",  # Your Netlify app URL
        "https://liftingchat.com",                      # Your main domain
        "https://www.liftingchat.com",                  # www subdomain
        "*"                                             # Allow all origins for testing (remove in production)
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Add request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    request_id = f"{time.time()}-{id(request)}"
    
    logger.info(f"[{request_id}] Request: {request.method} {request.url.path}")
    
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        logger.info(f"[{request_id}] Response: {response.status_code} (took {process_time:.3f}s)")
        return response
    except Exception as e:
        logger.error(f"[{request_id}] Error processing request: {str(e)}")
        raise

# Request models
class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = None

# Response models
class ChatResponse(BaseModel):
    response: str
    sources: List[Dict[str, str]] = []

# In-memory chatbot instance
chatbot_instance = YouTubeTranscriptChatbot()

# Helper function for future multi-tenant setup
def get_chatbot(creator_id: str = "mountaindog1"):
    """
    Get the appropriate chatbot instance based on creator ID.
    """
    return chatbot_instance

# Root endpoint
@app.get("/")
async def api_root():
    return {"status": "online", "message": "YouTube Creator Chatbot API"}

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    return {"status": "healthy"}

# Standard chat endpoint
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Standard chat endpoint with full response"""
    chatbot = get_chatbot()
    
    try:
        # Set a timeout for the AI response
        async with asyncio.timeout(90):  # 90 second timeout
            response = await chatbot.get_streaming_response(request.query)
        
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
        
        return {
            "response": main_response,
            "sources": sources
        }
    
    except asyncio.TimeoutError:
        logger.error(f"Response generation timed out for query: {request.query[:100]}...")
        return {
            "response": "I'm sorry, but it's taking me longer than expected to generate a response. Please try asking a simpler question or try again later.",
            "sources": []
        }
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Streaming chat endpoint
@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Streaming chat endpoint for real-time responses"""
    chatbot = get_chatbot()
    
    async def generate():
        queue = asyncio.Queue()
        completed = False
        
        def handle_chunk(chunk):
            queue.put_nowait({"type": "chunk", "content": chunk})
        
        async def process():
            nonlocal completed
            try:
                # Set a timeout for AI response
                async with asyncio.timeout(120):  # 2 minute timeout
                    await chatbot.get_streaming_response(request.query, handle_chunk)
            except asyncio.TimeoutError:
                await queue.put({"type": "error", "content": "Response generation timed out. Please try a simpler question."})
            except Exception as e:
                logger.error(f"Error in streaming response: {str(e)}")
                await queue.put({"type": "error", "content": f"Error: {str(e)}"})
            finally:
                # Always signal completion
                await queue.put({"type": "done"})
                completed = True
        
        # Start the processing task
        task = asyncio.create_task(process())
        
        # Yield chunks with proper SSE format
        try:
            while not completed or not queue.empty():
                try:
                    # Use wait_for to prevent blocking forever
                    message = await asyncio.wait_for(queue.get(), timeout=1.0)
                    
                    if message["type"] == "done":
                        # End of stream
                        yield "event: done\ndata: {}\n\n"
                        break
                    elif message["type"] == "error":
                        # Error message
                        yield f"event: error\ndata: {json.dumps({'content': message['content']})}\n\n"
                    else:
                        # Regular chunk
                        yield f"data: {json.dumps({'content': message['content']})}\n\n"
                        
                    # Small delay to prevent overwhelming the client
                    await asyncio.sleep(0.01)
                except asyncio.TimeoutError:
                    # No message available, yield a heartbeat to keep connection alive
                    yield "event: heartbeat\ndata: {}\n\n"
        except asyncio.CancelledError:
            # Client disconnected
            logger.info("Client disconnected from stream")
            task.cancel()
            raise
        finally:
            # Ensure task is properly cleaned up
            if not task.done():
                task.cancel()
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable Nginx buffering
        }
    )

# Run with: uvicorn app:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)))