import os
from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse  # Added HTMLResponse
from fastapi.staticfiles import StaticFiles  # Added this line
from fastapi.templating import Jinja2Templates  # Added this line
import json
import asyncio
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv

# Import chatbot (the complete Module 4 implementation)
from chatbot import YouTubeTranscriptChatbot

# Load environment variables
load_dotenv()

# Initialize FastAPI
app = FastAPI(
    title="YouTube Creator Chatbot",
    description="Chat with your favorite creator's content",
    version="1.0.0"
)

# NEW: Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# NEW: Set up templates
templates = Jinja2Templates(directory="./frontend")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development - restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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

# NEW: HTML routes
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/chat", response_class=HTMLResponse)
async def read_chat(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})

@app.get("/login", response_class=HTMLResponse)
async def read_login(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

# CHANGED: Added /api prefix to all API endpoints
@app.get("/api")
async def api_root():
    return {"status": "online", "message": "YouTube Creator Chatbot API"}

@app.post("/api/chat")
async def chat(request: ChatRequest):
    """Standard chat endpoint with full response"""
    chatbot = get_chatbot()  # Will use creator_id in the future
    
    try:
        response = await chatbot.get_streaming_response(request.query)
        
        # Extract sources from response if present
        sources = []
        response_parts = response.split("Sources:")
        if len(response_parts) > 1:
            main_response = response_parts[0].strip()
            sources_text = response_parts[1].strip()
            
            # Parse markdown links
            import re
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
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    """Streaming chat endpoint for real-time responses"""
    chatbot = get_chatbot()  # Will use creator_id in the future
    
    async def generate():
        queue = asyncio.Queue()
        
        def handle_chunk(chunk):
            queue.put_nowait(chunk)
        
        async def process():
            try:
                await chatbot.get_streaming_response(request.query, handle_chunk)
                queue.put_nowait(None)  # Signal completion
            except Exception as e:
                await queue.put(f"Error: {str(e)}")
                queue.put_nowait(None)
        
        asyncio.create_task(process())
        
        while True:
            chunk = await queue.get()
            if chunk is None:
                break
            yield f"data: {json.dumps({'content': chunk})}\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")

@app.get("/api/health")
async def health_check():
    """Health check endpoint for monitoring"""
    return {"status": "healthy"}

@app.get("/debug")
async def debug():
    import os
    files = os.listdir("./frontend")
    return {
        "current_directory": os.getcwd(),
        "files_in_frontend": files
    }

# Future endpoints for multi-tenant:
# - /creators (list available creators)
# - /auth (authentication)
# - /subscribe (payment processing)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)))