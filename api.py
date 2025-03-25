import os
import json
import time
import asyncio
import logging
from typing import List, Dict, Optional

from fastapi import FastAPI, HTTPException, Depends, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException, Depends, status, Cookie
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import secrets
import time
from datetime import datetime, timedelta

# Import from our chatbot implementation
# This assumes final-integration.py is saved as youtube_chatbot.py
from chatbot import YouTubeTranscriptChatbot

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ChatbotAPI")

# Load environment variables
load_dotenv()

# Create FastAPI app
app = FastAPI(
    title="YouTube Transcript Chatbot API",
    description="API for accessing chatbot capabilities based on YouTube creator transcripts",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development - restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create a global chatbot instance
chatbot = YouTubeTranscriptChatbot()

# Pydantic models for request/response
class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    sources: List[Dict[str, str]]

class UsageReportResponse(BaseModel):
    total_requests: int
    total_tokens: int
    total_cost: float
    last_24h_requests: int
    last_24h_tokens: int
    last_24h_cost: float
    daily_budget: float
    remaining_budget: float

# Simple user model for authentication
class User(BaseModel):
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    disabled: Optional[bool] = None

# Simple in-memory user database
fake_users_db = {
    "testuser": {
        "username": "testuser",
        "email": "test@example.com",
        "full_name": "Test User",
        "hashed_password": "fakehashednewpass123",
        "disabled": False,
    }
}

# Simple authentication
def fake_hash_password(password: str):
    return "fakehashed" + password

# Very simple token database - in memory (not persistent)
active_tokens = {}

# Function to get user based on username
def get_user(db, username: str):
    if username in db:
        user_dict = db[username]
        return User(**user_dict)

# OAuth2 password bearer scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")

# Function to get current user from token
def get_current_user(token: str = Depends(oauth2_scheme)):
    if token not in active_tokens:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    token_data = active_tokens[token]
    
    # Check if token has expired
    if token_data["expires"] < time.time():
        del active_tokens[token]
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    user = get_user(fake_users_db, token_data["user"])
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return user

# Simple session store - would use a real database in production
session_store = {}

def get_session(session_id: str):
    """Get or create a session"""
    if session_id not in session_store:
        session_store[session_id] = {
            "history": [],
            "created_at": time.time(),
            "last_access": time.time()
        }
    else:
        session_store[session_id]["last_access"] = time.time()
    
    return session_store[session_id]

# Cleanup old sessions periodically
async def cleanup_sessions():
    """Remove sessions older than 1 hour"""
    while True:
        current_time = time.time()
        session_ids = list(session_store.keys())
        
        for session_id in session_ids:
            session = session_store[session_id]
            if current_time - session["last_access"] > 3600:  # 1 hour
                del session_store[session_id]
                logger.info(f"Cleaned up inactive session: {session_id}")
        
        await asyncio.sleep(300)  # Check every 5 minutes

@app.on_event("startup")
async def startup_event():
    """Start background tasks on startup"""
    asyncio.create_task(cleanup_sessions())

@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "status": "online",
        "message": "YouTube Transcript Chatbot API is running"
    }

# Login endpoint
@app.post("/auth/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user_dict = fake_users_db.get(form_data.username)
    if not user_dict:
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    
    user = User(**user_dict)
    
    if user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    
    hashed_password = fake_hash_password(form_data.password)
    if not hashed_password == user_dict["hashed_password"]:
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    
    # Generate a token
    token = secrets.token_hex(16)
    active_tokens[token] = {
        "user": user.username,
        "expires": time.time() + 86400  # 24 hours from now
    }
    
    return {"access_token": token, "token_type": "bearer"}

# Protected endpoint example
@app.get("/users/me")
async def read_users_me(current_user: User = Depends(get_current_user)):
    return current_user

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Standard chat endpoint - returns full response"""
    try:
        # Get or create session
        session_id = request.session_id or "default"
        session = get_session(session_id)
        
        # Set chatbot history from session if it exists
        if session["history"]:
            chatbot.chat_history = session["history"]
        
        # Get response
        response = await chatbot.get_streaming_response(request.query)
        
        # Update session history
        session["history"] = chatbot.chat_history
        
        # Extract sources from response
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
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process chat request: {str(e)}"
        )

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Streaming chat endpoint - returns chunks as they're generated"""
    try:
        # Get or create session
        session_id = request.session_id or "default"
        session = get_session(session_id)
        
        # Set chatbot history from session if it exists
        if session["history"]:
            chatbot.chat_history = session["history"]
        
        # Create an async generator for streaming
        async def response_generator():
            # Queue for passing chunks between coroutines
            queue = asyncio.Queue()
            
            # Callback for handling chunks
            def handle_chunk(chunk):
                queue.put_nowait(chunk)
            
            # Create task for getting response
            async def get_response():
                try:
                    response = await chatbot.get_streaming_response(request.query, handle_chunk)
                    # Signal end of response
                    queue.put_nowait(None)
                    # Update session history
                    session["history"] = chatbot.chat_history
                except Exception as e:
                    logger.error(f"Error in streaming response: {str(e)}")
                    await queue.put("Error: " + str(e))
                    queue.put_nowait(None)
            
            # Start the task
            asyncio.create_task(get_response())
            
            # Yield chunks as they arrive
            while True:
                chunk = await queue.get()
                if chunk is None:
                    break
                yield f"data: {json.dumps({'content': chunk})}\n\n"
        
        return StreamingResponse(
            response_generator(),
            media_type="text/event-stream"
        )
    
    except Exception as e:
        logger.error(f"Error in streaming endpoint: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process streaming request: {str(e)}"
        )

@app.post("/chat/clear")
async def clear_chat(request: ChatRequest):
    """Clear chat history for a session"""
    try:
        session_id = request.session_id or "default"
        
        if session_id in session_store:
            session_store[session_id]["history"] = []
            logger.info(f"Cleared history for session: {session_id}")
        
        return {"status": "success", "message": "Chat history cleared"}
    
    except Exception as e:
        logger.error(f"Error clearing chat history: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to clear chat history: {str(e)}"
        )

@app.get("/usage", response_model=UsageReportResponse)
async def usage_report():
    """Get API usage statistics"""
    try:
        usage = chatbot.usage_tracker.get_usage_summary()
        daily_budget = getattr(chatbot, "DAILY_BUDGET", 1.0)
        
        return {
            "total_requests": usage["total_requests"],
            "total_tokens": usage["total_tokens"],
            "total_cost": usage["total_cost"],
            "last_24h_requests": usage["last_24h"]["requests"],
            "last_24h_tokens": usage["last_24h"]["tokens"],
            "last_24h_cost": usage["last_24h"]["cost"],
            "daily_budget": daily_budget,
            "remaining_budget": max(0, daily_budget - usage["last_24h"]["cost"])
        }
    
    except Exception as e:
        logger.error(f"Error generating usage report: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate usage report: {str(e)}"
        )

# Run with: uvicorn api:app --reload
# For production: gunicorn -k uvicorn.workers.UvicornWorker -w 4 api:app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)