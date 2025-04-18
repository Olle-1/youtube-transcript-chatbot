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
# import secrets # Removed placeholder import
import time
from datetime import datetime, timedelta
from sqlalchemy.orm import Session # Added for sync db session

# Import models and dependencies
from models.db_models import Tenant, get_sync_db # Added get_sync_db
from auth.dependencies import get_current_tenant, get_tenant_retriever, get_tenant_embeddings, get_current_user # Added get_current_user
from auth.utils import create_access_token, ACCESS_TOKEN_EXPIRE_MINUTES # Added JWT utils
from crud.user_crud import authenticate_user # Added DB auth function
import schemas # Added schemas import (assuming schemas.Token exists)
from schemas import user_schemas # Import user_schemas specifically
from langchain.vectorstores.base import VectorStoreRetriever # Added for type hint
from langchain_core.embeddings import Embeddings # Added for type hint


# Import from our chatbot implementation
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

# --- Placeholder Auth Removed ---
# The User BaseModel, fake_users_db, fake_hash_password, active_tokens,
# get_user, and the old get_current_user function were removed.
# A new get_current_active_user dependency using JWT decoding will be needed later (Step 6).

# OAuth2 password bearer scheme (still needed for dependency injection)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")


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

# Login endpoint - Refactored for DB Auth and JWT
@app.post("/auth/token", response_model=schemas.Token) # Use Token schema
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_sync_db) # Use sync session for auth
):
    # Authenticate user against the database using email (form_data.username) and password
    user = authenticate_user(db, email=form_data.username, password=form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    # Note: Add check for user.is_active if you implement that field

    # Determine tenant_id (handle case where user might not have one yet)
    # Using "default" as fallback based on plan. Adjust if None is not allowed or needs specific handling.
    tenant_id = user.tenant_id if user.tenant_id else "default"

    # Create JWT
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.email}, # Use email as JWT subject
        expires_delta=access_token_expires,
        tenant_id=tenant_id # Pass the tenant_id
    )
    return {"access_token": access_token, "token_type": "bearer"}


# Protected endpoint example - Needs refactoring (Step 6)
# The old get_current_user dependency was removed.
# This endpoint will fail until a new dependency using JWT decoding is created.
@app.get("/users/me", response_model=user_schemas.User) # Add response model
async def read_users_me(current_user: user_schemas.User = Depends(get_current_user)): # Add dependency
    """Fetches the profile of the currently authenticated user."""
    # The dependency handles fetching the user from the DB based on the JWT
    return current_user

@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    tenant: Tenant = Depends(get_current_tenant),
    retriever: VectorStoreRetriever = Depends(get_tenant_retriever),
    embeddings: Embeddings = Depends(get_tenant_embeddings)
):
    """Standard chat endpoint - returns full response"""
    try:
        # Get or create session
        session_id = request.session_id or "default"
        session = get_session(session_id)
        
        # Get history from session
        history = session.get("history", [])

        # Get response using injected dependencies and tenant prompt
        response = await chatbot.get_streaming_response(
            query=request.query,
            retriever=retriever,
            embeddings=embeddings,
            history=history,
            tenant_prompt_template=tenant.system_prompt
            # Callback is not needed for non-streaming endpoint
        )

        # Note: History update is removed as chatbot.py doesn't return updated history.
        # This needs to be addressed if session persistence is required.
        # session["history"] = updated_history # Needs updated_history from chatbot
        
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
async def chat_stream(
    request: ChatRequest,
    tenant: Tenant = Depends(get_current_tenant),
    retriever: VectorStoreRetriever = Depends(get_tenant_retriever),
    embeddings: Embeddings = Depends(get_tenant_embeddings)
):
    """Streaming chat endpoint - returns chunks as they're generated"""
    try:
        # Get or create session
        session_id = request.session_id or "default"
        session = get_session(session_id)
        
        # Get history from session
        history = session.get("history", [])
        
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
                    # Pass dependencies and tenant prompt to streaming response
                    response = await chatbot.get_streaming_response(
                        query=request.query,
                        retriever=retriever,
                        embeddings=embeddings,
                        history=history,
                        tenant_prompt_template=tenant.system_prompt,
                        callback=handle_chunk
                    )
                    # Signal end of response
                    queue.put_nowait(None)
                    # Note: History update is removed as chatbot.py doesn't return updated history.
                    # session["history"] = updated_history # Needs updated_history from chatbot
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