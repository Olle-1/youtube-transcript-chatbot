import os
import json
import asyncio
import re
import time
import logging
from fastapi import FastAPI, Request, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
# from starlette.middleware.base import BaseHTTPMiddleware # No longer needed with @app.middleware
from fastapi.responses import StreamingResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
# from fastapi.templating import Jinja2Templates # Will add back when serving frontend
from pydantic import BaseModel
from typing import Optional, List, Dict, Any # Added Any for callback type hint if needed later
from datetime import timedelta
from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import AsyncSession

# --- Database, Schemas, CRUD, Auth ---
from models.db_models import get_db, SessionLocal # Import DB session dependency AND Sync SessionLocal
import schemas.user_schemas as user_schemas
import schemas.chat_schemas as chat_schemas # Added
import crud.user_crud as user_crud
import crud.chat_crud as chat_crud # Added
import crud.tenant_crud as tenant_crud # Added Tenant CRUD
import auth.utils as auth_utils
from auth.dependencies import get_current_user, get_retriever, get_openai_embeddings # Added new dependencies
from langchain.vectorstores.base import VectorStoreRetriever # Added for type hint
from langchain_core.embeddings import Embeddings # Added for type hint

# --- Routers ---
from routers import admin # Added Admin Router

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
# --- Tenant Identification Middleware ---
# This middleware runs *after* CORS but *before* logging and routing.
@app.middleware("http")
async def tenant_middleware(request: Request, call_next):
    """
    Identifies the tenant based on the 'tenant_id' claim in the JWT token.
    Attaches tenant_id to request.state for protected routes.
    Excludes specific paths (auth, health, docs, static, root) from this check.
    """
    # Define paths to exclude from tenant check
    excluded_paths = [
        "/auth/login",
        "/auth/register",
        "/health",
        "/docs",          # Swagger UI
        "/openapi.json",  # OpenAPI schema
        "/",              # Root path serving frontend
        # Add other non-tenant-specific paths if needed
    ]
    # Also exclude static files path prefix
    static_path_prefix = "/static" # Matches the app.mount path below

    request_path = request.url.path

    # Check if the request path should be excluded
    if request_path in excluded_paths or request_path.startswith(static_path_prefix):
        logger.debug(f"Tenant check skipped for excluded path: {request_path}")
        response = await call_next(request)
        return response

    # Proceed with tenant identification for other paths
    auth_header = request.headers.get("Authorization")
    token = None
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Not authenticated",
        headers={"WWW-Authenticate": "Bearer"},
    )
    tenant_exception = HTTPException(
        status_code=status.HTTP_403_FORBIDDEN, # Use 403 if token is valid but tenant info is wrong/missing
        detail="Tenant identification failed",
    )

    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header.split("Bearer ")[1]
    else:
        # No token provided for a route that requires tenant identification
        logger.warning(f"Missing Authorization header for protected path: {request_path}")
        raise credentials_exception

    if not token: # Should be caught above, but defensive check
         logger.warning(f"Empty token after split for path: {request_path}")
         raise credentials_exception

    payload = auth_utils.decode_access_token(token)
    if payload is None:
        logger.warning(f"Invalid JWT token received for path: {request_path}")
        raise credentials_exception # Invalid token (expired, bad signature etc.)

    tenant_id = payload.get("tenant_id") # Assume tenant_id claim exists in the JWT payload
    if tenant_id is None:
        # Token is valid, but doesn't contain the necessary tenant information
        logger.warning(f"Missing 'tenant_id' claim in JWT for path: {request_path}")
        raise tenant_exception

    # --- Tenant Validation and Config Loading ---
    db = None # Initialize db to None
    try:
        db = SessionLocal() # Create a synchronous session for middleware
        tenant = tenant_crud.get_tenant_by_id(db, tenant_id=tenant_id)

        if not tenant: # Checks for None and implicitly is_active=False based on CRUD query
            logger.warning(f"Tenant validation failed: Tenant ID '{tenant_id}' not found or inactive for path: {request_path}")
            raise HTTPException(status_code=403, detail="Tenant not found or inactive") # Use 403 as per plan

        # Attach the full tenant object (or specific config) to request state
        request.state.tenant = tenant
        # You can access tenant details in subsequent dependencies/routes via request.state.tenant.pinecone_api_key etc.
        logger.info(f"Request validated for tenant: {tenant.name} (ID: {tenant_id}) on path: {request_path}")

    except HTTPException as http_exc:
        raise http_exc # Re-raise HTTP exceptions
    except Exception as e:
        # Catch potential DB errors or other issues during tenant lookup
        logger.error(f"Error during tenant validation for tenant ID '{tenant_id}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during tenant validation")
    finally:
        if db:
            db.close() # Ensure the synchronous session is closed
    # --- End Tenant Validation ---

    # Proceed with the request
    response = await call_next(request)
    return response
# --- End Tenant Identification Middleware ---


# Add request logging middleware
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

# --- Include Routers ---
app.include_router(admin.router)

# --- Request/Response Models (Existing) ---
class ChatRequest(BaseModel):
    query: str
    session_id: Optional[int] = None # Changed to int, matches DB session ID

class ChatResponse(BaseModel):
    response: str
    sources: List[Dict[str, str]] = []

# --- OAuth2 Scheme ---
# Points to the login endpoint
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

# In-memory chatbot instance
chatbot_instance = YouTubeTranscriptChatbot()

# Removed get_chatbot helper function, using global instance directly
# def get_chatbot(creator_id: str = "mountaindog1"):
#     """
#     Get the appropriate chatbot instance based on creator ID.
#     """
#     return chatbot_instance
# Root endpoint
# Serve frontend index.html
@app.get("/", response_class=HTMLResponse)
async def read_index():
    # Check if file exists to avoid errors if frontend isn't built/present
    index_path = "frontend/index.html"
    if not os.path.exists(index_path):
         return HTMLResponse(content="<h1>Backend Running</h1><p>Frontend not found at ./frontend/index.html</p>", status_code=404)
    return FileResponse(index_path)

# Serve static files (CSS, JS) from the frontend directory
# Mount this *after* the root endpoint to avoid conflicts
app.mount("/static", StaticFiles(directory="frontend"), name="static")

# API Root (Optional - can be removed if / serves the frontend)
# @app.get("/api")
# async def api_root():
#     return {"status": "online", "message": "YouTube Creator Chatbot API"}

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    return {"status": "healthy"}


# --- Authentication Endpoints ---

@app.post("/auth/register", response_model=user_schemas.User, status_code=status.HTTP_201_CREATED)
async def register_user(user: user_schemas.UserCreate, db: AsyncSession = Depends(get_db)):
    """Registers a new user."""
    db_user = await user_crud.get_user_by_email(db, email=user.email)
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    created_user = await user_crud.create_user(db=db, user=user)
    return created_user

@app.post("/auth/login", response_model=user_schemas.Token)
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(), db: AsyncSession = Depends(get_db)
):
    """Logs in a user and returns a JWT access token."""
    user = await user_crud.get_user_by_email(db, email=form_data.username) # username is email here
    if not user or not auth_utils.verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=auth_utils.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = auth_utils.create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/users/me", response_model=user_schemas.User)
async def read_users_me(current_user: user_schemas.User = Depends(get_current_user)):
    """Fetches the profile of the currently authenticated user."""
    return current_user


# --- Profile Settings Endpoints ---

@app.get("/profile/settings", response_model=user_schemas.UserUpdateProfile)
async def get_profile_settings(
    current_user: user_schemas.User = Depends(get_current_user)
):
    """Retrieves the current user's profile settings."""
    # The current_user object already has profile_settings loaded
    return {"profile_settings": current_user.profile_settings or {}}

@app.put("/profile/settings", response_model=user_schemas.User)
async def update_profile_settings(
    settings_update: user_schemas.UserUpdateProfile,
    db: AsyncSession = Depends(get_db),
    current_user: user_schemas.User = Depends(get_current_user)
):
    """Updates the current user's profile settings."""
    updated_user = await user_crud.update_user_profile_settings(
        db=db, user_id=current_user.id, settings_data=settings_update.profile_settings
    )
    if not updated_user:
        # This shouldn't happen if get_current_user worked, but handle defensively
        raise HTTPException(status_code=404, detail="User not found")
    return updated_user


# --- Chat Endpoints (Now Secured) ---

# --- Chat History Endpoints ---

@app.get("/chat/sessions", response_model=List[chat_schemas.ChatSession])
async def get_sessions(
    db: AsyncSession = Depends(get_db),
    current_user: user_schemas.User = Depends(get_current_user)
):
    """Retrieves all chat sessions for the current user."""
    sessions = await chat_crud.get_user_chat_sessions(db=db, user_id=current_user.id)
    return sessions

@app.get("/chat/sessions/{session_id}/messages", response_model=List[chat_schemas.ChatMessage])
async def get_session_messages(
    session_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: user_schemas.User = Depends(get_current_user)
):
    """Retrieves all messages for a specific chat session."""
    # get_chat_messages includes check for user ownership
    messages = await chat_crud.get_chat_messages(db=db, session_id=session_id, user_id=current_user.id)
    if not messages and not await chat_crud.get_chat_session(db, session_id, current_user.id):
         # If messages is empty AND session doesn't exist/belong to user, raise 404
         raise HTTPException(status_code=404, detail="Chat session not found or access denied")
    return messages


# --- Chat Processing Endpoints ---

# Standard chat endpoint (Commented out for now - focus on streaming)
# @app.post("/chat", response_model=ChatResponse)
# async def chat(
#     request: ChatRequest,
#     db: AsyncSession = Depends(get_db), # Added db
#     current_user: user_schemas.User = Depends(get_current_user)
# ):
#     """Standard chat endpoint - Needs update for history"""
#     # TODO: Implement session handling and message saving similar to chat_stream
#     chatbot = get_chatbot()
#     logger.warning("Non-streaming /chat endpoint called - history not implemented yet.")
#     try:
#         async with asyncio.timeout(90):
#             response = await chatbot.get_streaming_response(request.query) # Needs history arg
#         # ... (rest of parsing logic) ...
#         # TODO: Save user query and assistant response to DB
#         return { ... }
#     except Exception as e:
#         # ... (error handling) ...
#         return { ... }

# Streaming chat endpoint
@app.get("/chat/stream")  # Add GET support for EventSource
@app.post("/chat/stream") # Keep POST for API clients
async def chat_stream(
    # Use POST with body for consistency now
    request: ChatRequest, # Removed default None, now required in body
    db: AsyncSession = Depends(get_db), # Added db dependency
    current_user: user_schemas.User = Depends(get_current_user),
    retriever: VectorStoreRetriever = Depends(get_retriever), # Injected retriever
    embeddings: Embeddings = Depends(get_openai_embeddings) # Injected embeddings
):
    """
    Streaming chat endpoint for real-time responses.
    Handles chat session creation/retrieval and message persistence.
    """
    # chatbot = get_chatbot() # Removed, using global chatbot_instance
    session_id = request.session_id
    history = []

    # 1. Get or Create Chat Session & Load History
    if session_id:
        chat_session = await chat_crud.get_chat_session(db, session_id, current_user.id)
        if not chat_session:
            raise HTTPException(status_code=404, detail="Chat session not found or access denied")
        history_models = await chat_crud.get_chat_messages(db, session_id, current_user.id)
        # Convert history models to simple dict format expected by chatbot (adjust if needed)
        history = [{"role": msg.role, "content": msg.content} for msg in history_models]
    else:
        # Create a new session (consider generating a title later, e.g., from first query)
        chat_session = await chat_crud.create_chat_session(db, user_id=current_user.id)
        session_id = chat_session.id # Use the newly created session ID

    # 2. Save User Query
    await chat_crud.add_chat_message(
        db=db, session_id=session_id, role="user", content=request.query
    )
    
    # 3. Prepare and run streaming generation
    
    async def generate():
        queue = asyncio.Queue()
        completed = False
        
        full_response_content = ""
        sources_list = [] # To store parsed sources

        # Modified callback to accumulate response and parse sources
        def handle_chunk(chunk_data):
            nonlocal full_response_content, sources_list
            # Assuming chunk_data might be dict with 'content' and maybe 'sources' later
            # For now, assume it's just the text chunk
            content_chunk = ""
            if isinstance(chunk_data, str):
                 content_chunk = chunk_data
            elif isinstance(chunk_data, dict) and "content" in chunk_data:
                 content_chunk = chunk_data["content"]
                 # TODO: Potentially parse sources if chatbot sends them structured in chunks

            if content_chunk:
                 full_response_content += content_chunk
                 queue.put_nowait({"type": "chunk", "content": content_chunk})
        
        async def process():
            nonlocal completed
            try:
                # Set a timeout for AI response
                async with asyncio.timeout(120):  # 2 minute timeout
                    # TODO: Modify get_streaming_response to accept history
                    # await chatbot.get_streaming_response(request.query, history, handle_chunk)
                    # --- Placeholder until chatbot is modified ---
                    # Pass the loaded history to the chatbot
                    # Pass injected retriever and embeddings to the chatbot method
                    await chatbot_instance.get_streaming_response(
                        query=request.query,
                        retriever=retriever, # Pass injected retriever
                        embeddings=embeddings, # Pass injected embeddings
                        history=history,
                        callback=handle_chunk
                    )
            except asyncio.TimeoutError:
                await queue.put({"type": "error", "content": "Response generation timed out. Please try a simpler question."})
            except Exception as e:
                logger.error(f"Error in streaming response: {str(e)}")
                await queue.put({"type": "error", "content": f"Error: {str(e)}"})
            finally:
                completed = True
                # 4. Save Assistant Response (after stream finishes or errors)
                try:
                    # Basic source parsing from the full response (similar to non-streaming endpoint)
                    response_parts = full_response_content.split("Sources:")
                    final_content = response_parts[0].strip() if response_parts else full_response_content.strip()
                    if len(response_parts) > 1:
                        sources_text = response_parts[1].strip()
                        link_pattern = r'\[([^\]]+)\]\(([^)]+)\)'
                        matches = re.findall(link_pattern, sources_text)
                        sources_list = [{"title": title, "url": url} for title, url in matches]

                    if final_content: # Only save if there was some response content
                         await chat_crud.add_chat_message(
                              db=db,
                              session_id=session_id,
                              role="assistant",
                              content=final_content, # Save content without "Sources:" part
                              sources=sources_list if sources_list else None
                         )
                         logger.info(f"Saved assistant response for session {session_id}")
                    else:
                         logger.warning(f"No assistant response content generated for session {session_id}, not saving.")

                    # Send final "done" signal with session_id
                    await queue.put({"type": "done", "session_id": session_id})

                except Exception as db_err:
                     logger.error(f"Failed to save assistant message for session {session_id}: {db_err}", exc_info=True)
                     # Still send done signal, but maybe log error to client?
                     await queue.put({"type": "error", "content": f"Error saving chat message: {db_err}"})
                     await queue.put({"type": "done", "session_id": session_id}) # Ensure stream terminates
        
        # Start the processing task
        task = asyncio.create_task(process())
        
        # Yield chunks with proper SSE format
        try:
            while not completed or not queue.empty():
                try:
                    # Use wait_for to prevent blocking forever
                    message = await asyncio.wait_for(queue.get(), timeout=1.0)
                    
                    if message["type"] == "done":
                        # End of stream - include session_id
                        yield f"event: done\ndata: {json.dumps({'session_id': message.get('session_id', session_id)})}\n\n"
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
    
    # Add explicit CORS headers to help with browser issues
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers": "*"
        }
    )

# Run with: uvicorn app:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)))