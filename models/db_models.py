# models/db_models.py
import datetime
from sqlalchemy import (
    create_engine as create_sync_engine, # Alias sync engine
    Column, Integer, String, DateTime, ForeignKey, Text, JSON, Boolean
)
from sqlalchemy.orm import relationship, sessionmaker, declarative_base, Session as SyncSession # Import sync Session
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy_utils import EncryptedType # Added for encryption
import os
from dotenv import load_dotenv
import uuid # Added for potential UUID generation if needed later

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable not set")
# Load SECRET_KEY for encryption
SECRET_KEY = os.getenv("SECRET_KEY")
if not SECRET_KEY:
    raise ValueError("SECRET_KEY environment variable not set for encryption")

# Ensure correct URL prefix for psycopg (v3) driver for both sync/async
if "postgresql+psycopg://" in DATABASE_URL:
    db_url_for_psycopg = DATABASE_URL
elif "postgresql://" in DATABASE_URL:
    # Add the +psycopg suffix if only postgresql:// is present
    db_url_for_psycopg = DATABASE_URL.replace("postgresql://", "postgresql+psycopg://")
else:
    # Handle cases where the scheme might be missing entirely
    if "://" not in DATABASE_URL:
         db_url_for_psycopg = f"postgresql+psycopg://{DATABASE_URL}"
    else:
        # If it has a different, unsupported scheme
        raise ValueError(f"Unsupported DATABASE_URL format for psycopg driver: {DATABASE_URL}")

# Use the explicitly prefixed URL for both engines
async_db_url = db_url_for_psycopg
sync_db_url = db_url_for_psycopg # Use the same URL for sync with psycopg v3

# Async Engine and Session for FastAPI endpoints
async_engine = create_async_engine(async_db_url, echo=False) # Use the unified URL
AsyncSessionLocal = sessionmaker(
    bind=async_engine, class_=AsyncSession, expire_on_commit=False
)

# Sync Engine and Session for middleware, scripts, etc.
sync_engine = create_sync_engine(sync_db_url, echo=False) # Use the unified URL
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=sync_engine)

Base = declarative_base()

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    is_superuser = Column(Boolean, default=False, nullable=False) # Added for admin check
    tenant_id = Column(String, ForeignKey("tenants.id"), nullable=True, index=True) # Link to Tenant
    profile_settings = Column(JSON, nullable=True, default={}) # Store custom user info
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    chat_sessions = relationship("ChatSession", back_populates="user")

class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    title = Column(String, nullable=True) # Optional title for the session
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

    user = relationship("User", back_populates="chat_sessions")
    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan")

class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("chat_sessions.id"), nullable=False)
    role = Column(String, nullable=False)  # 'user' or 'assistant'
    content = Column(Text, nullable=False)
    sources = Column(JSON, nullable=True) # Store sources for assistant messages
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

    session = relationship("ChatSession", back_populates="messages")

# --- Tenant Model ---
class Tenant(Base):
    __tablename__ = "tenants"

    # Assuming tenant_id from JWT is a string (e.g., UUID)
    id = Column(String, primary_key=True, index=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, index=True, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    # Encrypt sensitive data like API keys
    pinecone_api_key = Column(EncryptedType(String, SECRET_KEY), nullable=True)
    pinecone_environment = Column(String, nullable=True)
    system_prompt = Column(Text, nullable=True) # Tenant-specific system prompt
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

# Helper function to get DB session (dependency for FastAPI)
# Async DB session dependency for FastAPI endpoints
async def get_db():
    async with AsyncSessionLocal() as session:
        yield session

# Sync DB session context manager (useful for scripts or middleware)
def get_sync_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Function to create tables (useful for initial setup or testing)
# We will use Alembic for proper migrations later
async def create_tables():
     # Use sync engine for table creation if run manually
     # Note: Alembic handles migrations in production
     async with async_engine.begin() as conn: # Still use async engine here for consistency if run via main()
         await conn.run_sync(Base.metadata.create_all)

if __name__ == "__main__":
    # Example of how to run create_tables manually if needed
    import asyncio
    async def main():
        print("Creating database tables...")
        await create_tables()
        print("Tables created.")
    # asyncio.run(main()) # Uncomment to run manually