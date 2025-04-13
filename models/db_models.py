# models/db_models.py
import datetime
from sqlalchemy import (
    create_engine, Column, Integer, String, DateTime, ForeignKey, Text, JSON
)
from sqlalchemy.orm import relationship, sessionmaker, declarative_base
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable not set")

# Use async engine for FastAPI
engine = create_async_engine(DATABASE_URL, echo=True) # echo=True for debugging SQL
AsyncSessionLocal = sessionmaker(
    bind=engine, class_=AsyncSession, expire_on_commit=False
)

Base = declarative_base()

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
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

# Helper function to get DB session (dependency for FastAPI)
async def get_db():
    async with AsyncSessionLocal() as session:
        yield session

# Function to create tables (useful for initial setup or testing)
# We will use Alembic for proper migrations later
async def create_tables():
     async with engine.begin() as conn:
         await conn.run_sync(Base.metadata.create_all)

if __name__ == "__main__":
    # Example of how to run create_tables manually if needed
    import asyncio
    async def main():
        print("Creating database tables...")
        await create_tables()
        print("Tables created.")
    # asyncio.run(main()) # Uncomment to run manually