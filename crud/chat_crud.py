# crud/chat_crud.py
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from models.db_models import ChatSession, ChatMessage, User
from typing import List, Optional, Dict, Any

async def create_chat_session(db: AsyncSession, user_id: int, title: Optional[str] = None) -> ChatSession:
    """Creates a new chat session for a user."""
    db_session = ChatSession(user_id=user_id, title=title)
    db.add(db_session)
    await db.commit()
    await db.refresh(db_session)
    return db_session

async def add_chat_message(
    db: AsyncSession,
    session_id: int,
    role: str,
    content: str,
    sources: Optional[List[Dict[str, str]]] = None
) -> ChatMessage:
    """Adds a new message to a chat session."""
    db_message = ChatMessage(
        session_id=session_id,
        role=role,
        content=content,
        sources=sources # Store sources as JSON
    )
    db.add(db_message)
    await db.commit()
    await db.refresh(db_message)
    return db_message

async def get_chat_session(db: AsyncSession, session_id: int, user_id: int) -> Optional[ChatSession]:
    """Gets a specific chat session for a user."""
    result = await db.execute(
        select(ChatSession)
        .filter(ChatSession.id == session_id, ChatSession.user_id == user_id)
    )
    return result.scalars().first()

async def get_chat_messages(db: AsyncSession, session_id: int, user_id: int) -> List[ChatMessage]:
    """Gets all messages for a specific chat session belonging to a user."""
    # First verify the session belongs to the user
    session = await get_chat_session(db, session_id, user_id)
    if not session:
        return [] # Or raise HTTPException(404)

    result = await db.execute(
        select(ChatMessage)
        .filter(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.timestamp.asc()) # Order messages chronologically
    )
    return result.scalars().all()

async def get_user_chat_sessions(db: AsyncSession, user_id: int) -> List[ChatSession]:
    """Gets all chat sessions for a user."""
    result = await db.execute(
        select(ChatSession)
        .filter(ChatSession.user_id == user_id)
        .order_by(ChatSession.updated_at.desc()) # Show most recent first
    )
    return result.scalars().all()