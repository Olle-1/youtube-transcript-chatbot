# schemas/chat_schemas.py
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime

# --- Chat Message Schemas ---

class ChatMessageBase(BaseModel):
    role: str
    content: str

class ChatMessage(ChatMessageBase):
    id: int
    session_id: int
    sources: Optional[List[Dict[str, str]]] = None
    timestamp: datetime

    class Config:
        from_attributes = True

# --- Chat Session Schemas ---

class ChatSessionBase(BaseModel):
    title: Optional[str] = None

class ChatSessionCreate(ChatSessionBase):
    # user_id will be taken from authenticated user
    pass

class ChatSession(ChatSessionBase):
    id: int
    user_id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

# Schema for returning a session with its messages
class ChatSessionWithMessages(ChatSession):
    messages: List[ChatMessage] = []