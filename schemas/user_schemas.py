# schemas/user_schemas.py
from pydantic import BaseModel, EmailStr, Field
from typing import Optional, Dict, Any
from datetime import datetime

# --- User Schemas ---

class UserBase(BaseModel):
    email: EmailStr

class UserCreate(UserBase):
    password: str = Field(..., min_length=8)

class UserUpdateProfile(BaseModel):
    profile_settings: Dict[str, Any]

class User(UserBase):
    id: int
    profile_settings: Optional[Dict[str, Any]] = {}
    created_at: datetime

    class Config:
        from_attributes = True # Renamed from orm_mode in Pydantic v2

# --- Token Schemas ---

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    email: Optional[EmailStr] = None
    # You might add user_id here as well depending on your needs