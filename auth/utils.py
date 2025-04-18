# auth/utils.py
from passlib.context import CryptContext
from datetime import datetime, timedelta, timezone
from typing import Optional
import os
from jose import JWTError, jwt
from dotenv import load_dotenv

load_dotenv()

# --- Password Hashing ---
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verifies a plain password against a hashed password."""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """Hashes a plain password."""
    return pwd_context.hash(password)

# --- JWT Token Handling ---
# SECRET_KEY is used for DB encryption (loaded in models/db_models.py)
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY") # Use dedicated key for JWT
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30 # Token validity period

if not JWT_SECRET_KEY:
    raise ValueError("JWT_SECRET_KEY environment variable not set for JWT")

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None, tenant_id: Optional[str] = None) -> str: # Added tenant_id parameter
    """Creates a JWT access token, optionally including a tenant ID."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    if tenant_id: # Add tenant_id to the payload if provided
        to_encode.update({"tenant_id": tenant_id})
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=ALGORITHM) # Use JWT_SECRET_KEY
    return encoded_jwt

def decode_access_token(token: str) -> Optional[dict]:
    """Decodes a JWT access token, returning the payload if valid."""
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[ALGORITHM]) # Use JWT_SECRET_KEY
        return payload
    except JWTError:
        return None # Indicates invalid token (expired, wrong signature, etc.)