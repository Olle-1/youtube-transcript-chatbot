# auth/dependencies.py
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError
from sqlalchemy.ext.asyncio import AsyncSession

# Assuming db_models, user_crud, auth_utils, user_schemas are importable
# Adjust paths if necessary based on your project structure
from models.db_models import User, get_db
import crud.user_crud as user_crud
import auth.utils as auth_utils
import schemas.user_schemas as user_schemas

# Define the OAuth2 scheme here or import from app.py if structured differently
# It points to the endpoint where the client gets the token
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

async def get_current_user(
    token: str = Depends(oauth2_scheme), db: AsyncSession = Depends(get_db)
) -> User:
    """
    Dependency to decode the JWT token, validate it, and fetch the user.
    Raises HTTPException if the token is invalid or the user doesn't exist.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    payload = auth_utils.decode_access_token(token)
    if payload is None:
        # This means the token was invalid (expired, bad signature, etc.)
        raise credentials_exception
    email: str = payload.get("sub")
    if email is None:
        # The 'sub' (subject, typically email/username) claim is missing
        raise credentials_exception
    
    # Validate the payload structure (optional but good practice)
    try:
        token_data = user_schemas.TokenData(email=email)
    except Exception: # Catch potential Pydantic validation errors
         raise credentials_exception

    user = await user_crud.get_user_by_email(db, email=token_data.email)
    if user is None:
        # User exists in token payload but not in DB (e.g., deleted after token issued)
        raise credentials_exception
    return user

# Example of a further dependency that checks if the user is "active"
# You might add an `is_active` field to your User model later
# async def get_current_active_user(
#     current_user: User = Depends(get_current_user)
# ) -> User:
#     """Checks if the fetched user is active."""
#     # if not current_user.is_active:
#     #     raise HTTPException(status_code=400, detail="Inactive user")
#     return current_user