# crud/user_crud.py
from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import Session # Add this import
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from models.db_models import User
from schemas.user_schemas import UserCreate
from auth.utils import get_password_hash, verify_password # Add verify_password import
from typing import Optional, Dict, Any

async def get_user_by_email(db: AsyncSession, email: str) -> Optional[User]:
    """Fetches a user by their email address."""
    result = await db.execute(select(User).filter(User.email == email))
    return result.scalars().first()

async def create_user(db: AsyncSession, user: UserCreate) -> User:
    """Creates a new user in the database."""
    hashed_password = get_password_hash(user.password)
    db_user = User(email=user.email, hashed_password=hashed_password)
    try:
        db.add(db_user)
        await db.commit()
        await db.refresh(db_user)
        return db_user
    except IntegrityError:
        await db.rollback()
        raise HTTPException(
            status_code=409,
            detail="Username or email already registered"
        )
    except SQLAlchemyError:
        await db.rollback()
        raise HTTPException(
            status_code=500,
            detail="An unexpected database error occurred during user creation"
        )

async def update_user_profile_settings(
    db: AsyncSession, user_id: int, settings_data: Dict[str, Any]
) -> Optional[User]:
    """Updates the profile_settings JSON field for a user."""
    result = await db.execute(select(User).filter(User.id == user_id))
    db_user = result.scalars().first()
    if db_user:
        # Merge new settings with existing ones (or replace entirely if desired)
        # For merging:
        existing_settings = db_user.profile_settings or {}
        existing_settings.update(settings_data)
        db_user.profile_settings = existing_settings
        # For replacing:
        # db_user.profile_settings = settings_data

        await db.commit()
        await db.refresh(db_user)
        return db_user
    return None


# --- Synchronous Authentication ---

def authenticate_user(db: Session, email: str, password: str) -> Optional[User]:
    """Authenticates a user using synchronous session."""
    # Use synchronous query with Session
    user = db.query(User).filter(User.email == email).first()
    if not user:
        return None # User not found
    if not verify_password(password, user.hashed_password):
        return None # Incorrect password
    return user # Authentication successful