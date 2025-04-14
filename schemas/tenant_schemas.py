# schemas/tenant_schemas.py

from pydantic import BaseModel, Field
from typing import Optional, List
import datetime
import uuid

# --- Base Schema ---
class TenantBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=100, description="Name of the tenant")
    pinecone_environment: Optional[str] = Field(None, max_length=100, description="Pinecone environment for the tenant")
    system_prompt: Optional[str] = Field(None, description="Custom system prompt for the tenant's chatbot")
    is_active: bool = Field(True, description="Whether the tenant is active")

    class Config:
        # Pydantic V2 replaces orm_mode with from_attributes
        from_attributes = True

# --- Schema for Creation ---
# Excludes fields automatically set or sensitive fields not set at creation
class TenantCreate(BaseModel): # Not inheriting Base to control fields precisely
    id: str = Field(..., description="Unique identifier for the tenant (e.g., UUID or specific string)")
    name: str = Field(..., min_length=1, max_length=100, description="Name of the tenant")
    pinecone_environment: Optional[str] = Field(None, max_length=100, description="Initial Pinecone environment")
    system_prompt: Optional[str] = Field(None, description="Initial custom system prompt")
    # is_active defaults to True in the model/CRUD
    # pinecone_api_key should be set via update if needed

    class Config:
        from_attributes = True


# --- Schema for Updates ---
# All fields are optional for partial updates
class TenantUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=100, description="New name for the tenant")
    pinecone_api_key: Optional[str] = Field(None, description="New Pinecone API key (will be encrypted)")
    pinecone_environment: Optional[str] = Field(None, max_length=100, description="New Pinecone environment")
    system_prompt: Optional[str] = Field(None, description="New custom system prompt")
    is_active: Optional[bool] = Field(None, description="Update active status")

    class Config:
        from_attributes = True


# --- Schema for Reading Tenant Data (Response Model) ---
# Includes fields from the database model, excluding sensitive ones like the API key
class Tenant(TenantBase): # Inherits common fields + is_active
    id: str = Field(..., description="Unique identifier for the tenant")
    created_at: datetime.datetime = Field(..., description="Timestamp of tenant creation")
    updated_at: datetime.datetime = Field(..., description="Timestamp of last tenant update")
    # Excludes pinecone_api_key for security

    class Config:
        from_attributes = True

# Optional: Schema for reading tenant data including sensitive info (Admin only)
# Not strictly required by the prompt, but could be useful.
# class TenantAdmin(Tenant):
#     # Potentially include encrypted key hash or other admin-specific info if needed
#     pass