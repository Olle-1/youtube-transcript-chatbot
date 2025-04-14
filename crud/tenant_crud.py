# crud/tenant_crud.py

from typing import List, Optional
from sqlalchemy.orm import Session
from models import db_models
from schemas import tenant_schemas # Assuming this will exist

def get_tenant_by_id(db: Session, tenant_id: str) -> Optional[db_models.Tenant]:
    """
    Retrieves an active tenant by its ID. (For general use)

    Args:
        db: The synchronous database session.
        tenant_id: The ID of the tenant to retrieve.

    Returns:
        The Tenant object if found and active, otherwise None.
    """
    # This version is for general use, might need an admin version later
    # that doesn't check is_active or returns all details.
    # For now, let's keep it as is, the admin GET endpoint can use a different query if needed.
    return db.query(db_models.Tenant)\
             .filter(db_models.Tenant.id == tenant_id, db_models.Tenant.is_active == True)\
             .first()

# --- Admin CRUD Functions ---

def get_tenant_by_id_admin(db: Session, tenant_id: str) -> Optional[db_models.Tenant]:
    """
    Retrieves any tenant (active or inactive) by its ID. For admin use.

    Args:
        db: The synchronous database session.
        tenant_id: The ID of the tenant to retrieve.

    Returns:
        The Tenant object if found, otherwise None.
    """
    return db.query(db_models.Tenant).filter(db_models.Tenant.id == tenant_id).first()


def get_tenants(db: Session, skip: int = 0, limit: int = 100) -> List[db_models.Tenant]:
    """
    Retrieves a list of all tenants (active and inactive). For admin use.

    Args:
        db: The synchronous database session.
        skip: Number of records to skip (for pagination).
        limit: Maximum number of records to return (for pagination).

    Returns:
        A list of Tenant objects.
    """
    return db.query(db_models.Tenant).offset(skip).limit(limit).all()


def create_tenant(db: Session, tenant_data: tenant_schemas.TenantCreate) -> db_models.Tenant:
    """
    Creates a new tenant.

    Args:
        db: The synchronous database session.
        tenant_data: Pydantic schema containing tenant creation data.

    Returns:
        The newly created Tenant object.
    """
    # Create the Tenant model instance.
    # Note: If TenantCreate includes pinecone_api_key, it will be automatically
    # handled by EncryptedType upon saving, assuming the model is set up correctly.
    # It's generally safer NOT to include sensitive keys directly in creation schemas.
    # Let's assume TenantCreate does NOT include pinecone_api_key for now.
    db_tenant = db_models.Tenant(
        id=tenant_data.id, # Allow specifying ID during creation
        name=tenant_data.name,
        pinecone_environment=tenant_data.pinecone_environment,
        system_prompt=tenant_data.system_prompt,
        is_active=True # New tenants are active by default
        # pinecone_api_key is not set here, should be updated via update_tenant if needed
    )
    db.add(db_tenant)
    db.commit()
    db.refresh(db_tenant)
    return db_tenant


def update_tenant(db: Session, tenant_id: str, tenant_update: tenant_schemas.TenantUpdate) -> Optional[db_models.Tenant]:
    """
    Updates an existing tenant.

    Args:
        db: The synchronous database session.
        tenant_id: The ID of the tenant to update.
        tenant_update: Pydantic schema containing the fields to update.

    Returns:
        The updated Tenant object, or None if the tenant was not found.
    """
    db_tenant = get_tenant_by_id_admin(db, tenant_id)
    if not db_tenant:
        return None

    update_data = tenant_update.model_dump(exclude_unset=True) # Use model_dump in Pydantic v2

    for key, value in update_data.items():
        # EncryptedType handles encryption automatically on assignment if the key matches
        setattr(db_tenant, key, value)

    db.commit()
    db.refresh(db_tenant)
    return db_tenant


def delete_tenant(db: Session, tenant_id: str) -> Optional[db_models.Tenant]:
    """
    Soft deletes a tenant by setting its is_active flag to False.

    Args:
        db: The synchronous database session.
        tenant_id: The ID of the tenant to delete (deactivate).

    Returns:
        The deactivated Tenant object, or None if the tenant was not found.
    """
    db_tenant = get_tenant_by_id_admin(db, tenant_id)
    if not db_tenant:
        return None

    db_tenant.is_active = False
    db.commit()
    db.refresh(db_tenant)
    return db_tenant