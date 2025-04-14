# routers/admin.py

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List, Optional

from models import db_models
from schemas import tenant_schemas
from crud import tenant_crud
from models.db_models import get_sync_db # Use sync session for CRUD
# Placeholder for the admin dependency - will be created in auth/dependencies.py
from auth.dependencies import get_current_active_admin_user

router = APIRouter(
    prefix="/admin",
    tags=["Admin Tenant Management"],
    # Add dependency here to protect all routes in this router
    dependencies=[Depends(get_current_active_admin_user)]
)

@router.post("/tenants/", response_model=tenant_schemas.Tenant, status_code=status.HTTP_201_CREATED)
def create_new_tenant(
    tenant: tenant_schemas.TenantCreate,
    db: Session = Depends(get_sync_db),
    # current_admin: db_models.User = Depends(get_current_active_admin_user) # Dependency already added at router level
):
    """
    Create a new tenant. Requires admin privileges.
    """
    # Optional: Check if tenant ID already exists (though primary key constraint should handle it)
    existing_tenant = tenant_crud.get_tenant_by_id_admin(db, tenant_id=tenant.id)
    if existing_tenant:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Tenant with ID '{tenant.id}' already exists.",
        )
    return tenant_crud.create_tenant(db=db, tenant_data=tenant)

@router.get("/tenants/", response_model=List[tenant_schemas.Tenant])
def read_tenants(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_sync_db),
    # current_admin: db_models.User = Depends(get_current_active_admin_user)
):
    """
    Retrieve a list of all tenants (active and inactive). Requires admin privileges.
    """
    tenants = tenant_crud.get_tenants(db, skip=skip, limit=limit)
    return tenants

@router.get("/tenants/{tenant_id}", response_model=tenant_schemas.Tenant)
def read_tenant(
    tenant_id: str,
    db: Session = Depends(get_sync_db),
    # current_admin: db_models.User = Depends(get_current_active_admin_user)
):
    """
    Retrieve details for a specific tenant by ID (active or inactive). Requires admin privileges.
    """
    db_tenant = tenant_crud.get_tenant_by_id_admin(db, tenant_id=tenant_id)
    if db_tenant is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Tenant not found")
    return db_tenant

@router.put("/tenants/{tenant_id}", response_model=tenant_schemas.Tenant)
def update_existing_tenant(
    tenant_id: str,
    tenant_update: tenant_schemas.TenantUpdate,
    db: Session = Depends(get_sync_db),
    # current_admin: db_models.User = Depends(get_current_active_admin_user)
):
    """
    Update an existing tenant's details. Requires admin privileges.
    """
    updated_tenant = tenant_crud.update_tenant(db=db, tenant_id=tenant_id, tenant_update=tenant_update)
    if updated_tenant is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Tenant not found")
    return updated_tenant

@router.delete("/tenants/{tenant_id}", response_model=tenant_schemas.Tenant)
def deactivate_tenant(
    tenant_id: str,
    db: Session = Depends(get_sync_db),
    # current_admin: db_models.User = Depends(get_current_active_admin_user)
):
    """
    Deactivate (soft delete) a tenant by setting is_active=False. Requires admin privileges.
    """
    deleted_tenant = tenant_crud.delete_tenant(db=db, tenant_id=tenant_id)
    if deleted_tenant is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Tenant not found")
    # Return the deactivated tenant object
    return deleted_tenant