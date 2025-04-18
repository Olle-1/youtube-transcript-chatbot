# Plan: Multi-Tenancy Authentication Refactor

**Goal:** Refactor authentication to use the database, add `tenant_id` to the `User` model and JWTs, and prepare for multi-tenant middleware checks.

## Phase 1: Backend Authentication Refactoring & Tenant Linking

1.  **Implement `authenticate_user` (Synchronous):**
    *   **File:** `crud/user_crud.py`
    *   **Action:** Add a *synchronous* function `authenticate_user(db: Session, email: str, password: str) -> Optional[models.User]`.
        *   Accept a synchronous `Session` (`from sqlalchemy.orm import Session`).
        *   Query the `User` table for the given `email` using the synchronous session.
        *   If the user exists, use `auth.utils.verify_password` to compare the provided password with the user's `hashed_password`.
        *   Return the `User` object on success, `None` otherwise.

2.  **Modify `User` Model:**
    *   **File:** `models/db_models.py`
    *   **Action:** Add the `tenant_id` column to the `User` model:
        ```python
        # Inside class User(Base):
        tenant_id = Column(String, ForeignKey("tenants.id"), nullable=True, index=True) # Added index
        # Optional: Add relationship back to Tenant if needed
        # tenant = relationship("Tenant")
        ```

3.  **Create Alembic Migration:**
    *   **Action:** Prepare the command to generate the database migration script.
    *   **Command:** `alembic revision --autogenerate -m "Add tenant_id to User model"`

4.  **Modify JWT Generation:**
    *   **File:** `auth/utils.py`
    *   **Action:** Update `create_access_token` to accept an optional `tenant_id` argument and add it to the JWT payload if provided.
        ```python
        # Modify the function signature and add tenant_id to the payload
        def create_access_token(data: dict, expires_delta: Optional[timedelta] = None, tenant_id: Optional[str] = None) -> str: # Added tenant_id
            to_encode = data.copy()
            if expires_delta:
                expire = datetime.now(timezone.utc) + expires_delta
            else:
                expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
            to_encode.update({"exp": expire})
            if tenant_id: # Add tenant_id if provided
                to_encode.update({"tenant_id": tenant_id})
            encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
            return encoded_jwt
        ```

## Phase 2: API Integration

5.  **Refactor Login Endpoint (`/auth/token`):**
    *   **File:** `api.py`
    *   **Action:**
        *   Remove the placeholder authentication logic (`fake_users_db`, `fake_hash_password`, `active_tokens`).
        *   Import necessary components: `authenticate_user` from `crud.user_crud`, `create_access_token` from `auth.utils`, `Session` from `sqlalchemy.orm`, `get_sync_db` from `models.db_models`, `schemas`.
        *   Modify the `login` function (`login_for_access_token`):
            ```python
            from crud.user_crud import authenticate_user # Add import
            from auth.utils import create_access_token # Add import
            from models.db_models import get_sync_db # Add import
            from sqlalchemy.orm import Session # Add import
            import schemas # Assuming schemas exist for Token response

            # ... (remove fake db/auth stuff)

            @app.post("/auth/token", response_model=schemas.Token) # Assuming a Token schema exists
            async def login_for_access_token(
                form_data: OAuth2PasswordRequestForm = Depends(),
                db: Session = Depends(get_sync_db) # Use sync session for auth
            ):
                user = authenticate_user(db, email=form_data.username, password=form_data.password)
                if not user:
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Incorrect username or password",
                        headers={"WWW-Authenticate": "Bearer"},
                    )

                # Determine tenant_id (handle case where user might not have one yet)
                tenant_id = user.tenant_id if user.tenant_id else "default" # Or handle differently if None is not allowed

                access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES) # Use constant from auth.utils if defined there
                access_token = create_access_token(
                    data={"sub": user.email}, # Use email or user.id as subject
                    expires_delta=access_token_expires,
                    tenant_id=tenant_id # Pass the tenant_id
                )
                return {"access_token": access_token, "token_type": "bearer"}
            ```

6.  **Update Token Verification/Usage:**
    *   **File:** `auth/dependencies.py` (or wherever the tenant is currently extracted)
    *   **Action:** Ensure the dependency that extracts tenant info (like `get_current_tenant`) uses `auth.utils.decode_access_token` to parse the JWT and retrieve the `tenant_id` claim.
    *   **File:** `api.py`
    *   **Action:** Refactor or replace any dependency relying on the old token system (like the old `get_current_user`) to use `decode_access_token` to validate the JWT and fetch the user based on the `sub` claim (email/id).

## Visualization (Simplified Flow)

```mermaid
graph TD
    subgraph Phase 1: Authentication & Model Changes
        A[1. Add sync authenticate_user to user_crud.py] --> B[2. Add tenant_id to User model in db_models.py];
        B --> C[3. Prepare Alembic migration command];
        C --> D[4. Modify create_access_token in auth/utils.py];
    end

    subgraph Phase 2: API Integration
        D --> E[5. Refactor /auth/token endpoint in api.py];
        E --> F[6. Update token decoding/usage in auth/dependencies.py];
    end

    subgraph User Flow
        G[User Login Request] --> E;
        E -- Calls --> A;
        A -- DB Check --> H{Database};
        E -- Generates JWT with tenant_id --> I[User Receives JWT];
        J[User API Request with JWT] --> F;
        F -- Decodes JWT --> K[Extracts user & tenant_id];
        K --> L[Process Request with Tenant Context];
    end