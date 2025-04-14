# Multi-Tenant Backend Architecture Plan (Railway)

This document outlines the proposed architecture, recommendations, and Standard Operating Procedures (SOPs) for deploying a single, multi-tenant backend service on Railway serving multiple distinct frontend applications. It also summarizes the implementation details and required post-deployment steps.

**Goal:** Build a robust, secure, scalable, and maintainable system while mitigating key risks associated with multi-tenancy.

## 1. Architecture Overview

A single backend service deployed on Railway will handle API requests from multiple frontend applications, potentially hosted on Railway or external platforms (Vercel, Netlify, etc.), each mapped to a unique custom domain (e.g., `app.clientA.com`). The backend identifies the originating tenant for each request and interacts with tenant-specific resources, such as dedicated Pinecone database instances and custom system prompts.

```mermaid
graph TD
    A[Incoming Request (app.clientA.com)] --> MW{Tenant ID Middleware};
    MW -->|Identify Tenant 'A'| TC[Set Request Tenant Context: 'A'];
    TC --> BL[Business Logic Layer];
    BL --> DAL[Data Access Layer];
    DAL -->|Use Tenant 'A' Config| PC_A[(Pinecone Client A)];
    DAL -->|Use Tenant 'A' ID/Filter| DB[(Database)];

    subgraph Request Lifecycle
        direction LR
        MW;
        TC;
        BL;
        DAL;
    end

    subgraph External Services
        PC_A;
        DB;
    end

    subgraph Monitoring & Alerting
        M[Monitoring System] -->|Watches| BL;
        M -->|Watches| DAL;
        M -->|Watches| PC_A;
        M -->|Watches| DB;
        M --> Alert[Alerting];
    end

    style MW fill:#f9d,stroke:#333,stroke-width:2px
    style TC fill:#f9d,stroke:#333,stroke-width:2px
    style BL fill:#ccf,stroke:#333,stroke-width:2px
    style DAL fill:#ccf,stroke:#333,stroke-width:2px
    style PC_A fill:#dfd,stroke:#333,stroke-width:1px
    style DB fill:#dfd,stroke:#333,stroke-width:1px
    style M fill:#fc9,stroke:#333,stroke-width:1px
```

## 2. Core Principles & Recommendations

*(Sections 2.1 - 2.5 remain largely the same as the initial plan, outlining best practices)*

### 2.1. Multi-Tenant vs. Duplicated Repositories
**Recommendation:** Multi-tenant architecture is strongly recommended for maintainability, consistency, and cost-effectiveness.

### 2.2. Mitigating Single Point of Failure & Deployment Risks
*   **Implementation:** Automated Testing, Monitoring & Alerting, Staged Rollouts (if possible), Health Checks.
*   **SOPs:** Code Reviews, Pre-Deployment Checklist, Rollback Procedure, Incident Response Plan.

### 2.3. Preventing Resource Contention ("Noisy Neighbor")
*   **Implementation:** Tenant-Aware Monitoring, API Rate Limiting, Async Task Processing, DB/API Optimization.
*   **SOPs:** Performance Monitoring, "Noisy Neighbor" Handling Process.

### 2.4. Ensuring Security & Tenant Isolation (Critical)
*   **Implementation:** Tenant Context Middleware, Strict DAL, Secure Secret Management (Encryption), Dependency Management, Input Validation.
*   **SOPs:** Security Code Reviews, Least Privilege, Security Audits, Data Handling Policy.

### 2.5. Planning for Scalability
*   **Implementation:** Stateless Design, Connection Pooling, Railway Scaling.
*   **SOPs:** Load Testing, Capacity Planning.

## 3. Implementation Summary

The multi-tenant architecture has been implemented as follows:

*   **Tenant Data Model (`models/db_models.py`):**
    *   A `Tenant` SQLAlchemy model was created.
    *   Fields include `id` (primary key, matches JWT claim), `name`, `is_active`, `pinecone_environment`.
    *   `pinecone_api_key` is stored encrypted using `sqlalchemy-utils.EncryptedType` and a `SECRET_KEY`.
    *   An optional `system_prompt` (Text) field was added to allow tenant-specific prompts.
*   **Database Migrations (Alembic):**
    *   Migrations were generated to:
        *   Create the `tenants` table (`8adfa00db3b0`).
        *   Add the `system_prompt` column to `tenants` (`66c9e4aafc9e`).
        *   Add an `is_superuser` boolean field to the `User` model (`3e87f40d465d`).
*   **Tenant Identification & Context (`app.py`, `crud/tenant_crud.py`, `auth/utils.py`):**
    *   FastAPI middleware (`@app.middleware("http")`) intercepts relevant requests.
    *   It extracts the JWT from the `Authorization` header.
    *   The token is decoded and validated using `auth.utils.decode_access_token`.
    *   The `tenant_id` claim is extracted.
    *   `crud.tenant_crud.get_tenant_by_id` is called to fetch the corresponding `Tenant` record from the database (using a manually managed session within the middleware).
    *   The tenant is validated (exists, `is_active`).
    *   The validated `Tenant` object (containing decrypted API key, environment, prompt) is attached to `request.state.tenant`.
*   **Tenant-Aware Logic (`auth/dependencies.py`, `chatbot.py`, `api.py`/`app.py`):**
    *   FastAPI dependencies were created:
        *   `get_current_tenant`: Retrieves the `Tenant` object from `request.state`.
        *   `get_pinecone_client`: Initializes Pinecone using the current tenant's API key and environment.
        *   `get_retriever`: Creates a Langchain retriever specific to the tenant's Pinecone index.
    *   Core application logic (e.g., chat endpoints in `api.py`/`app.py`, chatbot methods in `chatbot.py`) was refactored to inject these dependencies.
    *   Pinecone operations now use the tenant-specific client.
    *   The system prompt used by the chatbot logic now checks `tenant.system_prompt` and falls back to the default if it's not set.
*   **Admin Management (`routers/admin.py`, `crud/tenant_crud.py`, `schemas/tenant_schemas.py`, `auth/dependencies.py`):**
    *   Full CRUD functions (create, read, update, delete/deactivate) for tenants were added to `crud/tenant_crud.py`.
    *   Pydantic schemas (`schemas/tenant_schemas.py`) were defined for API request/response validation.
    *   A new API router (`routers/admin.py`) was created with endpoints under `/admin/tenants/` for managing tenants.
    *   These admin endpoints are secured using a dependency (`get_current_active_admin_user`) which verifies that the authenticated user has the `is_superuser` flag set to `True` in the database.

## 4. Post-Deployment Setup on Railway

After deploying this application to Railway, follow these steps to make it operational:

1.  **Set Environment Variables:**
    *   Navigate to your service settings in Railway and define the following environment variables:
        *   `SECRET_KEY`: **CRITICAL.** Generate a strong, unique secret key (e.g., using `openssl rand -hex 32` locally) and paste it here. This key is used to encrypt and decrypt sensitive tenant data (like Pinecone API keys) stored in the database. **Never commit this key to your code repository.**
        *   `DATABASE_URL`: Ensure this variable points to your production database (e.g., the connection string for your Railway Postgres service).
        *   `JWT_SECRET_KEY`, `ALGORITHM`, `ACCESS_TOKEN_EXPIRE_MINUTES`: Configure these for your JWT authentication.
        *   `OPENAI_API_KEY`: Your OpenAI API key.
        *   `BACKEND_CORS_ORIGINS`: A comma-separated list or JSON array string of allowed frontend origins (e.g., `["https://app.clientA.com","https://app.clientB.com"]`).
        *   Any other required environment variables for your application.
2.  **Run Database Migrations:**
    *   Open a shell connection to your running Railway service (via the Railway dashboard or CLI).
    *   Execute the Alembic command to apply all database schema changes:
        ```bash
        alembic upgrade head
        ```
    *   This will create the `tenants` table and add the `system_prompt` and `is_superuser` columns.
3.  **Create an Admin User:**
    *   You need at least one user designated as an administrator to manage tenants.
    *   **Method:** The exact method depends on your user setup. You might need to:
        *   Connect to the production database directly (e.g., using `psql` via Railway's CLI or a DB GUI) and manually set the `is_superuser` field to `true` for an existing user record.
        *   *Alternatively:* If you have a user registration flow, temporarily modify it (before deploying the final version) to allow setting this flag, create the admin user, then revert the change.
        *   *Alternatively:* Create a one-off script to run via the Railway shell that creates or updates a user with the flag set.
4.  **Create Tenants:**
    *   Log in to the application as the admin user created in the previous step.
    *   Use an API client (like Postman, Insomnia, or `curl`) to interact with the admin endpoints:
        *   Send a `POST` request to `/admin/tenants/` with a JSON body like:
          ```json
          {
            "id": "tenant_a_identifier", // MUST match the tenant_id claim in user JWTs
            "name": "Client A Name",
            "pinecone_environment": "us-west1-gcp", // Client A's Pinecone env
            "pinecone_api_key": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx", // Client A's Pinecone key
            "system_prompt": "You are a helpful assistant for Client A." // Optional
          }
          ```
        *   Repeat for each tenant you need to onboard.
5.  **Configure JWT Generation:**
    *   Ensure that your authentication system (where users log in) generates JWTs containing a `tenant_id` claim. The value of this claim **must exactly match** the `id` you used when creating the corresponding tenant record via the admin API.
6.  **Verify CORS:**
    *   Double-check that the `BACKEND_CORS_ORIGINS` environment variable correctly lists all frontend application domains that need to access the API.

Once these steps are completed, your multi-tenant application should be ready for use on Railway. Remember to also implement the monitoring and operational SOPs outlined earlier for long-term health.