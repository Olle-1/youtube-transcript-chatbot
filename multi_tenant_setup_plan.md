# Setup Plan for Multi-Tenancy

This plan outlines the steps required to configure the multi-tenant system after deployment, based on the `walkthrough-multitenant.md` documentation and observed runtime errors.

**Goal:** Resolve "Missing 'tenant_id' claim" errors and enable correct multi-tenant operation.

## Setup Steps

1.  **Verify Environment Variables (Walkthrough Step 1):**
    *   **Action:** Double-check that all required environment variables (`SECRET_KEY`, `DATABASE_URL`, `JWT_SECRET_KEY`, `OPENAI_API_KEY`, `BACKEND_CORS_ORIGINS`, etc.) are correctly set in your Railway deployment environment. Pay special attention to the `SECRET_KEY` and `JWT_SECRET_KEY`.
    *   **Reason:** Ensures the application has the necessary configuration and credentials.

2.  **Run Database Migrations (Walkthrough Step 2):**
    *   **Action:** Access the shell of your running Railway service and execute the command: `alembic upgrade head`.
    *   **Reason:** Applies necessary database schema changes (creates `tenants` table, adds `tenant_id` to `users`). This is critical for resolving the JWT errors.

3.  **Designate an Admin User (Walkthrough Step 3):**
    *   **Action:** Choose an existing user. Access the production database (e.g., via `railway connect postgresql`) and run SQL: `UPDATE users SET is_superuser = true WHERE email = 'your_admin_email@example.com';`
    *   **Reason:** An admin user is required to manage tenants via the API.

4.  **Create Tenant(s) via Admin API (Walkthrough Step 4):**
    *   **Action:**
        *   Log in as the admin user to get a JWT.
        *   Use an API client to send a `POST` request to `/admin/tenants/` (e.g., `https://your-backend-service-url.up.railway.app/admin/tenants/`) with the admin JWT.
        *   Include tenant details in the JSON body:
          ```json
          {
            "id": "unique_tenant_identifier", // e.g., "default_tenant"
            "name": "Default Tenant",
            "pinecone_environment": "your-pinecone-region",
            "pinecone_api_key": "your-pinecone-key",
            "system_prompt": null // Or a custom prompt
          }
          ```
    *   **Reason:** Creates the tenant records needed by the application.

5.  **Associate Users with Tenants (Implied):**
    *   **Action:** Access the production database. For each relevant user, update their record: `UPDATE users SET tenant_id = 'the_tenant_id_you_created' WHERE email = 'user_email@example.com';`
    *   **Reason:** Links users to tenants, ensuring `tenant_id` is available for JWT creation. **This is crucial for fixing the errors.**

6.  **Verify JWT and Functionality (Walkthrough Step 5 & Testing):**
    *   **Action:** Log in as a regular user. Optionally decode the JWT to confirm the `tenant_id` claim. Test previously failing chatbot functionality.
    *   **Reason:** Confirms the setup is complete and errors are resolved.

7.  **Verify CORS (Walkthrough Step 6):**
    *   **Action:** Ensure `BACKEND_CORS_ORIGINS` includes your frontend URL. Test frontend access and API communication.
    *   **Reason:** Ensures the frontend can interact with the backend.

## Diagram

```mermaid
graph TD
    A[1. Verify Env Vars] --> B(2. Run `alembic upgrade head`);
    B --> C(3. Set `is_superuser=true` for Admin);
    C --> D(4. Create Tenant via Admin API);
    D --> E(5. Update `users` table: `tenant_id` = 'created_tenant_id');
    E --> F(6. Test Login & API Calls);
    F --> G(7. Verify CORS);

    style B fill:#ccf,stroke:#333,stroke-width:2px
    style D fill:#f9f,stroke:#333,stroke-width:2px
    style E fill:#f9f,stroke:#333,stroke-width:2px