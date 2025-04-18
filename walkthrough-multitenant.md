# Walkthrough: Configuring and Using the Multi-Tenant System

This document provides step-by-step instructions for developers configuring, using, and maintaining the multi-tenant features of this application after deployment (e.g., on Railway).

## Prerequisites

*   The application backend has been successfully deployed (e.g., to Railway).
*   You have access to the Railway project dashboard or CLI to manage environment variables and run commands.
*   You have an API client (like Postman, Insomnia, or `curl`) for interacting with the backend API.
*   You have credentials for an existing user account that will be designated as an administrator.

## Step 1: Set Environment Variables

Secure configuration is crucial. Ensure the following environment variables are set correctly in your Railway service settings:

1.  **`SECRET_KEY`**:
    *   **Purpose:** Encrypts/decrypts sensitive tenant data (e.g., `pinecone_api_key`) stored in the database.
    *   **Action:** Generate a strong, unique secret key (e.g., run `openssl rand -hex 32` locally). Copy this value and set it as the `SECRET_KEY` environment variable in Railway.
    *   **CRITICAL:** Keep this key secure and **do not** commit it to version control. Losing it will make stored tenant secrets unrecoverable.
2.  **`DATABASE_URL`**:
    *   **Purpose:** Connection string for the application's primary database.
    *   **Action:** Verify this points to your production PostgreSQL database provided by Railway.
3.  **`JWT_SECRET_KEY`**, `ALGORITHM`, `ACCESS_TOKEN_EXPIRE_MINUTES`:
    *   **Purpose:** Configuration for user authentication JWTs.
    *   **Action:** Ensure these are set according to your security requirements. `JWT_SECRET_KEY` should also be a strong, unique secret.
4.  **`OPENAI_API_KEY`**:
    *   **Purpose:** API key for OpenAI services (used for embeddings, chat).
    *   **Action:** Set your valid OpenAI API key.
5.  **`BACKEND_CORS_ORIGINS`**:
    *   **Purpose:** Controls which frontend domains are allowed to make requests to the backend API.
    *   **Action:** Set this to a comma-separated list or a JSON array string of allowed origins (e.g., `https://app.clientA.com,https://app.clientB.com` or `["https://app.clientA.com","https://app.clientB.com"]`).
6.  **Other Variables:** Ensure any other application-specific variables are set (e.g., logging levels, external service URLs).

## Step 2: Run Database Migrations

Apply the necessary database schema changes using Alembic:

1.  **Access Shell:** Open a shell connection to your running Railway service (via the Railway dashboard or CLI).
2.  **Run Upgrade:** Execute the following command:
    ```bash
    alembic upgrade head
    ```
3.  **Verification:** This command should apply all pending migrations, including creating the `tenants` table and adding the `system_prompt` and `is_superuser` columns. Check the command output for success messages.

## Step 3: Create an Admin User

You need an administrative user to manage tenants via the API:

1.  **Identify User:** Choose an existing user account (or create one through your normal registration process if applicable) that will serve as the administrator. Note their user ID or email.
2.  **Set Admin Flag:** Access your production database (e.g., using `psql` via Railway CLI: `railway connect postgresql` then `psql $DATABASE_URL`). Update the chosen user's record in the `users` table:
    ```sql
    UPDATE users SET is_superuser = true WHERE email = 'admin_user@example.com';
    -- Or use the user's ID if more appropriate
    -- UPDATE users SET is_superuser = true WHERE id = 'user_id_here';
    ```
3.  **Verification:** Query the user record to confirm `is_superuser` is now `true`.

## Step 4: Create Tenants via Admin API

Use the secured admin endpoints to register tenant details:

1.  **Authenticate:** Log in to the application as the admin user designated in Step 3 to obtain a valid JWT access token.
2.  **Use API Client:** Configure your API client (e.g., Postman) to make requests to your deployed backend URL. Include the admin user's JWT in the `Authorization: Bearer <token>` header for all requests in this step.
3.  **Send POST Request:** Make a `POST` request to the `/admin/tenants/` endpoint.
    *   **URL:** `https://your-backend-service-url.up.railway.app/admin/tenants/`
    *   **Method:** `POST`
    *   **Headers:**
        *   `Authorization: Bearer <admin_user_jwt>`
        *   `Content-Type: application/json`
    *   **Body (JSON):**
        ```json
        {
          "id": "unique_tenant_identifier", // e.g., "client_a", "tenant123"
          "name": "Client A Full Name",
          "pinecone_environment": "your-pinecone-region", // e.g., "us-east-1-aws"
          "pinecone_api_key": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx", // The actual Pinecone API key for this tenant
          "system_prompt": "Optional: Custom system prompt for this tenant's chatbot." // Leave out or set to null to use default
        }
        ```
    *   **Important:**
        *   The `"id"` field **must** be unique for each tenant and **must match** the value that will be included in the `tenant_id` claim of JWTs for users belonging to this tenant (see Step 5).
        *   Provide the correct Pinecone environment and API key specific to this tenant. The API key will be encrypted before being stored in the database.
4.  **Repeat:** Repeat the `POST` request for each tenant you need to onboard.
5.  **Other Admin Operations:** You can also use the following admin endpoints (always requiring the admin JWT):
    *   `GET /admin/tenants/`: List all tenants.
    *   `GET /admin/tenants/{tenant_id}`: Get details for a specific tenant.
    *   `PUT /admin/tenants/{tenant_id}`: Update a tenant's details (name, prompt, active status, keys).
    *   `DELETE /admin/tenants/{tenant_id}`: Deactivate a tenant (sets `is_active=False`).

## Step 5: Configure JWT Generation

Ensure your user authentication flow correctly includes the tenant identifier in the JWTs it issues:

1.  **Locate JWT Creation Logic:** Find the code responsible for generating JWT access tokens when a user logs in (likely related to your `/login` or `/token` endpoint).
2.  **Add `tenant_id` Claim:** Modify the JWT payload (the data being encoded) to include a claim named `tenant_id`. The value of this claim **must exactly match** the `id` of the tenant that the logged-in user belongs to (as created in Step 4).
    ```python
    # Example modification during token creation (adjust to your code)
    user = get_user(...) # Fetch user details, including their associated tenant ID
    if user:
        tenant_id = user.tenant_id # Assuming user model has tenant_id relationship/field
        data_to_encode = {"sub": user.email, "tenant_id": tenant_id} # Add tenant_id here
        # Add expiry, etc.
        access_token = create_access_token(data=data_to_encode)
    ```
3.  **Verification:** After logging in as a regular user belonging to a specific tenant, decode their JWT (e.g., using jwt.io) to verify that the `tenant_id` claim is present and correct.

## Step 6: Verify CORS

Confirm that your frontend applications can communicate with the backend:

1.  **Check Variable:** Double-check the `BACKEND_CORS_ORIGINS` environment variable in Railway.
2.  **Test Frontend:** Access your deployed frontend applications (e.g., `https://app.clientA.com`) and perform actions that trigger API calls to the backend. Check your browser's developer console for any CORS-related errors.

## How it Works: Summary

*   **Request:** A user interacts with a frontend (e.g., `app.clientA.com`).
*   **Authentication:** The user logs in, receiving a JWT containing their `user_id` (`sub`) and their associated `tenant_id`.
*   **API Call:** The frontend makes an API call to the backend, including the JWT in the `Authorization` header.
*   **Middleware:** The backend's `tenant_middleware` intercepts the request.
    *   It extracts and validates the JWT.
    *   It reads the `tenant_id` claim.
    *   It queries the `tenants` table using the `tenant_id` to find the active tenant record.
    *   It loads the tenant's configuration (decrypted Pinecone key, environment, system prompt) into `request.state.tenant`.
*   **Dependencies:** FastAPI dependencies (`get_current_tenant`, `get_pinecone_client`, `get_retriever`) use the information from `request.state.tenant`.
*   **Business Logic:** Route handlers use the injected dependencies (like the tenant-specific Pinecone retriever or system prompt) to perform operations correctly scoped to that tenant.

By following these steps, you can successfully configure and utilize the multi-tenant capabilities of the application.