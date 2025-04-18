# Project Context: FastAPI Multi-Tenant Backend on Railway

This document provides comprehensive context about our project's infrastructure, deployment, and environment variables to help you understand the full technical landscape beyond just the codebase.

## Deployment Platform

Our application is deployed on **Railway**, a modern PaaS (Platform as a Service):

- **Deployment Model**: The codebase is linked to a GitHub repository. Changes pushed to the main branch are automatically deployed.
- **Project Structure**: We have two separate Railway projects:
  - `robust-joy` - Contains our main web application service
  - `sunny-smile` - Contains our PostgreSQL database service

## Database Architecture

- **Type**: PostgreSQL 16.8 (hosted on Railway)
- **Access**: The database exists in a separate Railway project from the application
- **Connection**: Our app connects to the database via the `DATABASE_URL` environment variable
- **Schema Management**: We use Alembic for database migrations 
- **Tables**: Currently includes users, chat_sessions, chat_messages, tenants, and alembic_version

## Environment Variables

### Database Service (`sunny-smile` project)
- `DATABASE_URL`: PostgreSQL connection string
- `DATABASE_PUBLIC_URL`: Public connection string
- `PGDATA`: PostgreSQL data directory
- `PGDATABASE`: Database name
- `PGHOST`: Database host
- `PGPASSWORD`: Database password
- `PGPORT`: Database port (likely 5432)
- `PGUSER`: Database username
- `POSTGRES_DB`: Same as PGDATABASE
- `POSTGRES_PASSWORD`: Same as PGPASSWORD
- `POSTGRES_USER`: Same as PGUSER

### Web Service (`robust-joy` project)
- `SECRET_KEY`: Used for encrypting sensitive tenant data in the database
- `JWT_SECRET_KEY`: Used for signing JWT tokens
- `ALGORITHM`: JWT algorithm (HS256)
- `ACCESS_TOKEN_EXPIRE_MINUTES`: JWT token expiry duration
- `OPENAI_API_KEY`: API key for OpenAI services
- `PINECONE_API_KEY`: API key for Pinecone vector database
- `INDEX_NAME`: Pinecone index name
- `DEEPSEEK_API_KEY`: API key for DeepSeek (if used)
- `BACKEND_CORS_ORIGINS`: Allowed frontend domains for CORS
- `DATABASE_URL`: Connection string to the PostgreSQL database
- `PORT`: Port the application listens on

## Multi-Tenant Architecture

The application has been recently modified to support a multi-tenant architecture:

- **Design Pattern**: Single backend service serving multiple distinct frontend applications
- **Tenant Identification**: Uses middleware to extract tenant ID from JWT claims
- **Resource Isolation**: Each tenant has dedicated configuration (Pinecone environment, API keys, etc.)
- **Data Model**: `Tenant` model in the database with encrypted API keys

## Authentication System

- **Method**: JWT (JSON Web Tokens) with Bearer scheme
- **Token Generation**: Located in `auth/utils.py` 
- **Secret Management**: Uses `JWT_SECRET_KEY` from environment variables
- **User Management**: Database-backed user system with hashed passwords
- **Admin Access**: Uses `is_superuser` flag in the User model

## Database Migrations

- **Tool**: Alembic
- **Configuration**: In the `alembic` directory
- **Migration Files**: In `alembic/versions/` 
- **Recent Migrations**:
  - `796c6b7bfada`: Initial migration (user, chat tables)
  - `8adfa00db3b0`: Add Tenant model
  - `66c9e4aafc9e`: Add system_prompt to Tenant model
  - `3e87f40d465d`: Add is_superuser field to User model

## Testing and Debugging

- **Local Development**: Ensure environment variables match Railway configuration
- **Database Access**: 
  - Install PostgreSQL client tools locally
  - Use Railway CLI for database connection: 
    ```
    railway link project --name="sunny-smile"
    railway connect
    ```
- **Logs**: Available in the Railway dashboard under the "Logs" tab for each service

## Deployment Workflow

1. Code changes are pushed to GitHub repository
2. Railway automatically builds and deploys the application
3. Database migrations should be run manually after deployment:
   ```
   # Connect to Railway app service
   railway shell
   
   # Run migrations
   alembic upgrade head
   ```

## External Services Integration

- **OpenAI**: Used for embeddings and chat capabilities
- **Pinecone**: Vector database for semantic search/retrieval
- **DeepSeek**: Potentially used for additional AI capabilities

## Security Considerations

- **Secret Management**: All API keys and sensitive data stored as Railway environment variables
- **Tenant Isolation**: Critical - data leakage between tenants must be prevented
- **API Key Encryption**: Tenant API keys are encrypted in the database using sqlalchemy-utils.EncryptedType

## Frontend Integration

- **CORS Configuration**: The `BACKEND_CORS_ORIGINS` environment variable controls which domains can access the API
- **Authentication Flow**: Frontend needs to include Authorization header with Bearer token

## Command-Line Access

- **Railway CLI**: Used for managing and accessing Railway services
- **Key Commands**:
  - `railway link`: Connect to a Railway project
  - `railway connect`: Connect to PostgreSQL database
  - `railway shell`: Open a shell in the application container
  - `railway run`: Run a command in the application environment
  - `railway vars`: Manage environment variables