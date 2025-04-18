# auth/dependencies.py
from fastapi import Depends, HTTPException, status, Request # Added Request
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError
from sqlalchemy.ext.asyncio import AsyncSession
import pinecone # Added pinecone
import os # Added os for potential fallback/error logging if needed
from typing import Optional # Added Optional
from langchain_pinecone import PineconeVectorStore # Added
from langchain_openai import OpenAIEmbeddings # Added
from langchain.vectorstores.base import VectorStoreRetriever # Added for type hint
import logging # Added for logging

# Configure logger (basic example, adjust as needed)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Assuming db_models, user_crud, auth_utils, user_schemas are importable
# Adjust paths if necessary based on your project structure
from models.db_models import User, Tenant, get_db # Added Tenant
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

# Dependency to get the current user and verify they are an admin
async def get_current_active_admin_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """
    Dependency that fetches the current authenticated user and verifies
    if they have superuser (admin) privileges.

    Raises:
        HTTPException 403: If the user is not a superuser.
    """
    if not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized - Admin privileges required"
        )
    # Optional: Add an is_active check here as well if needed
    # if not current_user.is_active:
    #     raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

# --- Tenant and Pinecone Dependencies ---

async def get_current_tenant(request: Request) -> Tenant:
    """
    Retrieves the tenant object attached to the request state by the middleware.
    Raises HTTPException if the tenant is not found (should not happen with middleware).
    """
    tenant = getattr(request.state, "tenant", None)
    if tenant is None:
        # This should ideally not happen if the middleware is applied correctly
        # to the routes using this dependency.
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Tenant context not found in request state. Middleware might be missing or failed.",
        )
    if not isinstance(tenant, Tenant):
         # Type check for safety
         raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Invalid tenant object type in request state: {type(tenant)}",
        )
    return tenant

def get_pinecone_client(tenant: Tenant = Depends(get_current_tenant)) -> pinecone.Index:
    """
    Initializes and returns a Pinecone Index client scoped to the current tenant.
    Uses credentials stored in the tenant object.
    """
    if not tenant.pinecone_api_key or not tenant.pinecone_environment or not tenant.pinecone_index_name:
        # Log this error as it indicates incomplete tenant configuration
        logger.error(f"Tenant {tenant.id} ({tenant.name}) missing Pinecone configuration.") # Keep as error, it's config issue
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Tenant '{tenant.name}' is missing required Pinecone configuration (API Key, Environment, or Index Name).",
        )
 
    try:
        logger.info(f"Initializing Pinecone for tenant {tenant.id} ({tenant.name}) in env {tenant.pinecone_environment}")
        pinecone.init(
            api_key=tenant.pinecone_api_key,
            environment=tenant.pinecone_environment
        )
        # Assuming you always want to work with a specific index per tenant
        logger.info(f"Getting Pinecone index '{tenant.pinecone_index_name}' for tenant {tenant.id}")
        index = pinecone.Index(tenant.pinecone_index_name)
        # Optional: Add a quick check to see if the index connection works
        # index.describe_index_stats()
        logger.info(f"Successfully initialized Pinecone index '{tenant.pinecone_index_name}' for tenant {tenant.id}")
        return index
    except Exception as e:
        # Log the detailed error from Pinecone
        logger.exception(f"Failed to initialize Pinecone for tenant {tenant.id} ({tenant.name}): {e}") # Use logger.exception
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Could not connect to Pinecone for tenant '{tenant.name}'. See server logs for details.", # Generic client message
        )

# --- Embedding and Retriever Dependencies ---

# Global OpenAI Embeddings (assuming key is global)
# Consider if this needs tenant-specific handling later
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.critical("OPENAI_API_KEY environment variable not set! Embeddings/Retriever will fail.")
    # Depending on application structure, might raise an error here or handle downstream

# Initialize embeddings globally if key exists
# Handle potential errors during initialization
openai_embeddings: Optional[OpenAIEmbeddings] = None
try:
    if OPENAI_API_KEY:
        openai_embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    else:
         logger.warning("OpenAI Embeddings not initialized due to missing API key.")
except Exception as e:
    logger.exception(f"Failed to initialize OpenAI Embeddings: {e}") # Use logger.exception
    openai_embeddings = None # Ensure it's None on error


def get_openai_embeddings() -> OpenAIEmbeddings:
    """Dependency to provide the globally initialized OpenAI Embeddings."""
    logger.debug("Entering get_openai_embeddings dependency")
    if openai_embeddings is None:
        logger.error("OpenAI Embeddings requested but not available (missing key or init error).") # Log specific reason
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="OpenAI Embeddings are not available. Please check server configuration." # Generic client message
        )
    logger.debug("Returning initialized OpenAI Embeddings")
    return openai_embeddings

def get_retriever(
    tenant: Tenant = Depends(get_current_tenant),
    # Pinecone client dependency ensures pinecone.init is called with tenant creds
    # We don't directly use the returned index object here, but rely on the side effect
    # of pinecone.init being called correctly before PineconeVectorStore.
    _pinecone_client: pinecone.Index = Depends(get_pinecone_client), # Renamed to avoid clash, underscore indicates it's used for side-effect
    embeddings: OpenAIEmbeddings = Depends(get_openai_embeddings)
) -> VectorStoreRetriever:
    """
    Creates and returns a Langchain VectorStoreRetriever scoped to the current tenant's
    Pinecone index and using the global OpenAI embeddings.
    """
    if not tenant.pinecone_index_name:
         raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Tenant '{tenant.name}' is missing Pinecone Index Name configuration.",
        )
 
    try:
        logger.info(f"Attempting to create PineconeVectorStore for tenant '{tenant.name}' index '{tenant.pinecone_index_name}'")
        vector_store = PineconeVectorStore(
            index_name=tenant.pinecone_index_name,
            embedding=embeddings,
            # namespace=tenant.pinecone_namespace, # Optional: Add if using namespaces per tenant
            text_key="text" # Ensure this matches your Pinecone setup
        )
        logger.info(f"PineconeVectorStore created for tenant '{tenant.name}'. Now creating retriever.")
        # Configure retriever as needed (e.g., MMR)
        retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 6,         # Number of final documents to return
                "fetch_k": 15,  # Number of documents to fetch initially for MMR analysis
                "lambda_mult": 0.7 # Diversity parameter (0=max diversity, 1=max relevance)
            }
        )
        logger.info(f"Successfully created retriever for tenant '{tenant.name}'")
        return retriever
    except Exception as e:
        # Log the detailed error
        logger.exception(f"Failed to create retriever for tenant {tenant.id} ({tenant.name}): {e}") # Use logger.exception
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Could not create vector store retriever for tenant '{tenant.name}'. See server logs for details.", # Generic client message
        )