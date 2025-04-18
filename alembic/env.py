import asyncio
import sys
import platform
from logging.config import fileConfig
from dotenv import load_dotenv # Add dotenv
import os # Make sure os is imported

# Set asyncio policy for Windows compatibility with psycopg/asyncpg
if platform.system() == "Windows":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from sqlalchemy import engine_from_config, create_engine # Add create_engine
from sqlalchemy import pool

from alembic import context

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
# Load .env file if it exists
load_dotenv()

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# add your model's MetaData object here
# for 'autogenerate' support
# from myapp import mymodel
# target_metadata = mymodel.Base.metadata
# Add import for Base from your models file
import sys
import os
# Ensure the project root is in the path to find 'models'
sys.path.insert(0, os.path.realpath(os.path.join(os.path.dirname(__file__), '..')))
from models.db_models import Base, sync_db_url # Import Base and the CORRECT sync URL

target_metadata = Base.metadata

# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    # Use DATABASE_URL directly as configured in alembic.ini via environment variable
    url = DATABASE_URL
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


# Use standard synchronous engine setup for online mode
def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    """
    # Use the sync_db_url directly imported from models.db_models, which handles URL parsing
    if not sync_db_url:
        raise ValueError("sync_db_url could not be imported or is empty. Check models/db_models.py and environment variables.")

    # Create engine using the imported synchronous URL
    connectable = create_engine(sync_db_url, poolclass=pool.NullPool)

    with connectable.connect() as connection:
        context.configure(
            connection=connection, target_metadata=target_metadata
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online() # Call the synchronous function directly
