# Plan to Fix Alembic Database Connection Error

## Problem

The application crashes during deployment on Railway with a `ModuleNotFoundError: No module named 'psycopg2'`. This occurs because the Alembic migration tool (`alembic/env.py`) is attempting to use the older `psycopg2` database driver, which isn't installed. The root cause is likely an incorrectly formatted database connection URL (e.g., starting with `postgresql+psycopg2://`) being used by Alembic, despite the correct `psycopg` v3 driver being installed (`psycopg[binary]`).

## Solution

Modify `alembic/env.py` to stop its redundant database URL parsing. Instead, it should import and utilize the correctly formatted synchronous database URL (`sync_db_url`, which starts with `postgresql://`) that is already being generated and validated within `models/db_models.py`. This ensures consistency and directs SQLAlchemy (via Alembic) to use the installed `psycopg` v3 driver.

## Diagram

```mermaid
graph TD
    A[Railway Deployment] --> B{Alembic Migration};
    B -- Uses --> C{Database URL};
    C -- Currently Incorrectly Parsed in --> D[alembic/env.py];
    D -- Causes --> E[SQLAlchemy tries psycopg2];
    E -- Leads to --> F[Error: ModuleNotFoundError psycopg2];

    G[models/db_models.py] -- Correctly Parses --> H[sync_db_url (postgresql://)];
    I[Proposed Fix: Modify alembic/env.py] -- To Use --> H;
    I -- Should Lead to --> J{Alembic uses correct URL};
    J -- Allows --> K[SQLAlchemy uses psycopg v3];
    K -- Resolves --> F;
```

## Next Steps

1.  Apply the code change to `alembic/env.py` (requires switching to Code mode).
2.  Commit the change to Git.
3.  Push the commit to the branch monitored by Railway.
4.  Monitor the Railway deployment.
5.  Verify that the Alembic migrations now run successfully (either automatically or via manual `alembic upgrade head`).