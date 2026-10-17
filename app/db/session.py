"""Compatibility imports for the shared PostgreSQL session."""

from sqlalchemy.orm import Session

from app.core.database import (
    SessionLocal,
    close_database_connections as close_db_connections,
    create_all_tables as init_db,
    engine,
    get_db,
)


def get_db_session() -> Session:
    """Create a session from the application's shared database engine."""
    return SessionLocal()
