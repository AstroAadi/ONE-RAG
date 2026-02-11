"""
Database Session Management
Handles PostgreSQL and SQLite connection and session lifecycle
"""

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool, QueuePool
from typing import Generator
import logging

from app.db.models import Base
from app.core.config import get_settings

logger = logging.getLogger(__name__)

# Get settings
settings = get_settings()

# Create engine based on database type
if settings.is_sqlite:
    # SQLite configuration
    engine = create_engine(
        settings.database_url,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        echo=settings.db_echo
    )
    
    # Enable foreign keys for SQLite
    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_conn, connection_record):
        """Enable foreign key constraints in SQLite"""
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()
        
elif settings.is_postgresql:
    # PostgreSQL configuration
    engine = create_engine(
        settings.database_url,
        poolclass=QueuePool,
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_max_overflow,
        pool_timeout=settings.db_pool_timeout,
        pool_recycle=settings.db_pool_recycle,
        pool_pre_ping=True,
        echo=settings.db_echo,
        connect_args={
            "connect_timeout": 10,
            "options": "-c statement_timeout=30000"
        }
    )
else:
    # Generic database configuration
    engine = create_engine(
        settings.database_url,
        poolclass=QueuePool,
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_max_overflow,
        echo=settings.db_echo
    )


# Create session factory
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)


def init_db() -> None:
    """
    Initialize database - create all tables
    
    Should be called on application startup
    """
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Database initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize database: {e}")
        raise


def get_db() -> Generator[Session, None, None]:
    """
    Dependency injection for database sessions
    
    Usage in FastAPI:
        @app.get("/items")
        def get_items(db: Session = Depends(get_db)):
            ...
    
    Yields:
        Database session that automatically closes after use
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_db_session() -> Session:
    """
    Get a database session for manual use
    
    Note: Remember to close the session manually!
    
    Returns:
        Database session
    """
    return SessionLocal()


def close_db_connections() -> None:
    """
    Close all database connections
    
    Should be called on application shutdown
    """
    try:
        engine.dispose()
        logger.info("✅ Database connections closed")
    except Exception as e:
        logger.error(f"❌ Error closing database connections: {e}")