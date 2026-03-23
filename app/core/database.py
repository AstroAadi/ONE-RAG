"""
Database Configuration with Connection Pooling
PostgreSQL and SQLite Support with proper pooling
"""

from sqlalchemy import create_engine, event, inspect, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool, NullPool
from contextlib import contextmanager
from typing import Generator
import logging
from app.core.config import settings

logger = logging.getLogger(__name__)

# Import Base from models
from app.db.models import Base


def get_engine():
    """Create SQLAlchemy engine with optimized connection pooling"""
    
    # Determine database type
    is_sqlite = settings.is_sqlite
    is_postgresql = settings.is_postgresql
    
    if is_sqlite:
        # SQLite configuration
        logger.warning("Using SQLite - not recommended for production with high concurrency")
        
        engine = create_engine(
            settings.database_url,
            connect_args={"check_same_thread": False},
            poolclass=NullPool,
            echo=settings.db_echo
        )
        
        # Enable foreign keys for SQLite
        @event.listens_for(engine, "connect")
        def set_sqlite_pragma(dbapi_conn, connection_record):
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()
            
    elif is_postgresql:
        # PostgreSQL configuration with connection pooling
        logger.info("Using PostgreSQL with connection pooling")
        
        engine = create_engine(
            settings.database_url,
            poolclass=QueuePool,
            pool_size=settings.db_pool_size,
            max_overflow=settings.db_max_overflow,
            pool_timeout=settings.db_pool_timeout,
            pool_recycle=settings.db_pool_recycle,
            pool_pre_ping=True,  # Verify connections before use
            echo=settings.db_echo,
            connect_args={
                "connect_timeout": 10,
                "options": "-c statement_timeout=30000"  # 30 second query timeout
            }
        )
    else:
        # Fallback for other databases (MySQL, etc.)
        engine = create_engine(
            settings.database_url,
            poolclass=QueuePool,
            pool_size=settings.db_pool_size,
            max_overflow=settings.db_max_overflow,
            pool_timeout=settings.db_pool_timeout,
            pool_recycle=settings.db_pool_recycle,
            pool_pre_ping=True,
            echo=settings.db_echo
        )
    
    logger.info(
        f"Database engine created: "
        f"pool_size={getattr(engine.pool, 'size', lambda: 'N/A')()}, "
        f"max_overflow={getattr(engine.pool, '_max_overflow', 'N/A')}"
    )
    
    return engine


# Create engine
engine = get_engine()

# Create session factory
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)


def get_db() -> Generator[Session, None, None]:
    """Dependency function for FastAPI to get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def get_db_context():
    """Context manager for database session (use outside FastAPI)"""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def create_all_tables():
    """Create all database tables using the models Base"""
    try:
        Base.metadata.create_all(bind=engine)
        _ensure_file_metadata_schema_compatibility()
        logger.info("✅ Database tables created/verified")
    except Exception as e:
        logger.error(f"❌ Failed to create database tables: {e}")
        raise


def _ensure_file_metadata_schema_compatibility():
    """
    Backfill columns that were added to the SQLAlchemy model after the
    underlying database table already existed.

    `create_all()` only creates missing tables; it does not alter existing ones.
    """
    inspector = inspect(engine)

    if not inspector.has_table("file_metadata"):
        return

    existing_columns = {
        column_info["name"]
        for column_info in inspector.get_columns("file_metadata")
    }

    if "parent_chunk_size" not in existing_columns:
        logger.warning(
            "Schema drift detected: file_metadata.parent_chunk_size is missing. "
            "Applying compatibility patch."
        )
        with engine.begin() as connection:
            connection.execute(
                text("ALTER TABLE file_metadata ADD COLUMN parent_chunk_size INTEGER")
            )
        logger.info("✅ Added file_metadata.parent_chunk_size column")


def drop_all_tables():
    """Drop all database tables (use with caution!)"""
    Base.metadata.drop_all(bind=engine)
    logger.warning("All database tables dropped")


def get_connection_pool_status() -> dict:
    """Get current connection pool status"""
    if isinstance(engine.pool, NullPool):
        return {
            "pool_type": "NullPool",
            "message": "No connection pooling (SQLite mode)"
        }
    
    pool = engine.pool
    
    return {
        "pool_type": pool.__class__.__name__,
        "pool_size": pool.size(),
        "checked_in": pool.checkedin(),
        "checked_out": pool.checkedout(),
        "overflow": pool.overflow(),
        "max_overflow": getattr(pool, '_max_overflow', None),
        "total_connections": pool.checkedin() + pool.checkedout() + pool.overflow()
    }


def check_database_health() -> tuple[bool, str]:
    """Check if database is accessible and healthy"""
    try:
        with get_db_context() as db:
            db.execute(text("SELECT 1"))
        return True, "Database connection healthy"
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return False, f"Database error: {str(e)}"


def close_database_connections():
    """Close all database connections and dispose of the engine"""
    engine.dispose()
    logger.info("Database connections closed")
