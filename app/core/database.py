"""
Database Configuration with Connection Pooling - CORRECTED
Uses the same Base as models.py to ensure tables are created
"""

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
from contextlib import contextmanager
from typing import Generator
import logging
from app.core.config import settings

logger = logging.getLogger(__name__)

# Import Base from models - CRITICAL!
from app.db.models import Base


def get_engine():
    """Create the PostgreSQL engine with connection pooling."""
    engine = create_engine(
        settings.database_url,
        poolclass=QueuePool,
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_max_overflow,
        pool_timeout=settings.db_pool_timeout,
        pool_recycle=settings.db_pool_recycle,
        pool_pre_ping=True,
        echo=False,
        connect_args={
            "connect_timeout": 10,
            "options": "-c statement_timeout=30000"
        }
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
        logger.info("✅ Database tables created/verified")
    except Exception as e:
        logger.error(f"❌ Failed to create database tables: {e}")
        raise


def drop_all_tables():
    """Drop all database tables (use with caution!)"""
    Base.metadata.drop_all(bind=engine)
    logger.warning("All database tables dropped")


def get_connection_pool_status() -> dict:
    """Get current connection pool status"""
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


# Health check function
def check_database_health() -> tuple[bool, str]:
    """Check if database is accessible and healthy"""
    try:
        with get_db_context() as db:
            db.execute(text("SELECT 1"))
        return True, "Database connection healthy"
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return False, f"Database error: {str(e)}"


# Cleanup function for application shutdown
def close_database_connections():
    """Close all database connections and dispose of the engine"""
    engine.dispose()
    logger.info("Database connections closed")