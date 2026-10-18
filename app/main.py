"""
Main FastAPI Application - CONCURRENT VERSION
Supports multiple concurrent users with background task processing
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging
import sys
import time
from datetime import datetime

for stream in (sys.stdout, sys.stderr):
    reconfigure = getattr(stream, "reconfigure", None)
    if callable(reconfigure):
        reconfigure(encoding="utf-8", errors="backslashreplace")

# Import configurations
from app.core.config import settings
from app.core.database import create_all_tables, close_database_connections, check_database_health
from app.core.task_manager import get_task_manager, shutdown_task_manager

# Import routes
from app.api.routes import (
    upload,
    query,
    task_routes,
    knowledge_base,
    knowledge_bases,
    chunking,
    chunk_edit,
    markdown_output,
    settings as settings_route
)

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(settings.log_file_path, encoding="utf-8"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager
    
    Startup:
    - Create database tables
    - Initialize task manager
    - Log startup info
    
    Shutdown:
    - Shutdown task manager
    - Close database connections
    - Cleanup resources
    """
    # Startup
    logger.info("=" * 70)
    logger.info("RAG SYSTEM STARTING")
    logger.info("=" * 70)
    
    # Create database tables
    try:
        create_all_tables()
        logger.info("✅ Database tables created/verified")
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")
        raise
    
    # Initialize task manager
    try:
        task_manager = get_task_manager()
        logger.info(
            f"✅ Task manager initialized: "
            f"I/O workers={settings.max_io_workers}, "
            f"CPU workers={settings.max_cpu_workers}"
        )
    except Exception as e:
        logger.error(f"❌ Task manager initialization failed: {e}")
        raise
    
    # Check database health
    db_healthy, db_message = check_database_health()
    if db_healthy:
        logger.info(f"✅ {db_message}")
    else:
        logger.warning(f"⚠️  {db_message}")
    
    logger.info("=" * 70)
    logger.info("RAG SYSTEM READY - Accepting requests")
    logger.info("=" * 70)
    
    yield
    
    # Shutdown
    logger.info("=" * 70)
    logger.info("RAG SYSTEM SHUTTING DOWN")
    logger.info("=" * 70)
    
    # Shutdown task manager (wait for tasks to complete)
    try:
        logger.info("Shutting down task manager...")
        shutdown_task_manager()
        logger.info("✅ Task manager shutdown complete")
    except Exception as e:
        logger.error(f"❌ Task manager shutdown error: {e}")
    
    # Close database connections
    try:
        logger.info("Closing database connections...")
        close_database_connections()
        logger.info("✅ Database connections closed")
    except Exception as e:
        logger.error(f"❌ Database shutdown error: {e}")
    
    logger.info("=" * 70)
    logger.info("RAG SYSTEM SHUTDOWN COMPLETE")
    logger.info("=" * 70)


# Create FastAPI application
app = FastAPI(
    title="RAG System API",
    description="Production-ready RAG system with support for thousands of concurrent users",
    version="2.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests and their processing time"""
    start_time = time.time()
    
    # Get user_id from header if available
    user_id = request.headers.get("X-User-ID")
    
    # Process request
    response = await call_next(request)
    
    # Calculate processing time
    process_time = time.time() - start_time
    
    # Log request
    logger.info(
        f"{request.method} {request.url.path} - "
        f"User: {user_id} - "
        f"Status: {response.status_code} - "
        f"Time: {process_time:.3f}s"
    )
    
    # Add processing time header
    response.headers["X-Process-Time"] = str(process_time)
    
    return response


# Exception handlers
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unhandled errors"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred. Please try again later.",
            "timestamp": datetime.utcnow().isoformat()
        }
    )


# Include routers.
# chunking must be registered before chunk_edit: both live under /api/chunks,
# and chunk_edit's "/{chunk_id}" would otherwise shadow "/preview", "/create"
# and "/finalize".
app.include_router(upload.router)
app.include_router(query.router)
app.include_router(task_routes.router)
app.include_router(knowledge_bases.router)
app.include_router(knowledge_base.router)
app.include_router(chunking.router)
app.include_router(chunk_edit.router)
app.include_router(markdown_output.router)
app.include_router(settings_route.router)


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with system information"""
    task_manager = get_task_manager()
    
    return {
        "message": "RAG System API - Concurrent Processing Enabled",
        "version": "2.0.0",
        "status": "online",
        "timestamp": datetime.utcnow().isoformat(),
        "features": {
            "concurrent_processing": True,
            "background_tasks": True,
            "connection_pooling": True,
            "worker_pools": {
                "io_workers": settings.max_io_workers,
                "cpu_workers": settings.max_cpu_workers
            }
        },
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "metrics": "/metrics",
            "upload": "/api/upload/file-upload",
            "upload_status": "/api/upload/status/{file_id}",
            "chunks_preview": "/api/chunks/preview",
            "chunks_create": "/api/chunks/create",
            "chunks_finalize": "/api/chunks/finalize",
            "query": "/api/query",
            "tasks": "/api/tasks",
            "knowledge_bases": "/api/knowledge-bases",
            "settings": "/api/settings/models"
        }
    }


# Health check endpoint
@app.get("/health")
async def health_check():
    """
    Comprehensive health check endpoint
    
    Checks:
    - Database connectivity
    - Task manager status
    - System resources
    """
    # Check database
    db_healthy, db_message = check_database_health()
    
    # Get task manager stats
    task_manager = get_task_manager()
    task_stats = task_manager.get_stats()
    
    # Determine overall health
    is_healthy = db_healthy
    status = "healthy" if is_healthy else "unhealthy"
    
    return {
        "status": status,
        "timestamp": datetime.utcnow().isoformat(),
        "components": {
            "database": {
                "healthy": db_healthy,
                "message": db_message
            },
            "task_manager": {
                "healthy": True,
                "stats": task_stats
            }
        },
        "system": {
            "version": "2.0.0",
            "mode": "concurrent",
            "io_workers": settings.max_io_workers,
            "cpu_workers": settings.max_cpu_workers
        }
    }


# Metrics endpoint (for monitoring)
@app.get("/metrics")
async def get_metrics():
    """
    System metrics endpoint
    
    Returns detailed metrics about:
    - Task processing
    - Database connections
    - Request statistics
    """
    from app.core.database import get_connection_pool_status
    
    task_manager = get_task_manager()
    
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "task_manager": task_manager.get_stats(),
        "database": get_connection_pool_status(),
        "configuration": {
            "io_workers": settings.max_io_workers,
            "cpu_workers": settings.max_cpu_workers,
            "db_pool_size": settings.db_pool_size,
            "db_max_overflow": settings.db_max_overflow
        }
    }


if __name__ == "__main__":
    import uvicorn
    
    # Run with uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        # reload=False,  # Disable in production
        # workers=1,  # Single worker (task manager handles concurrency)
        # log_level=settings.log_level.lower()
    )