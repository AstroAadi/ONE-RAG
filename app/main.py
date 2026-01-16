"""
Main Application Entry Point - UPDATED WITH DATABASE INITIALIZATION
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import chunk_edit, upload, query, knowledge_base, settings, markdown_output
from app.db.session import init_db, close_db_connections
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="RAG Knowledge Base API",
    description="RAG System with user isolation, database-backed history, and multi-format document support.",
    version="2.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(upload.router)
app.include_router(query.router)
app.include_router(knowledge_base.router)
app.include_router(settings.router)
app.include_router(markdown_output.router)
app.include_router(chunk_edit.router)


@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    logger.info("="*60)
    logger.info("Starting RAG Knowledge Base API v2.0")
    logger.info("="*60)
    
    # 1. Initialize database
    try:
        init_db()
        logger.info("✓ Database initialized successfully")
    except Exception as e:
        logger.error(f"✗ Failed to initialize database: {e}")
        raise
    
    # 2. Initialize extractor factory
    try:
        from app.services.extraction.extractor_factory import auto_register_extractors
        auto_register_extractors()
        logger.info("✓ Document extractors registered")
    except Exception as e:
        logger.warning(f"⚠ Could not auto-register extractors: {e}")
    
    # 3. Verify services
    try:
        from app.services.vector_store_service import VectorStoreService
        vector_store = VectorStoreService()
        count = vector_store.collection.count()
        logger.info(f"✓ Vector store initialized ({count} chunks)")
    except Exception as e:
        logger.warning(f"⚠ Vector store check failed: {e}")
    
    logger.info("="*60)
    logger.info("Application started successfully!")
    logger.info("API Documentation: http://localhost:8000/docs")
    logger.info("="*60)


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down RAG Knowledge Base API...")
    
    try:
        close_db_connections()
        logger.info("✓ Database connections closed")
    except Exception as e:
        logger.error(f"✗ Error during shutdown: {e}")
    
    logger.info("Application shutdown complete")


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "RAG Knowledge Base API v2.0",
        "status": "operational",
        "features": [
            "✓ Mandatory user authentication",
            "✓ Database-backed conversation history (FIFO)",
            "✓ User-isolated document storage",
            "✓ Multi-format document support (PDF, DOCX, PPTX, Images)",
            "✓ Hybrid search (BM25 + Semantic)",
            "✓ Vision-enabled LLM for image analysis",
            "✓ Per-user data management"
        ],
        "version": "2.0.0",
        "model": "Claude 3.5 Sonnet",
        "authentication": "Required (X-User-ID header)",
        "endpoints": {
            "health": "GET /api/health",
            "docs": "GET /docs",
            "upload_file": "POST /api/upload/",
            "process_document": "POST /api/upload/process/{file_id}",
            "query": "POST /api/query/",
            "conversation_query": "POST /api/query/conversation",
            "get_kb_status": "GET /api/knowledge-base/status",
            "delete_document": "DELETE /api/knowledge-base/document/{file_id}",
            "reset_user_data": "DELETE /api/knowledge-base/reset",
            "get_model": "GET /api/settings/model",
            "update_model": "POST /api/settings/model"
        },
        "documentation": "http://localhost:8000/docs"
    }


@app.get("/api/health")
async def health_check():
    """Comprehensive health check endpoint"""
    from datetime import datetime
    
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "2.0.0",
        "checks": {}
    }
    
    # Check database
    try:
        from app.db.session import get_db
        db = next(get_db())
        db.execute("SELECT 1")
        db.close()
        health_status["checks"]["database"] = "✓ operational"
    except Exception as e:
        health_status["checks"]["database"] = f"✗ error: {str(e)}"
        health_status["status"] = "unhealthy"
    
    # Check vector store
    try:
        from app.services.vector_store_service import VectorStoreService
        vector_store = VectorStoreService()
        count = vector_store.collection.count()
        health_status["checks"]["vector_store"] = f"✓ operational ({count} chunks indexed)"
    except Exception as e:
        health_status["checks"]["vector_store"] = f"✗ error: {str(e)}"
        health_status["status"] = "degraded"
    
    return health_status


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)