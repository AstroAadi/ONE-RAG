from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import upload, query, knowledge_base, settings, markdown_output
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Initialize FastAPI app
app = FastAPI(
    title="RAG Knowledge Base API",
    description="RAG System where users can upload documents, process them, and query the knowledge base using LLMs.",
    version="1.0.0"
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


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "RAG Knowledge Base API",
        "version": "1.0.0",
        "model": "Claude 3.5 Sonnet",
        "features": [
            "Simple RAG with single model",
            "PDF and DOCX support",
            "Vector search with ChromaDB",
            "Runtime model switching"
        ],
        "endpoints": {
            "upload_file": "POST /api/upload/",
            "process_document": "POST /api/upload/process/{file_id}",
            "query": "POST /api/query/",
            "get_status": "GET /api/knowledge-base/status",
            "delete_document": "DELETE /api/knowledge-base/document/{file_id}",
            "reset": "DELETE /api/knowledge-base/reset",
            "get_model": "GET /api/settings/model",
            "update_model": "POST /api/settings/model",
            "docs": "GET /docs"
        }
    }


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "rag_enabled": True
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)