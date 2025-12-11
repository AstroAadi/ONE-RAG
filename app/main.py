from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import upload, query, knowledge_base

# Initialize FastAPI app
app = FastAPI(
    title="RAG Knowledge Base API",
    description="API for building and querying a RAG-based knowledge base using AWS Bedrock",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(upload.router)
app.include_router(query.router)
app.include_router(knowledge_base.router)


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "RAG Knowledge Base API",
        "version": "1.0.0",
        "endpoints": {
            "upload_file": "POST /api/upload/",
            "process_document": "POST /api/upload/process/{file_id}",
            "query_knowledge_base": "POST /api/query/",
            "get_kb_status": "GET /api/knowledge-base/status",
            "delete_document": "DELETE /api/knowledge-base/document/{file_id}",
            "reset_kb": "DELETE /api/knowledge-base/reset",
            "docs": "GET /docs"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)