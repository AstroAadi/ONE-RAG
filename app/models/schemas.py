from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class FileUploadResponse(BaseModel):
    file_id: str
    filename: str
    file_type: str
    upload_time: datetime
    status: str
    message: str


class DocumentProcessResponse(BaseModel):
    file_id: str
    filename: str
    status: str
    markdown_path: Optional[str] = None
    chunks_created: int
    message: str


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Question to ask the knowledge base")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of relevant chunks to retrieve")


class SourceDocument(BaseModel):
    content: str
    filename: str
    chunk_id: str
    relevance_score: float


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[SourceDocument]
    processing_time: float


class KnowledgeBaseStatus(BaseModel):
    total_documents: int
    total_chunks: int
    documents: List[dict]