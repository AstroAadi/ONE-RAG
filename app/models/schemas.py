"""
Pydantic Schemas for RAG System with Hybrid Search
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any
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
    question: str = Field(..., min_length=1, description="Question to ask")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of chunks to retrieve")
    file_id: Optional[str] = Field(None, description="Optional file ID to search within")
    temperature: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="LLM temperature (0.0-1.0)")
    
    # Hybrid search weights
    keyword_weight: Optional[float] = Field(
        default=0.3, 
        ge=0.0, 
        le=1.0, 
        description="Weight for keyword (BM25) search. Default: 0.3"
    )
    semantic_weight: Optional[float] = Field(
        default=0.7, 
        ge=0.0, 
        le=1.0, 
        description="Weight for semantic search. Default: 0.7"
    )
    
    @field_validator('semantic_weight')
    @classmethod
    def validate_weights_sum(cls, v, info):
        """Ensure keyword_weight + semantic_weight = 1.0"""
        keyword_weight = info.data.get('keyword_weight', 0.3)
        total = keyword_weight + v
        
        if abs(total - 1.0) > 0.01:  # Allow small floating point differences
            # Auto-normalize
            normalized_semantic = v / total
            normalized_keyword = keyword_weight / total
            info.data['keyword_weight'] = normalized_keyword
            return normalized_semantic
        
        return v


class ConversationQueryRequest(QueryRequest):
    """Query request with conversation tracking"""
    conversation_id: str = Field(
        ...,
        description="Unique conversation ID for history tracking"
    )


class SourceDocument(BaseModel):
    content: str
    filename: str
    chunk_id: str
    relevance_score: float
    page_number: Optional[int] = 0
    chunk_type: Optional[str] = "text"
    section: Optional[str] = None
    
    # Hybrid search scores
    semantic_score: Optional[float] = None
    keyword_score: Optional[float] = None


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[SourceDocument]
    processing_time: float
    search_strategy: Optional[str] = "hybrid"
    confidence_score: Optional[float] = 1.0
    conversation_id: Optional[str] = None
    
    # Hybrid search info
    keyword_weight: Optional[float] = None
    semantic_weight: Optional[float] = None


class KnowledgeBaseStatus(BaseModel):
    total_documents: int
    total_chunks: int
    documents: List[dict]
    bm25_indexed: Optional[int] = 0


class DocumentStatistics(BaseModel):
    """Statistics for a processed document"""
    file_id: str
    filename: str
    total_pages: int
    total_chunks: int
    text_blocks: int
    tables: int
    images: int
    charts: int
    headings: int


class HallucinationCheck(BaseModel):
    """Result of hallucination detection"""
    is_hallucination: bool
    confidence_score: float
    unsupported_claims: List[str]
    warning_message: Optional[str]