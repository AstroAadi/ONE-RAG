"""
Enhanced Pydantic Schemas for Advanced RAG System
"""

from pydantic import BaseModel, Field
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
    structured_data_path: Optional[str] = None


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Question to ask")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of chunks to retrieve")
    file_id: Optional[str] = Field(None, description="Optional file ID to search within")


class EnhancedQueryRequest(QueryRequest):
    """Extended query request with advanced options"""
    search_mode: Optional[str] = Field(
        default="hybrid",
        description="Search strategy: 'semantic', 'parent', or 'hybrid'"
    )
    include_visual_context: bool = Field(
        default=True,
        description="Include image/chart descriptions in context"
    )
    metadata_filters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional metadata filters"
    )


class ConversationQueryRequest(EnhancedQueryRequest):
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
    has_parent_context: Optional[bool] = False


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[SourceDocument]
    processing_time: float
    search_strategy: Optional[str] = "hybrid"
    confidence_score: Optional[float] = 1.0
    conversation_id: Optional[str] = None


class KnowledgeBaseStatus(BaseModel):
    total_documents: int
    total_chunks: int
    total_child_chunks: Optional[int] = 0
    total_parent_chunks: Optional[int] = 0
    documents: List[dict]


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
    has_ai_analysis: bool


class ChunkMetadata(BaseModel):
    """Detailed chunk metadata"""
    chunk_id: str
    chunk_index: int
    chunk_type: str
    char_count: int
    page_number: int
    section_hierarchy: List[str]
    parent_chunk_id: Optional[str]
    is_parent: bool
    has_table: bool
    has_image: bool


class SearchFilters(BaseModel):
    """Filters for metadata search"""
    file_id: Optional[str] = None
    page_number: Optional[int] = None
    chunk_type: Optional[str] = None
    section: Optional[str] = None
    min_relevance: Optional[float] = 0.0


class QueryClassification(BaseModel):
    """Query intent classification result"""
    query_type: str  # factual, analytical, comparison, summary, definition
    needs_table_data: bool
    needs_visual_data: bool
    is_temporal: bool
    complexity: str  # simple, medium, complex
    suggested_search_mode: str


class HallucinationCheck(BaseModel):
    """Result of hallucination detection"""
    is_hallucination: bool
    confidence_score: float
    unsupported_claims: List[str]
    warning_message: Optional[str]