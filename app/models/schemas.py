"""
Updated Pydantic Schemas for Character-Based RAG System
- Chunking strategy selection
- Character-based limits only (no token limits except document parent)
- Delimiter selection
- Embedding model selection
"""

from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


# ==================== Enums ====================

class ProcessingStatus(str, Enum):
    """Document processing status"""
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ChunkingStrategy(str, Enum):
    """Chunking strategy options (character-based only)"""
    FIXED_CHARACTER = "fixed_character"
    PARENT_CHILD_PARAGRAPH = "parent_child_paragraph"
    PARENT_CHILD_DOCUMENT = "parent_child_document"


class SearchType(str, Enum):
    """Search type options"""
    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"


class EmbeddingModel(str, Enum):
    """Available embedding models"""
    COHERE_ENGLISH_V3 = "cohere.embed-english-v3"
    COHERE_MULTILINGUAL_V3 = "cohere.embed-multilingual-v3"
    AMAZON_TITAN_V1 = "amazon.titan-embed-text-v1"
    AMAZON_TITAN_V2 = "amazon.titan-embed-text-v2"


# ==================== Request Models ====================

class DocumentProcessRequest(BaseModel):
    """
    Document processing request with character-based chunking configuration
    """
    model_config = ConfigDict(str_strip_whitespace=True)
    
    # Chunking strategy
    chunking_strategy: ChunkingStrategy = Field(
        default=ChunkingStrategy.FIXED_CHARACTER,
        description="Chunking strategy to use"
    )
    
    # Character-based parameters (NO TOKEN LIMITS)
    max_chunk_chars: int = Field(
        default=2000,
        ge=500,
        le=10000,
        description="Maximum chunk size in characters (chunks can be smaller, never larger)"
    )
    
    overlap_chars: int = Field(
        default=200,
        ge=0,
        le=2000,
        description="Overlap size in characters (for fixed chunking)"
    )
    
    # Delimiter selection
    delimiter: str = Field(
        default="\n\n",
        min_length=1,
        max_length=10,
        description="Delimiter for splitting text (e.g., '\\n\\n', '\\n', '.', etc.)"
    )
    
    # Parent-child specific
    max_child_chunk_chars: int = Field(
        default=1000,
        ge=200,
        le=5000,
        description="Maximum child chunk size in characters (for parent-child strategies)"
    )
    
    # Embedding model selection
    embedding_model: EmbeddingModel = Field(
        default=EmbeddingModel.COHERE_ENGLISH_V3,
        description="Embedding model to use for generating embeddings"
    )
    
    @field_validator('overlap_chars')
    @classmethod
    def validate_overlap(cls, v, info):
        """Ensure overlap is reasonable compared to chunk size"""
        max_chars = info.data.get('max_chunk_chars', 2000)
        if v >= max_chars:
            raise ValueError(f"overlap_chars ({v}) must be less than max_chunk_chars ({max_chars})")
        return v


class FileUploadRequest(BaseModel):
    """File upload metadata (used internally)"""
    model_config = ConfigDict(from_attributes=True)
    
    user_id: str = Field(..., min_length=1, max_length=100)
    original_filename: str = Field(..., min_length=1, max_length=500)


class QueryRequest(BaseModel):
    """Query request with search type selection"""
    model_config = ConfigDict(str_strip_whitespace=True)
    
    question: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Question to ask the RAG system"
    )
    
    top_k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Number of chunks to retrieve"
    )
    
    file_id: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Optional: Restrict search to specific file"
    )
    
    temperature: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="LLM temperature (0.0-1.0)"
    )
    
    search_type: SearchType = Field(
        default=SearchType.HYBRID,
        description="Search type: 'dense', 'sparse', or 'hybrid'"
    )


class ConversationQueryRequest(QueryRequest):
    """Query request with conversation tracking"""
    conversation_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Unique conversation ID"
    )


# ==================== Response Models ====================

class FileUploadResponse(BaseModel):
    """Response after successful file upload"""
    model_config = ConfigDict(from_attributes=True)
    
    file_id: str
    filename: str
    file_type: str
    file_size_bytes: int
    upload_time: datetime
    status: ProcessingStatus
    message: str


class DocumentProcessResponse(BaseModel):
    """Response after document processing"""
    model_config = ConfigDict(from_attributes=True)
    
    file_id: str
    filename: str
    status: ProcessingStatus
    markdown_path: Optional[str] = None
    chunks_created: int
    total_pages: int = 0
    processing_time_seconds: Optional[float] = None
    statistics: Optional[Dict[str, int]] = None
    message: str
    
    # Chunking configuration used
    chunking_strategy: Optional[str] = Field(default=None)
    chunking_config: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Chunking configuration used (delimiter, max_chars, etc.)"
    )
    
    # Embedding model used
    embedding_model: Optional[str] = Field(default=None)
    
    # Chunk breakdown
    chunk_breakdown: Optional[Dict[str, int]] = Field(default=None)
    chunk_stats: Optional[Dict[str, Any]] = Field(default=None)


class SourceDocument(BaseModel):
    """Source document chunk in query results"""
    model_config = ConfigDict(from_attributes=True)
    
    content: str = Field(..., description="Chunk content")
    full_text: Optional[str] = Field(default=None, description="Complete chunk text")
    
    filename: str
    chunk_id: str
    file_id: str = Field(default="unknown")
    
    relevance_score: float
    
    page_number: int = Field(default=0, ge=0)
    chunk_type: str = Field(default="text")
    chunk_index: int = Field(default=0, ge=0)
    
    # Parent-child metadata
    is_parent: Optional[bool] = Field(default=None)
    parent_chunk_id: Optional[str] = Field(default=None)
    
    # Search scores
    semantic_score: Optional[float] = Field(default=None)
    keyword_score: Optional[float] = Field(default=None)
    
    @field_validator('relevance_score', mode='before')
    @classmethod
    def clamp_relevance_score(cls, v):
        """Ensure relevance score is within valid range"""
        try:
            score = float(v)
            return max(0.0, min(1.0, score))
        except (ValueError, TypeError):
            return 0.0


class QueryResponse(BaseModel):
    """Response to query request"""
    model_config = ConfigDict(from_attributes=True)
    
    question: str
    answer: str
    sources: List[SourceDocument]
    
    processing_time: float
    search_type: SearchType = Field(default=SearchType.HYBRID)
    confidence_score: float = Field(default=1.0, ge=0.0, le=1.0)
    
    conversation_id: Optional[str] = None


# ==================== Chunk Editing Models ====================

class UpdateChunkRequest(BaseModel):
    """Request to update/edit a chunk"""
    model_config = ConfigDict(str_strip_whitespace=True)
    
    chunk_id: str = Field(..., min_length=1, max_length=200)
    file_id: str = Field(..., min_length=1, max_length=100)
    new_text: str = Field(..., min_length=1, max_length=10000)
    preserve_metadata: bool = Field(default=True)


class ChunkDetailResponse(BaseModel):
    """Response with complete chunk details"""
    model_config = ConfigDict(from_attributes=True)
    
    chunk_id: str
    file_id: str
    text: str
    filename: str
    user_id: str
    
    chunk_type: str = Field(default="text")
    page_number: int = Field(default=0, ge=0)
    char_count: int = Field(..., ge=0)
    chunk_index: int = Field(default=0, ge=0)
    
    is_parent: Optional[bool] = Field(default=None)
    parent_chunk_id: Optional[str] = Field(default=None)
    
    created_at: str
    updated_at: Optional[str] = None
    edited: bool = Field(default=False)
    
    metadata: Dict[str, Any] = Field(default_factory=dict)


class UpdateChunkResponse(BaseModel):
    """Response after chunk update"""
    model_config = ConfigDict(from_attributes=True)
    
    message: str
    chunk_id: str
    file_id: str
    user_id: str
    old_char_count: int = Field(..., ge=0)
    new_char_count: int = Field(..., ge=0)
    updated_at: str


# ==================== Knowledge Base Models ====================

class DocumentInfo(BaseModel):
    """Document information in knowledge base"""
    model_config = ConfigDict(from_attributes=True)
    
    file_id: str
    filename: str
    file_type: str
    file_size_bytes: int
    status: ProcessingStatus
    chunks_created: int
    total_pages: int
    uploaded_at: datetime
    processed_at: Optional[datetime] = None
    
    word_count: int = Field(default=0, ge=0)
    image_count: int = Field(default=0, ge=0)
    table_count: int = Field(default=0, ge=0)
    
    chunking_strategy: Optional[str] = Field(default=None)
    embedding_model: Optional[str] = Field(default=None)


class KnowledgeBaseStatus(BaseModel):
    """Knowledge base status"""
    model_config = ConfigDict(from_attributes=True)
    
    total_documents: int = Field(..., ge=0)
    total_chunks: int = Field(..., ge=0)
    documents: List[DocumentInfo]
    
    total_size_bytes: Optional[int] = None
    avg_chunks_per_document: Optional[float] = None


# ==================== Error Response ====================

class ErrorResponse(BaseModel):
    """Standard error response"""
    model_config = ConfigDict(from_attributes=True)
    
    error: str
    message: str
    details: Optional[Dict[str, Any]] = Field(default=None)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ==================== Health Check ====================

class HealthCheckResponse(BaseModel):
    """Health check response"""
    model_config = ConfigDict(from_attributes=True)
    
    status: str = Field(..., pattern='^(healthy|degraded|unhealthy)$')
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    database: bool
    vector_store: bool
    llm_service: bool
    
    details: Optional[Dict[str, Any]] = None