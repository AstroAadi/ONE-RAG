"""
Pydantic Schemas for RAG System
Complete type safety with proper validation
Updated: Added chunk editing support with chunk_id, file_id, and full_text
"""

from pydantic import BaseModel, Field, field_validator, ConfigDict, model_validator
from typing import List, Optional, Dict, Any, Literal
from datetime import datetime
from enum import Enum


# ==================== Enums ====================

class ProcessingStatus(str, Enum):
    """Document processing status"""
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class FileCategory(str, Enum):
    """File category types"""
    DOCUMENT = "document"
    PRESENTATION = "presentation"
    IMAGE = "image"
    TEXT = "text"
    OTHER = "other"


class SearchType(str, Enum):
    """Search type options"""
    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"


# ==================== Request Models ====================

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
    
    # Search type selection
    search_type: SearchType = Field(
        default=SearchType.HYBRID,
        description="Search type: 'dense' (semantic only), 'sparse' (keyword only), or 'hybrid' (RRF fusion)"
    )


class ConversationQueryRequest(QueryRequest):
    """Query request with conversation tracking"""
    conversation_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Unique conversation ID"
    )


# ==================== NEW: Chunk Editing Request Models ====================

class ChunkRetrievalRequest(BaseModel):
    """Request to retrieve a specific chunk"""
    model_config = ConfigDict(str_strip_whitespace=True)
    
    chunk_id: str = Field(..., min_length=1, max_length=200, description="Unique chunk identifier")
    file_id: Optional[str] = Field(None, description="Optional file ID for validation")


class UpdateChunkRequest(BaseModel):
    """Request to update/edit a chunk"""
    model_config = ConfigDict(str_strip_whitespace=True)
    
    chunk_id: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Unique chunk identifier"
    )
    
    file_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Parent file ID (for validation)"
    )
    
    new_text: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="New chunk text content"
    )
    
    preserve_metadata: bool = Field(
        default=True,
        description="Whether to preserve existing metadata"
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


class SourceDocument(BaseModel):
    """
    ENHANCED: Source document chunk in query results with chunk editing support
    
    Now includes:
    - chunk_id: For identifying the exact chunk to edit
    - file_id: For parent file identification
    - full_text: Complete chunk text (not truncated) for editing
    """
    model_config = ConfigDict(from_attributes=True)
    
    # Core content
    content: str = Field(..., description="Chunk content (may be truncated for display)")
    
    # NEW: Full text for editing (not truncated)
    full_text: Optional[str] = Field(
        default=None,
        description="Complete chunk text (for editing purposes)"
    )
    
    # Identifiers
    filename: str = Field(..., description="Source filename")
    chunk_id: str = Field(..., description="Unique chunk ID (for editing)")
    
    # NEW: File ID for editing validation
    file_id: str = Field(
        default="unknown",
        description="Parent file ID (for editing validation)"
    )
    
    # Scores
    relevance_score: float = Field(..., description="Relevance score")
    
    # Optional metadata
    page_number: int = Field(default=0, ge=0)
    chunk_type: str = Field(default="text")
    chunk_index: int = Field(default=0, ge=0)
    
    # Search scores
    semantic_score: Optional[float] = Field(default=None)
    keyword_score: Optional[float] = Field(default=None)
    
    @field_validator('relevance_score', mode='before')
    @classmethod
    def clamp_relevance_score(cls, v):
        """Ensure relevance score is within valid range [0.0, 1.0]"""
        try:
            score = float(v)
            return max(0.0, min(1.0, score))
        except (ValueError, TypeError):
            return 0.0
    
    @field_validator('semantic_score', 'keyword_score', mode='before')
    @classmethod
    def clamp_optional_scores(cls, v):
        """Ensure optional scores are within valid range if provided"""
        if v is None:
            return None
        try:
            score = float(v)
            return max(0.0, min(1.0, score))
        except (ValueError, TypeError):
            return None


class QueryResponse(BaseModel):
    """Response to query request"""
    model_config = ConfigDict(from_attributes=True)
    
    question: str
    answer: str
    sources: List[SourceDocument]  # Now includes chunk_id, file_id, and full_text
    
    # Performance metrics
    processing_time: float = Field(..., description="Processing time in seconds")
    
    # Search metadata
    search_type: SearchType = Field(default=SearchType.HYBRID)
    confidence_score: float = Field(default=1.0, ge=0.0, le=1.0)
    
    # Optional conversation tracking
    conversation_id: Optional[str] = None


# ==================== NEW: Chunk Editing Response Models ====================

class ChunkDetailResponse(BaseModel):
    """Response with complete chunk details"""
    model_config = ConfigDict(from_attributes=True)
    
    chunk_id: str
    file_id: str
    text: str = Field(..., description="Complete chunk text")
    filename: str
    user_id: str
    
    # Metadata
    chunk_type: str = Field(default="text")
    page_number: int = Field(default=0, ge=0)
    char_count: int = Field(..., ge=0)
    chunk_index: int = Field(default=0, ge=0)
    
    # Timestamps
    created_at: str
    updated_at: Optional[str] = None
    
    # Edit tracking
    edited: bool = Field(default=False)
    
    # Additional metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)


class UpdateChunkResponse(BaseModel):
    """Response after chunk update"""
    model_config = ConfigDict(from_attributes=True)
    
    message: str
    chunk_id: str
    file_id: str
    user_id: str
    
    # Changes
    old_char_count: int = Field(..., ge=0)
    new_char_count: int = Field(..., ge=0)
    
    # Timestamp
    updated_at: str


class ChunkListItem(BaseModel):
    """Individual chunk in file chunk list"""
    model_config = ConfigDict(from_attributes=True)
    
    chunk_id: str
    chunk_index: int = Field(..., ge=0)
    chunk_type: str = Field(default="text")
    char_count: int = Field(..., ge=0)
    page_number: int = Field(default=0, ge=0)
    created_at: Optional[str] = None
    edited: bool = Field(default=False)


class FileChunksResponse(BaseModel):
    """Response with all chunks for a file"""
    model_config = ConfigDict(from_attributes=True)
    
    file_id: str
    filename: str
    total_chunks: int = Field(..., ge=0)
    chunks: List[ChunkListItem]


# ==================== Existing Models (Unchanged) ====================

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
    
    # Statistics
    word_count: int = Field(default=0, ge=0)
    image_count: int = Field(default=0, ge=0)
    table_count: int = Field(default=0, ge=0)


class KnowledgeBaseStatus(BaseModel):
    """Knowledge base status"""
    model_config = ConfigDict(from_attributes=True)
    
    total_documents: int = Field(..., ge=0)
    total_chunks: int = Field(..., ge=0)
    total_users: int = Field(default=0, ge=0)
    documents: List[DocumentInfo]
    
    # Optional statistics
    total_size_bytes: Optional[int] = None
    avg_chunks_per_document: Optional[float] = None


class UserDocumentsResponse(BaseModel):
    """User's documents list"""
    model_config = ConfigDict(from_attributes=True)
    
    user_id: str
    total_documents: int = Field(..., ge=0)
    documents: List[DocumentInfo]


# ==================== Error Response Models ====================

class ErrorResponse(BaseModel):
    """Standard error response"""
    model_config = ConfigDict(from_attributes=True)
    
    error: str = Field(..., description="Error code")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional details")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ==================== Statistics Models ====================

class DocumentStatistics(BaseModel):
    """Detailed document statistics"""
    model_config = ConfigDict(from_attributes=True)
    
    file_id: str
    filename: str
    
    # Content statistics
    total_pages: int = Field(..., ge=0)
    total_chunks: int = Field(..., ge=0)
    word_count: int = Field(default=0, ge=0)
    char_count: int = Field(default=0, ge=0)
    
    # Visual elements
    image_count: int = Field(default=0, ge=0)
    table_count: int = Field(default=0, ge=0)
    chart_count: int = Field(default=0, ge=0)
    
    # Processing info
    processing_time_seconds: Optional[float] = None
    extraction_method: Optional[str] = None


class SystemMetrics(BaseModel):
    """System-wide metrics"""
    model_config = ConfigDict(from_attributes=True)
    
    uptime_seconds: float = Field(..., ge=0.0)
    total_requests: int = Field(..., ge=0)
    total_errors: int = Field(..., ge=0)
    error_rate: float = Field(..., ge=0.0, le=1.0)
    
    # Per-endpoint metrics
    endpoints: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    
    # Optional error details
    errors: Optional[Dict[str, int]] = None


# ==================== Conversation Models ====================

class ConversationMessage(BaseModel):
    """Single message in conversation"""
    model_config = ConfigDict(from_attributes=True)
    
    role: str = Field(..., pattern='^(user|assistant)$')
    content: str = Field(..., min_length=1)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ConversationHistory(BaseModel):
    """Conversation history"""
    model_config = ConfigDict(from_attributes=True)
    
    conversation_id: str
    user_id: str
    messages: List[ConversationMessage]
    created_at: datetime
    updated_at: datetime


# ==================== Health Check Models ====================

class HealthCheckResponse(BaseModel):
    """Health check response"""
    model_config = ConfigDict(from_attributes=True)
    
    status: str = Field(..., pattern='^(healthy|degraded|unhealthy)$')
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Component health
    database: bool = Field(..., description="Database connectivity")
    vector_store: bool = Field(..., description="Vector store status")
    llm_service: bool = Field(..., description="LLM service availability")
    
    # Optional details
    details: Optional[Dict[str, Any]] = None