"""
Pydantic Schemas for RAG System
Complete type safety with proper validation
Updated: Added ChunkingStrategy enum and upload with chunking strategy selection
"""

from pydantic import BaseModel, Field, field_validator, ConfigDict, model_validator
from typing import List, Optional, Dict, Any, Literal
from datetime import datetime
from enum import Enum
import re


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


class ChunkingStrategy(str, Enum):
    """
    Available chunking strategies
    
    - DEFAULT: Token-based chunks using server defaults
    - FIXED_SIZE: Token-based fixed-size chunks
    - HIERARCHICAL: Parent/child split preserving document structure
    - SEMANTIC: Groups semantically similar sentences
    - NO_CHUNKING: Index the whole document as a single chunk
    """
    DEFAULT = "default"
    FIXED_SIZE = "fixed_size"
    HIERARCHICAL = "hierarchical"
    SEMANTIC = "semantic"
    NO_CHUNKING = "no_chunking"


# ==================== Request Models ====================

class FileUploadRequest(BaseModel):
    """File upload metadata (used internally)"""
    model_config = ConfigDict(from_attributes=True)
    
    user_id: str = Field(..., min_length=1, max_length=100)
    original_filename: str = Field(..., min_length=1, max_length=500)


class ProcessDocumentRequest(BaseModel):
    """
    Request to process a document with chunking strategy
    
    This is used as query parameter during /process/{file_id} call
    """
    model_config = ConfigDict(str_strip_whitespace=True)
    
    chunking_strategy: ChunkingStrategy = Field(
        default=ChunkingStrategy.DEFAULT,
        description="Chunking strategy to use for this document"
    )
    
    target_chunk_tokens: Optional[int] = Field(
        default=None,
        ge=100,
        le=2048,
        description="Target chunk size in tokens (default: 512)"
    )
    
    chunk_overlap_tokens: Optional[int] = Field(
        default=None,
        ge=0,
        le=500,
        description="Overlap between chunks in tokens (default: 50)"
    )


class QueryRequest(BaseModel):
    """Query request with search type selection"""
    model_config = ConfigDict(str_strip_whitespace=True)
    
    question: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Question to ask the RAG system"
    )
    
    knowledge_base_id: str = Field(
        ...,
        min_length=1,
        max_length=36,
        description="Knowledge base to search within"
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


# ==================== Chunk Editing Request Models ====================

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
    knowledge_base_id: str
    knowledge_base_name: str
    filename: str
    file_type: str
    file_size_bytes: int
    upload_time: datetime
    status: ProcessingStatus
    message: str
    task_id: Optional[str] = Field(
        default=None,
        description="Background extraction task id (poll /api/tasks/{task_id})"
    )


class ProcessingStatusResponse(BaseModel):
    """Extraction/processing status for a single file"""
    model_config = ConfigDict(from_attributes=True)

    file_id: str
    filename: str
    status: ProcessingStatus
    progress: float = Field(default=0.0, ge=0.0, le=100.0)
    message: Optional[str] = None
    task_id: Optional[str] = None

    knowledge_base_id: Optional[str] = None
    uploaded_at: Optional[datetime] = None
    processed_at: Optional[datetime] = None
    chunks_created: int = Field(default=0, ge=0)
    chunks_staged: int = Field(default=0, ge=0)
    total_pages: int = Field(default=0, ge=0)
    chunking_strategy: Optional[str] = None
    error_message: Optional[str] = None


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
    chunking_strategy: Optional[str] = Field(
        default=None,
        description="Chunking strategy used"
    )
    message: str


class SourceDocument(BaseModel):
    """
    Source document chunk in query results with chunk editing support
    
    Includes:
    - chunk_id: For identifying the exact chunk to edit
    - file_id: For parent file identification
    - full_text: Complete chunk text (not truncated) for editing
    """
    model_config = ConfigDict(from_attributes=True)
    
    # Core content
    content: str = Field(..., description="Chunk content (may be truncated for display)")
    
    # Full text for editing (not truncated)
    full_text: Optional[str] = Field(
        default=None,
        description="Complete chunk text (for editing purposes)"
    )
    
    # Identifiers
    filename: str = Field(..., description="Source filename")
    chunk_id: str = Field(..., description="Unique chunk ID (for editing)")
    
    # File ID for editing validation
    file_id: str = Field(
        default="unknown",
        description="Parent file ID (for editing validation)"
    )
    
    knowledge_base_id: Optional[str] = Field(
        default=None,
        description="Knowledge base the chunk belongs to"
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
    sources: List[SourceDocument]
    
    # Performance metrics
    processing_time: float = Field(..., description="Processing time in seconds")
    
    # Search metadata
    search_type: SearchType = Field(default=SearchType.HYBRID)
    confidence_score: float = Field(default=1.0, ge=0.0, le=1.0)
    
    # Scope of the search
    knowledge_base_id: Optional[str] = None
    
    # Optional conversation tracking
    conversation_id: Optional[str] = None


# ==================== Chunk Editing Response Models ====================

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


# ==================== Document Info Models ====================

class DocumentInfo(BaseModel):
    """Document information in knowledge base"""
    model_config = ConfigDict(from_attributes=True)
    
    file_id: str
    knowledge_base_id: Optional[str] = None
    filename: str
    file_type: str
    file_size_bytes: int
    status: ProcessingStatus
    chunks_created: int
    chunks_staged: int = Field(default=0, ge=0)
    total_pages: int
    uploaded_at: datetime
    processed_at: Optional[datetime] = None
    
    # Statistics
    word_count: int = Field(default=0, ge=0)
    image_count: int = Field(default=0, ge=0)
    table_count: int = Field(default=0, ge=0)
    
    # Chunking info
    chunking_strategy: Optional[str] = Field(
        default=None,
        description="Chunking strategy used"
    )


class KnowledgeBaseStatus(BaseModel):
    """Detailed status of a single knowledge base"""
    model_config = ConfigDict(from_attributes=True)
    
    knowledge_base_id: str
    knowledge_base_name: str
    user_id: str
    description: Optional[str] = None
    
    total_documents: int = Field(..., ge=0)
    total_chunks: int = Field(..., ge=0)
    documents: List[DocumentInfo]
    
    total_size_bytes: int = Field(default=0, ge=0)
    created_at: datetime
    last_accessed: Optional[datetime] = None


class KnowledgeBaseCreate(BaseModel):
    """Request to create a knowledge base"""
    model_config = ConfigDict(str_strip_whitespace=True)
    
    knowledge_base_name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = Field(default=None, max_length=2000)
    
    @field_validator('knowledge_base_name')
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Mirror the client-side rule so both ends agree on valid names"""
        if not re.fullmatch(r'[A-Za-z0-9 _-]+', v):
            raise ValueError(
                "Knowledge base name may only contain letters, numbers, "
                "spaces, hyphens, and underscores"
            )
        return v


class KnowledgeBaseUpdate(BaseModel):
    """Request to rename / re-describe a knowledge base"""
    model_config = ConfigDict(str_strip_whitespace=True)
    
    knowledge_base_name: Optional[str] = Field(default=None, min_length=1, max_length=255)
    description: Optional[str] = Field(default=None, max_length=2000)
    
    @field_validator('knowledge_base_name')
    @classmethod
    def validate_name(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        if not re.fullmatch(r'[A-Za-z0-9 _-]+', v):
            raise ValueError(
                "Knowledge base name may only contain letters, numbers, "
                "spaces, hyphens, and underscores"
            )
        return v


class KnowledgeBaseInfo(BaseModel):
    """Summary information about a knowledge base"""
    model_config = ConfigDict(from_attributes=True)
    
    knowledge_base_id: str
    user_id: str
    knowledge_base_name: str
    description: Optional[str] = None
    
    document_count: int = Field(default=0, ge=0)
    total_chunks: int = Field(default=0, ge=0)
    total_size_bytes: int = Field(default=0, ge=0)
    
    created_at: datetime
    updated_at: Optional[datetime] = None
    last_accessed: Optional[datetime] = None


class KnowledgeBaseList(BaseModel):
    """List of a user's knowledge bases"""
    model_config = ConfigDict(from_attributes=True)
    
    user_id: str
    total_knowledge_bases: int = Field(..., ge=0)
    knowledge_bases: List[KnowledgeBaseInfo]


# ==================== Chunk Staging Models ====================

class ChunkingRequest(BaseModel):
    """
    Request to chunk (or preview chunking of) an extracted document.

    Parameter relevance depends on the selected strategy:
    - fixed_size: chunk_size, chunk_overlap
    - semantic:   chunk_size, chunk_overlap, max_buffer_size, breakpoint_threshold
    - hierarchical: parent_chunk_size, child_chunk_size, child_chunk_overlap
    - default / no_chunking: no parameters
    """
    model_config = ConfigDict(str_strip_whitespace=True)
    
    file_id: str = Field(..., min_length=1, max_length=100)
    chunking_strategy: ChunkingStrategy = Field(default=ChunkingStrategy.DEFAULT)
    
    # fixed_size / semantic
    chunk_size: Optional[int] = Field(default=None, ge=50, le=4096)
    chunk_overlap: Optional[int] = Field(default=None, ge=0, le=1000)
    
    # semantic
    max_buffer_size: Optional[int] = Field(default=None, ge=1, le=64)
    breakpoint_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    
    # hierarchical
    parent_chunk_size: Optional[int] = Field(default=None, ge=100, le=8192)
    child_chunk_size: Optional[int] = Field(default=None, ge=50, le=4096)
    child_chunk_overlap: Optional[int] = Field(default=None, ge=0, le=1000)
    
    @model_validator(mode='after')
    def validate_overlaps(self):
        """Overlap must stay below the chunk size it applies to"""
        if (
            self.chunk_size is not None
            and self.chunk_overlap is not None
            and self.chunk_overlap >= self.chunk_size
        ):
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        
        if (
            self.child_chunk_size is not None
            and self.child_chunk_overlap is not None
            and self.child_chunk_overlap >= self.child_chunk_size
        ):
            raise ValueError("child_chunk_overlap must be smaller than child_chunk_size")
        
        if (
            self.parent_chunk_size is not None
            and self.child_chunk_size is not None
            and self.child_chunk_size > self.parent_chunk_size
        ):
            raise ValueError("child_chunk_size must not exceed parent_chunk_size")
        
        return self


class PreviewChunk(BaseModel):
    """A single chunk in a preview response"""
    model_config = ConfigDict(from_attributes=True)
    
    chunk_id: str
    chunk_index: int = Field(default=0, ge=0)
    content_preview: str = Field(..., description="Truncated chunk text")
    char_count: int = Field(default=0, ge=0)
    full_char_count: int = Field(default=0, ge=0)
    token_count: int = Field(default=0, ge=0)
    page_number: int = Field(default=0, ge=0)
    chunk_type: str = Field(default="text")
    parent_chunk_id: Optional[str] = None
    children: Optional[List["PreviewChunk"]] = None


class PreviewChunksResponse(BaseModel):
    """Response for a chunking preview (nothing is persisted)"""
    model_config = ConfigDict(from_attributes=True)
    
    file_id: str
    chunking_strategy: str
    total_chunks: int = Field(..., ge=0)
    preview_count: int = Field(..., ge=0)
    preview_chunks: List[PreviewChunk]
    message: str


class CreateChunksResponse(BaseModel):
    """Response after staging chunks for a document"""
    model_config = ConfigDict(from_attributes=True)
    
    file_id: str
    knowledge_base_id: Optional[str] = None
    chunks_created: int = Field(..., ge=0)
    chunk_ids: List[str] = Field(default_factory=list)
    total_chunk_ids: int = Field(default=0, ge=0)
    chunking_strategy: str
    chunking_time_seconds: float = Field(default=0.0, ge=0.0)
    status: str = "chunked"
    parameters: Dict[str, Any] = Field(default_factory=dict)
    message: str


class FinalizeChunksResponse(BaseModel):
    """Response after embedding + indexing staged chunks"""
    model_config = ConfigDict(from_attributes=True)
    
    file_id: str
    knowledge_base_id: Optional[str] = None
    indexed_chunks: int = Field(..., ge=0)
    embedding_model: str
    batch_size: int = Field(..., ge=1)
    indexing_time_seconds: float = Field(default=0.0, ge=0.0)
    status: str = "completed"
    message: str


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
    chunking_strategy: Optional[str] = None


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


# Resolve the self-referencing `children` field on PreviewChunk
PreviewChunk.model_rebuild()