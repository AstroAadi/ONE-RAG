"""
Pydantic Schemas for RAG System - MULTI-TENANT VERSION
Added Knowledge Base models for multi-tenant isolation
"""

from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import List, Optional, Dict, Any, Union, Literal,Annotated
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


class ChunkingStrategy(str, Enum):
    """Available chunking strategies"""
    DEFAULT = "default"
    FIXED_SIZE = "fixed_size"
    HIERARCHICAL = "hierarchical"
    SEMANTIC = "semantic"
    NO_CHUNKING = "no_chunking"


# ==================== NEW: Knowledge Base Models ====================

class KnowledgeBaseCreate(BaseModel):
    """Request to create a new knowledge base"""
    model_config = ConfigDict(str_strip_whitespace=True)
    
    knowledge_base_name: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Name of the knowledge base (unique per user)"
    )
    
    description: Optional[str] = Field(
        default=None,
        max_length=1000,
        description="Optional description of the knowledge base"
    )
    
    @field_validator('knowledge_base_name')
    @classmethod
    def validate_kb_name(cls, v: str) -> str:
        """Validate knowledge base name"""
        if not v or not v.strip():
            raise ValueError("Knowledge base name cannot be empty")
        
        # Check for invalid characters
        invalid_chars = ['/', '\\', '<', '>', ':', '"', '|', '?', '*']
        if any(char in v for char in invalid_chars):
            raise ValueError(f"Knowledge base name contains invalid characters: {invalid_chars}")
        
        return v.strip()


class KnowledgeBaseUpdate(BaseModel):
    """Request to update a knowledge base"""
    model_config = ConfigDict(str_strip_whitespace=True)
    
    knowledge_base_name: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=255,
        description="New name for the knowledge base"
    )
    
    description: Optional[str] = Field(
        default=None,
        max_length=1000,
        description="New description"
    )


class KnowledgeBaseInfo(BaseModel):
    """Knowledge base information"""
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
    """List of knowledge bases"""
    model_config = ConfigDict(from_attributes=True)
    
    user_id: str
    total_knowledge_bases: int = Field(..., ge=0)
    knowledge_bases: List[KnowledgeBaseInfo]


# ==================== Request Models ====================

class FileUploadRequest(BaseModel):
    """File upload with knowledge base"""
    model_config = ConfigDict(from_attributes=True)
    
    user_id: str = Field(..., min_length=1, max_length=100)
    knowledge_base_name: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Knowledge base name (will be created if doesn't exist)"
    )
    original_filename: str = Field(..., min_length=1, max_length=500)


class ProcessDocumentRequest(BaseModel):
    """Request to process a document"""
    model_config = ConfigDict(str_strip_whitespace=True)
    
    chunking_strategy: ChunkingStrategy = Field(
        default=ChunkingStrategy.DEFAULT,
        description="Chunking strategy to use"
    )
    
    target_chunk_tokens: Optional[int] = Field(
        default=None,
        ge=100,
        le=2048,
        description="Target chunk size in tokens"
    )
    
    chunk_overlap_tokens: Optional[int] = Field(
        default=None,
        ge=0,
        le=500,
        description="Overlap between chunks in tokens"
    )


class QueryRequest(BaseModel):
    """Query request with KB isolation"""
    model_config = ConfigDict(str_strip_whitespace=True)
    
    question: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Question to ask"
    )
    
    knowledge_base_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Knowledge base ID to query (REQUIRED)"
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
        description="Optional: Restrict to specific file"
    )
    
    temperature: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="LLM temperature"
    )
    
    search_type: SearchType = Field(
        default=SearchType.HYBRID,
        description="Search type"
    )


class ConversationQueryRequest(QueryRequest):
    """Conversational query with KB"""
    conversation_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Conversation ID"
    )


# ==================== Chunking Strategy Request Models ====================

class DefaultChunkingRequest(BaseModel):
    """Default chunking - no parameters needed"""
    model_config = ConfigDict(str_strip_whitespace=True)
    
    file_id: str = Field(..., description="File ID to create chunks for")
    chunking_strategy: Literal["default"] = Field(default="default", description="Chunking strategy (fixed: default)")


class FixedSizeChunkingRequest(BaseModel):
    """Fixed-size chunking with user-defined parameters"""
    model_config = ConfigDict(str_strip_whitespace=True)
    
    file_id: str = Field(..., description="File ID to create chunks for")
    chunking_strategy: Literal["fixed_size"] = Field(default="fixed_size", description="Chunking strategy (fixed: fixed_size)")
    chunk_size: int = Field(..., ge=100, le=2048, description="Chunk size in tokens")
    chunk_overlap: int = Field(..., ge=0, le=500, description="Overlap between chunks in tokens")


class SemanticChunkingRequest(BaseModel):
    """Semantic chunking - groups semantically similar content"""
    model_config = ConfigDict(str_strip_whitespace=True)
    
    file_id: str = Field(..., description="File ID to create chunks for")
    chunking_strategy: Literal["semantic"] = Field(default="semantic", description="Chunking strategy (fixed: semantic)")
    chunk_size: int = Field(..., ge=100, le=2048, description="Target chunk size in tokens")
    chunk_overlap: int = Field(..., ge=0, le=500, description="Overlap between chunks in tokens")
    max_buffer_size: int = Field(..., ge=1, le=20, description="Maximum number of sentences to compare for similarity")
    breakpoint_threshold: float = Field(..., ge=0.0, le=1.0, description="Similarity threshold for breaking chunks (0.5-0.7 recommended)")


class HierarchicalChunkingRequest(BaseModel):
    """Hierarchical chunking - creates parent-child relationships"""
    model_config = ConfigDict(str_strip_whitespace=True)
    
    file_id: str = Field(..., description="File ID to create chunks for")
    chunking_strategy: Literal["hierarchical"] = Field(default="hierarchical", description="Chunking strategy (fixed: hierarchical)")
    parent_chunk_size: int = Field(..., ge=500, le=4096, description="Size of parent chunks in tokens")
    child_chunk_size: int = Field(..., ge=100, le=1024, description="Size of child chunks in tokens")
    child_chunk_overlap: int = Field(..., ge=0, le=200, description="Overlap between child chunks in tokens")


class NoChunkingRequest(BaseModel):
    """No chunking - entire document as single chunk"""
    model_config = ConfigDict(str_strip_whitespace=True)
    
    file_id: str = Field(..., description="File ID to create chunks for")
    chunking_strategy: Literal["no_chunking"] = Field(default="no_chunking", description="Chunking strategy (fixed: no_chunking)")


# Union type for all chunking requests
ChunkingRequest = Annotated[
    Union[
        DefaultChunkingRequest,
        FixedSizeChunkingRequest,
        SemanticChunkingRequest,
        HierarchicalChunkingRequest,
        NoChunkingRequest
    ],
    Field(discriminator="chunking_strategy")
]


class ChunkCreationResponse(BaseModel):
    """Response after creating chunks"""
    model_config = ConfigDict(from_attributes=True)
    
    file_id: str
    knowledge_base_id: str
    chunks_created: int
    chunk_ids: List[str] = Field(description="First 10 chunk IDs")
    total_chunk_ids: int
    chunking_strategy: str
    chunking_time_seconds: float
    status: str
    parameters: Dict[str, Any] = Field(description="Parameters used for chunking")
    message: str


# ==================== Chunk Models ====================

class ChunkRetrievalRequest(BaseModel):
    """Request to retrieve a chunk"""
    model_config = ConfigDict(str_strip_whitespace=True)
    
    chunk_id: str = Field(..., min_length=1, max_length=200)
    file_id: Optional[str] = None


class UpdateChunkRequest(BaseModel):
    """Request to update a chunk"""
    model_config = ConfigDict(str_strip_whitespace=True)
    
    chunk_id: str = Field(..., min_length=1, max_length=200)
    file_id: str = Field(..., min_length=1, max_length=100)
    new_text: str = Field(..., min_length=1, max_length=10000)
    preserve_metadata: bool = Field(default=True)


# ==================== Response Models ====================

class FileUploadResponse(BaseModel):
    """Response after file upload"""
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


class DocumentProcessResponse(BaseModel):
    """Response after processing"""
    model_config = ConfigDict(from_attributes=True)
    
    file_id: str
    knowledge_base_id: str
    filename: str
    status: ProcessingStatus
    markdown_path: Optional[str] = None
    chunks_created: int
    total_pages: int = 0
    processing_time_seconds: Optional[float] = None
    statistics: Optional[Dict[str, int]] = None
    chunking_strategy: Optional[str] = None
    message: str


class SourceDocument(BaseModel):
    """Source document in query results"""
    model_config = ConfigDict(from_attributes=True)
    
    content: str
    full_text: Optional[str] = None
    filename: str
    chunk_id: str
    file_id: str = "unknown"
    knowledge_base_id: str = "unknown"
    relevance_score: float
    page_number: int = Field(default=0, ge=0)
    chunk_type: str = "text"
    chunk_index: int = Field(default=0, ge=0)
    semantic_score: Optional[float] = None
    keyword_score: Optional[float] = None
    
    @field_validator('relevance_score', mode='before')
    @classmethod
    def clamp_relevance_score(cls, v):
        try:
            return max(0.0, min(1.0, float(v)))
        except (ValueError, TypeError):
            return 0.0


class QueryResponse(BaseModel):
    """Query response"""
    model_config = ConfigDict(from_attributes=True)
    
    question: str
    answer: str
    sources: List[SourceDocument]
    processing_time: float
    search_type: SearchType = SearchType.HYBRID
    confidence_score: float = Field(default=1.0, ge=0.0, le=1.0)
    knowledge_base_id: str
    conversation_id: Optional[str] = None


class ChunkDetailResponse(BaseModel):
    """Complete chunk details"""
    model_config = ConfigDict(from_attributes=True)
    
    chunk_id: str
    file_id: str
    knowledge_base_id: str
    text: str
    filename: str
    user_id: str
    chunk_type: str = "text"
    page_number: int = Field(default=0, ge=0)
    char_count: int = Field(..., ge=0)
    chunk_index: int = Field(default=0, ge=0)
    created_at: str
    updated_at: Optional[str] = None
    edited: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)


class UpdateChunkResponse(BaseModel):
    """Response after chunk update"""
    model_config = ConfigDict(from_attributes=True)
    
    message: str
    chunk_id: str
    file_id: str
    knowledge_base_id: str
    user_id: str
    old_char_count: int = Field(..., ge=0)
    new_char_count: int = Field(..., ge=0)
    updated_at: str


class DocumentInfo(BaseModel):
    """Document information"""
    model_config = ConfigDict(from_attributes=True)
    
    file_id: str
    knowledge_base_id: str
    filename: str
    file_type: str
    file_size_bytes: int
    status: ProcessingStatus
    chunks_created: int
    total_pages: int
    uploaded_at: datetime
    processed_at: Optional[datetime] = None
    word_count: int = 0
    image_count: int = 0
    table_count: int = 0
    chunking_strategy: Optional[str] = None


class KnowledgeBaseStatus(BaseModel):
    """Knowledge base detailed status"""
    model_config = ConfigDict(from_attributes=True)
    
    knowledge_base_id: str
    knowledge_base_name: str
    user_id: str
    total_documents: int = Field(..., ge=0)
    total_chunks: int = Field(..., ge=0)
    total_size_bytes: int = Field(default=0, ge=0)
    documents: List[DocumentInfo]
    created_at: datetime
    last_accessed: Optional[datetime] = None


class ErrorResponse(BaseModel):
    """Standard error response"""
    model_config = ConfigDict(from_attributes=True)
    
    error: str
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class HealthCheckResponse(BaseModel):
    """Health check response"""
    model_config = ConfigDict(from_attributes=True)
    
    status: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    database: bool
    vector_store: bool
    llm_service: bool
    details: Optional[Dict[str, Any]] = None