"""
Database Models for RAG System - MULTI-TENANT VERSION
Added KnowledgeBase model for multi-tenant isolation
Each user can create multiple knowledge bases
"""

from sqlalchemy import (
    Column, String, Integer, DateTime, Boolean, Float, Text, ForeignKey, Enum,
    Index, UniqueConstraint
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
import enum

Base = declarative_base()


class ProcessingStatus(str, enum.Enum):
    """File processing status"""
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class FileType(str, enum.Enum):
    """File type categories"""
    DOCUMENT = "document"
    PRESENTATION = "presentation"
    IMAGE = "image"
    TEXT = "text"
    OTHER = "other"


class ChunkingStrategyEnum(str, enum.Enum):
    """Chunking strategies"""
    FIXED_SIZE = "fixed_size"
    SENTENCE = "sentence"
    PARAGRAPH = "paragraph"
    SLIDING_WINDOW = "sliding_window"
    RECURSIVE = "recursive"
    SEMANTIC = "semantic"


# ==================== NEW: KNOWLEDGE BASE MODEL ====================

class KnowledgeBase(Base):
    """
    Knowledge Base Model - Multi-tenant Isolation
    
    Each user can create multiple knowledge bases
    Knowledge bases are strictly isolated by user_id
    """
    __tablename__ = "knowledge_bases"
    
    # Primary identifiers
    knowledge_base_id = Column(String(36), primary_key=True, index=True)
    user_id = Column(String(100), nullable=False, index=True)
    
    # Knowledge base information
    knowledge_base_name = Column(String(255), nullable=False)
    knowledge_base_name_normalized = Column(String(255), nullable=False, index=True)
    description = Column(Text, nullable=True)
    
    # Statistics
    document_count = Column(Integer, default=0)
    total_chunks = Column(Integer, default=0)
    total_size_bytes = Column(Integer, default=0)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_accessed = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    files = relationship("FileMetadata", back_populates="knowledge_base", cascade="all, delete-orphan")
    
    # Unique constraint: user can't have duplicate KB names
    __table_args__ = (
        UniqueConstraint('user_id', 'knowledge_base_name_normalized', name='uix_user_kb_name'),
        Index('idx_user_kb', 'user_id', 'knowledge_base_id'),
    )
    
    def __repr__(self) -> str:
        return f"<KnowledgeBase(id={self.knowledge_base_id}, user={self.user_id}, name={self.knowledge_base_name})>"
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "knowledge_base_id": self.knowledge_base_id,
            "user_id": self.user_id,
            "knowledge_base_name": self.knowledge_base_name,
            "description": self.description,
            "document_count": self.document_count,
            "total_chunks": self.total_chunks,
            "total_size_bytes": self.total_size_bytes,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "last_accessed": self.last_accessed.isoformat() if self.last_accessed else None
        }


# ==================== UPDATED: FILE METADATA MODEL ====================

class FileMetadata(Base):
    """
    File metadata table with KB isolation
    
    Each file belongs to exactly one knowledge base
    """
    __tablename__ = "file_metadata"
    
    # Primary identifiers
    file_id = Column(String(36), primary_key=True, index=True)
    user_id = Column(String(100), nullable=False, index=True)
    knowledge_base_id = Column(
        String(36), 
        ForeignKey("knowledge_bases.knowledge_base_id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    
    # File information
    original_filename = Column(String(500), nullable=False)
    safe_filename = Column(String(500), nullable=False)
    file_extension = Column(String(20), nullable=False)
    file_type = Column(Enum(FileType), nullable=False)
    file_size_bytes = Column(Integer, nullable=False)
    
    # Storage paths
    storage_path = Column(String(1000), nullable=False)
    markdown_path = Column(String(1000), nullable=True)
    
    # Processing information
    status = Column(Enum(ProcessingStatus), default=ProcessingStatus.UPLOADED, index=True)
    chunks_created = Column(Integer, default=0)
    total_pages = Column(Integer, default=0)
    
    # Chunking strategy
    chunking_strategy = Column(
        Enum(ChunkingStrategyEnum),
        default=ChunkingStrategyEnum.FIXED_SIZE,
        nullable=True
    )
    target_chunk_tokens = Column(Integer, default=512)
    chunk_overlap_tokens = Column(Integer, default=50)
    
    # Timestamps
    uploaded_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    processed_at = Column(DateTime, nullable=True)
    last_accessed = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Processing metadata
    processing_metadata = Column(Text, nullable=True)
    error_message = Column(Text, nullable=True)
    
    # Statistics
    word_count = Column(Integer, default=0)
    image_count = Column(Integer, default=0)
    table_count = Column(Integer, default=0)
    
    # Relationships
    chunks = relationship("ChunkMetadata", back_populates="file", cascade="all, delete-orphan")
    knowledge_base = relationship("KnowledgeBase", back_populates="files")
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_user_kb_file', 'user_id', 'knowledge_base_id', 'file_id'),
        Index('idx_kb_status', 'knowledge_base_id', 'status'),
    )
    
    def __repr__(self) -> str:
        return f"<FileMetadata(file_id={self.file_id}, kb={self.knowledge_base_id}, filename={self.original_filename})>"
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "file_id": self.file_id,
            "user_id": self.user_id,
            "knowledge_base_id": self.knowledge_base_id,
            "filename": self.original_filename,  # Map to 'filename' for DocumentInfo schema
            "original_filename": self.original_filename,
            "file_extension": self.file_extension,
            "file_type": self.file_type.value if self.file_type else None,
            "file_size_bytes": self.file_size_bytes,
            "status": self.status.value if self.status else None,
            "chunks_created": self.chunks_created,
            "total_pages": self.total_pages,
            "chunking_strategy": self.chunking_strategy.value if self.chunking_strategy else None,
            "target_chunk_tokens": self.target_chunk_tokens,
            "chunk_overlap_tokens": self.chunk_overlap_tokens,
            "uploaded_at": self.uploaded_at.isoformat() if self.uploaded_at else None,
            "processed_at": self.processed_at.isoformat() if self.processed_at else None,
            "word_count": self.word_count,
            "image_count": self.image_count,
            "table_count": self.table_count
        }


class ChunkMetadata(Base):
    """
    Chunk metadata table
    
    Tracks individual chunks created from files
    Links to vector store chunk IDs
    """
    __tablename__ = "chunk_metadata"
    
    # Primary identifier
    chunk_id = Column(String(100), primary_key=True, index=True)
    
    # Foreign key to file
    file_id = Column(String(36), ForeignKey("file_metadata.file_id", ondelete="CASCADE"), nullable=False, index=True)
    
    # Chunk information
    chunk_index = Column(Integer, nullable=False)
    chunk_type = Column(String(20), default="text")
    char_count = Column(Integer, nullable=False)
    page_number = Column(Integer, default=0)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationship
    file = relationship("FileMetadata", back_populates="chunks")
    
    def __repr__(self) -> str:
        return f"<ChunkMetadata(chunk_id={self.chunk_id}, file_id={self.file_id}, index={self.chunk_index})>"


class QueryLog(Base):
    """
    Query logging table with KB tracking
    """
    __tablename__ = "query_logs"
    
    # Primary identifier
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # User and KB information
    user_id = Column(String(100), nullable=False, index=True)
    knowledge_base_id = Column(String(36), nullable=True, index=True)
    
    # Query information
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=True)
    
    # Search parameters
    top_k = Column(Integer, default=5)
    search_type = Column(String(20), default="hybrid")
    
    # Results
    chunks_retrieved = Column(Integer, default=0)
    processing_time_seconds = Column(Float, nullable=True)
    confidence_score = Column(Float, nullable=True)
    
    # Timestamps
    queried_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    
    # Optional conversation tracking
    conversation_id = Column(String(100), nullable=True, index=True)
    
    # Indexes
    __table_args__ = (
        Index('idx_user_kb_query', 'user_id', 'knowledge_base_id'),
    )
    
    def __repr__(self) -> str:
        return f"<QueryLog(id={self.id}, user={self.user_id}, kb={self.knowledge_base_id})>"


class ConversationHistory(Base):
    """
    Conversation history table with KB tracking
    """
    __tablename__ = "conversation_history"
    
    # Primary identifier
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Identifiers
    conversation_id = Column(String(100), nullable=False, index=True)
    user_id = Column(String(100), nullable=False, index=True)
    knowledge_base_id = Column(String(36), nullable=True, index=True)
    
    # Message information
    role = Column(String(20), nullable=False)
    content = Column(Text, nullable=False)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Indexes
    __table_args__ = (
        Index('idx_conv_user_kb', 'conversation_id', 'user_id', 'knowledge_base_id'),
    )
    
    def __repr__(self) -> str:
        return f"<ConversationHistory(conversation_id={self.conversation_id}, role={self.role})>"
