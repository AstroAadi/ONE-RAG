"""
Database Models for RAG System - SIMPLIFIED VERSION
Only 4 states: uploaded, processing, completed, failed
No granular tracking - keeps it simple!
"""

from sqlalchemy import (
    Column, String, Integer, DateTime, Boolean, Float, Text, ForeignKey, Enum,
    Index, UniqueConstraint, CheckConstraint
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
import enum

Base = declarative_base()


# ==================== SIMPLE 4-STATE STATUS ====================
class ProcessingStatus(str, enum.Enum):
    """File processing status - SIMPLE 4 STATES ONLY"""
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
    DEFAULT = "default"
    FIXED_SIZE = "fixed_size"
    HIERARCHICAL = "hierarchical"
    SEMANTIC = "semantic"
    NO_CHUNKING = "no_chunking"


# ==================== ENUM VALIDATORS (Production-safe) ====================

def validate_processing_status(status):
    """
    Convert and validate processing status to lowercase.
    Safe for database insertion.
    
    Args:
        status: Either a string or ProcessingStatus enum member
        
    Returns:
        Validated lowercase string value
        
    Raises:
        ValueError: If status is not valid
    """
    if isinstance(status, ProcessingStatus):
        return status.value
    
    status_str = str(status).lower().strip()
    valid_values = {s.value for s in ProcessingStatus}
    
    if status_str not in valid_values:
        raise ValueError(f"Invalid status '{status}'. Must be one of: {valid_values}")
    
    return status_str


def validate_file_type(file_type):
    """
    Convert and validate file type to lowercase.
    Safe for database insertion.
    
    Args:
        file_type: Either a string or FileType enum member
        
    Returns:
        Validated lowercase string value
        
    Raises:
        ValueError: If file_type is not valid
    """
    if isinstance(file_type, FileType):
        return file_type.value
    
    file_type_str = str(file_type).lower().strip()
    valid_values = {f.value for f in FileType}
    
    if file_type_str not in valid_values:
        raise ValueError(f"Invalid file type '{file_type}'. Must be one of: {valid_values}")
    
    return file_type_str


def validate_chunking_strategy(strategy):
    """
    Convert and validate chunking strategy to lowercase.
    Safe for database insertion.
    
    Args:
        strategy: Either a string or ChunkingStrategyEnum member
        
    Returns:
        Validated lowercase string value
        
    Raises:
        ValueError: If strategy is not valid
    """
    if isinstance(strategy, ChunkingStrategyEnum):
        return strategy.value
    
    strategy_str = str(strategy).lower().strip()
    valid_values = {s.value for s in ChunkingStrategyEnum}
    
    if strategy_str not in valid_values:
        raise ValueError(f"Invalid strategy '{strategy}'. Must be one of: {valid_values}")
    
    return strategy_str


# ==================== HELPER FUNCTIONS FOR DB INSERTION ====================

def get_processing_status_value(status):
    """
    Get safe lowercase string value for ProcessingStatus enum member.
    
    Args:
        status: ProcessingStatus enum member or string
        
    Returns:
        Lowercase string value safe for PostgreSQL insertion
    """
    if isinstance(status, ProcessingStatus):
        return status.value.lower()
    return str(status).lower().strip()


def get_file_type_value(file_type):
    """
    Get safe lowercase string value for FileType enum member.
    
    Args:
        file_type: FileType enum member or string
        
    Returns:
        Lowercase string value safe for PostgreSQL insertion
    """
    if isinstance(file_type, FileType):
        return file_type.value.lower()
    return str(file_type).lower().strip()


def get_chunking_strategy_value(strategy):
    """
    Get safe lowercase string value for ChunkingStrategyEnum member.
    
    Args:
        strategy: ChunkingStrategyEnum member or string
        
    Returns:
        Lowercase string value safe for PostgreSQL insertion
    """
    if isinstance(strategy, ChunkingStrategyEnum):
        return strategy.value.lower()
    return str(strategy).lower().strip()


# ==================== KNOWLEDGE BASE MODEL ====================

class KnowledgeBase(Base):
    """Knowledge Base Model - Multi-tenant Isolation"""
    __tablename__ = "knowledge_bases"
    
    # Primary identifiers
    knowledge_base_id = Column(String(36), primary_key=True, index=True)
    user_id = Column(String(100), nullable=False, index=True)
    
    # Knowledge base information
    knowledge_base_name = Column(String(255), nullable=False)
    knowledge_base_name_normalized = Column(String(255), nullable=False, index=True)
    description = Column(Text, nullable=True)
    
    # Statistics
    document_count = Column(Integer, default=0, nullable=False)
    total_chunks = Column(Integer, default=0, nullable=False)
    total_size_bytes = Column(Integer, default=0, nullable=False)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    last_accessed = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    files = relationship(
        "FileMetadata",
        back_populates="knowledge_base",
        cascade="all, delete-orphan",
        passive_deletes=True
    )
    
    # Indexes and constraints
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


# ==================== FILE METADATA MODEL (SIMPLIFIED) ====================

class FileMetadata(Base):
    """
    File metadata table - SIMPLIFIED
    
    Status flow: uploaded → processing → completed
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
    file_type = Column(String(20), nullable=False)  # String instead of Enum
    file_size_bytes = Column(Integer, nullable=False)
    
    # Storage paths
    storage_path = Column(String(1000), nullable=False)
    markdown_path = Column(String(1000), nullable=True)
    
    # STATUS: Simple 4 states only - String column (not Enum) to avoid psycopg2 issues
    status = Column(String(20), default="uploaded", nullable=False, index=True)
    
    # Processing information
    chunks_created = Column(Integer, default=0, nullable=False)
    total_pages = Column(Integer, default=0, nullable=False)
    
    # Chunking strategy - String column (not Enum) to avoid psycopg2 issues
    chunking_strategy = Column(
        String(20),
        default="default",
        nullable=True
    )
    target_chunk_tokens = Column(Integer, default=512, nullable=False)
    chunk_overlap_tokens = Column(Integer, default=50, nullable=False)
    
    # Timestamps
    uploaded_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    processed_at = Column(DateTime, nullable=True)
    last_accessed = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Processing metadata
    processing_metadata = Column(Text, nullable=True)
    error_message = Column(Text, nullable=True)
    
    # Statistics
    word_count = Column(Integer, default=0, nullable=False)
    image_count = Column(Integer, default=0, nullable=False)
    table_count = Column(Integer, default=0, nullable=False)
    
    # Relationships
    chunks = relationship(
        "ChunkMetadata",
        back_populates="file",
        cascade="all, delete-orphan",
        passive_deletes=True
    )
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
            "filename": self.original_filename,
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


# ==================== CHUNK METADATA MODEL ====================

class ChunkMetadata(Base):
    """Chunk metadata table"""
    __tablename__ = "chunk_metadata"
    
    # Primary identifier
    chunk_id = Column(String(100), primary_key=True, index=True)
    
    # Foreign key to file (CASCADE delete)
    file_id = Column(
        String(36),
        ForeignKey("file_metadata.file_id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    
    # Chunk information
    chunk_index = Column(Integer, nullable=False)
    chunk_type = Column(String(20), default="text", nullable=False)
    char_count = Column(Integer, nullable=False)
    page_number = Column(Integer, default=0, nullable=False)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationship
    file = relationship("FileMetadata", back_populates="chunks")
    
    # Indexes
    __table_args__ = (
        Index('idx_file_chunk', 'file_id', 'chunk_index'),
    )
    
    def __repr__(self) -> str:
        return f"<ChunkMetadata(chunk_id={self.chunk_id}, file_id={self.file_id}, index={self.chunk_index})>"


# ==================== QUERY LOG MODEL ====================

class QueryLog(Base):
    """Query logging table with knowledge base tracking"""
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
    top_k = Column(Integer, default=5, nullable=False)
    search_type = Column(String(20), default="hybrid", nullable=False)
    
    # Results
    chunks_retrieved = Column(Integer, default=0, nullable=False)
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


# ==================== CONVERSATION HISTORY MODEL ====================

class ConversationHistory(Base):
    """Conversation history table with knowledge base tracking"""
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
        Index('idx_conv_created', 'conversation_id', 'created_at'),
    )
    
    def __repr__(self) -> str:
        return f"<ConversationHistory(conversation_id={self.conversation_id}, role={self.role})>"
    

    