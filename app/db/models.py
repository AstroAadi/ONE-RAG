"""
Database Models for RAG System
Tracks file metadata, processing status, and user ownership
"""

from sqlalchemy import (
    Column, String, Integer, DateTime, Boolean, Float, Text, ForeignKey, Enum
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


class FileMetadata(Base):
    """
    File metadata table with user isolation
    
    Tracks all uploaded files and their processing status
    """
    __tablename__ = "file_metadata"
    
    # Primary identifiers
    file_id = Column(String(36), primary_key=True, index=True)
    user_id = Column(String(100), nullable=False, index=True)
    
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
    
    # Timestamps
    uploaded_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    processed_at = Column(DateTime, nullable=True)
    last_accessed = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Processing metadata (JSON-like text field)
    processing_metadata = Column(Text, nullable=True)  # Store JSON string
    error_message = Column(Text, nullable=True)
    
    # Statistics
    word_count = Column(Integer, default=0)
    image_count = Column(Integer, default=0)
    table_count = Column(Integer, default=0)
    
    # Relationships
    chunks = relationship("ChunkMetadata", back_populates="file", cascade="all, delete-orphan")
    
    def __repr__(self) -> str:
        return f"<FileMetadata(file_id={self.file_id}, user_id={self.user_id}, filename={self.original_filename})>"
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "file_id": self.file_id,
            "user_id": self.user_id,
            "original_filename": self.original_filename,
            "file_extension": self.file_extension,
            "file_type": self.file_type.value if self.file_type else None,
            "file_size_bytes": self.file_size_bytes,
            "status": self.status.value if self.status else None,
            "chunks_created": self.chunks_created,
            "total_pages": self.total_pages,
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
    Links to ChromaDB chunk IDs
    """
    __tablename__ = "chunk_metadata"
    
    # Primary identifier
    chunk_id = Column(String(100), primary_key=True, index=True)
    
    # Foreign key to file
    file_id = Column(String(36), ForeignKey("file_metadata.file_id", ondelete="CASCADE"), nullable=False, index=True)
    
    # Chunk information
    chunk_index = Column(Integer, nullable=False)
    chunk_type = Column(String(20), default="text")  # text, image, table, chart
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
    Query logging table for monitoring and analytics
    """
    __tablename__ = "query_logs"
    
    # Primary identifier
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # User information
    user_id = Column(String(100), nullable=False, index=True)
    
    # Query information
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=True)
    
    # Search parameters
    top_k = Column(Integer, default=5)
    keyword_weight = Column(Float, default=0.3)
    semantic_weight = Column(Float, default=0.7)
    
    # Results
    chunks_retrieved = Column(Integer, default=0)
    processing_time_seconds = Column(Float, nullable=True)
    confidence_score = Column(Float, nullable=True)
    
    # Timestamps
    queried_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    
    # Optional conversation tracking
    conversation_id = Column(String(100), nullable=True, index=True)
    
    def __repr__(self) -> str:
        return f"<QueryLog(id={self.id}, user_id={self.user_id}, question={self.question[:50]})>"


class ConversationHistory(Base):
    """
    Conversation history table
    Stores multi-turn conversations
    """
    __tablename__ = "conversation_history"
    
    # Primary identifier
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Identifiers
    conversation_id = Column(String(100), nullable=False, index=True)
    user_id = Column(String(100), nullable=False, index=True)
    
    # Message information
    role = Column(String(20), nullable=False)  # 'user' or 'assistant'
    content = Column(Text, nullable=False)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    def __repr__(self) -> str:
        return f"<ConversationHistory(conversation_id={self.conversation_id}, role={self.role})>"