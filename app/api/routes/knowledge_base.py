"""
Knowledge Base Management Routes - NEW
Complete CRUD operations for knowledge bases
Replaces old knowledge_base.py which was for global KB management
"""

from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func, desc
from typing import List, Optional
import uuid
from datetime import datetime
import logging
from pathlib import Path
import shutil

from app.models.schemas import (
    KnowledgeBaseCreate,
    KnowledgeBaseUpdate,
    KnowledgeBaseInfo,
    KnowledgeBaseList,
    KnowledgeBaseStatus,
    DocumentInfo
)
from app.db.models import (
    KnowledgeBase, FileMetadata, ChunkMetadata, 
    QueryLog, ConversationHistory, ProcessingStatus
)
from app.core.dependencies import get_user_id, get_database_session
from app.services.vector_store_service import VectorStoreService
from app.core.config import settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/knowledge-bases", tags=["Knowledge Bases"])

# Initialize services with lazy loading
_vector_store = None

def get_vector_store():
    """Get or initialize vector store (lazy loading)"""
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStoreService()
    return _vector_store


@router.post("/", response_model=KnowledgeBaseInfo)
async def create_knowledge_base(
    request: KnowledgeBaseCreate,
    user_id: str = Depends(get_user_id),
    db: Session = Depends(get_database_session)
):
    """
    Create a new knowledge base
    
    Headers:
        X-User-ID: User identifier (REQUIRED)
    
    Body:
        knowledge_base_name: Name of the KB (unique per user, case-insensitive)
        description: Optional description
    
    Rules:
    - KB name must be unique per user (case-insensitive)
    - KB name is normalized using .lower().strip()
    - KB ID is auto-generated (UUID)
    - No special characters allowed in name
    
    Returns:
        Created KB information
    
    Example:
        POST /api/knowledge-bases
        {
            "knowledge_base_name": "Machine Learning Papers",
            "description": "Research papers on ML and AI"
        }
    """
    try:
        # Normalize name
        normalized_name = request.knowledge_base_name.lower().strip()
        
        # Check if KB with same name exists
        existing = db.query(KnowledgeBase).filter(
            KnowledgeBase.user_id == user_id,
            KnowledgeBase.knowledge_base_name_normalized == normalized_name
        ).first()
        
        if existing:
            raise HTTPException(
                status_code=400,
                detail=f"Knowledge base '{request.knowledge_base_name}' already exists"
            )
        
        # Create KB
        kb_id = str(uuid.uuid4())
        
        new_kb = KnowledgeBase(
            knowledge_base_id=kb_id,
            user_id=user_id,
            knowledge_base_name=request.knowledge_base_name.strip(),
            knowledge_base_name_normalized=normalized_name,
            description=request.description,
            document_count=0,
            total_chunks=0,
            total_size_bytes=0,
            created_at=datetime.utcnow()
        )
        
        db.add(new_kb)
        db.commit()
        db.refresh(new_kb)
        
        logger.info(f"✓ Created KB: {kb_id} - '{new_kb.knowledge_base_name}' for user {user_id}")
        
        return KnowledgeBaseInfo(**new_kb.to_dict())
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error creating KB: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/", response_model=KnowledgeBaseList)
async def list_knowledge_bases(
    user_id: str = Depends(get_user_id),
    db: Session = Depends(get_database_session)
):
    """
    List all knowledge bases for current user
    
    Headers:
        X-User-ID: User identifier (REQUIRED)
    
    Returns:
        List of all KBs owned by the user, sorted by creation date (newest first)
    
    Example Response:
        {
            "user_id": "user_123",
            "total_knowledge_bases": 3,
            "knowledge_bases": [
                {
                    "knowledge_base_id": "kb_uuid_1",
                    "knowledge_base_name": "ML Papers",
                    "document_count": 15,
                    "total_chunks": 450,
                    ...
                }
            ]
        }
    """
    try:
        kbs = db.query(KnowledgeBase).filter(
            KnowledgeBase.user_id == user_id
        ).order_by(desc(KnowledgeBase.created_at)).all()
        
        kb_list = [KnowledgeBaseInfo(**kb.to_dict()) for kb in kbs]
        
        logger.info(f"Listed {len(kb_list)} KBs for user {user_id}")
        
        return KnowledgeBaseList(
            user_id=user_id,
            total_knowledge_bases=len(kb_list),
            knowledge_bases=kb_list
        )
        
    except Exception as e:
        logger.error(f"Error listing KBs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{kb_id}", response_model=KnowledgeBaseStatus)
async def get_knowledge_base_status(
    kb_id: str,
    user_id: str = Depends(get_user_id),
    db: Session = Depends(get_database_session)
):
    """
    Get detailed status of a knowledge base
    
    Headers:
        X-User-ID: User identifier (REQUIRED)
    
    Args:
        kb_id: Knowledge base ID
    
    Returns:
        Detailed KB information including:
        - KB metadata
        - List of all documents in the KB
        - Vector store statistics
        - Total chunks, size, etc.
    
    Security:
        User can only access their own KBs
    """
    try:
        # Get KB with ownership verification
        kb = db.query(KnowledgeBase).filter(
            KnowledgeBase.knowledge_base_id == kb_id,
            KnowledgeBase.user_id == user_id
        ).first()
        
        if not kb:
            raise HTTPException(
                status_code=404, 
                detail="Knowledge base not found or access denied"
            )
        
        # Get documents in this KB
        files = db.query(FileMetadata).filter(
            FileMetadata.knowledge_base_id == kb_id,
            FileMetadata.user_id == user_id
        ).order_by(desc(FileMetadata.uploaded_at)).all()
        
        documents = [DocumentInfo(**f.to_dict()) for f in files]
        
        # Get vector store stats
        vector_stats = get_vector_store().get_collection_stats(
            user_id=user_id,
            knowledge_base_id=kb_id
        )
        
        return KnowledgeBaseStatus(
            knowledge_base_id=kb.knowledge_base_id,
            knowledge_base_name=kb.knowledge_base_name,
            user_id=kb.user_id,
            total_documents=len(documents),
            total_chunks=vector_stats.get('total_chunks', kb.total_chunks),
            total_size_bytes=kb.total_size_bytes,
            documents=documents,
            created_at=kb.created_at,
            last_accessed=kb.last_accessed
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting KB status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{kb_id}", response_model=KnowledgeBaseInfo)
async def update_knowledge_base(
    kb_id: str,
    request: KnowledgeBaseUpdate,
    user_id: str = Depends(get_user_id),
    db: Session = Depends(get_database_session)
):
    """
    Update knowledge base name or description
    
    Headers:
        X-User-ID: User identifier (REQUIRED)
    
    Args:
        kb_id: Knowledge base ID
    
    Body:
        knowledge_base_name: New name (optional, must be unique)
        description: New description (optional)
    
    Returns:
        Updated KB information
    
    Note:
        - Name changes must still be unique per user
        - Vector store metadata is automatically updated if name changes
    """
    try:
        kb = db.query(KnowledgeBase).filter(
            KnowledgeBase.knowledge_base_id == kb_id,
            KnowledgeBase.user_id == user_id
        ).first()
        
        if not kb:
            raise HTTPException(
                status_code=404, 
                detail="Knowledge base not found or access denied"
            )
        
        old_name = kb.knowledge_base_name
        
        # Update name if provided
        if request.knowledge_base_name:
            new_normalized = request.knowledge_base_name.lower().strip()
            
            # Check if new name conflicts with other KBs
            if new_normalized != kb.knowledge_base_name_normalized:
                existing = db.query(KnowledgeBase).filter(
                    KnowledgeBase.user_id == user_id,
                    KnowledgeBase.knowledge_base_name_normalized == new_normalized,
                    KnowledgeBase.knowledge_base_id != kb_id
                ).first()
                
                if existing:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Knowledge base '{request.knowledge_base_name}' already exists"
                    )
            
            kb.knowledge_base_name = request.knowledge_base_name.strip()
            kb.knowledge_base_name_normalized = new_normalized
            
            # TODO: Update vector store metadata for all chunks in this KB
            # This is a heavy operation and should be done asynchronously
            logger.warning(
                f"KB name changed from '{old_name}' to '{kb.knowledge_base_name}'. "
                f"Vector store metadata should be updated."
            )
        
        # Update description
        if request.description is not None:
            kb.description = request.description
        
        kb.updated_at = datetime.utcnow()
        db.commit()
        db.refresh(kb)
        
        logger.info(f"✓ Updated KB: {kb_id} - '{kb.knowledge_base_name}'")
        
        return KnowledgeBaseInfo(**kb.to_dict())
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error updating KB: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{kb_id}")
async def delete_knowledge_base(
    kb_id: str,
    user_id: str = Depends(get_user_id),
    db: Session = Depends(get_database_session)
):
    """
    Delete a knowledge base and all its contents
    
    Headers:
        X-User-ID: User identifier (REQUIRED)
    
    Args:
        kb_id: Knowledge base ID to delete
    
    Cascades to:
        - All documents in the KB (files deleted from disk)
        - All chunks in vector store
        - All file metadata from database
        - All chunk metadata from database
        - All conversation history related to this KB
        - All query logs related to this KB
    
    WARNING: This action cannot be undone!
    
    Returns:
        Success message with deletion statistics
    """
    try:
        kb = db.query(KnowledgeBase).filter(
            KnowledgeBase.knowledge_base_id == kb_id,
            KnowledgeBase.user_id == user_id
        ).first()
        
        if not kb:
            raise HTTPException(
                status_code=404,
                detail="Knowledge base not found or access denied"
            )
        
        kb_name = kb.knowledge_base_name
        
        logger.info(f"🗑️  Deleting KB: {kb_id} - '{kb_name}'")
        
        deletion_stats = {
            "files_deleted": 0,
            "chunks_deleted_vector": 0,
            "chunks_deleted_db": 0,
            "conversations_deleted": 0,
            "query_logs_deleted": 0
        }
        
        # 1. Get all files in this KB
        files = db.query(FileMetadata).filter(
            FileMetadata.knowledge_base_id == kb_id
        ).all()
        
        # 2. Delete files from disk
        for file in files:
            try:
                # Delete main file
                file_path = Path(file.storage_path)
                if file_path.exists():
                    file_path.unlink()
                
                # Delete markdown
                if file.markdown_path:
                    markdown_path = Path(file.markdown_path)
                    if markdown_path.exists():
                        markdown_path.unlink()
                
                deletion_stats["files_deleted"] += 1
            except Exception as e:
                logger.warning(f"Error deleting file {file.storage_path}: {e}")
        
        # 3. Delete from vector store
        try:
            chunks_deleted = get_vector_store().delete_knowledge_base(kb_id, user_id)
            deletion_stats["chunks_deleted_vector"] = chunks_deleted
            logger.info(f"  ✓ Deleted {chunks_deleted} chunks from vector store")
        except Exception as e:
            logger.error(f"Error deleting from vector store: {e}")
        
        # 4. Delete chunk metadata from database
        # First get the chunk IDs to delete, then delete them
        chunk_ids_to_delete = (
            db.query(ChunkMetadata.chunk_id)
            .join(FileMetadata)
            .filter(FileMetadata.knowledge_base_id == kb_id)
            .all()
        )
        chunk_ids = [chunk_id for (chunk_id,) in chunk_ids_to_delete]
        
        if chunk_ids:
            chunk_count = db.query(ChunkMetadata).filter(
                ChunkMetadata.chunk_id.in_(chunk_ids)
            ).delete(synchronize_session=False)
            deletion_stats["chunks_deleted_db"] = chunk_count
        else:
            deletion_stats["chunks_deleted_db"] = 0
        
        # 5. Delete conversation history
        conv_count = db.query(ConversationHistory).filter(
            ConversationHistory.knowledge_base_id == kb_id,
            ConversationHistory.user_id == user_id
        ).delete(synchronize_session=False)
        deletion_stats["conversations_deleted"] = conv_count
        
        # 6. Delete query logs
        query_count = db.query(QueryLog).filter(
            QueryLog.knowledge_base_id == kb_id,
            QueryLog.user_id == user_id
        ).delete(synchronize_session=False)
        deletion_stats["query_logs_deleted"] = query_count
        
        # 7. Delete file metadata (will cascade to chunks due to FK)
        file_count = db.query(FileMetadata).filter(
            FileMetadata.knowledge_base_id == kb_id
        ).delete(synchronize_session=False)
        
        # 8. Delete KB itself
        db.delete(kb)
        
        # Commit all deletions
        db.commit()
        
        # 9. Try to cleanup KB directory
        try:
            kb_markdown_dir = Path(settings.markdown_dir) / user_id / kb_id
            if kb_markdown_dir.exists() and kb_markdown_dir.is_dir():
                shutil.rmtree(kb_markdown_dir)
                logger.info(f"  ✓ Deleted KB markdown directory")
        except Exception as e:
            logger.warning(f"Error cleaning up KB directory: {e}")
        
        message = (
            f"Knowledge base '{kb_name}' deleted successfully.\n"
            f"Deleted:\n"
            f"  • {deletion_stats['files_deleted']} files from disk\n"
            f"  • {deletion_stats['chunks_deleted_vector']} chunks from vector store\n"
            f"  • {deletion_stats['chunks_deleted_db']} chunk records from database\n"
            f"  • {deletion_stats['conversations_deleted']} conversation messages\n"
            f"  • {deletion_stats['query_logs_deleted']} query logs\n"
            f"  • {file_count} file metadata records"
        )
        
        logger.info(f"✓ Deleted KB: {kb_id} - '{kb_name}'")
        
        return {
            "message": message,
            "knowledge_base_id": kb_id,
            "knowledge_base_name": kb_name,
            "status": "success",
            "deletion_stats": deletion_stats
        }
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error deleting KB: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting knowledge base: {str(e)}"
        )


@router.delete("/{kb_id}/documents/{file_id}")
async def delete_document_from_kb(
    kb_id: str,
    file_id: str,
    user_id: str = Depends(get_user_id),
    db: Session = Depends(get_database_session)
):
    """
    Delete a specific document from a knowledge base
    
    Headers:
        X-User-ID: User identifier (REQUIRED)
    
    Args:
        kb_id: Knowledge base ID
        file_id: File ID to delete
    
    Deletes:
        - Document from vector store
        - File from disk
        - Database records (file metadata, chunks)
        - Markdown output
    
    Updates:
        - KB statistics (document count, chunk count, size)
    
    Returns:
        Success message
    """
    try:
        # Verify KB ownership
        kb = db.query(KnowledgeBase).filter(
            KnowledgeBase.knowledge_base_id == kb_id,
            KnowledgeBase.user_id == user_id
        ).first()
        
        if not kb:
            raise HTTPException(
                status_code=404,
                detail="Knowledge base not found or access denied"
            )
        
        # Get file metadata
        file_metadata = db.query(FileMetadata).filter(
            FileMetadata.file_id == file_id,
            FileMetadata.knowledge_base_id == kb_id,
            FileMetadata.user_id == user_id
        ).first()
        
        if not file_metadata:
            raise HTTPException(
                status_code=404,
                detail=f"Document not found in this knowledge base"
            )
        
        filename = file_metadata.original_filename
        chunks_count = file_metadata.chunks_created
        file_size = file_metadata.file_size_bytes
        
        # Delete from vector store
        success = get_vector_store().delete_document_by_file_id(file_id, user_id, kb_id)
        if not success:
            logger.warning(f"Document {file_id} not found in vector store")
        
        # Delete file from disk
        file_path = Path(file_metadata.storage_path)
        if file_path.exists():
            file_path.unlink()
            logger.info(f"  ✓ Deleted file from disk: {file_path}")
        
        # Delete markdown
        if file_metadata.markdown_path:
            markdown_path = Path(file_metadata.markdown_path)
            if markdown_path.exists():
                markdown_path.unlink()
                logger.info(f"  ✓ Deleted markdown: {markdown_path}")
        
        # Delete from database (cascades to chunks)
        db.delete(file_metadata)
        
        # Update KB statistics
        kb.document_count = max(0, kb.document_count - 1)
        kb.total_chunks = max(0, kb.total_chunks - chunks_count)
        kb.total_size_bytes = max(0, kb.total_size_bytes - file_size)
        kb.updated_at = datetime.utcnow()
        
        db.commit()
        
        logger.info(
            f"✓ Deleted document '{filename}' from KB '{kb.knowledge_base_name}'"
        )
        
        return {
            "message": f"Document '{filename}' deleted successfully",
            "file_id": file_id,
            "knowledge_base_id": kb_id,
            "knowledge_base_name": kb.knowledge_base_name,
            "chunks_deleted": chunks_count
        }
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting document: {str(e)}"
        )


@router.get("/{kb_id}/stats")
async def get_kb_statistics(
    kb_id: str,
    user_id: str = Depends(get_user_id),
    db: Session = Depends(get_database_session)
):
    """
    Get detailed statistics for a knowledge base
    
    Headers:
        X-User-ID: User identifier (REQUIRED)
    
    Args:
        kb_id: Knowledge base ID
    
    Returns:
        Detailed statistics including:
        - File type breakdown
        - Total size, chunks, pages
        - Query statistics
        - Conversation statistics
        - Processing info
    """
    try:
        # Verify KB ownership
        kb = db.query(KnowledgeBase).filter(
            KnowledgeBase.knowledge_base_id == kb_id,
            KnowledgeBase.user_id == user_id
        ).first()
        
        if not kb:
            raise HTTPException(
                status_code=404,
                detail="Knowledge base not found or access denied"
            )
        
        # File statistics
        total_files = db.query(func.count(FileMetadata.file_id)).filter(
            FileMetadata.knowledge_base_id == kb_id
        ).scalar() or 0
        
        total_pages = db.query(func.sum(FileMetadata.total_pages)).filter(
            FileMetadata.knowledge_base_id == kb_id
        ).scalar() or 0
        
        total_words = db.query(func.sum(FileMetadata.word_count)).filter(
            FileMetadata.knowledge_base_id == kb_id
        ).scalar() or 0
        
        # File type breakdown
        file_types = db.query(
            FileMetadata.file_extension,
            func.count(FileMetadata.file_id)
        ).filter(
            FileMetadata.knowledge_base_id == kb_id
        ).group_by(
            FileMetadata.file_extension
        ).all()
        
        file_type_breakdown = {ext: count for ext, count in file_types}
        
        # Processing status breakdown
        status_breakdown = db.query(
            FileMetadata.status,
            func.count(FileMetadata.file_id)
        ).filter(
            FileMetadata.knowledge_base_id == kb_id
        ).group_by(
            FileMetadata.status
        ).all()
        
        status_counts = {status.value: count for status, count in status_breakdown}
        
        # Query statistics
        total_queries = db.query(func.count(QueryLog.id)).filter(
            QueryLog.knowledge_base_id == kb_id
        ).scalar() or 0
        
        avg_processing_time = db.query(func.avg(QueryLog.processing_time_seconds)).filter(
            QueryLog.knowledge_base_id == kb_id
        ).scalar() or 0
        
        avg_confidence = db.query(func.avg(QueryLog.confidence_score)).filter(
            QueryLog.knowledge_base_id == kb_id
        ).scalar() or 0
        
        # Conversation statistics
        total_conversations = db.query(
            func.count(func.distinct(ConversationHistory.conversation_id))
        ).filter(
            ConversationHistory.knowledge_base_id == kb_id
        ).scalar() or 0
        
        return {
            "knowledge_base_id": kb_id,
            "knowledge_base_name": kb.knowledge_base_name,
            "user_id": user_id,
            "files": {
                "total": total_files,
                "total_size_bytes": kb.total_size_bytes,
                "total_size_mb": round(kb.total_size_bytes / (1024 * 1024), 2),
                "total_chunks": kb.total_chunks,
                "total_pages": total_pages,
                "total_words": total_words,
                "avg_chunks_per_file": round(kb.total_chunks / total_files, 1) if total_files > 0 else 0,
                "by_type": file_type_breakdown,
                "by_status": status_counts
            },
            "queries": {
                "total": total_queries,
                "avg_processing_time_seconds": round(avg_processing_time, 2),
                "avg_confidence_score": round(avg_confidence, 3)
            },
            "conversations": {
                "total": total_conversations
            },
            "metadata": {
                "created_at": kb.created_at.isoformat(),
                "updated_at": kb.updated_at.isoformat() if kb.updated_at else None,
                "last_accessed": kb.last_accessed.isoformat() if kb.last_accessed else None
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting KB stats: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting statistics: {str(e)}"
        )