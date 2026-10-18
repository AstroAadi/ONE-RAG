"""
Knowledge Base Management Route with USER ISOLATION
- All endpoints require X-User-ID header
- Reset deletes only user's data (not entire database)
- Status shows only user's documents
"""

from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session

from app.models.schemas import UserDocumentsResponse, DocumentInfo
from app.services.vector_store_service import VectorStoreService
from app.services.chunk_staging_service import delete_user_staging
from app.core.dependencies import get_user_id, get_database_session
from app.db.models import (
    FileMetadata, ChunkMetadata, ConversationHistory, QueryLog, KnowledgeBase
)
from pathlib import Path
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/knowledge-base", tags=["Knowledge Base"])

# Initialize services
vector_store = VectorStoreService()


@router.get("/status", response_model=UserDocumentsResponse)
async def get_all_user_documents(
    user_id: str = Depends(get_user_id),
    db: Session = Depends(get_database_session)
):
    """
    List every document the current user owns, across all knowledge bases.

    For a single knowledge base use GET /api/knowledge-bases/{kb_id} instead.

    Headers:
        X-User-ID: User identifier (REQUIRED)
    """
    try:
        files = db.query(FileMetadata).filter(
            FileMetadata.user_id == user_id
        ).order_by(FileMetadata.uploaded_at.desc()).all()

        documents = [
            DocumentInfo(
                file_id=file.file_id,
                knowledge_base_id=file.knowledge_base_id,
                filename=file.safe_filename,
                file_type=file.file_extension,
                file_size_bytes=file.file_size_bytes,
                status=file.status.value if file.status else "uploaded",
                chunks_created=file.chunks_created or 0,
                chunks_staged=file.chunks_staged or 0,
                total_pages=file.total_pages or 0,
                uploaded_at=file.uploaded_at,
                processed_at=file.processed_at,
                word_count=file.word_count or 0,
                image_count=file.image_count or 0,
                table_count=file.table_count or 0,
                chunking_strategy=(
                    file.chunking_strategy.value if file.chunking_strategy else None
                ),
            )
            for file in files
        ]

        return UserDocumentsResponse(
            user_id=user_id,
            total_documents=len(documents),
            documents=documents
        )
    
    except Exception as e:
        logger.error(f"Error retrieving user documents: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving user documents: {str(e)}"
        )


@router.delete("/document/{file_id}")
async def delete_document(
    file_id: str,
    user_id: str = Depends(get_user_id),
    db: Session = Depends(get_database_session)
):
    """
    Delete a specific document from the knowledge base (USER ISOLATED)
    
    Headers:
        X-User-ID: User identifier (REQUIRED)
    
    Args:
        file_id: The unique identifier of the file to delete
        
    Deletes:
    - Document from vector store
    - File from disk
    - Database records (file metadata, chunks)
    - Markdown output
    """
    try:
        # Get file metadata with user isolation
        file_metadata = db.query(FileMetadata).filter(
            FileMetadata.file_id == file_id,
            FileMetadata.user_id == user_id  # USER ISOLATION
        ).first()
        
        if not file_metadata:
            raise HTTPException(
                status_code=404,
                detail=f"Document with file_id '{file_id}' not found or does not belong to you"
            )
        
        # Delete from vector store
        success = vector_store.delete_document_by_file_id(file_id, user_id)
        if not success:
            logger.warning(f"Document {file_id} not found in vector store, continuing with deletion")
        
        # Delete file from disk
        file_path = Path(file_metadata.storage_path)
        if file_path.exists():
            file_path.unlink()
            logger.info(f"✓ Deleted file from disk: {file_path}")
        
        # Delete markdown if exists
        if file_metadata.markdown_path:
            markdown_path = Path(file_metadata.markdown_path)
            if markdown_path.exists():
                markdown_path.unlink()
                logger.info(f"✓ Deleted markdown: {markdown_path}")
        
        # Delete from database (cascades to chunks)
        db.delete(file_metadata)
        db.commit()
        logger.info(f"✓ Deleted database records for: {file_id}")
        
        return {
            "message": f"Document '{file_metadata.safe_filename}' deleted successfully",
            "file_id": file_id,
            "user_id": user_id
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


@router.delete("/reset")
async def reset_user_knowledge_base(
    user_id: str = Depends(get_user_id),
    db: Session = Depends(get_database_session)
):
    """
    Reset the knowledge base for CURRENT USER ONLY (not entire database!)
    
    Deletes (for current user only):
    - All documents from vector store
    - All files from disk
    - All database records (files, chunks, conversations, query logs)
    - All markdown outputs
    
    WARNING: This action cannot be undone!
    """
    try:
        deleted_counts = {
            "files": 0,
            "chunks_vector": 0,
            "chunks_db": 0,
            "conversations": 0,
            "query_logs": 0,
            "knowledge_bases": 0
        }
        
        # 1. Get all user's files (for vector store & disk deletion)
        files = db.query(FileMetadata).filter(
            FileMetadata.user_id == user_id
        ).all()
        
        logger.info(f"Resetting knowledge base for user {user_id}, found {len(files)} files")
        
        # 2. Delete from vector store (user-specific)
        for file in files:
            try:
                success = vector_store.delete_document_by_file_id(file.file_id, user_id)
                if success:
                    deleted_counts["chunks_vector"] += file.chunks_created or 0
            except Exception as e:
                logger.warning(f"Error deleting {file.file_id} from vector store: {e}")
        
        logger.info(f"✓ Deleted {deleted_counts['chunks_vector']} chunks from vector store")
        
        # 3. Delete files from disk
        for file in files:
            try:
                # Delete main file
                file_path = Path(file.storage_path)
                if file_path.exists():
                    file_path.unlink(missing_ok=True)
                
                # Delete markdown
                if file.markdown_path:
                    markdown_path = Path(file.markdown_path)
                    if markdown_path.exists():
                        markdown_path.unlink(missing_ok=True)
                
                deleted_counts["files"] += 1
            except Exception as e:
                logger.warning(f"Error deleting file {file.storage_path}: {e}")
        
        logger.info(f"✓ Deleted {deleted_counts['files']} files from disk")
        
        # 4. Delete chunk metadata from database (NO JOIN before delete!)
        chunk_count = db.query(ChunkMetadata).filter(
            ChunkMetadata.file_id.in_(
                db.query(FileMetadata.file_id).filter(FileMetadata.user_id == user_id)
            )
        ).delete(synchronize_session=False)
        deleted_counts["chunks_db"] = chunk_count
        logger.info(f"✓ Deleted {chunk_count} chunk records from database")
        
        # 5. Delete conversation history
        conv_count = db.query(ConversationHistory).filter(
            ConversationHistory.user_id == user_id
        ).delete(synchronize_session=False)
        deleted_counts["conversations"] = conv_count
        logger.info(f"✓ Deleted {conv_count} conversation messages")
        
        # 6. Delete query logs
        query_count = db.query(QueryLog).filter(
            QueryLog.user_id == user_id
        ).delete(synchronize_session=False)
        deleted_counts["query_logs"] = query_count
        logger.info(f"✓ Deleted {query_count} query logs")
        
        # 7. Delete file metadata
        file_count = db.query(FileMetadata).filter(
            FileMetadata.user_id == user_id
        ).delete(synchronize_session=False)
        logger.info(f"✓ Deleted {file_count} file metadata records")
        
        # 8. Delete the knowledge bases themselves (files are already gone)
        kb_count = db.query(KnowledgeBase).filter(
            KnowledgeBase.user_id == user_id
        ).delete(synchronize_session=False)
        deleted_counts["knowledge_bases"] = kb_count
        logger.info(f"✓ Deleted {kb_count} knowledge bases")
        
        # Commit everything
        db.commit()
        
        # 9. Drop any chunks staged but never finalized
        try:
            delete_user_staging(user_id)
            logger.info("✓ Deleted staged chunks")
        except Exception as e:
            logger.warning(f"Error deleting staged chunks: {e}")
        
        # 10. Cleanup user directories
        try:
            user_upload_dir = Path(settings.upload_dir) / user_id
            if user_upload_dir.exists() and user_upload_dir.is_dir():
                import shutil
                shutil.rmtree(user_upload_dir, ignore_errors=True)
                logger.info(f"✓ Deleted user upload directory")
            
            user_markdown_dir = Path(settings.markdown_dir) / user_id
            if user_markdown_dir.exists() and user_markdown_dir.is_dir():
                import shutil
                shutil.rmtree(user_markdown_dir, ignore_errors=True)
                logger.info(f"✓ Deleted user markdown directory")
        except Exception as e:
            logger.warning(f"Error cleaning up directories: {e}")
        
        message = (
            f"All data reset successfully for user {user_id}.\n"
            f"Deleted:\n"
            f"  • {deleted_counts['files']} files from disk\n"
            f"  • {deleted_counts['chunks_vector']} chunks from vector store\n"
            f"  • {deleted_counts['chunks_db']} chunk records from database\n"
            f"  • {deleted_counts['conversations']} conversation messages\n"
            f"  • {deleted_counts['query_logs']} query logs\n"
            f"  • {file_count} file metadata records\n"
            f"  • {deleted_counts['knowledge_bases']} knowledge bases"
        )
        
        return {
            "message": message,
            "status": "success",
            "user_id": user_id,
            "deleted_counts": deleted_counts
        }
    
    except Exception as e:
        db.rollback()
        logger.error(f"Error resetting knowledge base: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Error resetting knowledge base: {str(e)}"
        )


@router.get("/stats")
async def get_user_stats(
    user_id: str = Depends(get_user_id),
    db: Session = Depends(get_database_session)
):
    """
    Get detailed statistics for current user
    
    Headers:
        X-User-ID: User identifier (REQUIRED)
    
    Returns:
        Detailed statistics about user's documents, queries, and conversations
    """
    try:
        from sqlalchemy import func
        
        # File statistics
        total_files = db.query(func.count(FileMetadata.file_id)).filter(
            FileMetadata.user_id == user_id
        ).scalar() or 0
        
        total_bytes = db.query(func.sum(FileMetadata.file_size_bytes)).filter(
            FileMetadata.user_id == user_id
        ).scalar() or 0
        
        total_chunks = db.query(func.sum(FileMetadata.chunks_created)).filter(
            FileMetadata.user_id == user_id
        ).scalar() or 0
        
        # Query statistics
        total_queries = db.query(func.count(QueryLog.id)).filter(
            QueryLog.user_id == user_id
        ).scalar() or 0
        
        avg_processing_time = db.query(func.avg(QueryLog.processing_time_seconds)).filter(
            QueryLog.user_id == user_id
        ).scalar() or 0
        
        # Conversation statistics
        total_conversations = db.query(
            func.count(func.distinct(ConversationHistory.conversation_id))
        ).filter(
            ConversationHistory.user_id == user_id
        ).scalar() or 0
        
        # File type breakdown
        file_types = db.query(
            FileMetadata.file_extension,
            func.count(FileMetadata.file_id)
        ).filter(
            FileMetadata.user_id == user_id
        ).group_by(
            FileMetadata.file_extension
        ).all()
        
        file_type_breakdown = {ext: count for ext, count in file_types}
        
        return {
            "user_id": user_id,
            "files": {
                "total": total_files,
                "total_size_bytes": total_bytes,
                "total_size_mb": round(total_bytes / (1024 * 1024), 2),
                "total_chunks": total_chunks,
                "avg_chunks_per_file": round(total_chunks / total_files, 1) if total_files > 0 else 0,
                "by_type": file_type_breakdown
            },
            "queries": {
                "total": total_queries,
                "avg_processing_time_seconds": round(avg_processing_time, 2)
            },
            "conversations": {
                "total": total_conversations
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting user stats: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting statistics: {str(e)}"
        )