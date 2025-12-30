"""
Knowledge Base Management Route with USER ISOLATION
- All endpoints require X-User-ID header
- Reset deletes only user's data (not entire database)
- Status shows only user's documents
"""

from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session

from app.models.schemas import KnowledgeBaseStatus, DocumentInfo
from app.services.vector_store_service import VectorStoreService
from app.core.dependencies import get_user_id, get_database_session
from app.db.models import FileMetadata, ChunkMetadata, ConversationHistory, QueryLog
from pathlib import Path
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/knowledge-base", tags=["Knowledge Base"])

# Initialize services
vector_store = VectorStoreService()


@router.get("/status", response_model=KnowledgeBaseStatus)
async def get_knowledge_base_status(
    user_id: str = Depends(get_user_id),
    db: Session = Depends(get_database_session)
):
    """
    Get current status of the knowledge base for current user
    
    Headers:
        X-User-ID: User identifier (REQUIRED)
    
    Returns:
    - Total number of documents (user-specific)
    - Total number of chunks (user-specific)
    - List of documents with their IDs
    """
    try:
        # Get user's files from database
        files = db.query(FileMetadata).filter(
            FileMetadata.user_id == user_id
        ).all()
        
        # Get vector store stats (user-specific)
        vector_stats = vector_store.get_collection_stats(user_id=user_id)
        
        # Build document list
        documents = []
        for file in files:
            documents.append(DocumentInfo(
                file_id=file.file_id,
                filename=file.safe_filename,
                file_type=file.file_extension,
                file_size_bytes=file.file_size_bytes,
                status=file.status,
                chunks_created=file.chunks_created,
                total_pages=file.total_pages,
                uploaded_at=file.uploaded_at,
                processed_at=file.processed_at,
                word_count=file.word_count,
                image_count=file.image_count,
                table_count=file.table_count
            ))
        
        return KnowledgeBaseStatus(
            total_documents=len(files),
            total_chunks=vector_stats.get('total_chunks', 0),
            documents=documents
        )
    
    except Exception as e:
        logger.error(f"Error retrieving knowledge base status: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving knowledge base status: {str(e)}"
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
    
    Headers:
        X-User-ID: User identifier (REQUIRED)
    
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
            "query_logs": 0
        }
        
        # 1. Get all user's files
        files = db.query(FileMetadata).filter(
            FileMetadata.user_id == user_id
        ).all()
        
        logger.info(f"Resetting knowledge base for user {user_id}, found {len(files)} files")
        
        # 2. Delete from vector store (user-specific)
        for file in files:
            try:
                success = vector_store.delete_document_by_file_id(file.file_id, user_id)
                if success:
                    deleted_counts["chunks_vector"] += file.chunks_created
            except Exception as e:
                logger.warning(f"Error deleting {file.file_id} from vector store: {e}")
        
        logger.info(f"✓ Deleted {deleted_counts['chunks_vector']} chunks from vector store")
        
        # 3. Delete files from disk
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
                
                deleted_counts["files"] += 1
            except Exception as e:
                logger.warning(f"Error deleting file {file.storage_path}: {e}")
        
        logger.info(f"✓ Deleted {deleted_counts['files']} files from disk")
        
        # 4. Delete chunk metadata from database
        chunk_count = db.query(ChunkMetadata).join(FileMetadata).filter(
            FileMetadata.user_id == user_id
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
        
        # 7. Delete file metadata (last, as it has foreign key relationships)
        file_count = db.query(FileMetadata).filter(
            FileMetadata.user_id == user_id
        ).delete(synchronize_session=False)
        logger.info(f"✓ Deleted {file_count} file metadata records")
        
        # Commit all deletions
        db.commit()
        
        # 8. Try to cleanup user directories
        try:
            user_upload_dir = Path(settings.upload_dir) / user_id
            if user_upload_dir.exists() and user_upload_dir.is_dir():
                import shutil
                shutil.rmtree(user_upload_dir)
                logger.info(f"✓ Deleted user upload directory")
            
            user_markdown_dir = Path(settings.markdown_dir) / user_id
            if user_markdown_dir.exists() and user_markdown_dir.is_dir():
                import shutil
                shutil.rmtree(user_markdown_dir)
                logger.info(f"✓ Deleted user markdown directory")
        except Exception as e:
            logger.warning(f"Error cleaning up directories: {e}")
        
        message = (
            f"Knowledge base reset successfully for user {user_id}.\n"
            f"Deleted:\n"
            f"  • {deleted_counts['files']} files from disk\n"
            f"  • {deleted_counts['chunks_vector']} chunks from vector store\n"
            f"  • {deleted_counts['chunks_db']} chunk records from database\n"
            f"  • {deleted_counts['conversations']} conversation messages\n"
            f"  • {deleted_counts['query_logs']} query logs\n"
            f"  • {file_count} file metadata records"
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