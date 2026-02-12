"""
Chunks API - SIMPLIFIED VERSION
Only uses 4 states: uploaded, processing, completed, failed
No granular status tracking
"""

from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from typing import Optional, List
from datetime import datetime
import logging
import json

from app.core.dependencies import get_user_id, get_database_session
from app.core.task_manager import get_task_manager, TaskType
from app.db.models import (
    FileMetadata, ChunkMetadata, KnowledgeBase,
    ProcessingStatus, ChunkingStrategyEnum
)
from app.models.schemas import ChunkingStrategy
from app.services.chunking_service import EnhancedChunkingService
from app.services.vector_store_service import VectorStoreService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/chunks", tags=["Chunks"])


@router.post("/create")
async def create_chunks(
    file_id: str,
    chunking_strategy: ChunkingStrategy = ChunkingStrategy.FIXED_SIZE,
    target_chunk_tokens: Optional[int] = 512,
    chunk_overlap_tokens: Optional[int] = 50,
    force_recreate: bool = False,
    user_id: str = Depends(get_user_id),
    db: Session = Depends(get_database_session)
):
    """
    Create chunks from extracted document
    
    Status stays as COMPLETED (extraction already done)
    """
    try:
        # Get file metadata
        file_metadata = db.query(FileMetadata).filter(
            FileMetadata.file_id == file_id,
            FileMetadata.user_id == user_id
        ).first()
        
        if not file_metadata:
            raise HTTPException(status_code=404, detail=f"File not found: {file_id}")
        
        # Verify KB exists
        kb = db.query(KnowledgeBase).filter(
            KnowledgeBase.knowledge_base_id == file_metadata.knowledge_base_id,
            KnowledgeBase.user_id == user_id
        ).first()
        
        if not kb:
            raise HTTPException(status_code=404, detail=f"Knowledge base not found")
        
        # Check if file is completed (extraction done)
        if file_metadata.status != ProcessingStatus.COMPLETED:
            raise HTTPException(
                status_code=400,
                detail=f"File must be extracted first. Current status: {file_metadata.status.value}"
            )
        
        # Check if chunks already exist
        existing_chunks = db.query(ChunkMetadata).filter(
            ChunkMetadata.file_id == file_id
        ).count()
        
        if existing_chunks > 0 and not force_recreate:
            raise HTTPException(
                status_code=400,
                detail=f"File already has {existing_chunks} chunks. Use force_recreate=true to recreate."
            )
        
        # Delete existing chunks if force_recreate
        if force_recreate and existing_chunks > 0:
            logger.info(f"🗑️  Deleting {existing_chunks} existing chunks for file {file_id}")
            db.query(ChunkMetadata).filter(
                ChunkMetadata.file_id == file_id
            ).delete()
            
            # Update KB stats
            kb.total_chunks = max(0, kb.total_chunks - existing_chunks)
            db.commit()
        
        # Get markdown content
        markdown_content = None
        if file_metadata.markdown_path:
            try:
                with open(file_metadata.markdown_path, 'r', encoding='utf-8') as f:
                    markdown_content = f.read()
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Could not read markdown content: {str(e)}"
                )
        
        if not markdown_content:
            raise HTTPException(
                status_code=400,
                detail="No markdown content available. Re-run extraction."
            )
        
        logger.info(f"📊 Creating chunks for file {file_id} using {chunking_strategy.value}")
        
        # Create chunking service
        chunking_service = EnhancedChunkingService(
            target_chunk_tokens=target_chunk_tokens or 512,
            chunk_overlap_tokens=chunk_overlap_tokens or 50
        )
        
        # Prepare metadata
        base_metadata = {
            'user_id': user_id,
            'file_id': file_id,
            'filename': file_metadata.safe_filename,
            'knowledge_base_id': kb.knowledge_base_id,
            'knowledge_base_name': kb.knowledge_base_name
        }
        
        # Create chunks
        import time
        start_time = time.time()
        
        chunks = chunking_service.split_document_into_chunks(
            markdown_content,
            base_metadata=base_metadata,
            strategy=chunking_strategy
        )
        
        chunking_time = time.time() - start_time
        
        logger.info(f"✅ Created {len(chunks)} chunks in {chunking_time:.2f}s")
        
        # Save chunks to database
        chunk_ids = []
        
        for chunk in chunks:
            chunk_db = ChunkMetadata(
                chunk_id=chunk['chunk_id'],
                file_id=file_id,
                chunk_index=chunk['metadata']['chunk_index'],
                chunk_type=chunk['metadata'].get('chunk_type', 'text'),
                char_count=chunk['char_count'],
                page_number=chunk['metadata'].get('page_number', 0),
                created_at=datetime.utcnow()
            )
            db.add(chunk_db)
            chunk_ids.append(chunk['chunk_id'])
        
        # Map strategy - only 4 valid strategies
        strategy_map = {
            'default': ChunkingStrategyEnum.DEFAULT,
            'fixed_size': ChunkingStrategyEnum.FIXED_SIZE,
            'hierarchical': ChunkingStrategyEnum.HIERARCHICAL,
            'semantic': ChunkingStrategyEnum.SEMANTIC
        }
        
        # Update file metadata
        file_metadata.chunks_created = len(chunks)
        file_metadata.chunking_strategy = strategy_map.get(chunking_strategy.value)
        file_metadata.target_chunk_tokens = target_chunk_tokens or 512
        file_metadata.chunk_overlap_tokens = chunk_overlap_tokens or 50
        
        # Update KB stats
        kb.total_chunks += len(chunks)
        kb.updated_at = datetime.utcnow()
        
        db.commit()
        
        logger.info(f"🎉 Chunking complete: {len(chunks)} chunks saved")
        
        return {
            "file_id": file_id,
            "knowledge_base_id": kb.knowledge_base_id,
            "chunks_created": len(chunks),
            "chunk_ids": chunk_ids[:10],
            "total_chunk_ids": len(chunk_ids),
            "chunking_strategy": chunking_strategy.value,
            "chunking_time_seconds": round(chunking_time, 2),
            "status": "completed",
            "message": f"Created {len(chunks)} chunks successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"❌ Chunking failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Chunking failed: {str(e)}")


@router.post("/finalize")
async def finalize_chunks(
    file_id: str,
    chunk_ids: Optional[List[str]] = None,
    batch_size: int = 50,
    user_id: str = Depends(get_user_id),
    db: Session = Depends(get_database_session)
):
    """
    Index chunks to vector database
    
    Status stays COMPLETED
    """
    try:
        # Get file metadata
        file_metadata = db.query(FileMetadata).filter(
            FileMetadata.file_id == file_id,
            FileMetadata.user_id == user_id
        ).first()
        
        if not file_metadata:
            raise HTTPException(status_code=404, detail=f"File not found: {file_id}")
        
        # Verify KB
        kb = db.query(KnowledgeBase).filter(
            KnowledgeBase.knowledge_base_id == file_metadata.knowledge_base_id
        ).first()
        
        if not kb:
            raise HTTPException(status_code=404, detail="Knowledge base not found")
        
        # Check if file has chunks
        if file_metadata.chunks_created == 0:
            raise HTTPException(
                status_code=400,
                detail="No chunks found. Create chunks first via /api/chunks/create"
            )
        
        # Get chunks to index
        query = db.query(ChunkMetadata).filter(
            ChunkMetadata.file_id == file_id
        )
        
        if chunk_ids:
            query = query.filter(ChunkMetadata.chunk_id.in_(chunk_ids))
        
        chunks_to_index = query.all()
        
        if not chunks_to_index:
            raise HTTPException(
                status_code=400,
                detail="No chunks to index"
            )
        
        logger.info(f"📊 Indexing {len(chunks_to_index)} chunks to vector DB")
        
        # For large documents, use background task
        if len(chunks_to_index) > 100:
            task_manager = get_task_manager()
            
            task_id = task_manager.submit_task(
                func=index_chunks_background,
                task_type=TaskType.IO_BOUND,
                description=f"Index {len(chunks_to_index)} chunks",
                user_id=user_id,
                file_id=file_id,
                knowledge_base_id=kb.knowledge_base_id,
                chunk_ids=[c.chunk_id for c in chunks_to_index],
                batch_size=batch_size
            )
            
            return {
                "file_id": file_id,
                "chunks_to_index": len(chunks_to_index),
                "task_id": task_id,
                "status": "indexing",
                "message": f"Indexing {len(chunks_to_index)} chunks in background. Track: /api/tasks/{task_id}"
            }
        
        # For small documents, index immediately
        vector_store = VectorStoreService()
        
        # Read markdown to get chunk text
        markdown_content = None
        if file_metadata.markdown_path:
            with open(file_metadata.markdown_path, 'r', encoding='utf-8') as f:
                markdown_content = f.read()
        
        if not markdown_content:
            raise HTTPException(status_code=400, detail="Markdown content not found")
        
        # Re-chunk to get text content
        from app.services.chunking_service import EnhancedChunkingService
        chunking_service = EnhancedChunkingService(
            target_chunk_tokens=file_metadata.target_chunk_tokens,
            chunk_overlap_tokens=file_metadata.chunk_overlap_tokens
        )
        
        base_metadata = {
            'user_id': user_id,
            'file_id': file_id,
            'filename': file_metadata.safe_filename,
            'knowledge_base_id': kb.knowledge_base_id,
            'knowledge_base_name': kb.knowledge_base_name
        }
        
        chunks = chunking_service.split_document_into_chunks(
            markdown_content,
            base_metadata=base_metadata,
            strategy=ChunkingStrategy(file_metadata.chunking_strategy.value)
        )
        
        # Index to vector store
        import time
        start_time = time.time()
        
        chunks_indexed = vector_store.add_document_chunks(
            chunks,
            file_id,
            file_metadata.safe_filename,
            knowledge_base_id=kb.knowledge_base_id,
            knowledge_base_name=kb.knowledge_base_name,
            user_id=user_id
        )
        
        indexing_time = time.time() - start_time
        
        db.commit()
        
        # Get vector store stats
        vector_stats = vector_store.get_collection_stats(
            user_id=user_id,
            knowledge_base_id=kb.knowledge_base_id
        )
        
        logger.info(f"🎉 Indexed {chunks_indexed} chunks in {indexing_time:.2f}s")
        
        return {
            "file_id": file_id,
            "knowledge_base_id": kb.knowledge_base_id,
            "chunks_indexed": chunks_indexed,
            "indexing_time_seconds": round(indexing_time, 2),
            "vector_store_stats": vector_stats,
            "status": "completed",
            "message": f"Indexed {chunks_indexed} chunks successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"❌ Indexing failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Indexing failed: {str(e)}")


def index_chunks_background(
    user_id: str,
    file_id: str,
    knowledge_base_id: str,
    chunk_ids: List[str],
    batch_size: int = 50,
    progress_callback=None,
    task_id=None
):
    """Background indexing for large documents"""
    import time
    
    logger.info(f"🚀 [Task {task_id}] Indexing {len(chunk_ids)} chunks")
    
    try:
        from app.core.database import get_db_context
        from app.db.models import FileMetadata, ChunkMetadata, KnowledgeBase
        from app.services.vector_store_service import VectorStoreService
        
        start_time = time.time()
        
        with get_db_context() as db:
            file_metadata = db.query(FileMetadata).filter(
                FileMetadata.file_id == file_id
            ).first()
            
            kb = db.query(KnowledgeBase).filter(
                KnowledgeBase.knowledge_base_id == knowledge_base_id
            ).first()
            
            if not file_metadata or not kb:
                raise ValueError("File or KB not found")
            
            # Read markdown
            with open(file_metadata.markdown_path, 'r', encoding='utf-8') as f:
                markdown_content = f.read()
            
            # Re-chunk
            from app.services.chunking_service import EnhancedChunkingService
            from app.models.schemas import ChunkingStrategy
            
            chunking_service = EnhancedChunkingService(
                target_chunk_tokens=file_metadata.target_chunk_tokens,
                chunk_overlap_tokens=file_metadata.chunk_overlap_tokens
            )
            
            base_metadata = {
                'user_id': user_id,
                'file_id': file_id,
                'filename': file_metadata.safe_filename,
                'knowledge_base_id': knowledge_base_id,
                'knowledge_base_name': kb.knowledge_base_name
            }
            
            chunks = chunking_service.split_document_into_chunks(
                markdown_content,
                base_metadata=base_metadata,
                strategy=ChunkingStrategy(file_metadata.chunking_strategy.value)
            )
            
            # Index
            vector_store = VectorStoreService()
            
            total_indexed = vector_store.add_document_chunks(
                chunks,
                file_id,
                file_metadata.safe_filename,
                knowledge_base_id=knowledge_base_id,
                knowledge_base_name=kb.knowledge_base_name,
                user_id=user_id
            )
            
            if progress_callback:
                progress_callback(100.0)
            
            total_time = time.time() - start_time
            
            logger.info(f"🎉 [Task {task_id}] Indexed {total_indexed} chunks in {total_time:.2f}s")
            
            return {
                'success': True,
                'file_id': file_id,
                'chunks_indexed': total_indexed,
                'indexing_time_seconds': total_time
            }
            
    except Exception as e:
        logger.error(f"❌ [Task {task_id}] Indexing FAILED: {str(e)}", exc_info=True)
        raise