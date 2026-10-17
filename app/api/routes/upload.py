"""
Upload Route - FIXED VERSION
Proper argument passing to task manager
"""

from typing import Optional
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, Query
from sqlalchemy.orm import Session
from pathlib import Path
import uuid
import shutil
from datetime import datetime
import logging
import json

from app.models.schemas import (
    FileUploadResponse, ProcessingStatus, ChunkingStrategy
)
from app.utils.file_validator import FileValidator
from app.core.config import settings
from app.core.dependencies import get_user_id, get_database_session
from app.db.models import (
    FileMetadata, ProcessingStatus as DBProcessingStatus, FileType, ChunkingStrategyEnum
)
from app.core.task_manager import get_task_manager, TaskType, ProcessingStage

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/upload", tags=["Upload"])


@router.post("/", response_model=FileUploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    user_id: str = Depends(get_user_id),
    db: Session = Depends(get_database_session)
):
    """Upload document file (returns immediately)"""
    
    # Validate filename
    is_valid_name, name_error = FileValidator.validate_filename(file.filename)
    if not is_valid_name:
        raise HTTPException(status_code=400, detail=name_error)
    
    # Validate file type
    is_valid, error_message = FileValidator.validate_file(file)
    if not is_valid:
        raise HTTPException(status_code=400, detail=error_message)
    
    # Generate unique file ID
    file_id = str(uuid.uuid4())
    file_extension = FileValidator.get_file_extension(file.filename)
    safe_filename = FileValidator.get_safe_filename(file.filename)
    
    # Create user-specific upload directory
    user_upload_dir = Path(settings.upload_dir) / user_id
    user_upload_dir.mkdir(parents=True, exist_ok=True)
    
    unique_filename = f"{file_id}_{safe_filename}"
    upload_path = user_upload_dir / unique_filename
    
    try:
        # Save file
        with upload_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"📁 File saved: {upload_path} (user: {user_id})")
        
        # Validate file size
        file_size = upload_path.stat().st_size
        is_size_valid, size_error = FileValidator.validate_file_size(
            file_size,
            settings.max_file_size_mb
        )
        
        if not is_size_valid:
            upload_path.unlink()
            raise HTTPException(status_code=400, detail=size_error)
        
        # Get file category
        file_category = FileValidator.get_file_category(safe_filename)
        
        file_type_map = {
            'document': FileType.DOCUMENT,
            'presentation': FileType.PRESENTATION,
            'image': FileType.IMAGE,
            'text': FileType.TEXT,
            'unknown': FileType.OTHER
        }
        file_type_enum = file_type_map.get(file_category, FileType.OTHER)
        
        # Save to database
        file_metadata = FileMetadata(
            file_id=file_id,
            user_id=user_id,
            original_filename=file.filename,
            safe_filename=unique_filename,
            file_extension=file_extension,
            file_type=file_type_enum,
            file_size_bytes=file_size,
            storage_path=str(upload_path),
            status=DBProcessingStatus.UPLOADED,
            uploaded_at=datetime.utcnow()
        )
        
        db.add(file_metadata)
        db.commit()
        db.refresh(file_metadata)
        
        logger.info(f"✅ Database record created: file_id={file_id}, user_id={user_id}")
        
        return FileUploadResponse(
            file_id=file_id,
            filename=safe_filename,
            file_type=file_extension,
            file_size_bytes=file_size,
            upload_time=datetime.now(),
            status=ProcessingStatus.UPLOADED,
            message=f"File '{safe_filename}' uploaded. Use /process/{file_id} to process."
        )
    
    except HTTPException:
        db.rollback()
        raise
    except Exception as e:
        db.rollback()
        if upload_path.exists():
            upload_path.unlink()
        logger.error(f"❌ Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload error: {str(e)}")


@router.post("/process/{file_id}")
async def process_and_index_document(
    file_id: str,
    user_id: str = Depends(get_user_id),
    db: Session = Depends(get_database_session),
    chunking_strategy: ChunkingStrategy = Query(default=ChunkingStrategy.DEFAULT),
    target_chunk_tokens: Optional[int] = Query(default=None, ge=100, le=2048),
    chunk_overlap_tokens: Optional[int] = Query(default=None, ge=0, le=500)
):
    """
    Process document in background (returns immediately with task_id)
    
    FIXED: Proper argument passing - no duplication, all args in kwargs
    """
    
    # Get file metadata
    file_metadata = db.query(FileMetadata).filter(
        FileMetadata.file_id == file_id,
        FileMetadata.user_id == user_id
    ).first()
    
    if not file_metadata:
        raise HTTPException(status_code=404, detail=f"File not found: {file_id}")
    
    if file_metadata.status == DBProcessingStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"File already processed. Check /knowledge-base/document/{file_id}"
        )
    
    if file_metadata.status == DBProcessingStatus.PROCESSING:
        raise HTTPException(
            status_code=400,
            detail=f"File is already being processed. Check /tasks/file/{file_id}"
        )
    
    # Update status to PROCESSING
    file_metadata.status = DBProcessingStatus.PROCESSING
    db.commit()
    
    logger.info(
        f"🔄 Starting document processing: file_id={file_id}, "
        f"user_id={user_id}, strategy={chunking_strategy.value}"
    )
    
    # Get task manager
    task_manager = get_task_manager()
    
    # CORRECT: Pass all function arguments as keyword arguments
    # The task manager will inject progress_callback and task_id
    task_id = task_manager.submit_task(
        func=process_document_with_progress,
        task_type=TaskType.IO_BOUND,
        description=f"Process: {file_metadata.safe_filename}",
        user_id=user_id,
        file_id=file_id,
        # All function parameters as kwargs (NOT positional args!)
        # file_id=file_id,
        # user_id=user_id,
        chunking_strategy=chunking_strategy.value,
        target_chunk_tokens=target_chunk_tokens,
        chunk_overlap_tokens=chunk_overlap_tokens,
        storage_path=file_metadata.storage_path,
        safe_filename=file_metadata.safe_filename
    )
    
    logger.info(f"✅ Task submitted: task_id={task_id}, file_id={file_id}")
    
    return {
        "task_id": task_id,
        "file_id": file_id,
        "status": "processing",
        "message": "Document processing started",
        "status_endpoint": f"/api/tasks/{task_id}",
        "progress_endpoint": f"/api/tasks/{task_id}",
        "estimated_time": "Check progress for updates"
    }


def process_document_with_progress(
    file_id: str,
    user_id: str,
    chunking_strategy: str,
    target_chunk_tokens: Optional[int],
    chunk_overlap_tokens: Optional[int],
    storage_path: str,
    safe_filename: str,
    progress_callback=None,  # Injected by task manager
    task_id=None  # Injected by task manager
):
    """
    Process document with detailed progress tracking
    
    All parameters are passed as keyword arguments from submit_task
    """
    import time
    from pathlib import Path
    from datetime import datetime
    
    logger.info(f"🚀 [Task {task_id}] Starting document processing for {file_id}")
    
    start_time = time.time()
    
    try:
        # Import here to avoid serialization issues
        from app.services.extraction.extractor_factory import ExtractorFactory
        from app.utils.markdown_converter import MarkdownConverter
        from app.services.chunking_service import EnhancedChunkingService
        from app.services.vector_store_service import VectorStoreService
        from app.core.database import get_db_context
        from app.db.models import FileMetadata, ChunkMetadata, ChunkingStrategyEnum
        from app.models.schemas import ChunkingStrategy as SchemaChunkingStrategy
        
        # Update progress: Starting
        if progress_callback:
            progress_callback(
                progress=5.0,
                stage=ProcessingStage.EXTRACTING,
                details=f"Initializing extraction for {safe_filename}"
            )
        
        logger.info(f"📊 [Task {task_id}] Stage 1/5: Document Extraction")
        
        # Get database session
        with get_db_context() as db:
            file_metadata = db.query(FileMetadata).filter(
                FileMetadata.file_id == file_id
            ).first()
            
            if not file_metadata:
                raise ValueError(f"File not found: {file_id}")
            
            # STAGE 1: EXTRACTION
            file_extension = Path(storage_path).suffix.lstrip('.')
            
            logger.info(f"🔍 [Task {task_id}] Getting extractor for .{file_extension}")
            
            extractor = ExtractorFactory.get_extractor(file_extension)
            if not extractor:
                raise ValueError(f"No extractor for .{file_extension}")
            
            if progress_callback:
                progress_callback(
                    progress=10.0,
                    stage=ProcessingStage.EXTRACTING,
                    details=f"Extracting content from {file_extension.upper()} file"
                )
            
            logger.info(f"⚙️  [Task {task_id}] Running extraction...")
            extraction_start = time.time()
            
            # Extract document
            markdown_content, success, extraction_metadata = extractor.extract(storage_path)
            
            extraction_time = time.time() - extraction_start
            
            if not success:
                error_msg = extraction_metadata.get('error', 'Unknown extraction error')
                logger.error(f"❌ [Task {task_id}] Extraction failed: {error_msg}")
                raise Exception(f"Extraction failed: {error_msg}")
            
            total_pages = extraction_metadata.get('total_pages', 0)
            word_count = extraction_metadata.get('word_count', 0)
            image_count = extraction_metadata.get('images_extracted', 0)
            table_count = extraction_metadata.get('tables_extracted', 0)
            chart_count = extraction_metadata.get('charts_extracted', 0)
            
            logger.info(
                f"✅ [Task {task_id}] Extraction complete in {extraction_time:.2f}s: "
                f"{total_pages} pages, {word_count} words, {image_count} images, "
                f"{table_count} tables"
            )
            
            if progress_callback:
                progress_callback(
                    progress=30.0,
                    stage=ProcessingStage.EXTRACTING,
                    details=f"Extracted {total_pages} pages, {word_count} words",
                    total_pages=total_pages,
                    pages_processed=total_pages
                )
            
            # STAGE 2: SAVE MARKDOWN
            logger.info(f"📊 [Task {task_id}] Stage 2/5: Saving Markdown")
            
            markdown_converter = MarkdownConverter()
            user_markdown_dir = Path(settings.markdown_dir) / user_id
            user_markdown_dir.mkdir(parents=True, exist_ok=True)
            markdown_path = user_markdown_dir / f"{file_id}.md"
            
            markdown_saved = markdown_converter.save_markdown_file(
                markdown_content,
                str(markdown_path)
            )
            
            logger.info(f"✅ [Task {task_id}] Markdown saved: {markdown_path}")
            
            if progress_callback:
                progress_callback(
                    progress=40.0,
                    stage=ProcessingStage.CHUNKING,
                    details="Starting document chunking"
                )
            
            # STAGE 3: CHUNKING
            logger.info(f"📊 [Task {task_id}] Stage 3/5: Document Chunking")
            
            chunk_tokens = target_chunk_tokens or settings.target_chunk_tokens
            overlap_tokens = chunk_overlap_tokens or settings.chunk_overlap_tokens
            
            # Create chunking service with the specified token settings
            chunking_service = EnhancedChunkingService(
                target_chunk_tokens=chunk_tokens,
                chunk_overlap_tokens=overlap_tokens
            )
            
            strategy_enum = SchemaChunkingStrategy(chunking_strategy)
            
            logger.info(
                f"🔪 [Task {task_id}] Chunking with strategy={chunking_strategy}, "
                f"target_tokens={chunk_tokens}, overlap={overlap_tokens}"
            )
            
            chunking_start = time.time()
            
            # Create base metadata for all chunks (includes user_id for isolation)
            base_metadata = {
                'user_id': user_id,
                'file_id': file_id,
                'filename': safe_filename
            }
            
            chunks = chunking_service.split_document_into_chunks(
                markdown_content,
                base_metadata=base_metadata,
                strategy=strategy_enum
            )
            
            chunking_time = time.time() - chunking_start
            
            # Calculate chunk statistics manually
            if chunks:
                token_counts = [chunk.get('token_count', 0) for chunk in chunks]
                chunk_stats = {
                    'total_chunks': len(chunks),
                    'avg_tokens': sum(token_counts) / len(token_counts) if token_counts else 0,
                    'min_tokens': min(token_counts) if token_counts else 0,
                    'max_tokens': max(token_counts) if token_counts else 0
                }
            else:
                chunk_stats = {
                    'total_chunks': 0,
                    'avg_tokens': 0,
                    'min_tokens': 0,
                    'max_tokens': 0
                }
            
            logger.info(
                f"✅ [Task {task_id}] Chunking complete in {chunking_time:.2f}s: "
                f"{len(chunks)} chunks created, "
                f"avg={chunk_stats['avg_tokens']:.0f} tokens"
            )
            
            if progress_callback:
                progress_callback(
                    progress=60.0,
                    stage=ProcessingStage.EMBEDDING,
                    details=f"Created {len(chunks)} chunks, now embedding",
                    total_chunks=len(chunks),
                    chunks_created=len(chunks)
                )
            
            # STAGE 4: EMBEDDING & INDEXING
            logger.info(f"📊 [Task {task_id}] Stage 4/5: Embedding & Vector Indexing")
            
            simple_structure = {
                'filename': safe_filename,
                'file_id': file_id,
                'chunks': len(chunks)
            }
            
            vector_store = VectorStoreService()
            
            logger.info(f"🔢 [Task {task_id}] Generating embeddings for {len(chunks)} chunks...")
            
            if progress_callback:
                progress_callback(
                    progress=70.0,
                    stage=ProcessingStage.INDEXING,
                    details=f"Indexing {len(chunks)} chunks to vector store"
                )
            
            indexing_start = time.time()
            
            # Index to vector store
            chunks_added = vector_store.add_document_chunks(
                chunks,
                file_id,
                safe_filename,
                structured_data=simple_structure
            )
            
            indexing_time = time.time() - indexing_start
            
            logger.info(
                f"✅ [Task {task_id}] Indexing complete in {indexing_time:.2f}s: "
                f"{chunks_added} chunks indexed"
            )
            
            if progress_callback:
                progress_callback(
                    progress=85.0,
                    stage=ProcessingStage.FINALIZING,
                    details="Saving chunk metadata to database"
                )
            
            # STAGE 5: SAVE TO DATABASE
            logger.info(f"📊 [Task {task_id}] Stage 5/5: Saving Metadata to Database")
            
            db_start = time.time()
            
            # Save chunk metadata
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
            
            # Map chunking strategy
            strategy_map = {
                'default': ChunkingStrategyEnum.DEFAULT,
                'fixed_size': ChunkingStrategyEnum.FIXED_SIZE,
                'hierarchical': ChunkingStrategyEnum.HIERARCHICAL,
                'semantic': ChunkingStrategyEnum.SEMANTIC
            }
            
            # Update file metadata
            file_metadata.status = DBProcessingStatus.COMPLETED
            file_metadata.processed_at = datetime.utcnow()
            file_metadata.chunks_created = chunks_added
            file_metadata.total_pages = total_pages
            file_metadata.markdown_path = str(markdown_path) if markdown_saved else None
            file_metadata.word_count = word_count
            file_metadata.image_count = image_count
            file_metadata.table_count = table_count
            file_metadata.chunking_strategy = strategy_map.get(chunking_strategy)
            file_metadata.target_chunk_tokens = chunk_tokens
            file_metadata.chunk_overlap_tokens = overlap_tokens
            
            # Store processing metadata
            processing_meta = {
                'extraction_method': extractor.__class__.__name__,
                'extraction_time_seconds': extraction_time,
                'chunking_time_seconds': chunking_time,
                'indexing_time_seconds': indexing_time,
                'chunks_created': chunks_added,
                'chunking_strategy': chunking_strategy,
                'target_chunk_tokens': chunk_tokens,
                'chunk_overlap_tokens': overlap_tokens,
                'chunk_stats': chunk_stats,
                'extraction_metadata': extraction_metadata
            }
            file_metadata.processing_metadata = json.dumps(processing_meta)
            
            db.commit()
            
            db_time = time.time() - db_start
            
            logger.info(f"✅ [Task {task_id}] Database update complete in {db_time:.2f}s")
            
            total_time = time.time() - start_time
            
            if progress_callback:
                progress_callback(
                    progress=100.0,
                    stage=ProcessingStage.COMPLETE,
                    details=f"Processing complete in {total_time:.1f}s"
                )
            
            logger.info(
                f"🎉 [Task {task_id}] PROCESSING COMPLETE!\n"
                f"   Total time: {total_time:.2f}s\n"
                f"   - Extraction: {extraction_time:.2f}s\n"
                f"   - Chunking: {chunking_time:.2f}s\n"
                f"   - Indexing: {indexing_time:.2f}s\n"
                f"   - Database: {db_time:.2f}s\n"
                f"   Chunks: {chunks_added}, Pages: {total_pages}"
            )
            
            return {
                'success': True,
                'file_id': file_id,
                'filename': safe_filename,
                'chunks_created': chunks_added,
                'total_pages': total_pages,
                'processing_time_seconds': total_time,
                'stages': {
                    'extraction': extraction_time,
                    'chunking': chunking_time,
                    'indexing': indexing_time,
                    'database': db_time
                },
                'statistics': {
                    'images': image_count,
                    'tables': table_count,
                    'charts': chart_count,
                    'words': word_count
                },
                'chunking_strategy': chunking_strategy
            }
            
    except Exception as e:
        # Update database status to failed
        logger.error(f"❌ [Task {task_id}] PROCESSING FAILED: {str(e)}", exc_info=True)
        
        try:
            with get_db_context() as db:
                file_metadata = db.query(FileMetadata).filter(
                    FileMetadata.file_id == file_id
                ).first()
                
                if file_metadata:
                    file_metadata.status = DBProcessingStatus.FAILED
                    file_metadata.error_message = str(e)
                    db.commit()
                    logger.info(f"📝 [Task {task_id}] Updated database with error status")
        except Exception as db_error:
            logger.error(f"❌ [Task {task_id}] Failed to update error status: {db_error}")
        
        raise


@router.get("/status/{file_id}")
async def get_processing_status(
    file_id: str,
    user_id: str = Depends(get_user_id),
    db: Session = Depends(get_database_session)
):
    """Get processing status for a file"""
    file_metadata = db.query(FileMetadata).filter(
        FileMetadata.file_id == file_id,
        FileMetadata.user_id == user_id
    ).first()
    
    if not file_metadata:
        raise HTTPException(status_code=404, detail=f"File not found: {file_id}")
    
    return {
        "file_id": file_id,
        "filename": file_metadata.safe_filename,
        "status": file_metadata.status.value,
        "uploaded_at": file_metadata.uploaded_at,
        "processed_at": file_metadata.processed_at,
        "chunks_created": file_metadata.chunks_created,
        "total_pages": file_metadata.total_pages,
        "chunking_strategy": file_metadata.chunking_strategy.value if file_metadata.chunking_strategy else None,
        "error_message": file_metadata.error_message
    }


@router.get("/chunking-strategies")
async def get_available_chunking_strategies():
    """Get list of available chunking strategies"""
    return {
        "strategies": [
            {
                "name": "fixed_size",
                "display_name": "Fixed-size (Token-based)",
                "description": "Creates chunks of exact token size",
                "best_for": "General purpose",
                "parameters": ["target_chunk_tokens"]
            },
            {
                "name": "sentence",
                "display_name": "Sentence Chunking",
                "description": "Groups sentences together",
                "best_for": "Sentence context important",
                "parameters": ["target_chunk_tokens"]
            },
            {
                "name": "paragraph",
                "display_name": "Paragraph Chunking",
                "description": "Groups paragraphs together",
                "best_for": "Clear paragraph structure",
                "parameters": ["target_chunk_tokens"]
            },
            {
                "name": "sliding_window",
                "display_name": "Sliding Window",
                "description": "Creates overlapping chunks",
                "best_for": "Context across chunks critical",
                "parameters": ["target_chunk_tokens", "chunk_overlap_tokens"]
            },
            {
                "name": "recursive",
                "display_name": "Recursive Chunking",
                "description": "Hierarchical text splitting",
                "best_for": "Structured documents",
                "parameters": ["target_chunk_tokens"]
            },
            {
                "name": "semantic",
                "display_name": "Semantic Chunking",
                "description": "Groups semantically similar sentences",
                "best_for": "Topic coherence important",
                "parameters": ["target_chunk_tokens"]
            }
        ],
        "default_strategy": "fixed_size",
        "default_parameters": {
            "target_chunk_tokens": 512,
            "chunk_overlap_tokens": 50
        }
    }