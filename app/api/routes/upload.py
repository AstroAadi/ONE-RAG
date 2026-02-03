"""
Upload Route - FIXED VERSION
Fixes:
- Corrected task submission (no argument duplication)
- Simplified progress tracking
- Better error handling
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
from app.core.exceptions import (
    FileUploadError, FileSizeError, FileTypeError,
    DocumentProcessingError, ValidationError
)
from app.db.models import (
    FileMetadata, ProcessingStatus as DBProcessingStatus,
    FileType, ChunkingStrategyEnum
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
            raise FileSizeError(size_error)
        
        # Get file category
        file_category = FileValidator.get_file_category(safe_filename)
        
        file_type_map = {
            'document': FileType.DOCUMENT,
            'presentation': FileType.PRESENTATION,
            'image': FileType.IMAGE,
            'text': FileType.TEXT,
            'spreadsheet': FileType.OTHER,
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
    
    except (FileSizeError, FileTypeError) as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(e))
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
    chunking_strategy: ChunkingStrategy = Query(default=ChunkingStrategy.FIXED_SIZE),
    target_chunk_tokens: Optional[int] = Query(default=None, ge=100, le=2048),
    chunk_overlap_tokens: Optional[int] = Query(default=None, ge=0, le=500)
):
    """
    Process document in background
    
    FIXED: Proper argument passing - all args in kwargs, no duplication
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
    
    # FIXED: All arguments in kwargs, no duplication!
    task_id = task_manager.submit_task(
        func=process_document_with_progress,
        task_type=TaskType.IO_BOUND,
        description=f"Process: {file_metadata.safe_filename}",
        # ALL function arguments as kwargs:
        user_id=user_id,
        file_id=file_id,
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
        "progress_endpoint": f"/api/tasks/{task_id}"
    }


def process_document_with_progress(
    user_id: str,
    file_id: str,
    chunking_strategy: str,
    target_chunk_tokens: Optional[int],
    chunk_overlap_tokens: Optional[int],
    storage_path: str,
    safe_filename: str,
    progress_callback=None,  # Injected by task manager
    task_id=None  # Injected by task manager
):
    """
    Process document with simplified progress tracking
    
    All parameters passed as keyword arguments from submit_task
    progress_callback and task_id are injected by task manager
    """
    import time
    from pathlib import Path
    from datetime import datetime
    
    logger.info(f"🚀 [Task {task_id}] Starting processing for {file_id}")
    
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
        
        # SIMPLIFIED: Update progress at major milestones only
        if progress_callback:
            progress_callback(10.0, ProcessingStage.PROCESSING)
        
        logger.info(f"📊 [Task {task_id}] Stage: Document Extraction")
        
        # Get database session
        with get_db_context() as db:
            file_metadata = db.query(FileMetadata).filter(
                FileMetadata.file_id == file_id
            ).first()
            
            if not file_metadata:
                raise ValueError(f"File not found: {file_id}")
            
            # EXTRACTION
            file_extension = Path(storage_path).suffix.lstrip('.')
            extractor = ExtractorFactory.get_extractor(file_extension)
            
            if not extractor:
                raise ValueError(f"No extractor for .{file_extension}")
            
            logger.info(f"⚙️  [Task {task_id}] Running extraction...")
            extraction_start = time.time()
            
            markdown_content, success, extraction_metadata = extractor.extract(storage_path)
            
            extraction_time = time.time() - extraction_start
            
            if not success:
                error_msg = extraction_metadata.get('error', 'Unknown extraction error')
                raise Exception(f"Extraction failed: {error_msg}")
            
            total_pages = extraction_metadata.get('total_pages', 0)
            word_count = extraction_metadata.get('word_count', 0)
            image_count = extraction_metadata.get('images_extracted', 0)
            table_count = extraction_metadata.get('tables_extracted', 0)
            
            logger.info(f"✅ [Task {task_id}] Extraction: {total_pages} pages, {word_count} words")
            
            if progress_callback:
                progress_callback(30.0, ProcessingStage.PROCESSING)
            
            # SAVE MARKDOWN
            markdown_converter = MarkdownConverter()
            user_markdown_dir = Path(settings.markdown_dir) / user_id
            user_markdown_dir.mkdir(parents=True, exist_ok=True)
            markdown_path = user_markdown_dir / f"{file_id}.md"
            
            markdown_saved = markdown_converter.save_markdown_file(
                markdown_content,
                str(markdown_path)
            )
            
            if progress_callback:
                progress_callback(50.0, ProcessingStage.PROCESSING)
            
            # CHUNKING
            logger.info(f"📊 [Task {task_id}] Stage: Chunking")
            
            chunk_tokens = target_chunk_tokens or settings.target_chunk_tokens
            overlap_tokens = chunk_overlap_tokens or settings.chunk_overlap_tokens
            
            chunking_service = EnhancedChunkingService(
                target_chunk_tokens=chunk_tokens,
                chunk_overlap_tokens=overlap_tokens
            )
            
            strategy_enum = SchemaChunkingStrategy(chunking_strategy)
            
            chunking_start = time.time()
            
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
            
            logger.info(f"✅ [Task {task_id}] Created {len(chunks)} chunks")
            
            if progress_callback:
                progress_callback(70.0, ProcessingStage.PROCESSING)
            
            # EMBEDDING & INDEXING
            logger.info(f"📊 [Task {task_id}] Stage: Embedding & Indexing")
            
            vector_store = VectorStoreService()
            
            indexing_start = time.time()
            
            chunks_added = vector_store.add_document_chunks(
                chunks,
                file_id,
                safe_filename,
                structured_data={'filename': safe_filename, 'file_id': file_id}
            )
            
            indexing_time = time.time() - indexing_start
            
            logger.info(f"✅ [Task {task_id}] Indexed {chunks_added} chunks")
            
            if progress_callback:
                progress_callback(90.0, ProcessingStage.PROCESSING)
            
            # SAVE TO DATABASE
            db_start = time.time()
            
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
            
            # Map strategy
            strategy_map = {
                'fixed_size': ChunkingStrategyEnum.FIXED_SIZE,
                'sentence': ChunkingStrategyEnum.SENTENCE,
                'paragraph': ChunkingStrategyEnum.PARAGRAPH,
                'sliding_window': ChunkingStrategyEnum.SLIDING_WINDOW,
                'recursive': ChunkingStrategyEnum.RECURSIVE,
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
                'extraction_time_seconds': extraction_time,
                'chunking_time_seconds': chunking_time,
                'indexing_time_seconds': indexing_time,
                'chunks_created': chunks_added,
                'chunking_strategy': chunking_strategy
            }
            file_metadata.processing_metadata = json.dumps(processing_meta)
            
            db.commit()
            
            total_time = time.time() - start_time
            
            if progress_callback:
                progress_callback(100.0, ProcessingStage.COMPLETE)
            
            logger.info(
                f"🎉 [Task {task_id}] COMPLETE in {total_time:.2f}s\n"
                f"   Chunks: {chunks_added}, Pages: {total_pages}"
            )
            
            return {
                'success': True,
                'file_id': file_id,
                'filename': safe_filename,
                'chunks_created': chunks_added,
                'total_pages': total_pages,
                'processing_time_seconds': total_time
            }
            
    except Exception as e:
        logger.error(f"❌ [Task {task_id}] FAILED: {str(e)}", exc_info=True)
        
        # Update database
        try:
            with get_db_context() as db:
                file_metadata = db.query(FileMetadata).filter(
                    FileMetadata.file_id == file_id
                ).first()
                
                if file_metadata:
                    file_metadata.status = DBProcessingStatus.FAILED
                    file_metadata.error_message = str(e)
                    db.commit()
        except Exception as db_error:
            logger.error(f"❌ Failed to update error status: {db_error}")
        
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
                "best_for": "General purpose"
            },
            {
                "name": "sentence",
                "display_name": "Sentence Chunking",
                "description": "Groups sentences together",
                "best_for": "Sentence context important"
            },
            {
                "name": "paragraph",
                "display_name": "Paragraph Chunking",
                "description": "Groups paragraphs together",
                "best_for": "Clear paragraph structure"
            },
            {
                "name": "sliding_window",
                "display_name": "Sliding Window",
                "description": "Creates overlapping chunks",
                "best_for": "Context across chunks critical"
            },
            {
                "name": "recursive",
                "display_name": "Recursive Chunking",
                "description": "Hierarchical text splitting",
                "best_for": "Structured documents"
            },
            {
                "name": "semantic",
                "display_name": "Semantic Chunking",
                "description": "Groups semantically similar sentences",
                "best_for": "Topic coherence important"
            }
        ],
        "default_strategy": "fixed_size",
        "default_parameters": {
            "target_chunk_tokens": 512,
            "chunk_overlap_tokens": 50
        }
    }