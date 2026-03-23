"""
Upload Route - SIMPLIFIED VERSION
4 states only: uploaded, processing, completed, failed
Extraction → completed (no chunking/indexing in upload flow)
"""

from typing import Optional
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, Query, Form
from sqlalchemy.orm import Session
from pathlib import Path
import uuid
import shutil
from datetime import datetime
import logging

from app.models.schemas import FileUploadResponse, ProcessingStatus
from app.utils.file_validator import FileValidator
from app.core.config import settings
from app.core.dependencies import get_user_id, get_database_session
from app.core.exceptions import FileSizeError, FileTypeError
from app.db.models import (
    FileMetadata, ProcessingStatus as DBProcessingStatus,
    FileType, KnowledgeBase
)
from app.core.task_manager import get_task_manager, TaskType

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/upload", tags=["Upload"])



@router.post("/file-upload", response_model=FileUploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    knowledge_base_name: str = Form(...),
    user_id: str = Depends(get_user_id),
    db: Session = Depends(get_database_session)
):
    """
    Upload document file - SIMPLIFIED VERSION
    
    Status flow: uploaded → processing → completed
    After extraction completes, status = completed
    """
    
    # Validate inputs
    is_valid_name, name_error = FileValidator.validate_filename(file.filename)
    if not is_valid_name:
        raise HTTPException(status_code=400, detail=name_error)
    
    is_valid, error_message = FileValidator.validate_file(file)
    if not is_valid:
        raise HTTPException(status_code=400, detail=error_message)
    
    # Validate KB name
    if not knowledge_base_name or not knowledge_base_name.strip():
        raise HTTPException(status_code=400, detail="Knowledge base name cannot be empty")
    
    kb_name_stripped = knowledge_base_name.strip()
    if len(kb_name_stripped) > 255:
        raise HTTPException(status_code=400, detail="Knowledge base name too long (max 255 characters)")
    
    invalid_chars = ['/', '\\', '<', '>', ':', '"', '|', '?', '*']
    if any(char in kb_name_stripped for char in invalid_chars):
        raise HTTPException(
            status_code=400,
            detail=f"Knowledge base name contains invalid characters: {invalid_chars}"
        )
    
    # Generate file identifiers
    file_id = str(uuid.uuid4())
    file_extension = FileValidator.get_file_extension(file.filename)
    safe_filename = FileValidator.get_safe_filename(file.filename)
    
    # Create user upload directory
    user_upload_dir = Path(settings.upload_dir) / user_id
    user_upload_dir.mkdir(parents=True, exist_ok=True)
    
    unique_filename = f"{file_id}_{safe_filename}"
    upload_path = user_upload_dir / unique_filename
    
    try:
        # Save file to disk
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
        
        # Get or create knowledge base
        normalized_kb_name = kb_name_stripped.lower()
        
        kb = db.query(KnowledgeBase).filter(
            KnowledgeBase.user_id == user_id,
            KnowledgeBase.knowledge_base_name_normalized == normalized_kb_name
        ).first()
        
        if not kb:
            # Create new KB
            kb_id = str(uuid.uuid4())
            kb = KnowledgeBase(
                knowledge_base_id=kb_id,
                user_id=user_id,
                knowledge_base_name=kb_name_stripped,
                knowledge_base_name_normalized=normalized_kb_name,
                document_count=0,
                total_chunks=0,
                total_size_bytes=0,
                created_at=datetime.utcnow()
            )
            db.add(kb)
            db.flush()
            logger.info(f"✓ Created new KB: {kb.knowledge_base_id} - '{kb.knowledge_base_name}'")
        else:
            logger.info(f"✓ Using existing KB: {kb.knowledge_base_id} - '{kb.knowledge_base_name}'")
        
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
        
        # Create file metadata - simple status (no chunking params here)
        file_metadata = FileMetadata(
            file_id=file_id,
            user_id=user_id,
            knowledge_base_id=kb.knowledge_base_id,
            original_filename=file.filename,
            safe_filename=unique_filename,
            file_extension=file_extension,
            file_type=file_type_enum,
            file_size_bytes=file_size,
            storage_path=str(upload_path),
            status=DBProcessingStatus.PROCESSING,
            uploaded_at=datetime.utcnow()
        )
        
        db.add(file_metadata)
        
        # ✅ UPDATE KB STATS IMMEDIATELY
        kb.document_count += 1
        kb.total_size_bytes += file_size
        kb.updated_at = datetime.utcnow()
        
        db.commit()
        db.refresh(file_metadata)
        
        logger.info(
            f"✅ File uploaded: file_id={file_id}, kb_id={kb.knowledge_base_id}, "
            f"user_id={user_id}"
        )
        
        # Start extraction in background
        task_manager = get_task_manager()
        
        task_id = task_manager.submit_task(
            func=extract_document_background,
            task_type=TaskType.IO_BOUND,
            description=f"Extract: {file_metadata.safe_filename}",
            user_id=user_id,
            file_id=file_id,
            knowledge_base_id=kb.knowledge_base_id,
            storage_path=str(upload_path),
            safe_filename=unique_filename
        )
        
        logger.info(f"🚀 Extraction task started: task_id={task_id}")
        
        return FileUploadResponse(
            file_id=file_id,
            knowledge_base_id=kb.knowledge_base_id,
            knowledge_base_name=kb.knowledge_base_name,
            filename=safe_filename,
            file_type=file_extension,
            file_size_bytes=file_size,
            upload_time=datetime.now(),
            status=ProcessingStatus.PROCESSING,
            message=f"File uploaded. Extraction in progress. Track: /api/tasks/{task_id}"
        )
    
    except (FileSizeError, FileTypeError) as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        db.rollback()
        if upload_path.exists():
            upload_path.unlink()
        logger.error(f"❌ Upload error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Upload error: {str(e)}")


def extract_document_background(
    user_id: str,
    file_id: str,
    knowledge_base_id: str,
    storage_path: str,
    safe_filename: str,
    progress_callback=None,
    task_id=None
):
    """
    Extract document and (for most formats) mark as COMPLETED.

    CSV / TSV special path
    ----------------------
    For .csv and .tsv files this function bypasses the normal
    markdown → chunk → finalize pipeline entirely:
      1. Each row is converted to plain text.
      2. Embeddings are generated per row.
      3. Each row is stored as its own Qdrant point.
    The file is then marked COMPLETED with chunks_created set to the
    number of rows stored, so no further /chunks/create or
    /chunks/finalize calls are needed.

    All other formats follow the original flow:
      extract → save markdown → COMPLETED
    (chunking and indexing happen separately via /chunks/create and
    /chunks/finalize).
    """
    import time
    from pathlib import Path
    from datetime import datetime
    import json

    logger.info(f"🚀 [Task {task_id}] Starting extraction for {file_id}")
    start_time = time.time()

    try:
        from app.services.extraction.extractor_factory import ExtractorFactory
        from app.services.extraction.extractors.csv_extractor import CSVExtractor
        from app.utils.markdown_converter import MarkdownConverter
        from app.core.database import get_db_context
        from app.db.models import FileMetadata, ProcessingStatus as DBProcessingStatus

        if progress_callback:
            progress_callback(10.0)

        with get_db_context() as db:
            file_metadata = db.query(FileMetadata).filter(
                FileMetadata.file_id == file_id
            ).first()

            if not file_metadata:
                raise ValueError(f"File not found: {file_id}")

            # Resolve the KB name for Qdrant payload
            from app.db.models import KnowledgeBase
            kb = db.query(KnowledgeBase).filter(
                KnowledgeBase.knowledge_base_id == knowledge_base_id
            ).first()
            knowledge_base_name = kb.knowledge_base_name if kb else knowledge_base_id

            # Identify extractor
            file_extension = Path(storage_path).suffix.lstrip('.')
            extractor = ExtractorFactory.get_extractor(file_extension)

            if not extractor:
                format_status = ExtractorFactory.get_format_status(file_extension)
                if format_status["status"] == "unavailable":
                    raise ValueError(
                        f"Extractor for .{file_extension} is unavailable: "
                        f"{format_status['reason']}"
                    )
                raise ValueError(f"No extractor for .{file_extension}")

            logger.info(f"⚙️  [Task {task_id}] Running extraction ({extractor.__class__.__name__})...")

            if progress_callback:
                progress_callback(30.0)

            # ── CSV / TSV: row-level embedding + Qdrant storage ──────────
            if isinstance(extractor, CSVExtractor):
                logger.info(f"📊 [Task {task_id}] CSV/TSV detected — using row-level storage path")

                from app.services.vector_store_service import VectorStoreService
                vector_store = VectorStoreService()

                extraction_start = time.time()
                success, extraction_metadata = extractor.extract_and_store(
                    file_path=storage_path,
                    file_id=file_id,
                    filename=safe_filename,
                    knowledge_base_id=knowledge_base_id,
                    knowledge_base_name=knowledge_base_name,
                    user_id=user_id,
                    vector_store=vector_store,
                )
                extraction_time = time.time() - extraction_start

                if not success:
                    error_msg = extraction_metadata.get('error', 'CSV row storage failed')
                    raise Exception(f"CSV extraction failed: {error_msg}")

                rows_stored = extraction_metadata.get('total_stored', 0)
                row_count   = extraction_metadata.get('row_count', 0)

                logger.info(
                    f"✅ [Task {task_id}] CSV stored: {rows_stored}/{row_count} rows"
                )

                if progress_callback:
                    progress_callback(90.0)

                # Mark file as fully processed — no further steps needed
                processing_meta = {
                    'extraction_time_seconds': extraction_time,
                    'extractor_used': extractor.__class__.__name__,
                    'pipeline': 'csv_row_level_qdrant',
                    **extraction_metadata,
                }

                file_metadata.status          = DBProcessingStatus.COMPLETED
                file_metadata.processed_at    = datetime.utcnow()
                file_metadata.markdown_path   = None   # no markdown for CSV
                file_metadata.total_pages     = 1
                file_metadata.word_count      = row_count
                file_metadata.image_count     = 0
                file_metadata.table_count     = 0
                file_metadata.chunks_created  = rows_stored   # rows == chunks
                file_metadata.processing_metadata = json.dumps(processing_meta)

                # Keep KB chunk count in sync
                if kb:
                    kb.total_chunks  += rows_stored
                    kb.updated_at     = datetime.utcnow()

                db.commit()

                total_time = time.time() - start_time

                if progress_callback:
                    progress_callback(100.0)

                logger.info(
                    f"🎉 [Task {task_id}] CSV COMPLETED in {total_time:.2f}s — "
                    f"{rows_stored} rows indexed directly to Qdrant"
                )

                return {
                    'success': True,
                    'file_id': file_id,
                    'status': 'completed',
                    'pipeline': 'csv_row_level_qdrant',
                    'rows_stored': rows_stored,
                    'extraction_time_seconds': total_time,
                }

            # ── Standard path: extract → save markdown → COMPLETED ───────
            extraction_start = time.time()
            markdown_content, success, extraction_metadata = extractor.extract(storage_path)
            extraction_time = time.time() - extraction_start

            if not success:
                error_msg = extraction_metadata.get('error', 'Unknown extraction error')
                raise Exception(f"Extraction failed: {error_msg}")

            # Get stats
            total_pages = extraction_metadata.get('total_pages', 0)
            word_count  = extraction_metadata.get('word_count', 0)
            image_count = extraction_metadata.get('images_extracted', 0)
            table_count = extraction_metadata.get('tables_extracted', 0)

            logger.info(f"✅ [Task {task_id}] Extracted: {total_pages} pages, {word_count} words")

            if progress_callback:
                progress_callback(70.0)

            # Save markdown
            markdown_converter = MarkdownConverter()
            user_markdown_dir = Path(settings.markdown_dir) / user_id / knowledge_base_id
            user_markdown_dir.mkdir(parents=True, exist_ok=True)
            markdown_path = user_markdown_dir / f"{file_id}.md"

            markdown_saved = markdown_converter.save_markdown_file(
                markdown_content,
                str(markdown_path)
            )

            if progress_callback:
                progress_callback(90.0)

            # Update file metadata — COMPLETED after extraction only
            file_metadata.status       = DBProcessingStatus.COMPLETED
            file_metadata.processed_at = datetime.utcnow()
            file_metadata.markdown_path = str(markdown_path) if markdown_saved else None
            file_metadata.total_pages  = total_pages
            file_metadata.word_count   = word_count
            file_metadata.image_count  = image_count
            file_metadata.table_count  = table_count

            processing_meta = {
                'extraction_time_seconds': extraction_time,
                'extractor_used': extractor.__class__.__name__,
                **extraction_metadata
            }
            file_metadata.processing_metadata = json.dumps(processing_meta)

            db.commit()

            total_time = time.time() - start_time

            if progress_callback:
                progress_callback(100.0)

            logger.info(
                f"🎉 [Task {task_id}] COMPLETED in {total_time:.2f}s\n"
                f"   File: {safe_filename}\n"
                f"   Status: completed"
            )

            return {
                'success': True,
                'file_id': file_id,
                'status': 'completed',
                'total_pages': total_pages,
                'word_count': word_count,
                'extraction_time_seconds': total_time
            }

    except Exception as e:
        logger.error(f"❌ [Task {task_id}] FAILED: {str(e)}", exc_info=True)

        try:
            from app.core.database import get_db_context
            from app.db.models import ProcessingStatus as DBProcessingStatus
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
async def get_file_status(
    file_id: str,
    user_id: str = Depends(get_user_id),
    db: Session = Depends(get_database_session)
):
    """
    Get file processing status with detailed error information and retry guidance.
    
    Status values:
    - uploaded: File uploaded, extraction queued
    - processing: Extraction in progress
    - completed: Extraction completed successfully (ready for chunking)
    - failed: Extraction failed - use /api/chunks/retry-extraction to retry
    """
    file_metadata = db.query(FileMetadata).filter(
        FileMetadata.file_id == file_id,
        FileMetadata.user_id == user_id
    ).first()
    
    if not file_metadata:
        raise HTTPException(status_code=404, detail=f"File not found: {file_id}")
    
    kb = db.query(KnowledgeBase).filter(
        KnowledgeBase.knowledge_base_id == file_metadata.knowledge_base_id
    ).first()
    
    # Build response with guidance
    response = {
        "file_id": file_id,
        "filename": file_metadata.safe_filename,
        "knowledge_base_id": file_metadata.knowledge_base_id,
        "knowledge_base_name": kb.knowledge_base_name if kb else "Unknown",
        "status": file_metadata.status.value,
        "uploaded_at": file_metadata.uploaded_at.isoformat(),
        "processed_at": file_metadata.processed_at.isoformat() if file_metadata.processed_at else None,
        "total_pages": file_metadata.total_pages,
        "word_count": file_metadata.word_count,
        "chunks_created": file_metadata.chunks_created,
        "error_message": file_metadata.error_message
    }
    
    # Add next action guidance
    if file_metadata.status.value == "completed":
        response["next_action"] = "Chunking ready. POST to /api/chunks/create to create chunks."
    elif file_metadata.status.value == "processing":
        response["next_action"] = "Extraction in progress. Check back in a moment."
    elif file_metadata.status.value == "failed":
        response["next_action"] = f"Extraction failed. Use POST /api/chunks/retry-extraction to retry, or re-upload file."
        response["retry_endpoint"] = "/api/chunks/retry-extraction"
    elif file_metadata.status.value == "uploaded":
        response["next_action"] = "File uploaded. Extraction queued. Check status again shortly."
    
    return response
