"""
Upload Route

Phase 1 of ingestion: accept a file into a knowledge base and extract it to
markdown in the background.

    POST /api/upload/file-upload      upload into a knowledge base
    GET  /api/upload/status/{file_id} extraction progress
    GET  /api/upload/chunking-strategies

Extraction is deliberately separated from chunking and indexing (see
app/api/routes/chunking.py) so the user can review chunk boundaries before
embedding anything.
"""

from typing import Optional
from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException
from sqlalchemy.orm import Session
from pathlib import Path
import uuid
import shutil
from datetime import datetime
import logging
import json

from app.models.schemas import (
    FileUploadResponse, ProcessingStatus, ProcessingStatusResponse
)
from app.utils.file_validator import FileValidator
from app.core.config import settings
from app.core.dependencies import get_user_id, get_database_session
from app.db.models import (
    FileMetadata, ProcessingStatus as DBProcessingStatus, FileType
)
from app.core.task_manager import get_task_manager, TaskType, ProcessingStage
from app.services.knowledge_base_service import resolve_or_create_knowledge_base

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/upload", tags=["Upload"])


# Maps the extraction stage to a coarse progress value for clients that poll
# the file status endpoint rather than the task endpoint.
_STAGE_PROGRESS_FLOOR = {
    ProcessingStage.QUEUED.value: 10.0,
    ProcessingStage.EXTRACTING.value: 30.0,
    ProcessingStage.FINALIZING.value: 90.0,
    ProcessingStage.COMPLETE.value: 100.0,
}


def _latest_task_for_file(user_id: str, file_id: str) -> Optional[dict]:
    """Most recent background task recorded for a file, if any"""
    task_manager = get_task_manager()

    tasks = [
        task for task in task_manager.get_user_tasks(user_id, limit=200)
        if task.get("file_id") == file_id
    ]

    return tasks[0] if tasks else None


async def _store_upload(
    file: UploadFile,
    user_id: str,
    db: Session,
    knowledge_base_name: Optional[str],
    knowledge_base_id: Optional[str],
) -> FileUploadResponse:
    """
    Validate and persist an uploaded file, then start background extraction.

    Shared by the upload endpoints so both expose identical behaviour.
    """
    # Validate filename
    is_valid_name, name_error = FileValidator.validate_filename(file.filename)
    if not is_valid_name:
        raise HTTPException(status_code=400, detail=name_error)

    # Validate file type
    is_valid, error_message = FileValidator.validate_file(file)
    if not is_valid:
        raise HTTPException(status_code=400, detail=error_message)

    # Resolve the owning knowledge base before writing anything to disk
    try:
        knowledge_base, _ = resolve_or_create_knowledge_base(
            db,
            user_id=user_id,
            knowledge_base_id=knowledge_base_id,
            knowledge_base_name=knowledge_base_name,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    file_id = str(uuid.uuid4())
    file_extension = FileValidator.get_file_extension(file.filename)
    safe_filename = FileValidator.get_safe_filename(file.filename)

    user_upload_dir = Path(settings.upload_dir) / user_id
    user_upload_dir.mkdir(parents=True, exist_ok=True)

    unique_filename = f"{file_id}_{safe_filename}"
    upload_path = user_upload_dir / unique_filename

    try:
        with upload_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        logger.info(f"📁 File saved: {upload_path} (user: {user_id})")

        file_size = upload_path.stat().st_size
        is_size_valid, size_error = FileValidator.validate_file_size(
            file_size,
            settings.max_file_size_mb
        )

        if not is_size_valid:
            upload_path.unlink()
            raise HTTPException(status_code=400, detail=size_error)

        file_category = FileValidator.get_file_category(safe_filename)

        file_type_map = {
            'document': FileType.DOCUMENT,
            'presentation': FileType.PRESENTATION,
            'image': FileType.IMAGE,
            'text': FileType.TEXT,
            'unknown': FileType.OTHER
        }
        file_type_enum = file_type_map.get(file_category, FileType.OTHER)

        file_metadata = FileMetadata(
            file_id=file_id,
            user_id=user_id,
            knowledge_base_id=knowledge_base.knowledge_base_id,
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
        db.commit()
        db.refresh(file_metadata)

        logger.info(
            f"✅ Database record created: file_id={file_id}, "
            f"user_id={user_id}, kb={knowledge_base.knowledge_base_id}"
        )

    except HTTPException:
        db.rollback()
        if upload_path.exists():
            upload_path.unlink(missing_ok=True)
        raise
    except Exception as e:
        db.rollback()
        if upload_path.exists():
            upload_path.unlink(missing_ok=True)
        logger.error(f"❌ Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload error: {str(e)}")

    # Start extraction in the background; the request returns immediately.
    task_manager = get_task_manager()
    task_id = task_manager.submit_task(
        func=extract_document_with_progress,
        task_type=TaskType.IO_BOUND,
        description=f"Extract: {unique_filename}",
        user_id=user_id,
        file_id=file_id,
        storage_path=str(upload_path),
        safe_filename=unique_filename,
    )

    logger.info(f"✅ Extraction task submitted: task_id={task_id}, file_id={file_id}")

    return FileUploadResponse(
        file_id=file_id,
        knowledge_base_id=knowledge_base.knowledge_base_id,
        knowledge_base_name=knowledge_base.knowledge_base_name,
        filename=unique_filename,
        file_type=file_extension,
        file_size_bytes=file_size,
        upload_time=datetime.utcnow(),
        status=ProcessingStatus.PROCESSING,
        message=(
            f"'{safe_filename}' uploaded to '{knowledge_base.knowledge_base_name}'. "
            f"Extraction started."
        ),
        task_id=task_id,
    )


@router.post("/file-upload", response_model=FileUploadResponse)
async def upload_file_to_knowledge_base(
    file: UploadFile = File(...),
    knowledge_base_name: Optional[str] = Form(default=None),
    knowledge_base_id: Optional[str] = Form(default=None),
    user_id: str = Depends(get_user_id),
    db: Session = Depends(get_database_session)
):
    """
    Upload a document into a knowledge base and start extraction.

    The target knowledge base is resolved from `knowledge_base_id` when given,
    otherwise from `knowledge_base_name` (created on first use), otherwise the
    user's "Default" knowledge base.

    Returns immediately. Poll `/api/upload/status/{file_id}` until the status
    is `completed`, then call `/api/chunks/create`.
    """
    return await _store_upload(
        file=file,
        user_id=user_id,
        db=db,
        knowledge_base_name=knowledge_base_name,
        knowledge_base_id=knowledge_base_id,
    )


@router.post("/", response_model=FileUploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    knowledge_base_name: Optional[str] = Form(default=None),
    knowledge_base_id: Optional[str] = Form(default=None),
    user_id: str = Depends(get_user_id),
    db: Session = Depends(get_database_session)
):
    """
    Upload a document (alias of `/file-upload`).

    Kept so existing clients that post to `/api/upload/` keep working.
    """
    return await _store_upload(
        file=file,
        user_id=user_id,
        db=db,
        knowledge_base_name=knowledge_base_name,
        knowledge_base_id=knowledge_base_id,
    )


def extract_document_with_progress(
    storage_path: str,
    safe_filename: str,
    # The task manager injects file_id, user_id, progress_callback and task_id
    # into the kwargs of every submitted task, so they must all be accepted
    # here or the call fails with a TypeError before the body runs.
    file_id: str = "",
    user_id: str = "",
    progress_callback=None,
    task_id=None
):
    """
    Extract a document to markdown and record the result.

    This covers extraction only. Chunking and indexing are driven by the
    client through /api/chunks/create and /api/chunks/finalize, which lets the
    user choose a chunking strategy after seeing the extracted content.
    """
    import time

    logger.info(f"🚀 [Task {task_id}] Starting extraction for {file_id}")

    start_time = time.time()

    try:
        from app.services.extraction.extractor_factory import ExtractorFactory
        from app.utils.markdown_converter import MarkdownConverter
        from app.core.database import get_db_context

        if progress_callback:
            progress_callback(
                progress=5.0,
                stage=ProcessingStage.EXTRACTING,
                details=f"Initializing extraction for {safe_filename}"
            )

        with get_db_context() as db:
            file_metadata = db.query(FileMetadata).filter(
                FileMetadata.file_id == file_id
            ).first()

            if not file_metadata:
                raise ValueError(f"File not found: {file_id}")

            # ---- Extraction ----
            file_extension = Path(storage_path).suffix.lstrip('.')

            extractor = ExtractorFactory.get_extractor(file_extension)
            if not extractor:
                raise ValueError(f"No extractor for .{file_extension}")

            if progress_callback:
                progress_callback(
                    progress=15.0,
                    stage=ProcessingStage.EXTRACTING,
                    details=f"Extracting content from {file_extension.upper()} file"
                )

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
            chart_count = extraction_metadata.get('charts_extracted', 0)

            logger.info(
                f"✅ [Task {task_id}] Extraction complete in {extraction_time:.2f}s: "
                f"{total_pages} pages, {word_count} words, {image_count} images, "
                f"{table_count} tables"
            )

            if progress_callback:
                progress_callback(
                    progress=70.0,
                    stage=ProcessingStage.EXTRACTING,
                    details=f"Extracted {total_pages} pages, {word_count} words",
                    total_pages=total_pages,
                    pages_processed=total_pages
                )

            # ---- Save markdown ----
            if progress_callback:
                progress_callback(
                    progress=85.0,
                    stage=ProcessingStage.FINALIZING,
                    details="Saving extracted markdown"
                )

            markdown_converter = MarkdownConverter()
            user_markdown_dir = Path(settings.markdown_dir) / user_id
            user_markdown_dir.mkdir(parents=True, exist_ok=True)
            markdown_path = user_markdown_dir / f"{file_id}.md"

            markdown_saved = markdown_converter.save_markdown_file(
                markdown_content,
                str(markdown_path)
            )

            if not markdown_saved:
                raise Exception("Could not save extracted markdown to disk")

            logger.info(f"✅ [Task {task_id}] Markdown saved: {markdown_path}")

            # ---- Record result ----
            total_time = time.time() - start_time

            file_metadata.status = DBProcessingStatus.COMPLETED
            file_metadata.processed_at = datetime.utcnow()
            file_metadata.markdown_path = str(markdown_path)
            file_metadata.total_pages = total_pages
            file_metadata.word_count = word_count
            file_metadata.image_count = image_count
            file_metadata.table_count = table_count
            file_metadata.error_message = None

            file_metadata.processing_metadata = json.dumps({
                'extraction_method': extractor.__class__.__name__,
                'extraction_time_seconds': extraction_time,
                'total_time_seconds': total_time,
                'extraction_metadata': extraction_metadata,
            })

            db.commit()

            if progress_callback:
                progress_callback(
                    progress=100.0,
                    stage=ProcessingStage.COMPLETE,
                    details=f"Extraction complete in {total_time:.1f}s"
                )

            logger.info(
                f"🎉 [Task {task_id}] EXTRACTION COMPLETE in {total_time:.2f}s "
                f"({total_pages} pages)"
            )

            return {
                'success': True,
                'file_id': file_id,
                'filename': safe_filename,
                'total_pages': total_pages,
                'markdown_path': str(markdown_path),
                'extraction_time_seconds': extraction_time,
                'processing_time_seconds': total_time,
                'statistics': {
                    'images': image_count,
                    'tables': table_count,
                    'charts': chart_count,
                    'words': word_count,
                },
            }

    except Exception as e:
        logger.error(f"❌ [Task {task_id}] EXTRACTION FAILED: {str(e)}", exc_info=True)

        try:
            from app.core.database import get_db_context

            with get_db_context() as db:
                file_metadata = db.query(FileMetadata).filter(
                    FileMetadata.file_id == file_id
                ).first()

                if file_metadata:
                    file_metadata.status = DBProcessingStatus.FAILED
                    file_metadata.error_message = str(e)
                    db.commit()
        except Exception as db_error:
            logger.error(f"❌ [Task {task_id}] Could not record failure: {db_error}")

        raise


@router.get("/status/{file_id}", response_model=ProcessingStatusResponse)
async def get_processing_status(
    file_id: str,
    user_id: str = Depends(get_user_id),
    db: Session = Depends(get_database_session)
):
    """
    Extraction status and progress for a file.

    `progress` is taken from the live background task when one is running and
    falls back to the persisted state otherwise, so the value stays sensible
    after a restart or once the task record has been cleaned up.
    """
    file_metadata = db.query(FileMetadata).filter(
        FileMetadata.file_id == file_id,
        FileMetadata.user_id == user_id
    ).first()

    if not file_metadata:
        raise HTTPException(status_code=404, detail=f"File not found: {file_id}")

    status_value = file_metadata.status.value if file_metadata.status else "uploaded"

    task = _latest_task_for_file(user_id, file_id)

    # A task can die before its body runs (for example on a bad call), leaving
    # the row stuck at PROCESSING. Reconcile from the task record so clients
    # polling this endpoint are not left waiting forever.
    if (
        status_value == "processing"
        and task
        and task.get("status") in ("failed", "cancelled")
    ):
        error = task.get("error") or f"Extraction task {task['status']}"
        file_metadata.status = DBProcessingStatus.FAILED
        file_metadata.error_message = error
        db.commit()
        status_value = "failed"
        logger.warning(
            f"Marked file {file_id} as failed: extraction task "
            f"{task.get('task_id')} ended as {task.get('status')}"
        )

    if status_value == "completed":
        progress = 100.0
        message = "Upload and extraction completed."
    elif status_value == "failed":
        progress = float(task.get("progress", 0.0)) if task else 0.0
        message = file_metadata.error_message or "Document extraction failed."
    elif task:
        progress = float(task.get("progress", 0.0))
        stage = task.get("current_stage", "")
        progress = max(progress, _STAGE_PROGRESS_FLOOR.get(stage, 0.0))
        message = task.get("stage_details") or "Processing..."
    else:
        progress = 10.0
        message = "Processing..."

    return ProcessingStatusResponse(
        file_id=file_id,
        filename=file_metadata.safe_filename,
        status=status_value,
        progress=progress,
        message=message,
        task_id=task.get("task_id") if task else None,
        knowledge_base_id=file_metadata.knowledge_base_id,
        uploaded_at=file_metadata.uploaded_at,
        processed_at=file_metadata.processed_at,
        chunks_created=file_metadata.chunks_created or 0,
        chunks_staged=file_metadata.chunks_staged or 0,
        total_pages=file_metadata.total_pages or 0,
        chunking_strategy=(
            file_metadata.chunking_strategy.value
            if file_metadata.chunking_strategy else None
        ),
        error_message=file_metadata.error_message,
    )


@router.get("/chunking-strategies")
async def get_available_chunking_strategies():
    """
    Chunking strategies supported by /api/chunks/create and /api/chunks/preview,
    with the request parameters each one accepts.
    """
    return {
        "strategies": [
            {
                "name": "default",
                "display_name": "Default",
                "description": "Token-based chunks using the server defaults",
                "best_for": "General purpose",
                "parameters": []
            },
            {
                "name": "fixed_size",
                "display_name": "Fixed Size",
                "description": "Chunks of a target token size with optional overlap",
                "best_for": "Predictable chunk sizes",
                "parameters": ["chunk_size", "chunk_overlap"]
            },
            {
                "name": "hierarchical",
                "display_name": "Hierarchical",
                "description": "Large parent chunks split into smaller child chunks",
                "best_for": "Structured documents where context matters",
                "parameters": [
                    "parent_chunk_size",
                    "child_chunk_size",
                    "child_chunk_overlap"
                ]
            },
            {
                "name": "semantic",
                "display_name": "Semantic",
                "description": "Groups semantically similar passages together",
                "best_for": "Topic coherence",
                "parameters": [
                    "chunk_size",
                    "chunk_overlap",
                    "max_buffer_size",
                    "breakpoint_threshold"
                ]
            },
            {
                "name": "no_chunking",
                "display_name": "No Chunking",
                "description": "Index the whole document as a single chunk",
                "best_for": "Short documents",
                "parameters": []
            },
        ],
        "default_strategy": "default",
        "default_parameters": {
            "chunk_size": settings.target_chunk_tokens,
            "chunk_overlap": settings.chunk_overlap_tokens,
        }
    }
