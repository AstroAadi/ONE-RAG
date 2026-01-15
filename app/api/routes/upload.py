"""
Upload Route - FIXED VERSION
CRITICAL FIX: Do not add total_pages to chunk_metadata to prevent confusion with page_number
"""

from typing import Optional
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from sqlalchemy.orm import Session
from pathlib import Path
import uuid
import shutil
from datetime import datetime
import logging
import json

from app.models.schemas import FileUploadResponse, DocumentProcessResponse, ProcessingStatus
from app.utils.file_validator import FileValidator
from app.services.extraction.extractor_factory import ExtractorFactory
from app.utils.markdown_converter import MarkdownConverter
from app.services.chunking_service import SemanticChunkingService
from app.services.vector_store_service import VectorStoreService
from app.core.config import settings
from app.core.dependencies import get_user_id, get_database_session
from app.db.models import FileMetadata, ChunkMetadata, ProcessingStatus as DBProcessingStatus, FileType

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/upload", tags=["Upload"])

# Initialize services
markdown_converter = MarkdownConverter()
chunking_service = SemanticChunkingService(
    chunk_size=settings.chunk_size,
    chunk_overlap=settings.chunk_overlap
)
vector_store = VectorStoreService()


@router.post("/", response_model=FileUploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    user_id: str = Depends(get_user_id),
    db: Session = Depends(get_database_session)
):
    """
    Upload document file with proper user isolation
    
    Directory structure:
    - uploads/{user_id}/{filename} - Original file
    - uploads/{user_id}/extracted_images/{doc_stem}/ - Images (created during extraction)
    - markdown_output/{user_id}/{file_id}.md - Markdown output
    """
    
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
    
    # Get safe filename
    safe_filename = FileValidator.get_safe_filename(file.filename)
    
    # Create user-specific upload directory
    user_upload_dir = Path(settings.upload_dir) / user_id
    user_upload_dir.mkdir(parents=True, exist_ok=True)
    
    # FIXED: Use file_id in filename to ensure uniqueness
    # Format: {file_id}_{safe_filename}
    unique_filename = f"{file_id}_{safe_filename}"
    upload_path = user_upload_dir / unique_filename
    
    try:
        # Save file ONCE
        with upload_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"✅ Saved file: {upload_path}")
        
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
        
        # Map category to FileType enum
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
            safe_filename=unique_filename,  # Store unique filename
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
        
        logger.info(f"✅ Database record created for file_id: {file_id}")
        
        return FileUploadResponse(
            file_id=file_id,
            filename=safe_filename,
            file_type=file_extension,
            file_size_bytes=file_size,
            upload_time=datetime.now(),
            status=ProcessingStatus.UPLOADED,
            message=f"File '{safe_filename}' uploaded successfully. Use /process/{file_id} to extract and index."
        )
    
    except HTTPException:
        db.rollback()
        raise
    except Exception as e:
        db.rollback()
        if upload_path.exists():
            upload_path.unlink()
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload error: {str(e)}")


@router.post("/process/{file_id}", response_model=DocumentProcessResponse)
async def process_and_index_document(
    file_id: str,
    user_id: str = Depends(get_user_id),
    db: Session = Depends(get_database_session)
):
    """
    Process document and index to vector store
    
    FIXED:
    - No duplicate file saving
    - Proper image directory isolation
    - Consistent markdown output location
    - CRITICAL: Do not add total_pages to chunk_metadata (prevents page_number bug)
    """
    
    # Get file metadata from database
    file_metadata = db.query(FileMetadata).filter(
        FileMetadata.file_id == file_id,
        FileMetadata.user_id == user_id
    ).first()
    
    if not file_metadata:
        raise HTTPException(
            status_code=404,
            detail=f"File not found for ID: {file_id}"
        )
    
    # Check if already processed
    if file_metadata.status == DBProcessingStatus.COMPLETED:
        return DocumentProcessResponse(
            file_id=file_id,
            filename=file_metadata.safe_filename,
            status=ProcessingStatus.COMPLETED,
            markdown_path=file_metadata.markdown_path,
            chunks_created=file_metadata.chunks_created,
            total_pages=file_metadata.total_pages,
            statistics={
                'images': file_metadata.image_count,
                'tables': file_metadata.table_count,
                'words': file_metadata.word_count
            },
            message=f"Document already processed. {file_metadata.chunks_created} chunks indexed."
        )
    
    file_path = Path(file_metadata.storage_path)
    
    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"File not found on disk: {file_metadata.safe_filename}"
        )
    
    # Update status to processing
    file_metadata.status = DBProcessingStatus.PROCESSING
    db.commit()
    
    try:
        logger.info(f"📄 Processing: {file_metadata.safe_filename} (type: {file_metadata.file_extension})")
        
        # Get extractor (will handle image directory creation internally)
        extractor = ExtractorFactory.get_extractor(
            file_metadata.file_extension,
            config={
                'do_ocr': settings.enable_ocr,
                'do_table_structure': settings.infer_table_structure,
                'generate_images': settings.extract_images,
                'images_scale': settings.image_scale,
            }
        )
        
        if not extractor:
            supported_formats = ExtractorFactory.get_supported_formats()
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported file format: '.{file_metadata.file_extension}'. "
                       f"Supported formats: {', '.join(supported_formats)}"
            )
        
        logger.info(f"Using extractor: {extractor.__class__.__name__}")
        
        # Extract (extractor creates its own image directories)
        extracted_text, success, extraction_metadata = extractor.extract(str(file_path))
        
        if not success:
            error_msg = extraction_metadata.get('error', 'Unknown extraction error')
            file_metadata.status = DBProcessingStatus.FAILED
            file_metadata.error_message = error_msg
            db.commit()
            raise HTTPException(
                status_code=422,
                detail=f"Failed to extract content: {error_msg}"
            )
        
        if not extracted_text or not extracted_text.strip():
            file_metadata.status = DBProcessingStatus.FAILED
            file_metadata.error_message = "No content extracted"
            db.commit()
            raise HTTPException(
                status_code=422,
                detail="No content extracted from document."
            )
        
        logger.info(f"✅ Extracted {len(extracted_text)} characters")
        
        # Save markdown to user-specific directory
        markdown_content = markdown_converter.convert_text_to_markdown(
            extracted_text,
            file_metadata.original_filename,
            file_metadata.file_extension
        )
        
        # Create user-specific markdown directory
        user_markdown_dir = Path(settings.markdown_dir) / user_id
        user_markdown_dir.mkdir(parents=True, exist_ok=True)
        
        markdown_path = user_markdown_dir / f"{file_id}.md"
        markdown_saved = markdown_converter.save_markdown_file(
            markdown_content,
            str(markdown_path)
        )
        
        if markdown_saved:
            logger.info(f"✅ Saved markdown: {markdown_path.name}")
        
        # CRITICAL FIX: Prepare metadata for chunking
        # Do NOT include total_pages here - it will be confused with page_number
        chunk_metadata = {
            "file_id": file_id,
            "filename": file_metadata.safe_filename,
            "file_type": file_metadata.file_extension,
            "original_filename": file_metadata.original_filename,
            "user_id": user_id
        }
        
        # ❌ REMOVED: Do NOT add total_pages to chunk metadata
        # This was causing page_number to be set to total_pages instead of actual page
        # if 'total_pages' in extraction_metadata:
        #     chunk_metadata['total_pages'] = extraction_metadata['total_pages']
        
        logger.debug(f"Chunk metadata (without total_pages): {chunk_metadata}")
        
        # Count visual elements
        image_count = extracted_text.count('[Image Description]')
        table_count = extracted_text.count('**[TABLE]**') + extracted_text.count('| --- |')
        chart_count = extracted_text.count('**[CHART]**') + extracted_text.count('**[FIGURE]**')
        word_count = len(extracted_text.split())
        
        # Create structure for statistics
        simple_structure = {
            'statistics': {
                'images': image_count,
                'tables': table_count,
                'charts': chart_count
            }
        }
        
        # Chunk document
        logger.info("Creating chunks...")
        chunks = chunking_service.split_document_into_chunks(
            extracted_text,
            simple_structure,
            base_metadata=chunk_metadata
        )
        
        if not chunks:
            file_metadata.status = DBProcessingStatus.FAILED
            file_metadata.error_message = "No chunks created"
            db.commit()
            raise HTTPException(
                status_code=422, 
                detail="No chunks created."
            )
        
        logger.info(f"✅ Created {len(chunks)} chunks")
        
        # Verify page_number is correctly set
        page_numbers = [c['metadata'].get('page_number', 0) for c in chunks]
        logger.info(f"Page numbers in chunks: min={min(page_numbers)}, max={max(page_numbers)}, unique={len(set(page_numbers))}")
        
        # Index to vector store
        logger.info("Indexing to vector store...")
        chunks_added = vector_store.add_document_chunks(
            chunks,
            file_id,
            file_metadata.safe_filename,
            structured_data=simple_structure
        )
        
        logger.info(f"✅ Indexed {chunks_added} chunks")
        
        # Save chunk metadata to database
        for chunk in chunks:
            chunk_db = ChunkMetadata(
                chunk_id=chunk['chunk_id'],
                file_id=file_id,
                chunk_index=chunk['chunk_index'],
                chunk_type=chunk['metadata'].get('chunk_type', 'text'),
                char_count=chunk['char_count'],
                page_number=chunk['metadata'].get('page_number', 0),
                created_at=datetime.utcnow()
            )
            db.add(chunk_db)
        
        # Update file metadata with results
        file_metadata.status = DBProcessingStatus.COMPLETED
        file_metadata.processed_at = datetime.utcnow()
        file_metadata.chunks_created = chunks_added
        file_metadata.total_pages = extraction_metadata.get('total_pages', 0)  # Store here, not in chunks
        file_metadata.markdown_path = str(markdown_path) if markdown_saved else None
        file_metadata.word_count = word_count
        file_metadata.image_count = image_count
        file_metadata.table_count = table_count
        
        # Store processing metadata
        processing_meta = {
            'extraction_method': extractor.__class__.__name__,
            'chunks_created': chunks_added,
            'extraction_metadata': extraction_metadata
        }
        file_metadata.processing_metadata = json.dumps(processing_meta)
        
        db.commit()
        
        logger.info(f"✅ Processing complete for file_id: {file_id}")
        
        # Build response message
        message_parts = [
            f"✅ Document '{file_metadata.safe_filename}' processed successfully:",
            f"  • Format: {file_metadata.file_extension.upper()}",
            f"  • Chunks indexed: {chunks_added}",
        ]
        
        if table_count > 0:
            message_parts.append(f"  • Tables: {table_count}")
        
        if image_count > 0:
            message_parts.append(f"  • Images: {image_count}")
        
        # if 'total_pages' in extraction_metadata:
        #     message_parts.append(f"  • Pages: {extraction_metadata['total_pages']}")
        
        return DocumentProcessResponse(
            file_id=file_id,
            filename=file_metadata.safe_filename,
            status=ProcessingStatus.COMPLETED,
            markdown_path=str(markdown_path) if markdown_saved else None,
            chunks_created=chunks_added,
            total_pages=extraction_metadata.get('total_pages', 0),
            statistics={
                'images': image_count,
                'tables': table_count,
                'charts': chart_count,
                'words': word_count
            },
            message="\n".join(message_parts)
        )
    
    except HTTPException:
        raise
    except Exception as e:
        file_metadata.status = DBProcessingStatus.FAILED
        file_metadata.error_message = str(e)
        db.commit()
        
        logger.error(f"Processing error: {str(e)}")
        import traceback
        traceback.print_exc()
        
        raise HTTPException(
            status_code=500,
            detail=f"Processing error: {str(e)}"
        )


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
        raise HTTPException(
            status_code=404,
            detail=f"File not found for ID: {file_id}"
        )
    
    return {
        "file_id": file_id,
        "filename": file_metadata.safe_filename,
        "status": file_metadata.status.value,
        "uploaded_at": file_metadata.uploaded_at,
        "processed_at": file_metadata.processed_at,
        "chunks_created": file_metadata.chunks_created,
        "total_pages": file_metadata.total_pages,
        "error_message": file_metadata.error_message
    }