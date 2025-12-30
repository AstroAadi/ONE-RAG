"""
Upload Route with FULL DATABASE INTEGRATION and MANDATORY USER AUTH
- All endpoints require X-User-ID header
- Complete database tracking of uploads and processing
- User-isolated file storage
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
    user_id: str = Depends(get_user_id),  # MANDATORY USER AUTH
    db: Session = Depends(get_database_session)
):
    """
    Upload a document file (REQUIRES X-User-ID header)
    
    Supports multiple formats: PDF, DOCX, DOC, PPTX, PPT, images, markdown, etc.
    
    Headers:
        X-User-ID: User identifier (REQUIRED)
    
    Returns:
        FileUploadResponse with file_id for processing
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
    
    upload_path = user_upload_dir / safe_filename
    
    # Handle duplicate filenames
    if upload_path.exists():
        name_parts = safe_filename.rsplit('.', 1)
        if len(name_parts) == 2:
            safe_filename = f"{name_parts[0]}_{file_id[:8]}.{name_parts[1]}"
        else:
            safe_filename = f"{safe_filename}_{file_id[:8]}"
        upload_path = user_upload_dir / safe_filename
    
    try:
        # Save file
        with upload_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
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
        
        # SAVE TO DATABASE
        file_metadata = FileMetadata(
            file_id=file_id,
            user_id=user_id,
            original_filename=file.filename,
            safe_filename=safe_filename,
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
        
        logger.info(f"✓ File uploaded and saved to DB: {safe_filename} (user: {user_id}, file_id: {file_id})")
        
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
        # Rollback database on validation errors
        db.rollback()
        raise
    except Exception as e:
        # Rollback database and cleanup file on any error
        db.rollback()
        if upload_path.exists():
            upload_path.unlink()
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload error: {str(e)}")


@router.post("/process/{file_id}", response_model=DocumentProcessResponse)
async def process_and_index_document(
    file_id: str,
    user_id: str = Depends(get_user_id),  # MANDATORY USER AUTH
    db: Session = Depends(get_database_session)
):
    """
    Process document and index to vector store (REQUIRES X-User-ID header)
    
    - Extracts content from document
    - Chunks text semantically
    - Indexes to vector database
    - Saves all metadata to SQL database
    
    Headers:
        X-User-ID: User identifier (REQUIRED)
    
    Args:
        file_id: Unique file identifier from upload
        
    Returns:
        DocumentProcessResponse with processing status and statistics
    """
    
    # GET FILE METADATA FROM DATABASE (with user isolation)
    file_metadata = db.query(FileMetadata).filter(
        FileMetadata.file_id == file_id,
        FileMetadata.user_id == user_id  # USER ISOLATION
    ).first()
    
    if not file_metadata:
        raise HTTPException(
            status_code=404,
            detail=f"File not found for ID: {file_id}. Please upload the file first or check user ID."
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
    
    # UPDATE STATUS TO PROCESSING
    file_metadata.status = DBProcessingStatus.PROCESSING
    db.commit()
    
    try:
        logger.info(f"Processing: {file_metadata.safe_filename} (type: {file_metadata.file_extension}) for user: {user_id}")
        
        # Get appropriate extractor using factory
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
        
        # Extract text and metadata
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
                detail="No content extracted from document. The file may be empty or corrupted."
            )
        
        logger.info(f"✓ Extracted {len(extracted_text)} characters")
        
        # Save markdown
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
            logger.info(f"✓ Saved markdown: {markdown_path.name}")
        
        # Prepare metadata for chunking (WITH USER_ID)
        chunk_metadata = {
            "file_id": file_id,
            "filename": file_metadata.safe_filename,
            "file_type": file_metadata.file_extension,
            "original_filename": file_metadata.original_filename,
            "user_id": user_id  # CRITICAL FOR USER ISOLATION
        }
        
        # Add extraction metadata
        if 'total_pages' in extraction_metadata:
            chunk_metadata['total_pages'] = extraction_metadata['total_pages']
        
        # Count visual elements
        image_count = extracted_text.count('IMAGE_PATH:')
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
                detail="No chunks created. Document may be too short or improperly formatted."
            )
        
        logger.info(f"✓ Created {len(chunks)} chunks")
        
        # Index to vector store
        logger.info("Indexing to vector store...")
        chunks_added = vector_store.add_document_chunks(
            chunks,
            file_id,
            file_metadata.safe_filename,
            structured_data=simple_structure
        )
        
        logger.info(f"✓ Indexed {chunks_added} chunks to vector store")
        
        # SAVE CHUNK METADATA TO DATABASE
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
        
        # UPDATE FILE METADATA WITH RESULTS
        file_metadata.status = DBProcessingStatus.COMPLETED
        file_metadata.processed_at = datetime.utcnow()
        file_metadata.chunks_created = chunks_added
        file_metadata.total_pages = extraction_metadata.get('total_pages', 0)
        file_metadata.markdown_path = str(markdown_path) if markdown_saved else None
        file_metadata.word_count = word_count
        file_metadata.image_count = image_count
        file_metadata.table_count = table_count
        
        # Store processing metadata as JSON
        processing_meta = {
            'extraction_method': extractor.__class__.__name__,
            'chunks_created': chunks_added,
            'processing_time': None,  # Could add timing
            'extraction_metadata': extraction_metadata
        }
        file_metadata.processing_metadata = json.dumps(processing_meta)
        
        db.commit()
        
        logger.info(f"✓ Database updated with processing results")
        
        # Build response message
        message_parts = [
            f"✓ Document '{file_metadata.safe_filename}' processed successfully:",
            f"  • Format: {file_metadata.file_extension.upper()}",
            f"  • Chunks indexed: {chunks_added}",
        ]
        
        if table_count > 0:
            message_parts.append(f"  • Tables: {table_count}")
        
        if image_count > 0:
            message_parts.append(f"  • Images: {image_count}")
        
        if chart_count > 0:
            message_parts.append(f"  • Charts/Figures: {chart_count}")
        
        if 'total_pages' in extraction_metadata:
            message_parts.append(f"  • Pages: {extraction_metadata['total_pages']}")
        
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
        # UPDATE STATUS TO FAILED
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


@router.get("/list")
async def list_uploaded_files(
    user_id: str = Depends(get_user_id),
    db: Session = Depends(get_database_session)
):
    """
    List all uploaded files for the current user
    
    Headers:
        X-User-ID: User identifier (REQUIRED)
    
    Returns:
        List of uploaded files with metadata
    """
    
    # Query files from database with user isolation
    files = db.query(FileMetadata).filter(
        FileMetadata.user_id == user_id
    ).order_by(
        FileMetadata.uploaded_at.desc()
    ).all()
    
    file_list = []
    for file_meta in files:
        file_list.append({
            "file_id": file_meta.file_id,
            "filename": file_meta.safe_filename,
            "original_filename": file_meta.original_filename,
            "file_type": file_meta.file_extension,
            "file_category": file_meta.file_type.value,
            "size_bytes": file_meta.file_size_bytes,
            "size_mb": round(file_meta.file_size_bytes / (1024 * 1024), 2),
            "status": file_meta.status.value,
            "chunks_created": file_meta.chunks_created,
            "uploaded_at": file_meta.uploaded_at.isoformat(),
            "processed_at": file_meta.processed_at.isoformat() if file_meta.processed_at else None,
            "has_error": file_meta.error_message is not None
        })
    
    return {
        "user_id": user_id,
        "total_files": len(file_list),
        "files": file_list
    }


@router.delete("/{file_id}")
async def delete_uploaded_file(
    file_id: str,
    user_id: str = Depends(get_user_id),
    db: Session = Depends(get_database_session)
):
    """
    Delete an uploaded file (with user isolation)
    
    Deletes:
    - File from disk
    - Database record
    - Vector store entries
    - Chunk metadata
    
    Headers:
        X-User-ID: User identifier (REQUIRED)
    
    Args:
        file_id: File ID to delete
        
    Returns:
        Success message
    """
    
    # Get file metadata with user isolation
    file_metadata = db.query(FileMetadata).filter(
        FileMetadata.file_id == file_id,
        FileMetadata.user_id == user_id  # USER ISOLATION
    ).first()
    
    if not file_metadata:
        raise HTTPException(
            status_code=404,
            detail=f"File not found for ID: {file_id}"
        )
    
    try:
        # Delete from vector store
        vector_store.delete_document_by_file_id(file_id, user_id)
        logger.info(f"✓ Deleted from vector store: {file_id}")
        
        # Delete file from disk
        file_path = Path(file_metadata.storage_path)
        if file_path.exists():
            file_path.unlink()
            logger.info(f"✓ Deleted file from disk: {file_path.name}")
        
        # Delete markdown if exists
        if file_metadata.markdown_path:
            markdown_path = Path(file_metadata.markdown_path)
            if markdown_path.exists():
                markdown_path.unlink()
                logger.info(f"✓ Deleted markdown: {markdown_path.name}")
        
        # Delete from database (cascades to chunks)
        db.delete(file_metadata)
        db.commit()
        logger.info(f"✓ Deleted database record: {file_id}")
        
        return {
            "message": f"File '{file_metadata.safe_filename}' and all associated data deleted successfully",
            "file_id": file_id,
            "user_id": user_id
        }
        
    except Exception as e:
        db.rollback()
        logger.error(f"Error deleting file: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting file: {str(e)}"
        )


@router.get("/supported-formats")
async def get_supported_formats():
    """
    Get list of supported file formats
    
    No authentication required for this endpoint.
    
    Returns:
        Dictionary with supported formats and their categories
    """
    formats = ExtractorFactory.get_supported_formats()
    extractor_info = ExtractorFactory.get_extractor_info()
    
    # Categorize formats
    categories = {
        'documents': [],
        'presentations': [],
        'spreadsheets': [],
        'images': [],
        'other': []
    }
    
    for fmt in formats:
        dummy_filename = f"file.{fmt}"
        category = FileValidator.get_file_category(dummy_filename)
        
        if category == 'document':
            categories['documents'].append(fmt)
        elif category == 'presentation':
            categories['presentations'].append(fmt)
        elif category == 'spreadsheet':
            categories['spreadsheets'].append(fmt)
        elif category == 'image':
            categories['images'].append(fmt)
        else:
            categories['other'].append(fmt)
    
    return {
        "total_formats": len(formats),
        "formats": sorted(formats),
        "categories": categories,
        "extractors": extractor_info,
        "note": "All endpoints except this one require X-User-ID header"
    }