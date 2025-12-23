"""
Upload Route with ExtractorFactory Integration
Handles document upload and processing with automatic format detection
"""

from fastapi import APIRouter, UploadFile, File, HTTPException
from pathlib import Path
import uuid
import shutil
from datetime import datetime
import logging

from app.models.schemas import FileUploadResponse, DocumentProcessResponse
from app.utils.file_validator import FileValidator
from app.services.extraction.extractor_factory import ExtractorFactory
from app.utils.markdown_converter import MarkdownConverter
from app.services.chunking_service import SemanticChunkingService
from app.services.vector_store_service import VectorStoreService
from app.core.config import settings

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
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a document file
    
    Supports multiple formats: PDF, DOCX, DOC, PPTX, PPT, images, markdown, etc.
    
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
    upload_path = Path(settings.upload_dir) / safe_filename
    
    # Handle duplicate filenames
    if upload_path.exists():
        name_parts = safe_filename.rsplit('.', 1)
        if len(name_parts) == 2:
            safe_filename = f"{name_parts[0]}_{file_id[:8]}.{name_parts[1]}"
        else:
            safe_filename = f"{safe_filename}_{file_id[:8]}"
        upload_path = Path(settings.upload_dir) / safe_filename
    
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
        
        logger.info(f"File uploaded: {safe_filename} ({file_category})")
        
        return FileUploadResponse(
            file_id=file_id,
            filename=safe_filename,
            file_type=file_extension,
            upload_time=datetime.now(),
            status="uploaded",
            message=f"File '{safe_filename}' uploaded successfully. Use /process/{file_id} to extract and index."
        )
    
    except HTTPException:
        raise
    except Exception as e:
        if upload_path.exists():
            upload_path.unlink()
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload error: {str(e)}")


@router.post("/process/{file_id}", response_model=DocumentProcessResponse)
async def process_and_index_document(file_id: str):
    """
    Process document and index to vector store
    
    Uses ExtractorFactory to automatically select appropriate extractor
    based on file type.
    
    Args:
        file_id: Unique file identifier from upload
        
    Returns:
        DocumentProcessResponse with processing status and statistics
    """
    upload_dir = Path(settings.upload_dir)
    file_path = None
    file_extension = None
    actual_filename = None
    
    # Find uploaded file - try multiple strategies
    file_path, file_extension, actual_filename = _find_uploaded_file(upload_dir, file_id)
    
    if not file_path:
        raise HTTPException(
            status_code=404, 
            detail=f"File not found for ID: {file_id}. Please upload the file first."
        )
    
    try:
        logger.info(f"Processing: {actual_filename} (type: {file_extension})")
        
        # Get appropriate extractor using factory
        extractor = ExtractorFactory.get_extractor(
            file_extension,
            config={
                'do_ocr': settings.enable_ai_image_analysis,
                'do_table_structure': settings.infer_table_structure,
                'generate_images': settings.extract_images,
                'images_scale': 2.0,
            }
        )
        
        if not extractor:
            supported_formats = ExtractorFactory.get_supported_formats()
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported file format: '.{file_extension}'. "
                       f"Supported formats: {', '.join(supported_formats)}"
            )
        
        logger.info(f"Using extractor: {extractor.__class__.__name__}")
        
        # Extract text and metadata
        extracted_text, success, extraction_metadata = extractor.extract(str(file_path))
        
        if not success:
            error_msg = extraction_metadata.get('error', 'Unknown extraction error')
            raise HTTPException(
                status_code=422,
                detail=f"Failed to extract content: {error_msg}"
            )
        
        if not extracted_text or not extracted_text.strip():
            raise HTTPException(
                status_code=422,
                detail="No content extracted from document. The file may be empty or corrupted."
            )
        
        logger.info(f"✓ Extracted {len(extracted_text)} characters")
        
        # Save markdown
        markdown_content = markdown_converter.convert_text_to_markdown(
            extracted_text,
            actual_filename,
            file_extension
        )
        
        markdown_path = Path(settings.markdown_dir) / f"{file_id}.md"
        markdown_saved = markdown_converter.save_markdown_file(
            markdown_content,
            str(markdown_path)
        )
        
        if not markdown_saved:
            logger.warning("Failed to save markdown file, but continuing with processing")
        else:
            logger.info(f"✓ Saved markdown: {markdown_path.name}")
        
        # Prepare metadata for chunking
        chunk_metadata = {
            "file_id": file_id,
            "filename": actual_filename,
            "file_type": file_extension,
            "original_filename": actual_filename
        }
        
        # Add extraction metadata
        if 'total_pages' in extraction_metadata:
            chunk_metadata['total_pages'] = extraction_metadata['total_pages']
        
        # Count visual elements from extracted text
        image_count = extracted_text.count('IMAGE_PATH:')
        table_count = extracted_text.count('**[TABLE]**') + extracted_text.count('| --- |')
        chart_count = extracted_text.count('**[CHART]**') + extracted_text.count('**[FIGURE]**')
        
        # Create simple structure for statistics
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
            actual_filename,
            structured_data=simple_structure
        )
        
        logger.info(f"✓ Indexed {chunks_added} chunks to vector store")
        
        # Build response message
        message_parts = [
            f"✓ Document '{actual_filename}' processed successfully:",
            f"  • Format: {file_extension.upper()}",
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
            filename=actual_filename,
            status="processed",
            markdown_path=str(markdown_path) if markdown_saved else None,
            chunks_created=chunks_added,
            message="\n".join(message_parts)
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Processing error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Processing error: {str(e)}"
        )


def _find_uploaded_file(upload_dir: Path, file_id: str) -> tuple:
    """
    Find uploaded file using multiple strategies
    
    Args:
        upload_dir: Directory containing uploaded files
        file_id: File ID to search for
        
    Returns:
        Tuple of (file_path, file_extension, filename) or (None, None, None)
    """
    # Strategy 1: Try with file_id in filename
    for found_file in upload_dir.glob(f"*{file_id[:8]}*"):
        if found_file.is_file():
            file_extension = FileValidator.get_file_extension(found_file.name)
            return found_file, file_extension, found_file.name
    
    # Strategy 2: Try exact match with common extensions
    supported_formats = ExtractorFactory.get_supported_formats()
    for ext in supported_formats:
        potential_path = upload_dir / f"{file_id}.{ext}"
        if potential_path.exists():
            return potential_path, ext, potential_path.name
    
    # Strategy 3: Find most recent file (fallback)
    all_files = [f for f in upload_dir.glob("*") if f.is_file()]
    if all_files:
        most_recent = max(all_files, key=lambda p: p.stat().st_mtime)
        file_extension = FileValidator.get_file_extension(most_recent.name)
        
        # Only use if extension is supported
        if FileValidator.get_file_extension(most_recent.name, with_dot=True) in FileValidator.get_allowed_extensions():
            logger.warning(f"Could not find file for ID {file_id}, using most recent: {most_recent.name}")
            return most_recent, file_extension, most_recent.name
    
    return None, None, None


@router.get("/supported-formats")
async def get_supported_formats():
    """
    Get list of supported file formats
    
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
        "extractors": extractor_info
    }


@router.delete("/{file_id}")
async def delete_uploaded_file(file_id: str):
    """
    Delete an uploaded file (before processing)
    
    Args:
        file_id: File ID to delete
        
    Returns:
        Success message
    """
    upload_dir = Path(settings.upload_dir)
    file_path, file_extension, actual_filename = _find_uploaded_file(upload_dir, file_id)
    
    if not file_path:
        raise HTTPException(
            status_code=404,
            detail=f"File not found for ID: {file_id}"
        )
    
    try:
        file_path.unlink()
        logger.info(f"Deleted file: {actual_filename}")
        
        return {
            "message": f"File '{actual_filename}' deleted successfully",
            "file_id": file_id
        }
    except Exception as e:
        logger.error(f"Error deleting file: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting file: {str(e)}"
        )


@router.get("/list")
async def list_uploaded_files():
    """
    List all uploaded files (not yet processed)
    
    Returns:
        List of uploaded files with metadata
    """
    upload_dir = Path(settings.upload_dir)
    
    files = []
    for file_path in upload_dir.glob("*"):
        if file_path.is_file() and not file_path.name.startswith('.'):
            stat = file_path.stat()
            files.append({
                "filename": file_path.name,
                "size_bytes": stat.st_size,
                "size_mb": round(stat.st_size / (1024 * 1024), 2),
                "uploaded_time": datetime.fromtimestamp(stat.st_mtime),
                "file_type": FileValidator.get_file_extension(file_path.name),
                "category": FileValidator.get_file_category(file_path.name)
            })
    
    # Sort by upload time (most recent first)
    files.sort(key=lambda x: x['uploaded_time'], reverse=True)
    
    return {
        "total_files": len(files),
        "files": files
    }