"""
Clean Upload Route
Simple document upload and processing
"""

from fastapi import APIRouter, UploadFile, File, HTTPException
from pathlib import Path
import uuid
import shutil
from datetime import datetime
import logging

from app.models.schemas import FileUploadResponse, DocumentProcessResponse
from app.utils.file_validator import FileValidator
from app.services.text_extractor import DoclingDocumentExtractor as DocumentTextExtractor
from app.utils.markdown_converter import MarkdownConverter
from app.services.chunking_service import SemanticChunkingService
from app.services.vector_store_service import EnhancedVectorStoreService
from app.core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/upload", tags=["Upload"])

# Initialize services
markdown_converter = MarkdownConverter()
chunking_service = SemanticChunkingService(
    chunk_size=settings.chunk_size,
    chunk_overlap=settings.chunk_overlap
)
vector_store = EnhancedVectorStoreService()


@router.post("/", response_model=FileUploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """Upload a document file"""
    # Validate file
    is_valid, error_message = FileValidator.validate_file(file)
    if not is_valid:
        raise HTTPException(status_code=400, detail=error_message)
    
    # Generate unique file ID
    file_id = str(uuid.uuid4())
    file_extension = FileValidator.get_file_extension(file.filename)
    
    # Save with actual filename
    original_filename = file.filename
    upload_path = Path(settings.upload_dir) / original_filename
    
    # Handle duplicate filenames
    if upload_path.exists():
        name_parts = original_filename.rsplit('.', 1)
        if len(name_parts) == 2:
            original_filename = f"{name_parts[0]}_{file_id[:8]}.{name_parts[1]}"
        else:
            original_filename = f"{original_filename}_{file_id[:8]}"
        upload_path = Path(settings.upload_dir) / original_filename
    
    try:
        with upload_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Validate file size
        file_size = upload_path.stat().st_size
        is_size_valid, size_error = FileValidator.validate_file_size(file_size)
        
        if not is_size_valid:
            upload_path.unlink()
            raise HTTPException(status_code=400, detail=size_error)
        
        return FileUploadResponse(
            file_id=file_id,
            filename=original_filename,
            file_type=file_extension,
            upload_time=datetime.now(),
            status="uploaded",
            message=f"File '{original_filename}' uploaded. Use /process/{file_id} to extract and index."
        )
    
    except Exception as e:
        if upload_path.exists():
            upload_path.unlink()
        raise HTTPException(status_code=500, detail=f"Upload error: {str(e)}")


@router.post("/process/{file_id}", response_model=DocumentProcessResponse)
async def process_and_index_document(file_id: str):
    """Process document and index to vector store"""
    upload_dir = Path(settings.upload_dir)
    file_path = None
    file_extension = None
    actual_filename = None
    
    # Find uploaded file
    for ext in ['pdf', 'docx', 'doc']:
        # Try UUID filename
        potential_path = upload_dir / f"{file_id}.{ext}"
        if potential_path.exists():
            file_path = potential_path
            file_extension = ext
            actual_filename = potential_path.name
            break
    
    # Try versioned filename
    if not file_path:
        for file in upload_dir.glob(f"*{file_id[:8]}*"):
            if file.is_file():
                file_path = file
                file_extension = file.suffix.lstrip('.')
                actual_filename = file.name
                break
    
    # Last resort: most recent file
    if not file_path:
        for ext in ['pdf', 'docx', 'doc']:
            files = list(upload_dir.glob(f"*.{ext}"))
            if files:
                file_path = max(files, key=lambda p: p.stat().st_mtime)
                file_extension = ext
                actual_filename = file.name
                logger.warning(f"Using most recent file: {actual_filename}")
                break
    
    if not file_path:
        raise HTTPException(status_code=404, detail="File not found. Please upload first.")
    
    try:
        logger.info(f"Processing: {actual_filename}")
        
        # Extract text and images
        extracted_text, success = DocumentTextExtractor.extract_text_from_document(
            str(file_path),
            file_extension
        )
        
        if not success or not extracted_text:
            raise HTTPException(
                status_code=422,
                detail="Failed to extract text from document."
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
            raise HTTPException(status_code=500, detail="Failed to save markdown")
        
        logger.info(f"✓ Saved markdown: {markdown_path.name}")
        
        # Chunk document
        chunk_metadata = {
            "file_id": file_id,
            "filename": actual_filename,
            "file_type": file_extension,
            "original_filename": actual_filename
        }
        
        # Count visual elements
        image_count = extracted_text.count('IMAGE_PATH:')
        table_count = extracted_text.count('**[TABLE]**')
        chart_count = extracted_text.count('**[CHART]**') + extracted_text.count('**[FIGURE]**')
        
        simple_structure = {
            'statistics': {
                'images': image_count,
                'tables': table_count,
                'charts': chart_count
            }
        }
        
        chunks = chunking_service.split_document_into_chunks(
            extracted_text,
            simple_structure,
            base_metadata=chunk_metadata
        )
        
        if not chunks:
            raise HTTPException(status_code=422, detail="No chunks created")
        
        logger.info(f"✓ Created {len(chunks)} chunks")
        
        # Index to vector store
        chunks_added = vector_store.add_document_chunks(
            chunks,
            file_id,
            actual_filename,
            structured_data=simple_structure
        )
        
        logger.info(f"✓ Indexed {chunks_added} chunks")
        
        # Build response
        message_parts = [
            f"Document '{actual_filename}' processed:",
            f"- Chunks indexed: {chunks_added}",
            f"- Tables: {table_count}",
            f"- Images/Charts: {image_count + chart_count}",
        ]
        
        return DocumentProcessResponse(
            file_id=file_id,
            filename=actual_filename,
            status="processed",
            markdown_path=str(markdown_path),
            chunks_created=chunks_added,
            message="\n".join(message_parts)
        )
    
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Processing error: {str(e)}"
        )