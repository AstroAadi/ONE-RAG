"""
Streamlined Upload Route
- NO JSON file generation
- Only markdown output
- Fast processing
"""

from fastapi import APIRouter, UploadFile, File, HTTPException
from pathlib import Path
import uuid
import shutil
from datetime import datetime
import logging

from app.models.schemas import FileUploadResponse, DocumentProcessResponse
from app.utils.file_validator import FileValidator
from app.services.text_extractor import PageByPagePDFExtractor as DocumentTextExtractor
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
    """
    Upload a document file (PDF or DOCX)
    FIXED: Save with actual filename, use UUID only for markdown
    """
    # Validate file
    is_valid, error_message = FileValidator.validate_file(file)
    if not is_valid:
        raise HTTPException(status_code=400, detail=error_message)
    
    # Generate unique file ID for internal tracking
    file_id = str(uuid.uuid4())
    file_extension = FileValidator.get_file_extension(file.filename)
    
    # Save uploaded file with ACTUAL filename (not UUID)
    # This allows users to identify which document they uploaded
    original_filename = file.filename
    upload_path = Path(settings.upload_dir) / original_filename
    
    # If file with same name exists, append UUID to make unique
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
            filename=original_filename,  # Return actual filename
            file_type=file_extension,
            upload_time=datetime.now(),
            status="uploaded",
            message=f"File '{original_filename}' uploaded successfully. Use /process endpoint with file_id to extract and index."
        )
    
    except Exception as e:
        if upload_path.exists():
            upload_path.unlink()
        raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")


@router.post("/process/{file_id}", response_model=DocumentProcessResponse)
async def process_and_index_document(file_id: str):
    """
    Streamlined document processing:
    FIXED: Find file by actual name, use UUID only for markdown/internal tracking
    - Extract text and images
    - Generate markdown with image references (NO JSON)
    - Chunk with semantic boundaries
    - Index to vector store
    """
    # Find uploaded file - now saved with actual filename
    upload_dir = Path(settings.upload_dir)
    file_path = None
    file_extension = None
    actual_filename = None
    
    # Try to find by file_id first (for backwards compatibility)
    for ext in ['pdf', 'docx', 'doc']:
        potential_path = upload_dir / f"{file_id}.{ext}"
        if potential_path.exists():
            file_path = potential_path
            file_extension = ext
            actual_filename = potential_path.name
            break
    
    # If not found by UUID, search for files containing the file_id (versioned filenames)
    if not file_path:
        for file in upload_dir.glob(f"*{file_id[:8]}*"):
            if file.is_file():
                file_path = file
                file_extension = file.suffix.lstrip('.')
                actual_filename = file.name
                break
    
    # Last resort: find any recently uploaded file
    if not file_path:
        for ext in ['pdf', 'docx', 'doc']:
            files = list(upload_dir.glob(f"*.{ext}"))
            if files:
                # Get most recent
                file_path = max(files, key=lambda p: p.stat().st_mtime)
                file_extension = ext
                actual_filename = file_path.name
                logger.warning(f"Could not find file with ID {file_id}, using most recent: {actual_filename}")
                break
    
    if not file_path:
        raise HTTPException(status_code=404, detail="File not found. Please upload first.")
    
    try:
        print(f"Processing {actual_filename} (file_id: {file_id})...")
        
        # Step 1: Extract text + images (returns markdown only, no JSON)
        extracted_text, success = DocumentTextExtractor.extract_text_from_document(
            str(file_path),
            file_extension,
            aws_region=settings.aws_region,
            aws_profile=settings.aws_profile_name if hasattr(settings, 'aws_profile_name') else None
        )
        
        if not success or not extracted_text:
            raise HTTPException(
                status_code=422,
                detail="Failed to extract text from document."
            )
        
        print(f"✓ Extracted {len(extracted_text)} characters")
        
        # Step 2: Save markdown with UUID filename for internal tracking
        markdown_content = markdown_converter.convert_text_to_markdown(
            extracted_text,
            actual_filename,  # Use actual filename in markdown header
            file_extension
        )
        
        # Use UUID for markdown filename
        markdown_path = Path(settings.markdown_dir) / f"{file_id}.md"
        markdown_saved = markdown_converter.save_markdown_file(
            markdown_content,
            str(markdown_path)
        )
        
        if not markdown_saved:
            raise HTTPException(status_code=500, detail="Failed to save markdown file")
        
        print(f"✓ Saved markdown: {markdown_path.name}")
        
        # Step 3: Semantic chunking (no structured_data needed)
        chunk_metadata = {
            "file_id": file_id,
            "filename": actual_filename,  # Store actual filename for user reference
            "file_type": file_extension,
            "original_filename": actual_filename
        }
        
        # Count images and tables from markdown
        image_count = extracted_text.count('IMAGE_PATH:')
        table_count = extracted_text.count('**[TABLE]**')
        chart_count = extracted_text.count('**[CHART]**') + extracted_text.count('**[FIGURE]**')
        
        # Create simple structured data for chunking
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
            raise HTTPException(status_code=422, detail="No chunks created from document")
        
        print(f"✓ Created {len(chunks)} chunks")
        
        # Step 4: Index to vector store
        chunks_added = vector_store.add_document_chunks(
            chunks,
            file_id,
            actual_filename,  # Use actual filename
            structured_data=simple_structure
        )
        
        print(f"✓ Indexed {chunks_added} chunks to vector store")
        
        # Build response message
        message_parts = [
            f"Document '{actual_filename}' processed successfully:",
            f"- Total chunks indexed: {chunks_added}",
            f"- Tables detected: {table_count}",
            f"- Images/Charts: {image_count + chart_count}",
        ]
        
        # Check if image analysis was used
        if image_count + chart_count > 0:
            if settings.enable_ai_image_analysis:
                message_parts.append("- AI image analysis: Enabled")
            else:
                message_parts.append("- AI image analysis: Disabled (enable in .env for detailed visual descriptions)")
        
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
            detail=f"Error processing document: {str(e)}"
        )