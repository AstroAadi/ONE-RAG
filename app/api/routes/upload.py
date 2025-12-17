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

from app.models.schemas import FileUploadResponse, DocumentProcessResponse
from app.utils.file_validator import FileValidator
from app.services.text_extractor import PageByPagePDFExtractor as DocumentTextExtractor
from app.utils.markdown_converter import MarkdownConverter
from app.services.chunking_service import SemanticChunkingService
from app.services.vector_store_service import EnhancedVectorStoreService
from app.core.config import settings

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
    """Upload a document file (PDF or DOCX)"""
    # Validate file
    is_valid, error_message = FileValidator.validate_file(file)
    if not is_valid:
        raise HTTPException(status_code=400, detail=error_message)
    
    # Generate unique file ID
    file_id = str(uuid.uuid4())
    file_extension = FileValidator.get_file_extension(file.filename)
    
    # Save uploaded file
    upload_path = Path(settings.upload_dir) / f"{file_id}.{file_extension}"
    
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
            filename=file.filename,
            file_type=file_extension,
            upload_time=datetime.now(),
            status="uploaded",
            message="File uploaded successfully. Use /process endpoint to extract and index."
        )
    
    except Exception as e:
        if upload_path.exists():
            upload_path.unlink()
        raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")


@router.post("/process/{file_id}", response_model=DocumentProcessResponse)
async def process_and_index_document(file_id: str):
    """
    Streamlined document processing:
    - Extract text and images
    - Generate markdown with image references (NO JSON)
    - Chunk with semantic boundaries
    - Index to vector store
    """
    # Find uploaded file
    upload_dir = Path(settings.upload_dir)
    file_path = None
    file_extension = None
    
    for ext in ['pdf', 'docx', 'doc']:
        potential_path = upload_dir / f"{file_id}.{ext}"
        if potential_path.exists():
            file_path = potential_path
            file_extension = ext
            break
    
    if not file_path:
        raise HTTPException(status_code=404, detail="File not found. Please upload first.")
    
    try:
        print(f"Processing {file_path.name}...")
        
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
        
        # Step 2: Save markdown (with image references embedded)
        markdown_content = markdown_converter.convert_text_to_markdown(
            extracted_text,
            f"{file_id}.{file_extension}",
            file_extension
        )
        
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
            "filename": f"{file_id}.{file_extension}",
            "file_type": file_extension,
            "original_filename": file_path.name
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
            f"{file_id}.{file_extension}",
            structured_data=simple_structure
        )
        
        print(f"✓ Indexed {chunks_added} chunks to vector store")
        
        # Build response message
        message_parts = [
            f"Document processed successfully:",
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
            filename=f"{file_id}.{file_extension}",
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