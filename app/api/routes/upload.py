from fastapi import APIRouter, UploadFile, File, HTTPException
from pathlib import Path
import uuid
import shutil
from datetime import datetime

from app.models.schemas import FileUploadResponse, DocumentProcessResponse
from app.utils.file_validator import FileValidator
from app.services.text_extractor import DocumentTextExtractor
from app.utils.markdown_converter import MarkdownConverter
from app.services.chunking_service import TextChunkingService
from app.services.vector_store_service import VectorStoreService
from app.core.config import settings

router = APIRouter(prefix="/api/upload", tags=["Upload"])

# Initialize services
text_extractor = DocumentTextExtractor()
markdown_converter = MarkdownConverter()
chunking_service = TextChunkingService(
    chunk_size=settings.chunk_size,
    chunk_overlap=settings.chunk_overlap
)
vector_store = VectorStoreService()


@router.post("/", response_model=FileUploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a document file (PDF or DOCX)
    
    Returns file_id for processing
    """
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
        
        # Get file size
        file_size = upload_path.stat().st_size
        is_size_valid, size_error = FileValidator.validate_file_size(file_size)
        
        if not is_size_valid:
            upload_path.unlink()  # Delete the file
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
    Process uploaded document: extract text, convert to markdown, chunk, and index
    
    Steps:
    1. Extract text from document
    2. Convert to markdown
    3. Chunk the text
    4. Generate embeddings and store in vector DB
    """
    # Find the uploaded file
    upload_dir = Path(settings.upload_dir)
    file_path = None
    original_filename = None
    file_extension = None
    
    for ext in ['pdf', 'docx', 'doc']:
        potential_path = upload_dir / f"{file_id}.{ext}"
        if potential_path.exists():
            file_path = potential_path
            file_extension = ext
            original_filename = file_id  # We don't store original name, using file_id
            break
    
    if not file_path:
        raise HTTPException(status_code=404, detail="File not found. Please upload first.")
    
    try:
        # Step 1: Extract text from document
        extracted_text, success = text_extractor.extract_text_from_document(
            str(file_path),
            file_extension
        )
        
        if not success or not extracted_text:
            raise HTTPException(
                status_code=422,
                detail="Failed to extract text from document. File might be corrupted or empty."
            )
        
        # Step 2: Convert to markdown
        markdown_content = markdown_converter.convert_text_to_markdown(
            extracted_text,
            f"{file_id}.{file_extension}",
            file_extension
        )
        
        # Save markdown file
        markdown_path = Path(settings.markdown_dir) / f"{file_id}.md"
        markdown_saved = markdown_converter.save_markdown_file(
            markdown_content,
            str(markdown_path)
        )
        
        if not markdown_saved:
            raise HTTPException(status_code=500, detail="Failed to save markdown file")
        
        # Step 3: Chunk the text
        chunk_metadata = {
            "file_id": file_id,
            "filename": f"{file_id}.{file_extension}",
            "file_type": file_extension
        }
        
        chunks = chunking_service.split_text_into_chunks(
            extracted_text,
            metadata=chunk_metadata
        )
        
        if not chunks:
            raise HTTPException(status_code=422, detail="No chunks created from document")
        
        # Step 4: Add to vector store
        chunks_added = vector_store.add_document_chunks_to_vector_store(
            chunks,
            file_id,
            f"{file_id}.{file_extension}"
        )
        
        return DocumentProcessResponse(
            file_id=file_id,
            filename=f"{file_id}.{file_extension}",
            status="processed",
            markdown_path=str(markdown_path),
            chunks_created=chunks_added,
            message=f"Document processed successfully. {chunks_added} chunks indexed."
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing document: {str(e)}"
        )