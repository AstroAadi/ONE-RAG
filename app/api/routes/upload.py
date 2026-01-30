"""
Upload Route - ENHANCED VERSION with Chunking Strategy Selection
User can select from 6 different chunking strategies when processing documents
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
import time

from app.models.schemas import (
    FileUploadResponse, DocumentProcessResponse, ProcessingStatus, ChunkingStrategy
)
from app.utils.file_validator import FileValidator
from app.services.extraction.extractor_factory import ExtractorFactory
from app.utils.markdown_converter import MarkdownConverter
from app.services.chunking_service import EnhancedChunkingService
from app.services.vector_store_service import VectorStoreService
from app.core.config import settings
from app.core.dependencies import get_user_id, get_database_session
from app.db.models import (
    FileMetadata, ChunkMetadata, ProcessingStatus as DBProcessingStatus, 
    FileType, ChunkingStrategyEnum
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/upload", tags=["Upload"])

# Initialize services
markdown_converter = MarkdownConverter()
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
    
    Note: Chunking strategy is selected during the /process/{file_id} call
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
    
    # Use file_id in filename to ensure uniqueness
    unique_filename = f"{file_id}_{safe_filename}"
    upload_path = user_upload_dir / unique_filename
    
    try:
        # Save file
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
        
        logger.info(f"✅ Database record created for file_id: {file_id}")
        
        return FileUploadResponse(
            file_id=file_id,
            filename=safe_filename,
            file_type=file_extension,
            file_size_bytes=file_size,
            upload_time=datetime.now(),
            status=ProcessingStatus.UPLOADED,
            message=f"File '{safe_filename}' uploaded successfully. Use /process/{file_id} to extract and index with your preferred chunking strategy."
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
    db: Session = Depends(get_database_session),
    chunking_strategy: ChunkingStrategy = Query(
        default=ChunkingStrategy.FIXED_SIZE,
        description="Chunking strategy: fixed_size, sentence, paragraph, sliding_window, recursive, or semantic"
    ),
    target_chunk_tokens: Optional[int] = Query(
        default=None,
        ge=100,
        le=2048,
        description="Target chunk size in tokens (default: 512)"
    ),
    chunk_overlap_tokens: Optional[int] = Query(
        default=None,
        ge=0,
        le=500,
        description="Overlap between chunks in tokens (default: 50, only for sliding_window)"
    )
):
    """
    Process document and index to vector store with selected chunking strategy
    
    Query Parameters:
    - chunking_strategy: Choose from 6 strategies
      * fixed_size: Token-based fixed-size chunks (no overlap)
      * sentence: Groups sentences up to target size
      * paragraph: Groups paragraphs up to target size
      * sliding_window: Fixed-size with overlapping windows
      * recursive: Hierarchical split by separators
      * semantic: Groups semantically similar sentences
    
    - target_chunk_tokens: Target chunk size (default: 512)
    - chunk_overlap_tokens: Overlap for sliding_window (default: 50)
    
    Example:
        POST /process/{file_id}?chunking_strategy=semantic&target_chunk_tokens=600
    """
    
    start_time = time.time()
    
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
            chunking_strategy=file_metadata.chunking_strategy.value if file_metadata.chunking_strategy else None,
            message=f"Document already processed using {file_metadata.chunking_strategy.value if file_metadata.chunking_strategy else 'unknown'} strategy. {file_metadata.chunks_created} chunks indexed."
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
        logger.info(
            f"📄 Processing: {file_metadata.safe_filename} "
            f"(type: {file_metadata.file_extension}, strategy: {chunking_strategy.value})"
        )
        
        # Set chunking parameters
        chunk_tokens = target_chunk_tokens or settings.chunk_size or 512
        overlap_tokens = chunk_overlap_tokens or settings.chunk_overlap or 50
        
        # Initialize chunking service with parameters
        chunking_service = EnhancedChunkingService(
            target_chunk_tokens=chunk_tokens,
            chunk_overlap_tokens=overlap_tokens
        )
        
        # Get extractor
        extractor = ExtractorFactory.get_extractor(
            file_metadata.file_extension,
            config={
                'do_ocr': settings.enable_ocr,
                'do_table_structure': settings.infer_table_structure,
                'generate_images': settings.extract_images,
                'user_id': user_id,
                'file_id': file_id
            }
        )
        
        # Extract content
        logger.info(f"Extracting content from {file_metadata.file_extension}...")
        extracted_text, success, extraction_metadata = extractor.extract(str(file_path))
        
        if not success or extraction_metadata.get('status') == 'error':
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
        
        # Save markdown
        markdown_content = markdown_converter.convert_text_to_markdown(
            extracted_text,
            file_metadata.original_filename,
            file_metadata.file_extension
        )
        
        user_markdown_dir = Path(settings.markdown_dir) / user_id
        user_markdown_dir.mkdir(parents=True, exist_ok=True)
        
        markdown_path = user_markdown_dir / f"{file_id}.md"
        markdown_saved = markdown_converter.save_markdown_file(
            markdown_content,
            str(markdown_path)
        )
        
        if markdown_saved:
            logger.info(f"✅ Saved markdown: {markdown_path.name}")
        
        # Prepare metadata for chunking
        chunk_metadata = {
            "file_id": file_id,
            "filename": file_metadata.safe_filename,
            "file_type": file_metadata.file_extension,
            "original_filename": file_metadata.original_filename,
            "user_id": user_id
        }
        
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
        
        # Chunk document with selected strategy
        logger.info(f"Creating chunks using {chunking_strategy.value} strategy...")
        chunks = chunking_service.split_document_into_chunks(
            extracted_text,
            simple_structure,
            base_metadata=chunk_metadata,
            strategy=chunking_strategy
        )
        
        if not chunks:
            file_metadata.status = DBProcessingStatus.FAILED
            file_metadata.error_message = "No chunks created"
            db.commit()
            raise HTTPException(
                status_code=422, 
                detail="No chunks created."
            )
        
        # Get chunking statistics
        chunk_stats = chunking_service.get_chunk_stats(chunks)
        logger.info(
            f"✅ Created {len(chunks)} chunks | "
            f"Avg: {chunk_stats['avg_tokens']:.0f} tokens | "
            f"Min: {chunk_stats['min_tokens']} | "
            f"Max: {chunk_stats['max_tokens']}"
        )
        
        # Verify page numbers
        page_numbers = [c['metadata'].get('page_number', 0) for c in chunks]
        logger.info(
            f"Page numbers: min={min(page_numbers)}, "
            f"max={max(page_numbers)}, "
            f"unique={len(set(page_numbers))}"
        )
        
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
                chunk_index=chunk['metadata']['chunk_index'],
                chunk_type=chunk['metadata'].get('chunk_type', 'text'),
                char_count=chunk['char_count'],
                page_number=chunk['metadata'].get('page_number', 0),
                created_at=datetime.utcnow()
            )
            db.add(chunk_db)
        
        # Map ChunkingStrategy to ChunkingStrategyEnum
        strategy_map = {
            ChunkingStrategy.FIXED_SIZE: ChunkingStrategyEnum.FIXED_SIZE,
            ChunkingStrategy.SENTENCE: ChunkingStrategyEnum.SENTENCE,
            ChunkingStrategy.PARAGRAPH: ChunkingStrategyEnum.PARAGRAPH,
            ChunkingStrategy.SLIDING_WINDOW: ChunkingStrategyEnum.SLIDING_WINDOW,
            ChunkingStrategy.RECURSIVE: ChunkingStrategyEnum.RECURSIVE,
            ChunkingStrategy.SEMANTIC: ChunkingStrategyEnum.SEMANTIC
        }
        
        # Update file metadata with results
        file_metadata.status = DBProcessingStatus.COMPLETED
        file_metadata.processed_at = datetime.utcnow()
        file_metadata.chunks_created = chunks_added
        file_metadata.total_pages = extraction_metadata.get('total_pages', 0)
        file_metadata.markdown_path = str(markdown_path) if markdown_saved else None
        file_metadata.word_count = word_count
        file_metadata.image_count = image_count
        file_metadata.table_count = table_count
        file_metadata.chunking_strategy = strategy_map.get(chunking_strategy)
        file_metadata.target_chunk_tokens = chunk_tokens
        file_metadata.chunk_overlap_tokens = overlap_tokens
        
        # Store processing metadata
        processing_meta = {
            'extraction_method': extractor.__class__.__name__,
            'chunks_created': chunks_added,
            'chunking_strategy': chunking_strategy.value,
            'target_chunk_tokens': chunk_tokens,
            'chunk_overlap_tokens': overlap_tokens,
            'chunk_stats': chunk_stats,
            'extraction_metadata': extraction_metadata
        }
        file_metadata.processing_metadata = json.dumps(processing_meta)
        
        db.commit()
        
        processing_time = time.time() - start_time
        
        logger.info(
            f"✅ Processing complete for file_id: {file_id} "
            f"({processing_time:.2f}s)"
        )
        
        # Build response message
        message_parts = [
            f"✅ Document '{file_metadata.safe_filename}' processed successfully:",
            f"  • Format: {file_metadata.file_extension.upper()}",
            f"  • Chunking strategy: {chunking_strategy.value}",
            f"  • Chunks indexed: {chunks_added}",
            f"  • Avg chunk size: {chunk_stats['avg_tokens']:.0f} tokens",
        ]
        
        if table_count > 0:
            message_parts.append(f"  • Tables: {table_count}")
        
        if image_count > 0:
            message_parts.append(f"  • Images: {image_count}")
        
        return DocumentProcessResponse(
            file_id=file_id,
            filename=file_metadata.safe_filename,
            status=ProcessingStatus.COMPLETED,
            markdown_path=str(markdown_path) if markdown_saved else None,
            chunks_created=chunks_added,
            total_pages=extraction_metadata.get('total_pages', 0),
            processing_time_seconds=processing_time,
            statistics={
                'images': image_count,
                'tables': table_count,
                'charts': chart_count,
                'words': word_count
            },
            chunking_strategy=chunking_strategy.value,
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
        "chunking_strategy": file_metadata.chunking_strategy.value if file_metadata.chunking_strategy else None,
        "target_chunk_tokens": file_metadata.target_chunk_tokens,
        "chunk_overlap_tokens": file_metadata.chunk_overlap_tokens,
        "error_message": file_metadata.error_message
    }


@router.get("/chunking-strategies")
async def get_available_chunking_strategies():
    """
    Get list of available chunking strategies with descriptions
    
    Returns information about all 6 chunking strategies
    """
    return {
        "strategies": [
            {
                "name": "fixed_size",
                "display_name": "Fixed-size (Token-based)",
                "description": "Creates chunks of exact token size with no overlap. Simple and efficient.",
                "best_for": "General purpose, consistent chunk sizes",
                "parameters": ["target_chunk_tokens"]
            },
            {
                "name": "sentence",
                "display_name": "Sentence Chunking",
                "description": "Groups sentences together until target size is reached. Preserves sentence boundaries.",
                "best_for": "Documents where sentence context is important",
                "parameters": ["target_chunk_tokens"]
            },
            {
                "name": "paragraph",
                "display_name": "Paragraph Chunking",
                "description": "Groups paragraphs together. Preserves paragraph structure and context.",
                "best_for": "Documents with clear paragraph structure, essays, articles",
                "parameters": ["target_chunk_tokens"]
            },
            {
                "name": "sliding_window",
                "display_name": "Sliding Window",
                "description": "Creates overlapping chunks with specified overlap. Ensures continuity across chunks.",
                "best_for": "When context across chunks is critical, Q&A systems",
                "parameters": ["target_chunk_tokens", "chunk_overlap_tokens"]
            },
            {
                "name": "recursive",
                "display_name": "Recursive Chunking",
                "description": "Hierarchically splits text by separators (sections, paragraphs, sentences, words). Preserves document structure.",
                "best_for": "Structured documents, technical documentation",
                "parameters": ["target_chunk_tokens"]
            },
            {
                "name": "semantic",
                "display_name": "Semantic Chunking",
                "description": "Groups semantically similar sentences together using keyword overlap. Creates contextually coherent chunks.",
                "best_for": "Documents where topic coherence is important, research papers",
                "parameters": ["target_chunk_tokens"]
            }
        ],
        "default_strategy": "fixed_size",
        "default_parameters": {
            "target_chunk_tokens": 512,
            "chunk_overlap_tokens": 50
        }
    }