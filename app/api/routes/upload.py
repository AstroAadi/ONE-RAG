"""
Updated Upload Route - Character-Based Chunking with User Configuration
- Character-based limits only (no token limits except document parent: 10K)
- User selects delimiter
- User selects embedding model
- User defines maximum chunk sizes (chunks can be smaller)
"""

from typing import Optional
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, Body
from sqlalchemy.orm import Session
from pathlib import Path
import uuid
import shutil
from datetime import datetime
import logging
import json

from app.models.schemas import (
    FileUploadResponse, 
    DocumentProcessResponse, 
    DocumentProcessRequest,
    ProcessingStatus,
    ChunkingStrategy,
    EmbeddingModel
)
from app.utils.file_validator import FileValidator
from app.services.extraction.extractor_factory import ExtractorFactory
from app.utils.markdown_converter import MarkdownConverter
from app.services.chunking_service import CharacterBasedChunkingService
from app.services.vector_store_service import VectorStoreService
from app.core.config import settings
from app.core.dependencies import get_user_id, get_database_session
from app.db.models import FileMetadata, ChunkMetadata, ProcessingStatus as DBProcessingStatus, FileType

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
    - markdown_output/{user_id}/{file_id}.md - Markdown output
    
    Note: Chunking configuration is selected during processing
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
    safe_filename = FileValidator.get_safe_filename(file.filename)
    
    # Create user-specific upload directory
    user_upload_dir = Path(settings.upload_dir) / user_id
    user_upload_dir.mkdir(parents=True, exist_ok=True)
    
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
            message=(
                f"File '{safe_filename}' uploaded successfully. "
                f"Use /process/{file_id} to process with your chosen configuration."
            )
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
    process_request: DocumentProcessRequest = Body(
        default=DocumentProcessRequest(),
        description="Processing configuration with chunking and embedding settings"
    ),
    user_id: str = Depends(get_user_id),
    db: Session = Depends(get_database_session)
):
    """
    Process document and index to vector store with user-selected configuration
    
    CHARACTER-BASED CHUNKING ONLY (No token limits except document parent: 10K tokens)
    
    Body Parameters:
        chunking_strategy: Strategy to use
            - "fixed_character": Fixed character-based chunking with delimiter
            - "parent_child_paragraph": Paragraph-level parents with child chunks
            - "parent_child_document": Document-level parent (10K token limit) with children
        
        max_chunk_chars: Maximum chunk size in characters (500-10000)
            - Chunks can be SMALLER but NEVER LARGER than this limit
        
        overlap_chars: Overlap size in characters (0-2000)
            - Only applies to fixed_character strategy
        
        delimiter: Text delimiter for splitting (e.g., "\\n\\n", "\\n", ".", etc.)
            - Default: "\\n\\n" (double newline = paragraphs)
            - Used by all strategies
        
        max_child_chunk_chars: Maximum child chunk size (200-5000)
            - Only applies to parent-child strategies
            - Children can be SMALLER but NEVER LARGER
        
        embedding_model: Embedding model to use
            - "cohere.embed-english-v3" (default)
            - "cohere.embed-multilingual-v3"
            - "amazon.titan-embed-text-v1"
            - "amazon.titan-embed-text-v2"
    
    Examples:
        # Default configuration
        POST /process/{file_id}
        {}
        
        # Fixed chunking with custom delimiter
        POST /process/{file_id}
        {
            "chunking_strategy": "fixed_character",
            "max_chunk_chars": 2500,
            "overlap_chars": 250,
            "delimiter": "\\n",
            "embedding_model": "cohere.embed-english-v3"
        }
        
        # Parent-child paragraph with custom child size
        POST /process/{file_id}
        {
            "chunking_strategy": "parent_child_paragraph",
            "max_child_chunk_chars": 1200,
            "delimiter": "\\n\\n",
            "embedding_model": "cohere.embed-multilingual-v3"
        }
        
        # Document-level parent (auto-limited to 10K tokens)
        POST /process/{file_id}
        {
            "chunking_strategy": "parent_child_document",
            "max_child_chunk_chars": 800,
            "delimiter": "\\n\\n"
        }
    """
    
    # Get file metadata
    file_metadata = db.query(FileMetadata).filter(
        FileMetadata.file_id == file_id,
        FileMetadata.user_id == user_id
    ).first()
    
    if not file_metadata:
        raise HTTPException(status_code=404, detail=f"File not found for ID: {file_id}")
    
    # Check if already processed
    if file_metadata.status == DBProcessingStatus.COMPLETED:
        processing_meta = {}
        if file_metadata.processing_metadata:
            try:
                processing_meta = json.loads(file_metadata.processing_metadata)
            except:
                pass
        
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
            message=f"Document already processed. {file_metadata.chunks_created} chunks indexed.",
            chunking_strategy=processing_meta.get('chunking_strategy'),
            chunking_config=processing_meta.get('chunking_config'),
            embedding_model=processing_meta.get('embedding_model')
        )
    
    file_path = Path(file_metadata.storage_path)
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found on disk: {file_metadata.safe_filename}")
    
    # Update status to processing
    file_metadata.status = DBProcessingStatus.PROCESSING
    db.commit()
    
    try:
        start_time = datetime.utcnow()
        
        logger.info(
            f"📄 Processing: {file_metadata.safe_filename}\n"
            f"  Strategy: {process_request.chunking_strategy.value}\n"
            f"  Max chunk: {process_request.max_chunk_chars} chars\n"
            f"  Delimiter: {repr(process_request.delimiter)}\n"
            f"  Embedding: {process_request.embedding_model.value}"
        )
        
        # Extract content
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
        
        extracted_text, extraction_metadata = extractor.extract_content(str(file_path))
        
        if extraction_metadata.get('success') == False:
            error_msg = extraction_metadata.get('error', 'Unknown extraction error')
            file_metadata.status = DBProcessingStatus.FAILED
            file_metadata.error_message = error_msg
            db.commit()
            raise HTTPException(status_code=422, detail=f"Failed to extract content: {error_msg}")
        
        if not extracted_text or not extracted_text.strip():
            file_metadata.status = DBProcessingStatus.FAILED
            file_metadata.error_message = "No content extracted"
            db.commit()
            raise HTTPException(status_code=422, detail="No content extracted from document.")
        
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
        markdown_saved = markdown_converter.save_markdown_file(markdown_content, str(markdown_path))
        
        if markdown_saved:
            logger.info(f"✅ Saved markdown: {markdown_path.name}")
        
        # Prepare metadata
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
        
        simple_structure = {
            'statistics': {
                'images': image_count,
                'tables': table_count,
                'charts': chart_count
            }
        }
        
        # ========== CREATE CHUNKING SERVICE WITH USER CONFIG ==========
        
        chunking_service = CharacterBasedChunkingService(
            max_chunk_chars=process_request.max_chunk_chars,
            overlap_chars=process_request.overlap_chars,
            delimiter=process_request.delimiter,
            max_child_chunk_chars=process_request.max_child_chunk_chars,
            max_document_parent_tokens=10000,  # Fixed: 10K token limit for document parent
            tokenizer_model="cl100k_base"
        )
        
        logger.info(
            f"Chunking with:\n"
            f"  Strategy: {process_request.chunking_strategy.value}\n"
            f"  Max chunk: {process_request.max_chunk_chars} chars\n"
            f"  Max child: {process_request.max_child_chunk_chars} chars\n"
            f"  Overlap: {process_request.overlap_chars} chars\n"
            f"  Delimiter: {repr(process_request.delimiter)}"
        )
        
        # Chunk document
        chunks = chunking_service.split_document_into_chunks(
            markdown_content,
            simple_structure,
            base_metadata=chunk_metadata,
            strategy=process_request.chunking_strategy,
            embedding_model=process_request.embedding_model.value
        )
        
        if not chunks:
            file_metadata.status = DBProcessingStatus.FAILED
            file_metadata.error_message = "No chunks created"
            db.commit()
            raise HTTPException(status_code=422, detail="No chunks created.")
        
        logger.info(f"✅ Created {len(chunks)} chunks")
        
        # Get statistics
        chunk_stats = chunking_service.get_chunk_stats(chunks)
        
        # Count parent/child
        parent_chunks = [c for c in chunks if c['metadata'].get('is_parent', False)]
        child_chunks = [c for c in chunks if not c['metadata'].get('is_parent', False)]
        
        chunk_breakdown = None
        if parent_chunks:
            chunk_breakdown = {
                'parent_chunks': len(parent_chunks),
                'child_chunks': len(child_chunks),
                'total_chunks': len(chunks)
            }
            logger.info(f"  Breakdown: {len(parent_chunks)} parents, {len(child_chunks)} children")
        
        # Index to vector store
        logger.info(f"Indexing to vector store with embedding model: {process_request.embedding_model.value}")
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
                is_parent=chunk['metadata'].get('is_parent'),
                parent_chunk_id=chunk['metadata'].get('parent_chunk_id'),
                child_index=chunk['metadata'].get('child_index'),
                created_at=datetime.utcnow()
            )
            db.add(chunk_db)
        
        # Calculate processing time
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        # Update file metadata
        file_metadata.status = DBProcessingStatus.COMPLETED
        file_metadata.processed_at = datetime.utcnow()
        file_metadata.chunks_created = chunks_added
        file_metadata.total_pages = extraction_metadata.get('total_pages', 0)
        file_metadata.markdown_path = str(markdown_path) if markdown_saved else None
        file_metadata.word_count = word_count
        file_metadata.image_count = image_count
        file_metadata.table_count = table_count
        
        # Store processing metadata
        processing_meta = {
            'extraction_method': extractor.__class__.__name__,
            'chunks_created': chunks_added,
            'extraction_metadata': extraction_metadata,
            'chunking_strategy': process_request.chunking_strategy.value,
            'chunking_config': {
                'max_chunk_chars': process_request.max_chunk_chars,
                'overlap_chars': process_request.overlap_chars,
                'delimiter': process_request.delimiter,
                'max_child_chunk_chars': process_request.max_child_chunk_chars
            },
            'embedding_model': process_request.embedding_model.value,
            'chunk_breakdown': chunk_breakdown,
            'chunk_stats': {
                'avg_chars': chunk_stats.get('avg_chars', 0),
                'min_chars': chunk_stats.get('min_chars', 0),
                'max_chars': chunk_stats.get('max_chars', 0),
                'total_chars': chunk_stats.get('total_chars', 0)
            }
        }
        file_metadata.processing_metadata = json.dumps(processing_meta)
        
        db.commit()
        
        logger.info(f"✅ Processing complete for file_id: {file_id}")
        
        # Build response message
        message_parts = [
            f"✅ Document '{file_metadata.safe_filename}' processed successfully:",
            f"  • Format: {file_metadata.file_extension.upper()}",
            f"  • Chunking: {process_request.chunking_strategy.value}",
            f"  • Max chunk: {process_request.max_chunk_chars} chars",
            f"  • Delimiter: {repr(process_request.delimiter)}",
            f"  • Embedding: {process_request.embedding_model.value}",
            f"  • Chunks: {chunks_added}",
        ]
        
        if chunk_breakdown:
            message_parts.append(f"  • Parents: {chunk_breakdown['parent_chunks']}")
            message_parts.append(f"  • Children: {chunk_breakdown['child_chunks']}")
        
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
            chunking_strategy=process_request.chunking_strategy.value,
            chunking_config=processing_meta['chunking_config'],
            embedding_model=process_request.embedding_model.value,
            chunk_breakdown=chunk_breakdown,
            chunk_stats=chunk_stats,
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
        
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


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
        raise HTTPException(status_code=404, detail=f"File not found for ID: {file_id}")
    
    processing_meta = {}
    if file_metadata.processing_metadata:
        try:
            processing_meta = json.loads(file_metadata.processing_metadata)
        except:
            pass
    
    return {
        "file_id": file_id,
        "filename": file_metadata.safe_filename,
        "status": file_metadata.status.value,
        "uploaded_at": file_metadata.uploaded_at,
        "processed_at": file_metadata.processed_at,
        "chunks_created": file_metadata.chunks_created,
        "total_pages": file_metadata.total_pages,
        "error_message": file_metadata.error_message,
        "chunking_strategy": processing_meta.get('chunking_strategy'),
        "chunking_config": processing_meta.get('chunking_config'),
        "embedding_model": processing_meta.get('embedding_model'),
        "chunk_breakdown": processing_meta.get('chunk_breakdown')
    }


@router.get("/chunking-options")
async def get_chunking_options():
    """
    Get available chunking strategies, delimiters, and embedding models
    """
    return {
        "strategies": [
            {
                "name": ChunkingStrategy.FIXED_CHARACTER.value,
                "display_name": "Fixed Character Chunking",
                "description": "Split by delimiter, combine until max chars, apply overlap",
                "parameters": ["max_chunk_chars", "overlap_chars", "delimiter"],
                "note": "Chunks can be smaller than max, never larger",
                "default": True
            },
            {
                "name": ChunkingStrategy.PARENT_CHILD_PARAGRAPH.value,
                "display_name": "Paragraph-Level Parent-Child",
                "description": "Each paragraph is parent, children split by delimiter up to max chars",
                "parameters": ["max_child_chunk_chars", "delimiter"],
                "note": "No size limit on parents, children limited to max chars"
            },
            {
                "name": ChunkingStrategy.PARENT_CHILD_DOCUMENT.value,
                "display_name": "Document-Level Parent-Child",
                "description": "Full document is parent (10K token limit), children split by delimiter",
                "parameters": ["max_child_chunk_chars", "delimiter"],
                "note": "Parent auto-truncated to 10,000 tokens if needed"
            }
        ],
        "common_delimiters": [
            {"value": "\\n\\n", "name": "Double newline (paragraphs)", "default": True},
            {"value": "\\n", "name": "Single newline (lines)"},
            {"value": ". ", "name": "Period + space (sentences)"},
            {"value": "\\n---\\n", "name": "Horizontal rule (sections)"},
            {"value": "\\n# ", "name": "Markdown header"}
        ],
        "embedding_models": [
            {
                "value": EmbeddingModel.COHERE_ENGLISH_V3.value,
                "name": "Cohere English v3",
                "dimensions": 1024,
                "max_input": 512,
                "default": True
            },
            {
                "value": EmbeddingModel.COHERE_MULTILINGUAL_V3.value,
                "name": "Cohere Multilingual v3",
                "dimensions": 1024,
                "max_input": 512
            },
            {
                "value": EmbeddingModel.AMAZON_TITAN_V1.value,
                "name": "Amazon Titan v1",
                "dimensions": 1536,
                "max_input": 8192
            },
            {
                "value": EmbeddingModel.AMAZON_TITAN_V2.value,
                "name": "Amazon Titan v2",
                "dimensions": 1024,
                "max_input": 8192
            }
        ],
        "limits": {
            "max_chunk_chars": {"min": 500, "max": 10000, "default": 2000},
            "overlap_chars": {"min": 0, "max": 2000, "default": 200},
            "max_child_chunk_chars": {"min": 200, "max": 5000, "default": 1000},
            "document_parent_tokens": 10000  # Fixed limit for document-level parent
        }
    }