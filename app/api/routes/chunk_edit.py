"""
Chunk Editing Route - NEW FEATURE
Allows users to edit individual chunks in the vector database
"""

from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field
from typing import Optional
import logging
from datetime import datetime

from app.services.vector_store_service import VectorStoreService
from app.core.dependencies import get_user_id, get_database_session
from app.db.models import ChunkMetadata

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/chunks", tags=["Chunk Management"])

# Initialize vector store
vector_store = VectorStoreService()


class ChunkDetailResponse(BaseModel):
    """Response model for chunk details"""
    chunk_id: str
    file_id: str
    text: str
    filename: str
    user_id: str
    chunk_type: str = "text"
    page_number: int = 0
    char_count: int
    chunk_index: int
    created_at: str
    edited: bool = False
    metadata: dict = {}


class UpdateChunkRequest(BaseModel):
    """Request model for updating a chunk"""
    file_id: str = Field(..., description="Parent file ID (for validation)")
    chunk_id: str = Field(..., description="Unique chunk identifier")
    new_text: str = Field(..., min_length=1, max_length=10000, description="New chunk text")
    preserve_metadata: bool = Field(
        default=True,
        description="Whether to preserve existing metadata"
    )


class UpdateChunkResponse(BaseModel):
    """Response model for chunk update"""
    message: str
    chunk_id: str
    file_id: str
    user_id: str
    old_char_count: int
    new_char_count: int
    updated_at: str


@router.get("/{chunk_id}", response_model=ChunkDetailResponse)
async def get_chunk_details(
    chunk_id: str,
    user_id: str = Depends(get_user_id),
    file_id: Optional[str] = None
):
    """
    Retrieve complete details of a specific chunk
    
    Headers:
        X-User-ID: User identifier (REQUIRED)
    
    Query Parameters:
        file_id: Optional file ID for additional validation
    
    Returns:
        Complete chunk data including:
        - chunk_id: Unique identifier
        - file_id: Parent file ID
        - text: Full chunk text
        - All metadata fields
    """
    try:
        # Retrieve chunk from vector store
        chunk_data = vector_store.get_chunk_by_id(
            chunk_id=chunk_id,
            user_id=user_id
        )
        
        if not chunk_data:
            raise HTTPException(
                status_code=404,
                detail=f"Chunk not found or access denied: {chunk_id}"
            )
        
        # Validate file_id if provided
        if file_id and chunk_data['file_id'] != file_id:
            raise HTTPException(
                status_code=400,
                detail=f"File ID mismatch: chunk belongs to file {chunk_data['file_id']}"
            )
        
        return ChunkDetailResponse(**chunk_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving chunk {chunk_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving chunk: {str(e)}"
        )


@router.put("/edit", response_model=UpdateChunkResponse)
async def edit_chunk(
    request: UpdateChunkRequest,
    user_id: str = Depends(get_user_id),
    db: Session = Depends(get_database_session)
):
    """
    Edit an existing chunk in the vector database
    
    Headers:
        X-User-ID: User identifier (REQUIRED)
    
    Request Body:
        {
            "file_id": "uuid-of-file",
            "chunk_id": "unique-chunk-id", 
            "new_text": "Updated chunk text here...",
            "preserve_metadata": true
        }
    
    Process:
    1. Validates chunk ownership and file_id
    2. Re-embeds the new text (both dense and sparse vectors)
    3. Updates the vector in Qdrant using upsert
    4. Updates metadata in database (if applicable)
    5. Preserves all metadata while replacing vectors
    
    Returns:
        Success message with update details
    """
    try:
        # Retrieve existing chunk to get old char count
        logger.info(
            f"Edit chunk request: chunk_id={request.chunk_id}, "
            f"file_id={request.file_id}, user_id={user_id}"
        )
        
        existing_chunk = vector_store.get_chunk_by_id(
            chunk_id=request.chunk_id,
            user_id=user_id
        )
        
        if not existing_chunk:
            raise HTTPException(
                status_code=404,
                detail=f"Chunk not found or access denied: {request.chunk_id}"
            )
        
        old_char_count = existing_chunk.get('char_count', 0)
        
        # Perform the update in vector store
        success = vector_store.update_chunk(
            chunk_id=request.chunk_id,
            file_id=request.file_id,
            new_text=request.new_text,
            user_id=user_id,
            preserve_metadata=request.preserve_metadata
        )
        
        if not success:
            raise HTTPException(
                status_code=500,
                detail="Failed to update chunk in vector store"
            )
        
        # Update chunk metadata in database (if exists)
        try:
            chunk_meta = db.query(ChunkMetadata).filter(
                ChunkMetadata.chunk_id == request.chunk_id
            ).first()
            
            if chunk_meta:
                chunk_meta.char_count = len(request.new_text)
                # Could add an 'edited' flag or 'updated_at' field if schema supports
                db.commit()
                logger.info(f"Updated chunk metadata in database: {request.chunk_id}")
        except Exception as db_error:
            logger.warning(f"Could not update database metadata: {db_error}")
            # Don't fail the request if DB update fails - vector update succeeded
        
        updated_at = datetime.utcnow().isoformat()
        
        return UpdateChunkResponse(
            message=f"Chunk '{request.chunk_id}' updated successfully",
            chunk_id=request.chunk_id,
            file_id=request.file_id,
            user_id=user_id,
            old_char_count=old_char_count,
            new_char_count=len(request.new_text),
            updated_at=updated_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error editing chunk {request.chunk_id}: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Error editing chunk: {str(e)}"
        )


@router.get("/file/{file_id}/chunks")
async def list_file_chunks(
    file_id: str,
    user_id: str = Depends(get_user_id),
    db: Session = Depends(get_database_session),
    limit: int = 100
):
    """
    List all chunks for a specific file
    
    Headers:
        X-User-ID: User identifier (REQUIRED)
    
    Query Parameters:
        limit: Maximum number of chunks to return (default: 100)
    
    Returns:
        List of chunks with basic metadata
    """
    try:
        from app.db.models import FileMetadata
        
        # Verify file ownership
        file_meta = db.query(FileMetadata).filter(
            FileMetadata.file_id == file_id,
            FileMetadata.user_id == user_id
        ).first()
        
        if not file_meta:
            raise HTTPException(
                status_code=404,
                detail=f"File not found or access denied: {file_id}"
            )
        
        # Get chunks from database
        chunks = db.query(ChunkMetadata).filter(
            ChunkMetadata.file_id == file_id
        ).limit(min(limit, 1000)).all()
        
        chunk_list = []
        for chunk in chunks:
            chunk_list.append({
                "chunk_id": chunk.chunk_id,
                "chunk_index": chunk.chunk_index,
                "chunk_type": chunk.chunk_type,
                "char_count": chunk.char_count,
                "page_number": chunk.page_number,
                "created_at": chunk.created_at.isoformat() if chunk.created_at else None
            })
        
        return {
            "file_id": file_id,
            "filename": file_meta.original_filename,
            "total_chunks": len(chunk_list),
            "chunks": chunk_list
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing chunks for file {file_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error listing chunks: {str(e)}"
        )