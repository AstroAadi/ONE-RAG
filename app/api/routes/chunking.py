"""
Chunking Routes - preview / create / finalize

These three endpoints implement the second and third phases of ingestion:

    POST /api/chunks/preview   chunk in memory and return a sample (no writes)
    POST /api/chunks/create    chunk and stage the result for finalization
    POST /api/chunks/finalize  embed the staged chunks and index them

Splitting "create" from "finalize" lets the user review chunk boundaries and
re-chunk with different settings before paying the embedding cost.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import logging
import time

from fastapi import APIRouter, Body, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.dependencies import get_user_id, get_database_session
from app.db.models import (
    ChunkingStrategyEnum,
    ChunkMetadata,
    FileMetadata,
    ProcessingStatus as DBProcessingStatus,
)
from app.models.schemas import (
    ChunkingRequest,
    ChunkingStrategy,
    CreateChunksResponse,
    FinalizeChunksResponse,
    PreviewChunk,
    PreviewChunksResponse,
)
from app.services.chunk_staging_service import (
    delete_staged_chunks,
    load_staged_chunks,
    save_staged_chunks,
)
from app.services.chunking_service import EnhancedChunkingService
from app.services.vector_store_service import VectorStoreService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/chunks", tags=["Chunking"])

vector_store = VectorStoreService()

# Number of chunks returned by the preview endpoint
PREVIEW_LIMIT = 5

# Characters of chunk text included per preview entry
PREVIEW_CHAR_LIMIT = 1200

STRATEGY_DB_ENUM = {
    ChunkingStrategy.DEFAULT: ChunkingStrategyEnum.DEFAULT,
    ChunkingStrategy.FIXED_SIZE: ChunkingStrategyEnum.FIXED_SIZE,
    ChunkingStrategy.HIERARCHICAL: ChunkingStrategyEnum.HIERARCHICAL,
    ChunkingStrategy.SEMANTIC: ChunkingStrategyEnum.SEMANTIC,
    ChunkingStrategy.NO_CHUNKING: ChunkingStrategyEnum.NO_CHUNKING,
}


# ==================== Helpers ====================

def _load_extracted_file(
    db: Session,
    file_id: str,
    user_id: str
) -> FileMetadata:
    """
    Fetch a file that is ready to be chunked.

    Chunking requires a completed extraction, so this rejects files that are
    still uploading/extracting or whose extraction failed.
    """
    file_metadata = db.query(FileMetadata).filter(
        FileMetadata.file_id == file_id,
        FileMetadata.user_id == user_id
    ).first()

    if not file_metadata:
        raise HTTPException(status_code=404, detail=f"File not found: {file_id}")

    if file_metadata.status == DBProcessingStatus.PROCESSING:
        raise HTTPException(
            status_code=409,
            detail=(
                "Document extraction is still running. "
                f"Poll /api/upload/status/{file_id} until it completes."
            )
        )

    if file_metadata.status == DBProcessingStatus.FAILED:
        raise HTTPException(
            status_code=409,
            detail=(
                "Document extraction failed: "
                f"{file_metadata.error_message or 'unknown error'}"
            )
        )

    if file_metadata.status != DBProcessingStatus.COMPLETED:
        raise HTTPException(
            status_code=409,
            detail=(
                "Document has not been extracted yet. "
                f"Poll /api/upload/status/{file_id} until it completes."
            )
        )

    return file_metadata


def _read_markdown(file_metadata: FileMetadata) -> str:
    """Read the extracted markdown for a document"""
    if not file_metadata.markdown_path:
        raise HTTPException(
            status_code=409,
            detail="No extracted markdown is available for this document"
        )

    markdown_path = Path(file_metadata.markdown_path)

    if not markdown_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Extracted markdown file is missing; re-upload the document"
        )

    try:
        return markdown_path.read_text(encoding="utf-8")
    except OSError as e:
        logger.error(f"Could not read markdown {markdown_path}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Could not read extracted markdown: {str(e)}"
        )


def _resolve_parameters(request: ChunkingRequest) -> Dict[str, Any]:
    """
    Resolve the effective chunking parameters for the requested strategy.

    Only the parameters that apply to the chosen strategy are returned, so the
    response documents exactly what was used.
    """
    strategy = request.chunking_strategy

    if strategy in (ChunkingStrategy.DEFAULT, ChunkingStrategy.NO_CHUNKING):
        return {}

    if strategy == ChunkingStrategy.FIXED_SIZE:
        return {
            "chunk_size": request.chunk_size or settings.target_chunk_tokens,
            "chunk_overlap": (
                request.chunk_overlap
                if request.chunk_overlap is not None
                else settings.chunk_overlap_tokens
            ),
        }

    if strategy == ChunkingStrategy.SEMANTIC:
        return {
            "chunk_size": request.chunk_size or settings.target_chunk_tokens,
            "chunk_overlap": (
                request.chunk_overlap
                if request.chunk_overlap is not None
                else settings.chunk_overlap_tokens
            ),
            "max_buffer_size": request.max_buffer_size or 8,
            "breakpoint_threshold": (
                request.breakpoint_threshold
                if request.breakpoint_threshold is not None
                else 0.25
            ),
        }

    # Hierarchical
    return {
        "parent_chunk_size": request.parent_chunk_size or (
            settings.target_chunk_tokens * 4
        ),
        "child_chunk_size": request.child_chunk_size or settings.target_chunk_tokens,
        "child_chunk_overlap": (
            request.child_chunk_overlap
            if request.child_chunk_overlap is not None
            else settings.chunk_overlap_tokens
        ),
    }


def _build_chunks(
    markdown: str,
    request: ChunkingRequest,
    file_metadata: FileMetadata,
    parameters: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Run the chunking service with the resolved parameters.

    Chunk IDs are namespaced by file_id because the chunking service derives
    them from content and index alone, which could otherwise collide across
    two documents that share identical text.
    """
    strategy = request.chunking_strategy

    target_tokens = parameters.get("chunk_size", settings.target_chunk_tokens)

    # The smallest chunk the request will actually produce. For hierarchical
    # chunking that is the child size, which is typically well below the
    # document-level target.
    smallest_chunk = parameters.get("child_chunk_size", target_tokens)

    # The configured minimum is a floor for the default chunk size; applying it
    # unchanged to a smaller requested size would discard every chunk, so scale
    # it down to fit. Likewise the maximum must leave room for the request.
    min_chunk_tokens = min(settings.min_chunk_tokens, max(1, smallest_chunk // 4))
    max_chunk_tokens = max(settings.max_chunk_tokens, target_tokens * 2)

    chunker = EnhancedChunkingService(
        target_chunk_tokens=target_tokens,
        chunk_overlap_tokens=parameters.get(
            "chunk_overlap", settings.chunk_overlap_tokens
        ),
        tokenizer_model=settings.tokenizer_model,
        min_chunk_tokens=min_chunk_tokens,
        max_chunk_tokens=max_chunk_tokens,
    )

    base_metadata = {
        "user_id": file_metadata.user_id,
        "file_id": file_metadata.file_id,
        "knowledge_base_id": file_metadata.knowledge_base_id or "",
        "filename": file_metadata.safe_filename,
    }

    chunks = chunker.split_document_into_chunks(
        markdown,
        base_metadata=base_metadata,
        strategy=strategy,
        parent_chunk_tokens=parameters.get("parent_chunk_size"),
        child_chunk_tokens=parameters.get("child_chunk_size"),
        child_chunk_overlap_tokens=parameters.get("child_chunk_overlap"),
        breakpoint_threshold=parameters.get("breakpoint_threshold"),
        max_buffer_size=parameters.get("max_buffer_size"),
    )

    # Namespace chunk IDs to this file
    prefix = file_metadata.file_id
    for position, chunk in enumerate(chunks):
        chunk["chunk_id"] = f"{prefix}_{chunk['chunk_id']}_{position}"
        chunk.setdefault("metadata", {})
        chunk["metadata"]["chunk_index"] = chunk["metadata"].get("chunk_index", position)
        chunk["chunk_index"] = chunk["metadata"]["chunk_index"]

        parent_id = chunk["metadata"].get("parent_chunk_id")
        if parent_id:
            chunk["metadata"]["parent_chunk_id"] = f"{prefix}_{parent_id}"

    return chunks


def _is_embedding_failure(message: str) -> bool:
    """Whether an indexing error came from the embedding step."""
    lowered = message.lower()
    return any(
        marker in lowered
        for marker in ("embedding", "empty vectors", "402", "payment required")
    )


def _embedding_remedy(message: str) -> str:
    """Turn an embedding failure into a concrete next step."""
    lowered = message.lower()

    if "402" in lowered or "payment required" in lowered or "credits" in lowered:
        return (
            "The HuggingFace Inference API reports no remaining credits. "
            "Set EMBEDDING_BACKEND=local in the backend .env to run the "
            "embedding model on this machine instead (no credits needed), "
            "or add credits to the HuggingFace account."
        )

    if "401" in lowered or "403" in lowered or "token" in lowered:
        return (
            "The HuggingFace API rejected the credentials. Check "
            "HUGGINGFACE_API_TOKEN in the backend .env, or set "
            "EMBEDDING_BACKEND=local to embed on this machine."
        )

    if "could not load local embedding model" in lowered:
        return (
            "The local embedding model could not be loaded. Confirm there is "
            "disk space for the model download and network access to "
            "huggingface.co on first run."
        )

    return f"Underlying error: {message}"


def _to_preview_chunk(chunk: Dict[str, Any]) -> PreviewChunk:
    """Map an internal chunk dict to the preview response shape"""
    text = chunk.get("text", "")
    metadata = chunk.get("metadata", {})

    return PreviewChunk(
        chunk_id=str(chunk.get("chunk_id", "")),
        chunk_index=int(metadata.get("chunk_index", 0) or 0),
        content_preview=text[:PREVIEW_CHAR_LIMIT],
        char_count=len(text[:PREVIEW_CHAR_LIMIT]),
        full_char_count=int(chunk.get("char_count", len(text))),
        token_count=int(chunk.get("token_count", 0) or 0),
        page_number=int(metadata.get("page_number", 0) or 0),
        chunk_type=str(metadata.get("chunk_type", "text")),
        parent_chunk_id=metadata.get("parent_chunk_id"),
    )


def _build_preview_tree(chunks: List[Dict[str, Any]]) -> List[PreviewChunk]:
    """
    Shape the preview so hierarchical results show parents with their children.

    For flat strategies this returns the first PREVIEW_LIMIT chunks. For
    hierarchical chunking, children are grouped under synthetic parent entries
    so the client can render the hierarchy it was given.
    """
    has_parents = any(
        chunk.get("metadata", {}).get("parent_chunk_id") for chunk in chunks
    )

    if not has_parents:
        return [_to_preview_chunk(chunk) for chunk in chunks[:PREVIEW_LIMIT]]

    # Preserve parent order of first appearance
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for chunk in chunks:
        parent_id = chunk.get("metadata", {}).get("parent_chunk_id") or "ungrouped"
        grouped.setdefault(parent_id, []).append(chunk)

    parents: List[PreviewChunk] = []

    for parent_id, children in list(grouped.items())[:PREVIEW_LIMIT]:
        child_previews = [_to_preview_chunk(child) for child in children]

        combined_chars = sum(
            int(child.get("char_count", 0) or 0) for child in children
        )
        combined_tokens = sum(
            int(child.get("token_count", 0) or 0) for child in children
        )
        first_child_text = children[0].get("text", "") if children else ""

        parents.append(PreviewChunk(
            chunk_id=parent_id,
            chunk_index=len(parents),
            content_preview=first_child_text[:PREVIEW_CHAR_LIMIT],
            char_count=len(first_child_text[:PREVIEW_CHAR_LIMIT]),
            full_char_count=combined_chars,
            token_count=combined_tokens,
            page_number=int(
                children[0].get("metadata", {}).get("page_number", 0) or 0
            ) if children else 0,
            chunk_type="parent",
            children=child_previews,
        ))

    return parents


# ==================== Endpoints ====================

@router.post("/preview", response_model=PreviewChunksResponse)
async def preview_chunks(
    request: ChunkingRequest,
    user_id: str = Depends(get_user_id),
    db: Session = Depends(get_database_session)
):
    """
    Chunk a document in memory and return a sample.

    Nothing is written: this is a dry run so the user can compare strategies
    and parameters before committing. Safe to call repeatedly.
    """
    file_metadata = _load_extracted_file(db, request.file_id, user_id)
    markdown = _read_markdown(file_metadata)

    parameters = _resolve_parameters(request)

    try:
        chunks = _build_chunks(markdown, request, file_metadata, parameters)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Preview chunking failed for {request.file_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Chunk preview failed: {str(e)}"
        )

    if not chunks:
        raise HTTPException(
            status_code=422,
            detail=(
                "Chunking produced no chunks. The document may be empty or the "
                "parameters may be too restrictive."
            )
        )

    preview = _build_preview_tree(chunks)

    return PreviewChunksResponse(
        file_id=request.file_id,
        chunking_strategy=request.chunking_strategy.value,
        total_chunks=len(chunks),
        preview_count=len(preview),
        preview_chunks=preview,
        message=(
            f"Previewing {len(preview)} of {len(chunks)} chunks "
            f"({request.chunking_strategy.value}). Nothing was saved."
        )
    )


@router.post("/create", response_model=CreateChunksResponse)
async def create_chunks(
    request: ChunkingRequest,
    user_id: str = Depends(get_user_id),
    db: Session = Depends(get_database_session)
):
    """
    Chunk a document and stage the result for finalization.

    Chunks are persisted to staging and recorded in chunk_metadata, but they
    are NOT embedded or searchable until /finalize runs. Re-running this
    replaces any previously staged chunks for the document.
    """
    file_metadata = _load_extracted_file(db, request.file_id, user_id)
    markdown = _read_markdown(file_metadata)

    parameters = _resolve_parameters(request)

    start = time.time()

    try:
        chunks = _build_chunks(markdown, request, file_metadata, parameters)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chunking failed for {request.file_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Chunk creation failed: {str(e)}"
        )

    if not chunks:
        raise HTTPException(
            status_code=422,
            detail=(
                "Chunking produced no chunks. The document may be empty or the "
                "parameters may be too restrictive."
            )
        )

    chunking_time = time.time() - start

    try:
        save_staged_chunks(
            user_id=user_id,
            file_id=request.file_id,
            chunks=chunks,
            chunking_strategy=request.chunking_strategy.value,
            parameters=parameters,
        )
    except OSError as e:
        logger.error(f"Could not stage chunks for {request.file_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Could not stage chunks: {str(e)}"
        )

    try:
        # Replace any chunk metadata from a previous chunking run
        db.query(ChunkMetadata).filter(
            ChunkMetadata.file_id == request.file_id
        ).delete(synchronize_session=False)

        for chunk in chunks:
            metadata = chunk.get("metadata", {})
            db.add(ChunkMetadata(
                chunk_id=chunk["chunk_id"],
                file_id=request.file_id,
                chunk_index=int(metadata.get("chunk_index", 0) or 0),
                chunk_type=str(metadata.get("chunk_type", "text")),
                char_count=int(chunk.get("char_count", 0) or 0),
                page_number=int(metadata.get("page_number", 0) or 0),
            ))

        file_metadata.chunks_staged = len(chunks)
        file_metadata.chunking_strategy = STRATEGY_DB_ENUM.get(
            request.chunking_strategy, ChunkingStrategyEnum.DEFAULT
        )

        if "chunk_size" in parameters:
            file_metadata.target_chunk_tokens = parameters["chunk_size"]
        if "chunk_overlap" in parameters:
            file_metadata.chunk_overlap_tokens = parameters["chunk_overlap"]

        # These chunks are not indexed yet
        file_metadata.chunks_created = 0
        file_metadata.indexed_at = None

        db.commit()
    except Exception as e:
        db.rollback()
        logger.error(f"Could not record chunk metadata for {request.file_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Could not record chunk metadata: {str(e)}"
        )

    chunk_ids = [chunk["chunk_id"] for chunk in chunks]

    logger.info(
        f"✅ Staged {len(chunks)} chunks for {request.file_id} "
        f"({request.chunking_strategy.value}) in {chunking_time:.2f}s"
    )

    return CreateChunksResponse(
        file_id=request.file_id,
        knowledge_base_id=file_metadata.knowledge_base_id,
        chunks_created=len(chunks),
        chunk_ids=chunk_ids[:100],
        total_chunk_ids=len(chunk_ids),
        chunking_strategy=request.chunking_strategy.value,
        chunking_time_seconds=round(chunking_time, 3),
        status="chunked",
        parameters=parameters,
        message=(
            f"Created {len(chunks)} chunks. "
            f"Call /api/chunks/finalize to embed and index them."
        )
    )


@router.post("/finalize", response_model=FinalizeChunksResponse)
async def finalize_chunks(
    file_id: str = Query(..., description="File whose staged chunks to index"),
    batch_size: int = Query(
        default=50, ge=1, le=500,
        description="Chunks embedded per batch"
    ),
    embedding_model: Optional[str] = Query(
        default=None,
        description="Embedding model id (defaults to the configured model)"
    ),
    chunk_ids: Optional[List[str]] = Body(
        default=None,
        description="Optional subset of staged chunk IDs to index"
    ),
    user_id: str = Depends(get_user_id),
    db: Session = Depends(get_database_session)
):
    """
    Embed the staged chunks for a document and index them into the vector store.

    After this succeeds the document is searchable and its staging file is
    cleared. The configured embedding model is used; `embedding_model` is
    accepted and echoed back but changing it requires re-indexing every
    document in the collection, so a mismatch is rejected.
    """
    file_metadata = db.query(FileMetadata).filter(
        FileMetadata.file_id == file_id,
        FileMetadata.user_id == user_id
    ).first()

    if not file_metadata:
        raise HTTPException(status_code=404, detail=f"File not found: {file_id}")

    staged = load_staged_chunks(user_id, file_id)

    if not staged or not staged.get("chunks"):
        raise HTTPException(
            status_code=409,
            detail=(
                "No staged chunks for this document. "
                "Call /api/chunks/create before finalizing."
            )
        )

    chunks: List[Dict[str, Any]] = staged["chunks"]

    # Optional subset selection
    if chunk_ids:
        requested = set(chunk_ids)
        chunks = [chunk for chunk in chunks if chunk.get("chunk_id") in requested]

        if not chunks:
            raise HTTPException(
                status_code=422,
                detail="None of the requested chunk_ids are staged for this document"
            )

    # The vector store collection is built for one embedding model; indexing a
    # document with a different one would make its vectors incomparable.
    effective_model = embedding_model or settings.embedding_model_id
    if effective_model != settings.embedding_model_id:
        raise HTTPException(
            status_code=409,
            detail=(
                f"The vector collection is indexed with "
                f"'{settings.embedding_model_id}'. Change the embedding model "
                f"via /api/settings/models/embedding and re-index all documents "
                f"before using '{effective_model}'."
            )
        )

    start = time.time()
    indexed = 0

    try:
        for offset in range(0, len(chunks), batch_size):
            batch = chunks[offset:offset + batch_size]

            indexed += vector_store.add_document_chunks(
                batch,
                file_id,
                file_metadata.safe_filename,
                structured_data={
                    "file_id": file_id,
                    "filename": file_metadata.safe_filename,
                    "chunks": len(batch),
                }
            )

            logger.info(
                f"Indexed {indexed}/{len(chunks)} chunks for {file_id}"
            )
    except Exception as e:
        logger.error(f"Indexing failed for {file_id}: {e}", exc_info=True)

        # The staging file is deliberately left in place so the client can
        # retry finalization once the underlying problem is resolved.
        message = str(e)

        if _is_embedding_failure(message):
            raise HTTPException(
                status_code=503,
                detail=(
                    "Embeddings are unavailable, so nothing was indexed. "
                    "Your chunks are still saved - retry finalizing once this "
                    "is resolved. "
                    f"{_embedding_remedy(message)}"
                )
            )

        raise HTTPException(
            status_code=500,
            detail=f"Failed to index chunks: {message}"
        )

    indexing_time = time.time() - start

    try:
        file_metadata.chunks_created = indexed
        file_metadata.chunks_staged = 0
        file_metadata.indexed_at = datetime.utcnow()
        file_metadata.status = DBProcessingStatus.COMPLETED
        db.commit()
    except Exception as e:
        db.rollback()
        logger.error(f"Could not update file metadata after indexing: {e}")

    # Staged chunks are now in the vector store
    delete_staged_chunks(user_id, file_id)

    logger.info(
        f"🎉 Finalized {file_id}: {indexed} chunks indexed in {indexing_time:.2f}s"
    )

    return FinalizeChunksResponse(
        file_id=file_id,
        knowledge_base_id=file_metadata.knowledge_base_id,
        indexed_chunks=indexed,
        embedding_model=effective_model,
        batch_size=batch_size,
        indexing_time_seconds=round(indexing_time, 3),
        status="completed",
        message=(
            f"Indexed {indexed} chunks. The document is now searchable."
        )
    )
