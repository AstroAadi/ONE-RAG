"""
Chunks API
"""

from fastapi import APIRouter, HTTPException, Depends, Body
from app.models.schemas import ChunkingRequest
from sqlalchemy.orm import Session
from typing import Any, Optional, List, Dict
from datetime import datetime
from pathlib import Path
import logging

from app.core.dependencies import get_user_id, get_database_session
from app.core.task_manager import get_task_manager, TaskType
from app.db.models import (
    FileMetadata, ChunkMetadata, KnowledgeBase,
    ProcessingStatus, ChunkingStrategyEnum
)
from app.services.chunking_service import EnhancedChunkingService, ChunkingStrategy
from app.services.vector_store_service import VectorStoreService

logger = logging.getLogger(__name__)
router = APIRouter()


# ==================== ZIP HELPER ====================

def _get_zip_file_chunks(file_metadata, context: str = "") -> Optional[List[Dict]]:
    """
    If the file is a ZIP, call ZIPExtractor.extract_chunks() and return the
    per-file chunk list.  Returns None for non-ZIP files or on any error so
    callers can gracefully fall back to flat-markdown chunking.
    """
    if not file_metadata.safe_filename.lower().endswith('.zip'):
        return None
    try:
        from app.services.extraction.extractors.zip_extractor import ZIPExtractor
        zip_extractor = ZIPExtractor()
        zip_file_chunks, zip_ok, _ = zip_extractor.extract_chunks(file_metadata.file_path)
        if not zip_ok or not zip_file_chunks:
            logger.warning(f"ZIP extract_chunks returned no data{' (' + context + ')' if context else ''} "
                           f"— falling back to flat markdown")
            return None
        logger.info(f"ZIP extract_chunks: {len(zip_file_chunks)} file(s) extracted{' (' + context + ')' if context else ''}")
        return zip_file_chunks
    except Exception as e:
        logger.warning(f"ZIP extract_chunks error{' (' + context + ')' if context else ''}: {e} "
                       f"— falling back to flat markdown")
        return None


# ==================== PREVIEW ENDPOINT ====================

@router.post("/preview", tags=["Chunks - Preview"])
async def preview_chunks(
    request: ChunkingRequest = Body(...),
    user_id: str = Depends(get_user_id),
    db: Session = Depends(get_database_session)
):
    """
    Preview the first few chunks that would be created for a given file and chunking strategy.
    For hierarchical strategy → returns nested structure (parents containing children).
    For other strategies → returns flat list like before.

        Example request bodies:
        1. Default:
        {
            "file_id": "example_file_id",
            "chunking_strategy": "default"
        }
        2. Fixed Size:
        {
            "file_id": "example_file_id",
            "chunking_strategy": "fixed_size",
            "chunk_size": 1000,
            "chunk_overlap": 100
        }
        3. Semantic:
        {
            "file_id": "example_file_id",
            "chunking_strategy": "semantic",
            "chunk_size": 1000,
            "chunk_overlap": 100,
            "max_buffer_size": 256,
            "breakpoint_threshold": 0.8
        }
        4. Hierarchical:
        {
            "file_id": "example_file_id",
            "chunking_strategy": "hierarchical",
            "parent_chunk_size": 500,
            "child_chunk_size": 200,
            "child_chunk_overlap": 30
        }
        5. No Chunking:
        {
            "file_id": "example_file_id",
            "chunking_strategy": "no_chunking"
        }
        6. JSON:
        {
            "file_id": "example_file_id",
            "chunking_strategy": "json",
            "child_chunk_size": 100,
            "context_prefix": true
        }
    """
    import time

    strategy = getattr(request, "chunking_strategy", "default")

    # ── DB lookups ───────────────────────────────────────────────────────────
    file_metadata = db.query(FileMetadata).filter(
        FileMetadata.file_id == request.file_id,
        FileMetadata.user_id == user_id
    ).first()
    if not file_metadata:
        raise HTTPException(status_code=404, detail=f"File not found: {request.file_id}")

    kb = db.query(KnowledgeBase).filter(
        KnowledgeBase.knowledge_base_id == file_metadata.knowledge_base_id,
        KnowledgeBase.user_id == user_id
    ).first()
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")

    if file_metadata.status != ProcessingStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"File must be extracted first. Current status: {file_metadata.status.value}"
        )

    # ── CSV / TSV fast-path: no markdown exists → sample rows from Qdrant ────
    #
    # Files processed via the CSV row-level pipeline (csv_extractor.py) are
    # stored directly in Qdrant during extraction and never produce a markdown
    # file.  For these files we skip all chunking logic and instead fetch a
    # small sample of already-indexed points so the caller gets a meaningful
    # preview regardless of which chunking_strategy value was sent.
    if not file_metadata.markdown_path:
        if file_metadata.status != ProcessingStatus.COMPLETED:
            # Should already be caught above, but be explicit.
            raise HTTPException(
                status_code=400,
                detail="File extraction has not completed yet."
            )

        SAMPLE_LIMIT = 5
        start_time = time.time()

        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue

            vector_store = VectorStoreService()
            points, _ = vector_store.client.scroll(
                collection_name=vector_store.collection_name,
                scroll_filter=Filter(must=[
                    FieldCondition(key="file_id",  match=MatchValue(value=request.file_id)),
                    FieldCondition(key="user_id",  match=MatchValue(value=user_id)),
                ]),
                limit=SAMPLE_LIMIT,
                with_payload=True,
                with_vectors=False,
            )
        except Exception as e:
            logger.error(f"❌ Qdrant sample fetch failed for {request.file_id}: {e}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Could not fetch sample rows from vector store: {e}"
            )

        preview_time = time.time() - start_time
        total_stored = file_metadata.chunks_created or 0

        if not points:
            raise HTTPException(
                status_code=404,
                detail=(
                    "File has been extracted but no indexed rows were found in the "
                    "vector store. The file may still be indexing, or indexing may "
                    "have failed."
                )
            )

        sample_rows = [
            {
                "chunk_index":  pt.payload.get("chunk_index", idx),
                "chunk_id":     pt.payload.get("chunk_id", ""),
                "chunk_type":   pt.payload.get("chunk_type", "csv_row"),
                "char_count":   len(pt.payload.get("text", "")),
                "page_number":  pt.payload.get("page_number", 1),
                "content":      pt.payload.get("text", ""),
                "row_index":    pt.payload.get("row_index", None),
            }
            for idx, pt in enumerate(points)
        ]

        logger.info(
            f"👁️ CSV preview for file {request.file_id}: "
            f"showing {len(sample_rows)} sample rows of {total_stored} total"
        )

        return {
            "file_id":               request.file_id,
            "knowledge_base_id":     kb.knowledge_base_id,
            "chunking_strategy":     "csv_row_level",
            "parameters":            {"note": "Each row is stored as a separate vector. No chunking applied."},
            "preview_chunks":        sample_rows,
            "preview_structure":     "flat",
            "preview_count_flat":    len(sample_rows),
            "estimated_total_chunks": total_stored,
            "preview_time_seconds":  round(preview_time, 3),
            "note": (
                "Sample rows fetched directly from the vector store. "
                "CSV/TSV files are indexed row-by-row and do not use the "
                "markdown chunking pipeline."
            ),
        }

    # ── Standard path: read markdown and run chunking ─────────────────────────
    try:
        with open(file_metadata.markdown_path, 'r', encoding='utf-8') as f:
            markdown_content = f.read()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not read markdown content: {e}")

    if not markdown_content:
        raise HTTPException(status_code=400, detail="No markdown content available. Re-run extraction.")

    base_metadata = {
        'user_id': user_id,
        'file_id': request.file_id,
        'filename': file_metadata.safe_filename,
        'knowledge_base_id': kb.knowledge_base_id,
        'knowledge_base_name': kb.knowledge_base_name,
    }

    # ── Build constructor / call kwargs ──────────────────────────────────────
    constructor_kwargs: dict = {}
    call_kwargs: dict = {}
    params: dict = {}

    if strategy == "default":
        constructor_kwargs = {'target_chunk_tokens': 300, 'chunk_overlap_tokens': 30}
        params = {"chunk_size": 300, "chunk_overlap": 30}

    elif strategy == "fixed_size":
        chunk_size = getattr(request, "chunk_size", None)
        chunk_overlap = getattr(request, "chunk_overlap", None)
        if chunk_size is None or chunk_overlap is None:
            raise HTTPException(status_code=400,
                detail="fixed_size strategy requires: chunk_size, chunk_overlap")
        constructor_kwargs = {'target_chunk_tokens': chunk_size, 'chunk_overlap_tokens': chunk_overlap}
        params = {"chunk_size": chunk_size, "chunk_overlap": chunk_overlap}

    elif strategy == "semantic":
        chunk_size = getattr(request, "chunk_size", None)
        chunk_overlap = getattr(request, "chunk_overlap", None)
        max_buffer_size = getattr(request, "max_buffer_size", None)
        breakpoint_threshold = getattr(request, "breakpoint_threshold", None)
        if None in (chunk_size, chunk_overlap, max_buffer_size, breakpoint_threshold):
            raise HTTPException(status_code=400,
                detail="semantic strategy requires: chunk_size, chunk_overlap, max_buffer_size, breakpoint_threshold")
        constructor_kwargs = {'target_chunk_tokens': chunk_size, 'chunk_overlap_tokens': chunk_overlap}
        call_kwargs = {'max_buffer_size': max_buffer_size, 'breakpoint_threshold': breakpoint_threshold}
        params = {"chunk_size": chunk_size, "chunk_overlap": chunk_overlap,
                  "max_buffer_size": max_buffer_size, "breakpoint_threshold": breakpoint_threshold}

    elif strategy == "hierarchical":
        parent_chunk_size = getattr(request, "parent_chunk_size", None)
        child_chunk_size = getattr(request, "child_chunk_size", None)
        child_chunk_overlap = getattr(request, "child_chunk_overlap", None)
        if None in (parent_chunk_size, child_chunk_size, child_chunk_overlap):
            raise HTTPException(status_code=400,
                detail="hierarchical strategy requires: parent_chunk_size, child_chunk_size, child_chunk_overlap")
        constructor_kwargs = {'target_chunk_tokens': child_chunk_size, 'chunk_overlap_tokens': child_chunk_overlap}
        call_kwargs = {'parent_chunk_size': parent_chunk_size,
                       'child_chunk_size': child_chunk_size,
                       'child_chunk_overlap': child_chunk_overlap}
        params = {"parent_chunk_size": parent_chunk_size,
                  "child_chunk_size": child_chunk_size,
                  "child_chunk_overlap": child_chunk_overlap}

    elif strategy == "no_chunking":
        params = {"note": "Full document as single chunk"}

    elif strategy == "json":
        child_chunk_size = getattr(request, "child_chunk_size", None)
        context_prefix   = getattr(request, "context_prefix", True)
        if child_chunk_size is None:
            raise HTTPException(
                status_code=400,
                detail="json strategy requires: child_chunk_size"
            )
        call_kwargs = {
            "json_child_tokens":   child_chunk_size,
            "json_context_prefix": context_prefix,
        }
        params = {
            "child_chunk_size": child_chunk_size,
            "context_prefix":   context_prefix,
        }

    else:
        raise HTTPException(status_code=400, detail=f"Invalid chunking_strategy: {strategy}")

    # ── Detect ZIP: extract per-file chunks ──────────────────────────────────
    zip_file_chunks = _get_zip_file_chunks(file_metadata, context="preview")

    # ── Run chunking (no DB writes) ──────────────────────────────────────────
    try:
        start_time = time.time()

        if strategy == "no_chunking":
            svc = EnhancedChunkingService()
            if zip_file_chunks:
                all_chunks = svc.split_document_into_chunks(
                    markdown_text="",
                    base_metadata=base_metadata,
                    strategy=ChunkingStrategy.NO_CHUNKING,
                    zip_file_chunks=zip_file_chunks,
                )
            else:
                all_chunks = svc._chunk_no_chunking(markdown_content, base_metadata)
        elif strategy == "json":
            # JSON strategy: no constructor kwargs — uses service defaults
            chunking_service = EnhancedChunkingService()
            all_chunks = chunking_service.split_document_into_chunks(
                markdown_content,
                base_metadata=base_metadata,
                strategy=ChunkingStrategy.JSON,
                zip_file_chunks=zip_file_chunks,
                **call_kwargs,
            )
        else:
            chunking_service = EnhancedChunkingService(**constructor_kwargs)
            strategy_enum = ChunkingStrategy(strategy)
            all_chunks = chunking_service.split_document_into_chunks(
                markdown_content,
                base_metadata=base_metadata,
                strategy=strategy_enum,
                zip_file_chunks=zip_file_chunks,
                **call_kwargs,
            )

        preview_time = time.time() - start_time

    except Exception as e:
        logger.error(f"❌ Preview chunking failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Preview chunking failed: {str(e)}")

    # ── Prepare preview data ─────────────────────────────────────────────────
    preview_data = []

    if strategy in ("hierarchical", "json"):
        parent_map = {}  # chunk_id → preview parent dict
        all_parent_ids = {
            chunk["chunk_id"]
            for chunk in all_chunks
            if chunk["metadata"].get("chunk_type") == "parent"
        }
        MAX_PARENTS = 4
        MAX_CHILDREN_PER_PARENT = 10

        # Step 1: Collect parents (first N parents)
        parent_count = 0
        for chunk in all_chunks:
            if chunk["metadata"].get("chunk_type") != "parent":
                continue
            if parent_count >= MAX_PARENTS:
                break

            item = {
                "chunk_index": chunk["metadata"]["chunk_index"],
                "chunk_id": chunk["chunk_id"],
                "chunk_type": "parent",
                "char_count": chunk["char_count"],
                "page_number": chunk["metadata"].get("page_number", 0),
                "content_preview": chunk["text"][:1200] + (" …" if len(chunk["text"]) > 1200 else ""),
                "full_char_count": len(chunk["text"]),
                "children": []
            }
            parent_map[chunk["chunk_id"]] = item
            preview_data.append(item)
            parent_count += 1

        # Step 2: Attach children to the correct parent using parent_chunk_id.
        # Limit children per parent so preview stays readable without cutting a
        # parent off halfway through its child list.
        for chunk in all_chunks:
            if chunk["metadata"].get("chunk_type") != "child":
                continue

            parent_id = chunk["metadata"].get("parent_chunk_id")
            if parent_id in parent_map:
                parent_item = parent_map[parent_id]
                if len(parent_item["children"]) >= MAX_CHILDREN_PER_PARENT:
                    parent_item["children_truncated"] = True
                    parent_item["omitted_children_count"] = parent_item.get("omitted_children_count", 0) + 1
                    continue

                child_item = {
                    "chunk_index": chunk["metadata"]["chunk_index"],
                    "chunk_id": chunk["chunk_id"],
                    "chunk_type": "child",
                    "char_count": chunk["char_count"],
                    "page_number": chunk["metadata"].get("page_number", 0),
                    "content_preview": chunk["text"][:900] + (" …" if len(chunk["text"]) > 900 else ""),
                    "full_char_count": len(chunk["text"]),
                }
                # Extra JSON-specific fields
                if strategy == "json":
                    child_item["breadcrumb"] = chunk["metadata"].get("breadcrumb", "")
                    child_item["path"] = chunk["metadata"].get("path", [])
                parent_item["children"].append(child_item)

        # Optional: show true orphans only. If a parent exists in the full
        # chunk set but was omitted from the limited preview, that child is
        # not actually orphaned.
        for chunk in all_chunks:
            parent_id = chunk["metadata"].get("parent_chunk_id")
            if (
                chunk["metadata"].get("chunk_type") == "child" and
                parent_id not in all_parent_ids
            ):
                preview_data.append({
                    "warning": "orphan child - parent missing from chunk set",
                    "chunk_id": chunk["chunk_id"],
                    "parent_chunk_id": parent_id,
                    "chunk_index": chunk["metadata"]["chunk_index"],
                    "char_count": chunk["char_count"],
                    "content_preview": chunk["text"][:500] + " …"
                })

    else:
        # Flat structure for non-hierarchical strategies
        PREVIEW_LIMIT = 30
        preview_chunks_data = all_chunks[:PREVIEW_LIMIT]
        preview_data = [
            {
                "chunk_index": chunk["metadata"]["chunk_index"],
                "chunk_id": chunk["chunk_id"],
                "chunk_type": chunk["metadata"].get("chunk_type", "text"),
                "char_count": chunk["char_count"],
                "page_number": chunk["metadata"].get("page_number", 0),
                "content": chunk["text"],
            }
            for chunk in preview_chunks_data
        ]

    # ── Logging ──────────────────────────────────────────────────────────────
    flat_count = (
        sum(1 + len(p.get("children", [])) for p in preview_data)
        if strategy in ("hierarchical", "json")
        else len(preview_data)
    )

    logger.info(
        f"👁️ Preview for file {request.file_id} ({strategy}): "
        f"showing {flat_count} items of {len(all_chunks)} total chunks"
    )

    # ── Response ─────────────────────────────────────────────────────────────
    return {
        "file_id": request.file_id,
        "knowledge_base_id": kb.knowledge_base_id,
        "chunking_strategy": strategy,
        "parameters": params,
        "preview_chunks": preview_data,
        "preview_structure": (
            "nested (parents → children)"
            if strategy in ("hierarchical", "json")
            else "flat"
        ),
        "preview_count_flat": flat_count,
        "estimated_total_chunks": len(all_chunks),
        "preview_time_seconds": round(preview_time, 2),
        "note": "This is a preview only. No data has been saved. Call /create to persist chunks.",
    }
 

# ==================== SINGLE UNIFIED ENDPOINT ====================

@router.post("/create", tags=["Chunks - Create"])
async def create_chunks(
    request: ChunkingRequest = Body(...),
    user_id: str = Depends(get_user_id),
    db: Session = Depends(get_database_session)
):
    """
    Unified chunk creation endpoint. Accepts all chunking strategies.

    Example request bodies:
1. Default:
{
    "file_id": "example_file_id",
    "chunking_strategy": "default"
}
2. Fixed Size:
{
    "file_id": "example_file_id",
    "chunking_strategy": "fixed_size",
    "chunk_size": 1000,
    "chunk_overlap": 100
}
3. Semantic:
{
    "file_id": "example_file_id",
    "chunking_strategy": "semantic",
    "chunk_size": 1000,
    "chunk_overlap": 100,
    "max_buffer_size": 256,
    "breakpoint_threshold": 0.8
}
4. Hierarchical:
{
    "file_id": "example_file_id",
    "chunking_strategy": "hierarchical",
    "parent_chunk_size": 500,
    "child_chunk_size": 200,
    "child_chunk_overlap": 30
}
5. No Chunking:
{
    "file_id": "example_file_id",
    "chunking_strategy": "no_chunking"
}
6. JSON:
{
    "file_id": "example_file_id",
    "chunking_strategy": "json",
    "child_chunk_size": 100,
    "context_prefix": true
}
"""
    strategy = getattr(request, "chunking_strategy", "default")
    if strategy == "default":
        return await _create_chunks_internal(
            file_id=request.file_id,
            chunking_strategy="default",
            chunk_size=None,
            chunk_overlap=None,
            max_buffer_size=None,
            breakpoint_threshold=None,
            parent_chunk_size=None,
            child_chunk_size=None,
            child_chunk_overlap=None,
            json_child_tokens=None,
            json_context_prefix=True,
            user_id=user_id,
            db=db,
        )
    elif strategy == "fixed_size":
        return await _create_chunks_internal(
            file_id=request.file_id,
            chunking_strategy="fixed_size",
            chunk_size=getattr(request, "chunk_size", None),
            chunk_overlap=getattr(request, "chunk_overlap", None),
            max_buffer_size=None,
            breakpoint_threshold=None,
            parent_chunk_size=None,
            child_chunk_size=None,
            child_chunk_overlap=None,
            json_child_tokens=None,
            json_context_prefix=True,
            user_id=user_id,
            db=db,
        )
    elif strategy == "semantic":
        return await _create_chunks_internal(
            file_id=request.file_id,
            chunking_strategy="semantic",
            chunk_size=getattr(request, "chunk_size", None),
            chunk_overlap=getattr(request, "chunk_overlap", None),
            max_buffer_size=getattr(request, "max_buffer_size", None),
            breakpoint_threshold=getattr(request, "breakpoint_threshold", None),
            parent_chunk_size=None,
            child_chunk_size=None,
            child_chunk_overlap=None,
            json_child_tokens=None,
            json_context_prefix=True,
            user_id=user_id,
            db=db,
        )
    elif strategy == "hierarchical":
        return await _create_chunks_internal(
            file_id=request.file_id,
            chunking_strategy="hierarchical",
            chunk_size=None,
            chunk_overlap=None,
            max_buffer_size=None,
            breakpoint_threshold=None,
            parent_chunk_size=getattr(request, "parent_chunk_size", None),
            child_chunk_size=getattr(request, "child_chunk_size", None),
            child_chunk_overlap=getattr(request, "child_chunk_overlap", None),
            json_child_tokens=None,
            json_context_prefix=True,
            user_id=user_id,
            db=db,
        )
    elif strategy == "no_chunking":
        return await _create_chunks_internal(
            file_id=request.file_id,
            chunking_strategy="no_chunking",
            chunk_size=None,
            chunk_overlap=None,
            max_buffer_size=None,
            breakpoint_threshold=None,
            parent_chunk_size=None,
            child_chunk_size=None,
            child_chunk_overlap=None,
            json_child_tokens=None,
            json_context_prefix=True,
            user_id=user_id,
            db=db,
        )
    elif strategy == "json":
        child_chunk_size = getattr(request, "child_chunk_size", None)
        if child_chunk_size is None:
            raise HTTPException(
                status_code=400,
                detail="json strategy requires: child_chunk_size"
            )
        return await _create_chunks_internal(
            file_id=request.file_id,
            chunking_strategy="json",
            chunk_size=None,
            chunk_overlap=None,
            max_buffer_size=None,
            breakpoint_threshold=None,
            parent_chunk_size=None,
            child_chunk_size=child_chunk_size,
            child_chunk_overlap=None,
            json_child_tokens=child_chunk_size,
            json_context_prefix=getattr(request, "context_prefix", True),
            user_id=user_id,
            db=db,
        )
    else:
        raise HTTPException(status_code=400, detail=f"Invalid chunking_strategy: {strategy}")


# ==================== INTERNAL IMPLEMENTATION ====================

async def _create_chunks_internal(
    file_id: str,
    chunking_strategy: str,
    chunk_size: Optional[int],
    chunk_overlap: Optional[int],
    max_buffer_size: Optional[int],
    breakpoint_threshold: Optional[float],
    parent_chunk_size: Optional[int],
    child_chunk_size: Optional[int],
    child_chunk_overlap: Optional[int],
    json_child_tokens: Optional[int],
    json_context_prefix: bool,
    user_id: str,
    db: Session,
):
    """
    Core chunking logic shared by all strategy endpoints.
    chunking_strategy is a plain string like "default", "fixed_size", etc.
    """
    try:
        # ── DB lookups ───────────────────────────────────────────────────
        file_metadata = db.query(FileMetadata).filter(
            FileMetadata.file_id == file_id,
            FileMetadata.user_id == user_id
        ).first()
        if not file_metadata:
            raise HTTPException(status_code=404, detail=f"File not found: {file_id}")

        kb = db.query(KnowledgeBase).filter(
            KnowledgeBase.knowledge_base_id == file_metadata.knowledge_base_id,
            KnowledgeBase.user_id == user_id
        ).first()
        if not kb:
            raise HTTPException(status_code=404, detail="Knowledge base not found")

        if file_metadata.status != ProcessingStatus.COMPLETED:
            raise HTTPException(
                status_code=400,
                detail=f"File must be extracted first. Current status: {file_metadata.status.value}"
            )

        existing_chunks = db.query(ChunkMetadata).filter(
            ChunkMetadata.file_id == file_id
        ).count()
        if existing_chunks > 0:
            raise HTTPException(
                status_code=400,
                detail=f"File already has {existing_chunks} chunks. Delete them first."
            )

        # BUG-1 CHECK: document_count is already incremented during upload.
        # _create_chunks_internal must NOT touch document_count here.

        # ── Read markdown ────────────────────────────────────────────────
        if not file_metadata.markdown_path:
            raise HTTPException(status_code=400, detail="No markdown content available. Re-run extraction.")
        try:
            with open(file_metadata.markdown_path, 'r', encoding='utf-8') as f:
                markdown_content = f.read()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Could not read markdown content: {e}")

        if not markdown_content:
            raise HTTPException(status_code=400, detail="No markdown content available. Re-run extraction.")

        logger.info(f"📊 Creating chunks for file {file_id} using {chunking_strategy}")

        base_metadata = {
            'user_id': user_id,
            'file_id': file_id,
            'filename': file_metadata.safe_filename,
            'knowledge_base_id': kb.knowledge_base_id,
            'knowledge_base_name': kb.knowledge_base_name,
        }

        # ── Validate params & build constructor / call kwargs ────────────
        constructor_kwargs: dict = {}   # → EnhancedChunkingService(...)
        call_kwargs: dict = {}          # → split_document_into_chunks(...)

        if chunking_strategy == "default":
            constructor_kwargs = {'target_chunk_tokens': 300, 'chunk_overlap_tokens': 30}
            logger.info("Using DEFAULT: 300 tokens, 30 overlap")

        elif chunking_strategy == "fixed_size":
            if chunk_size is None or chunk_overlap is None:
                raise HTTPException(status_code=400,
                    detail="fixed_size strategy requires: chunk_size, chunk_overlap")
            constructor_kwargs = {
                'target_chunk_tokens': chunk_size,
                'chunk_overlap_tokens': chunk_overlap,
            }
            logger.info(f"Using FIXED_SIZE: {chunk_size} tokens, {chunk_overlap} overlap")

        elif chunking_strategy == "semantic":
            if None in (chunk_size, chunk_overlap, max_buffer_size, breakpoint_threshold):
                raise HTTPException(status_code=400,
                    detail="semantic strategy requires: chunk_size, chunk_overlap, "
                           "max_buffer_size, breakpoint_threshold")
            constructor_kwargs = {
                'target_chunk_tokens': chunk_size,
                'chunk_overlap_tokens': chunk_overlap,
            }
            call_kwargs = {
                'max_buffer_size': max_buffer_size,
                'breakpoint_threshold': breakpoint_threshold,
            }
            logger.info(f"Using SEMANTIC: size={chunk_size}, overlap={chunk_overlap}, "
                        f"buffer={max_buffer_size}, threshold={breakpoint_threshold}")

        elif chunking_strategy == "hierarchical":
            if None in (parent_chunk_size, child_chunk_size, child_chunk_overlap):
                raise HTTPException(status_code=400,
                    detail="hierarchical strategy requires: parent_chunk_size, "
                           "child_chunk_size, child_chunk_overlap")
            constructor_kwargs = {
                'target_chunk_tokens': child_chunk_size,
                'chunk_overlap_tokens': child_chunk_overlap,
            }
            call_kwargs = {
                'parent_chunk_size': parent_chunk_size,
                'child_chunk_size': child_chunk_size,
                'child_chunk_overlap': child_chunk_overlap,
            }
            logger.info(f"Using HIERARCHICAL: parent={parent_chunk_size}, "
                        f"child={child_chunk_size}, overlap={child_chunk_overlap}")

        elif chunking_strategy == "no_chunking":
            logger.info("Using NO_CHUNKING: entire document as single chunk")

        elif chunking_strategy == "json":
            if json_child_tokens is None:
                raise HTTPException(
                    status_code=400,
                    detail="json strategy requires: child_chunk_size (json_child_tokens)"
                )
            call_kwargs = {
                "json_child_tokens":   json_child_tokens,
                "json_context_prefix": json_context_prefix,
            }
            logger.info(
                f"Using JSON: child_tokens={json_child_tokens}, "
                f"context_prefix={json_context_prefix}"
            )

        # ── Detect ZIP: extract per-file chunks ──────────────────────────
        import time
        start_time = time.time()

        zip_file_chunks = _get_zip_file_chunks(file_metadata, context="create")

        # ── Create chunks ────────────────────────────────────────────────
        if chunking_strategy == "no_chunking":
            svc = EnhancedChunkingService()
            if zip_file_chunks:
                chunks = svc.split_document_into_chunks(
                    markdown_text="",
                    base_metadata=base_metadata,
                    strategy=ChunkingStrategy.NO_CHUNKING,
                    zip_file_chunks=zip_file_chunks,
                )
            else:
                chunks = svc._chunk_no_chunking(markdown_content, base_metadata)
        elif chunking_strategy == "json":
            chunking_service = EnhancedChunkingService()
            chunks = chunking_service.split_document_into_chunks(
                markdown_content,
                base_metadata=base_metadata,
                strategy=ChunkingStrategy.JSON,
                zip_file_chunks=zip_file_chunks,
                **call_kwargs,
            )
        else:
            chunking_service = EnhancedChunkingService(**constructor_kwargs)
            strategy_enum = ChunkingStrategy(chunking_strategy)
            chunks = chunking_service.split_document_into_chunks(
                markdown_content,
                base_metadata=base_metadata,
                strategy=strategy_enum,
                zip_file_chunks=zip_file_chunks,
                **call_kwargs,
            )

        chunking_time = time.time() - start_time

        logger.info(f"✅ Created {len(chunks)} chunks in {chunking_time:.2f}s")

        # ── Persist chunks ───────────────────────────────────────────────
        chunk_ids: List[str] = []
        for chunk in chunks:
            chunk_db = ChunkMetadata(
                chunk_id=chunk['chunk_id'],
                file_id=file_id,
                chunk_index=chunk['metadata']['chunk_index'],
                chunk_type=chunk['metadata'].get('chunk_type', 'text'),
                char_count=chunk['char_count'],
                token_count=chunk.get('token_count'),
                page_number=chunk['metadata'].get('page_number', 0),
                created_at=datetime.utcnow(),
            )
            db.add(chunk_db)
            chunk_ids.append(chunk['chunk_id'])

        strategy_map = {
            'default':      ChunkingStrategyEnum.DEFAULT,
            'fixed_size':   ChunkingStrategyEnum.FIXED_SIZE,
            'hierarchical': ChunkingStrategyEnum.HIERARCHICAL,
            'semantic':     ChunkingStrategyEnum.SEMANTIC,
            'no_chunking':  ChunkingStrategyEnum.NO_CHUNKING,
            'json':         ChunkingStrategyEnum.JSON,
        }

        normalized_strategy = str(chunking_strategy).lower()
        if normalized_strategy not in strategy_map:
            logger.warning(f"Final DB normalization: invalid chunking_strategy '{chunking_strategy}' remapped to 'default'.")
            normalized_strategy = 'default'

        file_metadata.chunks_created = len(chunks)
        file_metadata.chunking_strategy = strategy_map.get(normalized_strategy)

        if chunking_strategy == "default":
            file_metadata.target_chunk_tokens = 300
            file_metadata.chunk_overlap_tokens = 30
            file_metadata.parent_chunk_size = None
        elif chunking_strategy == "fixed_size":
            file_metadata.target_chunk_tokens = chunk_size
            file_metadata.chunk_overlap_tokens = chunk_overlap
            file_metadata.parent_chunk_size = None
        elif chunking_strategy == "semantic":
            file_metadata.target_chunk_tokens = chunk_size
            file_metadata.chunk_overlap_tokens = chunk_overlap
            file_metadata.parent_chunk_size = None
        elif chunking_strategy == "hierarchical":
            file_metadata.target_chunk_tokens = child_chunk_size
            file_metadata.chunk_overlap_tokens = child_chunk_overlap
            file_metadata.parent_chunk_size = parent_chunk_size
        elif chunking_strategy == "no_chunking":
            file_metadata.target_chunk_tokens = 0
            file_metadata.chunk_overlap_tokens = 0
            file_metadata.parent_chunk_size = None
        elif chunking_strategy == "json":
            file_metadata.target_chunk_tokens = json_child_tokens
            file_metadata.chunk_overlap_tokens = 0
            file_metadata.parent_chunk_size = None

        import json
        existing_meta: dict = {}
        if file_metadata.processing_metadata:
            try:
                existing_meta = json.loads(file_metadata.processing_metadata)
            except Exception:
                existing_meta = {}
        existing_meta['chunking_call_kwargs'] = call_kwargs
        file_metadata.processing_metadata = json.dumps(existing_meta)

        kb.total_chunks += len(chunks)
        kb.updated_at = datetime.utcnow()
        db.commit()

        logger.info(f"🎉 Chunking complete: {len(chunks)} chunks saved")

        # ── Build response ───────────────────────────────────────────────
        params: dict
        if chunking_strategy == "default":
            params = {"chunk_size": 300, "chunk_overlap": 30}
        elif chunking_strategy == "fixed_size":
            params = {"chunk_size": chunk_size, "chunk_overlap": chunk_overlap}
        elif chunking_strategy == "semantic":
            params = {
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "max_buffer_size": max_buffer_size,
                "breakpoint_threshold": breakpoint_threshold,
            }
        elif chunking_strategy == "hierarchical":
            params = {
                "parent_chunk_size": parent_chunk_size,
                "child_chunk_size": child_chunk_size,
                "child_chunk_overlap": child_chunk_overlap,
            }
        elif chunking_strategy == "json":
            params = {
                "child_chunk_size": json_child_tokens,
                "context_prefix":   json_context_prefix,
            }
        else:
            params = {"note": "Full document as single chunk"}

        return {
            "file_id": file_id,
            "knowledge_base_id": kb.knowledge_base_id,
            "chunks_created": len(chunks),
            "chunk_ids_sample": chunk_ids[:10],
            "total_chunk_ids": len(chunk_ids),
            "chunking_strategy": chunking_strategy,
            "chunking_time_seconds": round(chunking_time, 2),
            "status": "completed",
            "parameters": params,
            "message": f"Created {len(chunks)} chunks successfully",
        }

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"❌ Chunking failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Chunking failed: {str(e)}")


# ==================== FINALIZE ENDPOINT ====================

@router.post("/finalize", tags=["Chunks - Finalize"])
async def finalize_chunks(
    file_id: str,
    embedding_model: str = "cohere.embed-english-v3",
    chunk_ids: Optional[List[str]] = None,
    batch_size: int = 50,
    user_id: str = Depends(get_user_id),
    db: Session = Depends(get_database_session)
):
    """
    Index chunks to the vector database using the specified embedding model.
    For files with > 100 chunks the work runs in a background task.
    """
    try:
        file_metadata = db.query(FileMetadata).filter(
            FileMetadata.file_id == file_id,
            FileMetadata.user_id == user_id
        ).first()
        if not file_metadata:
            raise HTTPException(status_code=404, detail=f"File not found: {file_id}")

        kb = db.query(KnowledgeBase).filter(
            KnowledgeBase.knowledge_base_id == file_metadata.knowledge_base_id
        ).first()
        if not kb:
            raise HTTPException(status_code=404, detail="Knowledge base not found")

        if file_metadata.chunks_created == 0:
            raise HTTPException(status_code=400,
                detail="No chunks found. Create chunks first via /api/chunks/create")

        query = db.query(ChunkMetadata).filter(ChunkMetadata.file_id == file_id)
        if chunk_ids:
            query = query.filter(ChunkMetadata.chunk_id.in_(chunk_ids))
        chunks_to_index = query.all()

        if not chunks_to_index:
            raise HTTPException(status_code=400, detail="No chunks to index")

        logger.info(f"📊 Indexing {len(chunks_to_index)} chunks with model: {embedding_model}")

        if len(chunks_to_index) > 100:
            task_manager = get_task_manager()
            task_id = task_manager.submit_task(
                func=index_chunks_background,
                task_type=TaskType.IO_BOUND,
                description=f"Index {len(chunks_to_index)} chunks",
                user_id=user_id,
                file_id=file_id,
                knowledge_base_id=kb.knowledge_base_id,
                chunk_ids=[c.chunk_id for c in chunks_to_index],
                embedding_model=embedding_model,
                batch_size=batch_size,
            )
            return {
                "file_id": file_id,
                "chunks_to_index": len(chunks_to_index),
                "task_id": task_id,
                "status": "indexing",
                "message": (f"Indexing {len(chunks_to_index)} chunks in background. "
                            f"Track: /api/tasks/{task_id}"),
            }

        # ── Inline indexing for small files ──────────────────────────────
        vector_store = VectorStoreService(embedding_model_id=embedding_model)

        if not file_metadata.markdown_path:
            raise HTTPException(status_code=400, detail="Markdown content not found")
        with open(file_metadata.markdown_path, 'r', encoding='utf-8') as f:
            markdown_content = f.read()
        if not markdown_content:
            raise HTTPException(status_code=400, detail="Markdown content is empty")

        chunks = _rechunk_for_indexing(file_metadata, kb, user_id, markdown_content)

        import time
        start_time = time.time()
        chunks_indexed = vector_store.add_document_chunks(
            chunks, file_id, file_metadata.safe_filename,
            knowledge_base_id=kb.knowledge_base_id,
            knowledge_base_name=kb.knowledge_base_name,
            user_id=user_id,
        )
        indexing_time = time.time() - start_time
        db.commit()

        vector_stats = vector_store.get_collection_stats(
            user_id=user_id, knowledge_base_id=kb.knowledge_base_id
        )
        logger.info(f"🎉 Indexed {chunks_indexed} chunks in {indexing_time:.2f}s")

        return {
            "file_id": file_id,
            "knowledge_base_id": kb.knowledge_base_id,
            "chunks_indexed": chunks_indexed,
            "embedding_model": embedding_model,
            "indexing_time_seconds": round(indexing_time, 2),
            "vector_store_stats": vector_stats,
            "status": "completed",
            "message": f"Indexed {chunks_indexed} chunks successfully using {embedding_model}",
        }

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"❌ Indexing failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Indexing failed: {str(e)}")


# ==================== RE-CHUNK HELPER ====================

def _rechunk_for_indexing(
    file_metadata,
    kb,
    user_id: str,
    markdown_content: str,
) -> List[Dict]:
    """
    Re-produce the exact same chunks that were originally created,
    using the strategy + params stored on file_metadata.
    """
    import json

    strategy_str = file_metadata.chunking_strategy.value
    strategy_enum = ChunkingStrategy(strategy_str)

    base_metadata = {
        'user_id': user_id,
        'file_id': file_metadata.file_id,
        'filename': file_metadata.safe_filename,
        'knowledge_base_id': kb.knowledge_base_id,
        'knowledge_base_name': kb.knowledge_base_name,
    }

    target  = file_metadata.target_chunk_tokens  or 300
    overlap = file_metadata.chunk_overlap_tokens or 20

    # ── Detect ZIP: extract per-file chunks ──────────────────────────────────
    zip_file_chunks = _get_zip_file_chunks(file_metadata, context="rechunk")

    # ── Restore call-time kwargs saved during /create ─────────────────────────
    # Must be loaded here — before any early-return guard — so every strategy
    # path (including the json fast-path below) can reference saved_call_kwargs.
    saved_call_kwargs: dict = {}
    if file_metadata.processing_metadata:
        try:
            meta = json.loads(file_metadata.processing_metadata)
            saved_call_kwargs = meta.get('chunking_call_kwargs', {})
        except Exception:
            saved_call_kwargs = {}

    # ── no_chunking path ─────────────────────────────────────────────────────
    if strategy_str == "no_chunking":
        svc = EnhancedChunkingService()
        if zip_file_chunks:
            return svc.split_document_into_chunks(
                markdown_text="",
                base_metadata=base_metadata,
                strategy=ChunkingStrategy.NO_CHUNKING,
                zip_file_chunks=zip_file_chunks,
            )
        return svc._chunk_no_chunking(markdown_content, base_metadata)

    # ── json path (no constructor kwargs needed) ──────────────────────────────
    if strategy_str == "json":
        svc = EnhancedChunkingService()
        # Prefer saved kwargs from /create; fall back to stored token size
        json_call_kwargs = saved_call_kwargs if saved_call_kwargs else {
            'json_child_tokens':   file_metadata.target_chunk_tokens or 100,
            'json_context_prefix': True,
        }
        return svc.split_document_into_chunks(
            markdown_content,
            base_metadata=base_metadata,
            strategy=ChunkingStrategy.JSON,
            zip_file_chunks=zip_file_chunks,
            **json_call_kwargs,
        )

    chunking_service = EnhancedChunkingService(
        target_chunk_tokens=target,
        chunk_overlap_tokens=overlap,
    )

    call_kwargs: dict = {}

    if strategy_str == "hierarchical":
        stored_parent = file_metadata.parent_chunk_size
        call_kwargs = {
            'parent_chunk_size': stored_parent,
            'child_chunk_size': target,
            'child_chunk_overlap': overlap,
        }

    elif strategy_str == "semantic":
        call_kwargs = saved_call_kwargs

    return chunking_service.split_document_into_chunks(
        markdown_content,
        base_metadata=base_metadata,
        strategy=strategy_enum,
        zip_file_chunks=zip_file_chunks,
        **call_kwargs,
    )


# ==================== BACKGROUND INDEXING ====================

def index_chunks_background(
    user_id: str,
    file_id: str,
    knowledge_base_id: str,
    chunk_ids: List[str],
    embedding_model: str = "cohere.embed-english-v3",
    batch_size: int = 50,
    progress_callback=None,
    task_id=None,
):
    """Background indexing for large documents."""
    import time
    logger.info(f"🚀 [Task {task_id}] Indexing {len(chunk_ids)} chunks")

    try:
        from app.core.database import get_db_context
        from app.db.models import FileMetadata, KnowledgeBase
        from app.services.vector_store_service import VectorStoreService

        start_time = time.time()

        with get_db_context() as db:
            file_metadata = db.query(FileMetadata).filter(
                FileMetadata.file_id == file_id
            ).first()
            kb = db.query(KnowledgeBase).filter(
                KnowledgeBase.knowledge_base_id == knowledge_base_id
            ).first()
            if not file_metadata or not kb:
                raise ValueError("File or KB not found")

            with open(file_metadata.markdown_path, 'r', encoding='utf-8') as f:
                markdown_content = f.read()

            chunks = _rechunk_for_indexing(file_metadata, kb, user_id, markdown_content)

            vector_store = VectorStoreService(embedding_model_id=embedding_model)
            total_indexed = vector_store.add_document_chunks(
                chunks, file_id, file_metadata.safe_filename,
                knowledge_base_id=knowledge_base_id,
                knowledge_base_name=kb.knowledge_base_name,
                user_id=user_id,
            )

            if progress_callback:
                progress_callback(100.0)

            total_time = time.time() - start_time
            logger.info(f"🎉 [Task {task_id}] Indexed {total_indexed} chunks in {total_time:.2f}s")

            return {
                'success': True,
                'file_id': file_id,
                'chunks_indexed': total_indexed,
                'indexing_time_seconds': total_time,
            }

    except Exception as e:
        logger.error(f"❌ [Task {task_id}] Indexing FAILED: {str(e)}", exc_info=True)
        raise
