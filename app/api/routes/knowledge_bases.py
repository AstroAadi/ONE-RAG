"""
Knowledge Base CRUD Routes

Exposes knowledge bases as first-class, user-owned resources:

    GET    /api/knowledge-bases/                        list
    POST   /api/knowledge-bases/                        create
    GET    /api/knowledge-bases/{kb_id}                 detailed status
    PUT    /api/knowledge-bases/{kb_id}                 rename / re-describe
    DELETE /api/knowledge-bases/{kb_id}                 delete KB + contents
    DELETE /api/knowledge-bases/{kb_id}/documents/{id}  delete one document
    GET    /api/knowledge-bases/{kb_id}/stats           usage statistics

Every endpoint requires the X-User-ID header and is scoped to that user.
"""

from datetime import datetime
from pathlib import Path
from typing import List
import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.dependencies import get_user_id, get_database_session
from app.db.models import FileMetadata, KnowledgeBase
from app.models.schemas import (
    DocumentInfo,
    KnowledgeBaseCreate,
    KnowledgeBaseInfo,
    KnowledgeBaseList,
    KnowledgeBaseStatus,
    KnowledgeBaseUpdate,
)
from app.services.knowledge_base_service import (
    create_knowledge_base,
    find_knowledge_base_by_name,
    get_knowledge_base,
    summarize_knowledge_base,
    touch_knowledge_base,
)
from app.services.vector_store_service import VectorStoreService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/knowledge-bases", tags=["Knowledge Bases"])

vector_store = VectorStoreService()


def _require_knowledge_base(
    db: Session,
    kb_id: str,
    user_id: str
) -> KnowledgeBase:
    """Load a knowledge base or raise 404"""
    knowledge_base = get_knowledge_base(db, kb_id, user_id)

    if not knowledge_base:
        raise HTTPException(
            status_code=404,
            detail=f"Knowledge base not found: {kb_id}"
        )

    return knowledge_base


def _to_document_info(file: FileMetadata) -> DocumentInfo:
    """Map a FileMetadata row to the API document shape"""
    return DocumentInfo(
        file_id=file.file_id,
        knowledge_base_id=file.knowledge_base_id,
        filename=file.safe_filename,
        file_type=file.file_extension,
        file_size_bytes=file.file_size_bytes,
        status=file.status.value if file.status else "uploaded",
        chunks_created=file.chunks_created or 0,
        chunks_staged=file.chunks_staged or 0,
        total_pages=file.total_pages or 0,
        uploaded_at=file.uploaded_at,
        processed_at=file.processed_at,
        word_count=file.word_count or 0,
        image_count=file.image_count or 0,
        table_count=file.table_count or 0,
        chunking_strategy=(
            file.chunking_strategy.value if file.chunking_strategy else None
        ),
    )


def _delete_file_artifacts(file: FileMetadata) -> None:
    """Remove the uploaded file, its markdown, and any staged chunks from disk"""
    for raw_path in (file.storage_path, file.markdown_path):
        if not raw_path:
            continue
        try:
            Path(raw_path).unlink(missing_ok=True)
        except OSError as e:
            logger.warning(f"Could not delete {raw_path}: {e}")

    # Staged (not yet indexed) chunks
    try:
        from app.services.chunk_staging_service import delete_staged_chunks
        delete_staged_chunks(file.user_id, file.file_id)
    except Exception as e:
        logger.debug(f"No staged chunks to delete for {file.file_id}: {e}")


@router.get("/", response_model=KnowledgeBaseList)
async def list_knowledge_bases(
    user_id: str = Depends(get_user_id),
    db: Session = Depends(get_database_session)
):
    """
    List all knowledge bases for the current user, most recent first.

    Document counts, chunk counts, and sizes are aggregated per knowledge base
    so the client can render the list without a follow-up request per KB.
    """
    try:
        knowledge_bases: List[KnowledgeBase] = db.query(KnowledgeBase).filter(
            KnowledgeBase.user_id == user_id
        ).order_by(KnowledgeBase.created_at.desc()).all()

        summaries = [
            KnowledgeBaseInfo(**summarize_knowledge_base(db, kb))
            for kb in knowledge_bases
        ]

        return KnowledgeBaseList(
            user_id=user_id,
            total_knowledge_bases=len(summaries),
            knowledge_bases=summaries
        )

    except Exception as e:
        logger.error(f"Error listing knowledge bases: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error listing knowledge bases: {str(e)}"
        )


@router.post("/", response_model=KnowledgeBaseInfo, status_code=201)
async def create_new_knowledge_base(
    request: KnowledgeBaseCreate,
    user_id: str = Depends(get_user_id),
    db: Session = Depends(get_database_session)
):
    """
    Create a knowledge base.

    Names must be unique per user (case-insensitive); a duplicate returns 409
    rather than silently creating a second KB with the same name.
    """
    existing = find_knowledge_base_by_name(db, request.knowledge_base_name, user_id)

    if existing:
        raise HTTPException(
            status_code=409,
            detail=(
                f"A knowledge base named "
                f"'{existing.knowledge_base_name}' already exists"
            )
        )

    try:
        knowledge_base = create_knowledge_base(
            db,
            user_id=user_id,
            name=request.knowledge_base_name,
            description=request.description
        )
    except IntegrityError:
        db.rollback()
        raise HTTPException(
            status_code=409,
            detail="A knowledge base with this name already exists"
        )
    except Exception as e:
        db.rollback()
        logger.error(f"Error creating knowledge base: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error creating knowledge base: {str(e)}"
        )

    return KnowledgeBaseInfo(**summarize_knowledge_base(db, knowledge_base))


@router.get("/{kb_id}", response_model=KnowledgeBaseStatus)
async def get_knowledge_base_status(
    kb_id: str,
    user_id: str = Depends(get_user_id),
    db: Session = Depends(get_database_session)
):
    """
    Get a knowledge base with the full list of its documents.

    `total_chunks` counts only chunks that have been indexed into the vector
    store, so a document that is uploaded or merely chunked contributes 0.
    """
    knowledge_base = _require_knowledge_base(db, kb_id, user_id)

    try:
        files: List[FileMetadata] = db.query(FileMetadata).filter(
            FileMetadata.knowledge_base_id == kb_id,
            FileMetadata.user_id == user_id
        ).order_by(FileMetadata.uploaded_at.desc()).all()

        documents = [_to_document_info(file) for file in files]

        total_size_bytes = sum(file.file_size_bytes or 0 for file in files)
        total_chunks = sum(file.chunks_created or 0 for file in files)

        touch_knowledge_base(db, knowledge_base)

        return KnowledgeBaseStatus(
            knowledge_base_id=knowledge_base.knowledge_base_id,
            knowledge_base_name=knowledge_base.knowledge_base_name,
            user_id=user_id,
            description=knowledge_base.description,
            total_documents=len(documents),
            total_chunks=total_chunks,
            documents=documents,
            total_size_bytes=total_size_bytes,
            created_at=knowledge_base.created_at,
            last_accessed=knowledge_base.last_accessed
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error loading knowledge base {kb_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error loading knowledge base: {str(e)}"
        )


@router.put("/{kb_id}", response_model=KnowledgeBaseInfo)
async def update_knowledge_base(
    kb_id: str,
    request: KnowledgeBaseUpdate,
    user_id: str = Depends(get_user_id),
    db: Session = Depends(get_database_session)
):
    """Rename a knowledge base and/or change its description"""
    knowledge_base = _require_knowledge_base(db, kb_id, user_id)

    if request.knowledge_base_name is None and request.description is None:
        raise HTTPException(
            status_code=400,
            detail="Provide knowledge_base_name and/or description to update"
        )

    try:
        if request.knowledge_base_name is not None:
            new_name = " ".join(request.knowledge_base_name.split())

            clash = find_knowledge_base_by_name(db, new_name, user_id)
            if clash and clash.knowledge_base_id != kb_id:
                raise HTTPException(
                    status_code=409,
                    detail=f"A knowledge base named '{new_name}' already exists"
                )

            knowledge_base.knowledge_base_name = new_name

        if request.description is not None:
            knowledge_base.description = request.description

        knowledge_base.updated_at = datetime.utcnow()

        db.commit()
        db.refresh(knowledge_base)

        return KnowledgeBaseInfo(**summarize_knowledge_base(db, knowledge_base))

    except HTTPException:
        db.rollback()
        raise
    except IntegrityError:
        db.rollback()
        raise HTTPException(
            status_code=409,
            detail="A knowledge base with this name already exists"
        )
    except Exception as e:
        db.rollback()
        logger.error(f"Error updating knowledge base {kb_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error updating knowledge base: {str(e)}"
        )


@router.delete("/{kb_id}")
async def delete_knowledge_base(
    kb_id: str,
    user_id: str = Depends(get_user_id),
    db: Session = Depends(get_database_session)
):
    """
    Delete a knowledge base and everything inside it.

    Removes vector store chunks, files and markdown on disk, and the database
    rows (chunk metadata cascades from file metadata). This cannot be undone.
    """
    knowledge_base = _require_knowledge_base(db, kb_id, user_id)
    kb_name = knowledge_base.knowledge_base_name

    try:
        files: List[FileMetadata] = db.query(FileMetadata).filter(
            FileMetadata.knowledge_base_id == kb_id,
            FileMetadata.user_id == user_id
        ).all()

        # 1. Vector store chunks for the whole KB
        chunks_deleted = 0
        try:
            chunks_deleted = vector_store.delete_by_knowledge_base(kb_id, user_id)
        except Exception as e:
            logger.warning(f"Error deleting KB {kb_id} chunks from vector store: {e}")

        # 2. On-disk artifacts
        for file in files:
            _delete_file_artifacts(file)

        documents_deleted = len(files)

        # 3. Database rows. Deleting the KB cascades to files, which cascades
        #    to chunk metadata.
        db.delete(knowledge_base)
        db.commit()

        logger.info(
            f"✅ Deleted knowledge base {kb_id} ('{kb_name}'): "
            f"{documents_deleted} documents, {chunks_deleted} chunks"
        )

        return {
            "message": f"Knowledge base '{kb_name}' deleted successfully",
            "knowledge_base_id": kb_id,
            "user_id": user_id,
            "documents_deleted": documents_deleted,
            "chunks_deleted": chunks_deleted
        }

    except HTTPException:
        db.rollback()
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error deleting knowledge base {kb_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting knowledge base: {str(e)}"
        )


@router.delete("/{kb_id}/documents/{file_id}")
async def delete_document_from_knowledge_base(
    kb_id: str,
    file_id: str,
    user_id: str = Depends(get_user_id),
    db: Session = Depends(get_database_session)
):
    """
    Delete a single document from a knowledge base.

    Removes its chunks from the vector store, its files from disk, and its
    database rows.
    """
    _require_knowledge_base(db, kb_id, user_id)

    file = db.query(FileMetadata).filter(
        FileMetadata.file_id == file_id,
        FileMetadata.user_id == user_id,
        FileMetadata.knowledge_base_id == kb_id
    ).first()

    if not file:
        raise HTTPException(
            status_code=404,
            detail=f"Document '{file_id}' not found in knowledge base '{kb_id}'"
        )

    filename = file.safe_filename

    try:
        try:
            vector_store.delete_document_by_file_id(file_id, user_id)
        except Exception as e:
            logger.warning(f"Error deleting {file_id} from vector store: {e}")

        _delete_file_artifacts(file)

        db.delete(file)
        db.commit()

        logger.info(f"✅ Deleted document {file_id} from KB {kb_id}")

        return {
            "message": f"Document '{filename}' deleted successfully",
            "file_id": file_id,
            "knowledge_base_id": kb_id,
            "user_id": user_id
        }

    except Exception as e:
        db.rollback()
        logger.error(f"Error deleting document {file_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting document: {str(e)}"
        )


@router.get("/{kb_id}/stats")
async def get_knowledge_base_stats(
    kb_id: str,
    user_id: str = Depends(get_user_id),
    db: Session = Depends(get_database_session)
):
    """Usage statistics for a single knowledge base"""
    knowledge_base = _require_knowledge_base(db, kb_id, user_id)

    try:
        summary = summarize_knowledge_base(db, knowledge_base)

        # Document counts by extraction status
        status_rows = db.query(
            FileMetadata.status,
            func.count(FileMetadata.file_id)
        ).filter(
            FileMetadata.knowledge_base_id == kb_id,
            FileMetadata.user_id == user_id
        ).group_by(FileMetadata.status).all()

        documents_by_status = {
            (status.value if status else "unknown"): count
            for status, count in status_rows
        }

        # Document counts by file extension
        type_rows = db.query(
            FileMetadata.file_extension,
            func.count(FileMetadata.file_id)
        ).filter(
            FileMetadata.knowledge_base_id == kb_id,
            FileMetadata.user_id == user_id
        ).group_by(FileMetadata.file_extension).all()

        documents_by_type = {ext: count for ext, count in type_rows}

        document_count = summary["document_count"]
        total_chunks = summary["total_chunks"]

        return {
            "knowledge_base_id": kb_id,
            "knowledge_base_name": knowledge_base.knowledge_base_name,
            "user_id": user_id,
            "total_documents": document_count,
            "total_chunks": total_chunks,
            "total_size_bytes": summary["total_size_bytes"],
            "total_size_mb": round(summary["total_size_bytes"] / (1024 * 1024), 2),
            "avg_chunks_per_document": (
                round(total_chunks / document_count, 1) if document_count else 0
            ),
            "documents_by_status": documents_by_status,
            "documents_by_type": documents_by_type,
            "created_at": (
                knowledge_base.created_at.isoformat()
                if knowledge_base.created_at else None
            ),
            "last_accessed": (
                knowledge_base.last_accessed.isoformat()
                if knowledge_base.last_accessed else None
            ),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting KB stats for {kb_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error getting knowledge base statistics: {str(e)}"
        )
