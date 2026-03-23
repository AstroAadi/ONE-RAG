import logging
import json
import re
import uuid
from datetime import datetime
from pathlib import Path

import httpx
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api_upload import APISourceInput
from app.core.config import settings
from app.core.dependencies import get_database_session, get_user_id
from app.db.models import (
    FileMetadata,
    FileType,
    KnowledgeBase,
    ProcessingStatus as DBProcessingStatus,
)
from app.services.api_source_service import APISourceService
from app.utils.file_validator import FileValidator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/sources", tags=["Sources"])


def _normalise_kb_name(source_name: str) -> str:
    cleaned = re.sub(r'[\\/<>:"|?*]+', " ", (source_name or "").strip())
    cleaned = " ".join(cleaned.split())
    return (cleaned[:255] or "API Source")


def _get_knowledge_base_by_id(
    db: Session,
    user_id: str,
    knowledge_base_id: str,
) -> KnowledgeBase:
    kb = db.query(KnowledgeBase).filter(
        KnowledgeBase.knowledge_base_id == knowledge_base_id,
        KnowledgeBase.user_id == user_id,
    ).first()
    if not kb:
        raise HTTPException(
            status_code=404,
            detail=f"Knowledge base not found: {knowledge_base_id}",
        )
    return kb


def _get_or_create_knowledge_base(
    db: Session,
    user_id: str,
    knowledge_base_name: str,
) -> KnowledgeBase:
    normalized_name = knowledge_base_name.lower()
    kb = db.query(KnowledgeBase).filter(
        KnowledgeBase.user_id == user_id,
        KnowledgeBase.knowledge_base_name_normalized == normalized_name,
    ).first()

    if kb:
        return kb

    kb = KnowledgeBase(
        knowledge_base_id=str(uuid.uuid4()),
        user_id=user_id,
        knowledge_base_name=knowledge_base_name,
        knowledge_base_name_normalized=normalized_name,
        document_count=0,
        total_chunks=0,
        total_size_bytes=0,
        created_at=datetime.utcnow(),
    )
    db.add(kb)
    db.flush()
    return kb


def _persist_api_source_result(
    db: Session,
    user_id: str,
    payload: APISourceInput,
    extraction_result: dict,
) -> dict:
    if payload.knowledge_base_id:
        kb = _get_knowledge_base_by_id(db, user_id, payload.knowledge_base_id)
    else:
        knowledge_base_name = _normalise_kb_name(
            payload.knowledge_base_name or payload.source_name
        )
        kb = _get_or_create_knowledge_base(db, user_id, knowledge_base_name)

    file_id = str(uuid.uuid4())
    base_name = FileValidator.get_safe_filename(
        f"{payload.source_name or 'api_source'}.json"
    )
    unique_filename = f"{file_id}_{base_name}"

    user_upload_dir = Path(settings.upload_dir) / user_id
    user_upload_dir.mkdir(parents=True, exist_ok=True)
    storage_path = user_upload_dir / unique_filename

    user_markdown_dir = Path(settings.markdown_dir) / user_id / kb.knowledge_base_id
    user_markdown_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = user_markdown_dir / f"{file_id}.json"

    markdown_text = extraction_result["markdown_text"]
    storage_path.write_text(markdown_text, encoding="utf-8")
    markdown_path.write_text(markdown_text, encoding="utf-8")

    file_size_bytes = len(markdown_text.encode("utf-8"))
    now = datetime.utcnow()

    file_metadata = FileMetadata(
        file_id=file_id,
        user_id=user_id,
        knowledge_base_id=kb.knowledge_base_id,
        original_filename=base_name,
        safe_filename=unique_filename,
        file_extension="json",
        file_type=FileType.TEXT,
        file_size_bytes=file_size_bytes,
        storage_path=str(storage_path),
        markdown_path=str(markdown_path),
        status=DBProcessingStatus.COMPLETED,
        uploaded_at=now,
        processed_at=now,
        total_pages=extraction_result.get("pages_fetched", 0),
        processing_metadata=json.dumps(
            {
                "source_type": "api",
                "source_name": payload.source_name,
                "url": str(payload.url),
                "http_method": payload.http_method.value,
                "json_path": payload.json_path,
                "field_paths": payload.field_paths,
                "page_urls": extraction_result.get("page_urls", []),
            },
            ensure_ascii=False,
        ),
        word_count=len(markdown_text.split()),
    )
    db.add(file_metadata)

    kb.document_count += 1
    kb.total_size_bytes += file_size_bytes
    kb.updated_at = now

    db.commit()
    db.refresh(file_metadata)

    return {
        "file_id": file_metadata.file_id,
        "knowledge_base_id": kb.knowledge_base_id,
        "knowledge_base_name": kb.knowledge_base_name,
        "markdown_path": file_metadata.markdown_path,
        "storage_path": file_metadata.storage_path,
        "status": file_metadata.status.value,
    }


@router.post("/api/extract")
async def extract_api_source(
    payload: APISourceInput,
    user_id: str = Depends(get_user_id),
    db: Session = Depends(get_database_session),
):
    """
    Call the configured API endpoint and return the extracted JSON data.

    The response is already filtered by `json_path` and `field_paths`, so the
    returned `data` and `markdown_text` are ready for downstream chunking.
    """
    try:
        service = APISourceService()
        result = await service.fetch_and_extract(payload)
        persisted_file = _persist_api_source_result(db, user_id, payload, result)
        return {
            **persisted_file,
            "user_id": user_id,
            **result,
        }
    except httpx.HTTPStatusError as exc:
        logger.error("API source request failed: %s", exc, exc_info=True)
        detail = exc.response.text if exc.response is not None else str(exc)
        raise HTTPException(
            status_code=exc.response.status_code if exc.response is not None else 502,
            detail=f"API source request failed: {detail}",
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("API source extraction failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"API source extraction failed: {exc}",
        ) from exc
