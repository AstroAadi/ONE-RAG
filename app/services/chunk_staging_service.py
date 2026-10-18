"""
Chunk Staging Service

Ingestion is a three-phase flow so the user can inspect and adjust chunking
before paying the cost of embedding:

    1. upload   -> extract the document to markdown
    2. create   -> split the markdown into chunks and STAGE them here
    3. finalize -> embed the staged chunks and index them into the vector store

Staged chunks live as one JSON file per document under a per-user directory.
This keeps the (potentially large) chunk text out of the relational database
while still surviving a process restart between steps 2 and 3.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import logging

from app.core.config import settings

logger = logging.getLogger(__name__)

STAGING_DIR_NAME = "chunk_staging"


def _staging_root() -> Path:
    """Root directory for all staged chunks"""
    return Path(settings.markdown_dir).parent / STAGING_DIR_NAME


def staging_path(user_id: str, file_id: str) -> Path:
    """Path of the staging file for one document"""
    return _staging_root() / user_id / f"{file_id}.json"


def save_staged_chunks(
    user_id: str,
    file_id: str,
    chunks: List[Dict[str, Any]],
    chunking_strategy: str,
    parameters: Optional[Dict[str, Any]] = None
) -> Path:
    """
    Persist chunks for a document, replacing any previously staged set.

    Re-running the create step with different settings is expected, so this
    overwrites rather than appends.
    """
    path = staging_path(user_id, file_id)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "file_id": file_id,
        "user_id": user_id,
        "chunking_strategy": chunking_strategy,
        "parameters": parameters or {},
        "chunk_count": len(chunks),
        "chunks": chunks,
    }

    # Write to a temporary file first so an interrupted write cannot leave a
    # half-written staging file behind.
    temp_path = path.with_suffix(".json.tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False)
    temp_path.replace(path)

    logger.info(f"📦 Staged {len(chunks)} chunks for file {file_id} at {path}")

    return path


def load_staged_chunks(user_id: str, file_id: str) -> Optional[Dict[str, Any]]:
    """
    Load the staged chunk set for a document.

    Returns None when nothing has been staged (i.e. the create step has not
    run, or the staged set was already consumed and cleared).
    """
    path = staging_path(user_id, file_id)

    if not path.exists():
        return None

    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as e:
        logger.error(f"Could not read staged chunks for {file_id}: {e}")
        return None


def delete_staged_chunks(user_id: str, file_id: str) -> bool:
    """Remove the staging file for a document. Returns True if one was removed."""
    path = staging_path(user_id, file_id)

    try:
        if path.exists():
            path.unlink()
            logger.debug(f"Removed staged chunks for {file_id}")
            return True
    except OSError as e:
        logger.warning(f"Could not delete staged chunks for {file_id}: {e}")

    return False


def delete_user_staging(user_id: str) -> None:
    """Remove all staged chunks for a user (used when resetting their data)"""
    user_dir = _staging_root() / user_id

    if not user_dir.exists():
        return

    import shutil
    shutil.rmtree(user_dir, ignore_errors=True)
    logger.info(f"Removed staging directory for user {user_id}")
