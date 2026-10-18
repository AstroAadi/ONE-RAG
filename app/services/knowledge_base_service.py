"""
Knowledge Base helpers

A knowledge base is a named, user-owned collection of documents. Queries are
always scoped to exactly one knowledge base, which keeps retrieval results
from bleeding across unrelated document sets.

These helpers are shared by the knowledge base routes and the upload route
(which needs to resolve or create a KB while accepting a file).
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional, Tuple
import logging
import uuid

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.db.models import FileMetadata, KnowledgeBase

logger = logging.getLogger(__name__)

DEFAULT_KNOWLEDGE_BASE_NAME = "Default"


def normalize_kb_name(name: str) -> str:
    """Case-insensitive, whitespace-insensitive form used for uniqueness checks"""
    return " ".join(name.split()).lower()


def get_knowledge_base(
    db: Session,
    knowledge_base_id: str,
    user_id: str
) -> Optional[KnowledgeBase]:
    """Fetch a knowledge base, enforcing user ownership"""
    return db.query(KnowledgeBase).filter(
        KnowledgeBase.knowledge_base_id == knowledge_base_id,
        KnowledgeBase.user_id == user_id
    ).first()


def find_knowledge_base_by_name(
    db: Session,
    name: str,
    user_id: str
) -> Optional[KnowledgeBase]:
    """
    Find a user's knowledge base by name, ignoring case and extra whitespace.
    """
    target = normalize_kb_name(name)

    candidates = db.query(KnowledgeBase).filter(
        KnowledgeBase.user_id == user_id
    ).all()

    for candidate in candidates:
        if normalize_kb_name(candidate.knowledge_base_name) == target:
            return candidate

    return None


def create_knowledge_base(
    db: Session,
    user_id: str,
    name: str,
    description: Optional[str] = None
) -> KnowledgeBase:
    """
    Create a knowledge base. Caller is responsible for rejecting duplicates
    beforehand if it wants a friendly error message.
    """
    knowledge_base = KnowledgeBase(
        knowledge_base_id=str(uuid.uuid4()),
        user_id=user_id,
        knowledge_base_name=" ".join(name.split()),
        description=description,
        created_at=datetime.utcnow(),
        last_accessed=datetime.utcnow()
    )

    db.add(knowledge_base)
    db.commit()
    db.refresh(knowledge_base)

    logger.info(
        f"✅ Created knowledge base: {knowledge_base.knowledge_base_id} "
        f"('{knowledge_base.knowledge_base_name}') for user {user_id}"
    )

    return knowledge_base


def resolve_or_create_knowledge_base(
    db: Session,
    user_id: str,
    knowledge_base_id: Optional[str] = None,
    knowledge_base_name: Optional[str] = None
) -> Tuple[KnowledgeBase, bool]:
    """
    Resolve the knowledge base a new upload belongs to.

    Resolution order:
      1. explicit knowledge_base_id (must exist and belong to the user)
      2. knowledge_base_name - reused if it already exists, created otherwise
      3. the user's "Default" knowledge base, created on first use

    Returns (knowledge_base, created) where `created` reports whether a new
    knowledge base was inserted.

    Raises:
        ValueError: if knowledge_base_id is given but not found for this user.
    """
    if knowledge_base_id and knowledge_base_id.strip():
        knowledge_base = get_knowledge_base(
            db, knowledge_base_id.strip(), user_id
        )
        if not knowledge_base:
            raise ValueError(
                f"Knowledge base not found: {knowledge_base_id}"
            )
        return knowledge_base, False

    name = (knowledge_base_name or "").strip() or DEFAULT_KNOWLEDGE_BASE_NAME

    existing = find_knowledge_base_by_name(db, name, user_id)
    if existing:
        return existing, False

    return create_knowledge_base(db, user_id, name), True


def touch_knowledge_base(db: Session, knowledge_base: KnowledgeBase) -> None:
    """Record that a knowledge base was just read or written"""
    knowledge_base.last_accessed = datetime.utcnow()
    db.commit()


def summarize_knowledge_base(
    db: Session,
    knowledge_base: KnowledgeBase
) -> dict:
    """
    Aggregate document counts and sizes for a knowledge base.

    Counts come from the relational store (authoritative for ownership and
    file sizes) rather than the vector store, so they stay correct for
    documents that are uploaded but not yet indexed.
    """
    row = db.query(
        func.count(FileMetadata.file_id),
        func.coalesce(func.sum(FileMetadata.file_size_bytes), 0),
        func.coalesce(func.sum(FileMetadata.chunks_created), 0)
    ).filter(
        FileMetadata.knowledge_base_id == knowledge_base.knowledge_base_id,
        FileMetadata.user_id == knowledge_base.user_id
    ).one()

    document_count, total_size_bytes, total_chunks = row

    return {
        "knowledge_base_id": knowledge_base.knowledge_base_id,
        "user_id": knowledge_base.user_id,
        "knowledge_base_name": knowledge_base.knowledge_base_name,
        "description": knowledge_base.description,
        "document_count": int(document_count or 0),
        "total_chunks": int(total_chunks or 0),
        "total_size_bytes": int(total_size_bytes or 0),
        "created_at": knowledge_base.created_at,
        "updated_at": knowledge_base.updated_at,
        "last_accessed": knowledge_base.last_accessed,
    }
