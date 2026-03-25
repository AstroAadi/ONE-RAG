"""
Query Route - MULTI-TENANT VERSION
All queries now require knowledge_base_id for strict KB isolation.

Improvements in this version:
- Query preprocessing: extracts structured IDs, years, and noisy stop-phrases
  before embedding so the vector store gets a clean semantic query.
- Post-filtering: after vector search returns candidates, extracted metadata
  filters (e.g. record IDs, ISBNs, years) are applied in Python to narrow
  results — no changes required to VectorStoreService.
- LLM still receives the original user question for natural-language answering.
"""

from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from app.models.schemas import (
    QueryRequest, QueryResponse, SourceDocument,
    ConversationQueryRequest, SearchType
)
from app.services.vector_store_service import VectorStoreService
from app.services.llm_service import LLMService
from app.core.dependencies import get_user_id, get_database_session
from app.core.exceptions import (
    SearchError, LLMGenerationError, RAGException
)
from app.db.models import QueryLog, KnowledgeBase

import re
import time
import hashlib
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Tuple, List, Dict, Optional

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/query", tags=["Query"])


# ==================== LAZY SERVICE LOADING ====================

_vector_store: Optional[VectorStoreService] = None
_llm_service: Optional[LLMService] = None


def get_vector_store() -> VectorStoreService:
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStoreService()
    return _vector_store


def get_llm_service() -> LLMService:
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service


# ==================== QUERY PREPROCESSING ====================

@dataclass
class ParsedQuery:
    """Result of preprocessing a raw user question."""
    semantic_query: str                        # cleaned string for vector embedding
    filters: dict = field(default_factory=dict)  # exact-match metadata filters
    original: str = ""                         # untouched original question


# --------------- Patterns that should become metadata filters ----------------
# Each entry: (regex_pattern, metadata_field_name)
# Groups inside the pattern: the LAST group is taken as the filter value.
_FILTER_PATTERNS: List[Tuple[str, str]] = [
    # Book / record IDs  →  e.g. "bk-110", "prod-007"
    (r'\b([a-z]{2,6}-\d{2,6})\b',                  'record_id'),
    # Explicit file_id param  →  e.g. "file_id: abc123"
    (r'\b(?:file[_-]?id)\s*[:\s]\s*([a-zA-Z0-9_-]+)', 'file_id'),
    # ISBN  →  e.g. "ISBN 978-0-13-235088-4"
    (r'\b(?:isbn[\s:]*)([\d\-X]{10,17})\b',          'isbn'),
    # A bare 4-digit year used as filter intent  →  e.g. "published in 2008"
    (r'\b(?:published\s+in|year\s*[=:]?\s*)(\d{4})\b', 'published_year'),
    # Page number intent  →  e.g. "on page 42"
    (r'\b(?:on\s+page|page\s+number\s*)(\d+)\b',      'page_number'),
]

# --------------- Boilerplate phrases that pollute embeddings -----------------
_STOP_PHRASES: List[str] = [
    r'\b(?:tell\s+me|can\s+you\s+tell\s+me|i\s+(?:want|need)\s+to\s+know)\b',
    r'\b(?:please\s+(?:tell\s+me|find|show|give|provide))\b',
    r'\b(?:what\s+is\s+the(?:\s+name\s+of)?|what\s+are\s+the)\b',
    r'\b(?:who\s+is\s+the|who\s+are\s+the)\b',
    r'\b(?:give\s+me|show\s+me|let\s+me\s+know|find\s+(?:me|out))\b',
    r'\b(?:how\s+much|how\s+many)\b',
    r'\b(?:could\s+you|would\s+you|can\s+you)\b',
    r'\b(?:i\s+was\s+wondering|i\s+(?:am|\'m)\s+looking\s+for)\b',
    r'\b(?:is\s+there|are\s+there|do\s+you\s+(?:know|have))\b',
    # Filler conjunctions left behind after stripping IDs
    r'\b(?:and|with|having|for|about|of|in|the|a|an)\b(?=\s*$|\s*,|\s*\?)',
]

# Punctuation / whitespace cleanup applied last
_CLEANUP_PATTERN = re.compile(r'[\s,?.!]+')


def _extract_filters(text: str) -> Tuple[str, dict]:
    """
    Scan *text* for structured-ID patterns and record them in a filters dict.

    Crucially, the matched values are kept in the returned text — they are
    NOT removed. This means tokens like ``bk-110`` remain in the semantic
    query so the sparse (SPLADE) search can keyword-match them directly.
    The filters dict is used separately for post-filtering the results.
    """
    filters: dict = {}

    for pattern, field_name in _FILTER_PATTERNS:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            raw = matches[0]
            value = raw if isinstance(raw, str) else raw[-1]
            filters[field_name] = value.strip()
            # ← intentionally NOT removing the match from `text`

    return text, filters


def _strip_stop_phrases(text: str) -> str:
    """Remove question-boilerplate phrases that add noise to the embedding."""
    for phrase in _STOP_PHRASES:
        text = re.sub(phrase, ' ', text, flags=re.IGNORECASE)
    return text


def _normalize(text: str) -> str:
    """Collapse repeated spaces/punctuation and strip trailing junk."""
    # Replace multi-whitespace and leftover punctuation runs with a single space
    text = _CLEANUP_PATTERN.sub(' ', text).strip(' ,?.!')
    return text


def preprocess_query(question: str) -> ParsedQuery:
    """
    Convert a raw user question into a :class:`ParsedQuery`.

    Pipeline
    --------
    1. Scan for structured IDs / metadata hints → ``filters`` dict.
       The IDs are kept in the text so the sparse (SPLADE) leg of the
       hybrid search can keyword-match them.
    2. Strip common question boilerplate (stop-phrases).
    3. Normalize whitespace / punctuation.
    4. Fallback to a lightly cleaned version of the original if nothing remains.

    The returned ``semantic_query`` is what gets embedded and sent to the
    vector store.  ``filters`` are used for post-filtering after retrieval.
    The ``original`` question is kept intact for the LLM prompt.

    Examples
    --------
    >>> preprocess_query("What is the title of the book with book_id bk-110?")
    ParsedQuery(
        semantic_query="title book book_id bk-110",   # ID kept for SPLADE
        filters={"record_id": "bk-110"},              # also used for post-filter
        original="What is the title of the book with book_id bk-110?"
    )
    """
    # Step 1 – record structured IDs in filters (kept in text for SPLADE)
    working, filters = _extract_filters(question)

    # Step 2 – remove boilerplate stop-phrases
    working = _strip_stop_phrases(working)

    # Step 3 – clean up remaining punctuation / whitespace
    semantic_query = _normalize(working)

    # Step 4 – fallback guard: if we stripped everything meaningful, use a
    #           softened version of the original rather than an empty string
    if not semantic_query or len(semantic_query) < 4:
        semantic_query = _normalize(
            re.sub(r'\b(?:what|who|how|when|where|which|is|are|the|a|an)\b',
                   ' ', question, flags=re.IGNORECASE)
        ) or question.strip()

    logger.debug(
        "Query preprocessing | original=%r → semantic=%r | filters=%s",
        question[:80], semantic_query, filters
    )

    return ParsedQuery(
        semantic_query=semantic_query,
        filters=filters,
        original=question,
    )


# ==================== QUERY RESULT CACHE ====================

class QueryCache:
    """KB-aware, TTL-based in-memory query cache."""

    def __init__(self, ttl_seconds: int = 300):
        self._cache: dict = {}
        self.ttl = ttl_seconds

    # ------------------------------------------------------------------
    def _make_key(
        self,
        user_id: str,
        kb_id: str,
        question: str,
        top_k: int,
        file_id: Optional[str] = None,
    ) -> str:
        data = f"{user_id}:{kb_id}:{question}:{top_k}:{file_id or 'all'}"
        return hashlib.sha256(data.encode()).hexdigest()

    # ------------------------------------------------------------------
    def get(
        self,
        user_id: str,
        kb_id: str,
        question: str,
        top_k: int,
        file_id: Optional[str] = None,
    ) -> Optional[dict]:
        key = self._make_key(user_id, kb_id, question, top_k, file_id)
        entry = self._cache.get(key)
        if entry:
            result, ts = entry
            if datetime.utcnow() - ts < timedelta(seconds=self.ttl):
                logger.info("✓ Cache hit for query in KB %s", kb_id)
                return result
            del self._cache[key]
        return None

    # ------------------------------------------------------------------
    def set(
        self,
        user_id: str,
        kb_id: str,
        question: str,
        top_k: int,
        result: dict,
        file_id: Optional[str] = None,
    ) -> None:
        key = self._make_key(user_id, kb_id, question, top_k, file_id)
        self._cache[key] = (result, datetime.utcnow())

    # ------------------------------------------------------------------
    def clear_kb(self, kb_id: str) -> None:
        """Evict every cached entry that belongs to *kb_id*."""
        stale = [k for k, (v, _) in self._cache.items()
                 if v.get("knowledge_base_id") == kb_id]
        for key in stale:
            del self._cache[key]
        logger.info("Cache cleared for KB %s (%d entries removed)", kb_id, len(stale))


# Global singleton cache
query_cache = QueryCache(ttl_seconds=300)


# ==================== HELPER FUNCTIONS ====================

def safe_get_int(metadata: dict, key: str, default: int = 0) -> int:
    try:
        value = metadata.get(key, default)
        return int(value) if value is not None else default
    except (ValueError, TypeError):
        return default


def safe_get_str(metadata: dict, key: str, default: str = '') -> str:
    try:
        value = metadata.get(key, default)
        return str(value) if value is not None else default
    except (ValueError, TypeError):
        return default


def safe_get_score(score) -> float:
    try:
        return max(0.0, min(1.0, float(score)))
    except (ValueError, TypeError):
        return 0.0


def build_source_documents(
    documents: List[str],
    metadatas: List[Dict],
    scores: List[float],
    knowledge_base_id: str,
) -> Tuple[List[SourceDocument], str]:
    """Build :class:`SourceDocument` list and a concatenated context string."""
    sources: List[SourceDocument] = []
    context_parts: List[str] = []

    for i, (doc, metadata, score) in enumerate(zip(documents, metadatas, scores)):
        header = f"[Source {i + 1}]"

        page = safe_get_int(metadata, 'page_number', 0)
        if page > 0:
            header += f" [Page {page}]"

        chunk_type = safe_get_str(metadata, 'chunk_type', 'text')
        if chunk_type != 'text':
            header += f" [{chunk_type.upper()}]"

        context_parts.append(f"{header}:\n{doc}\n")

        sources.append(SourceDocument(
            content=doc[:500] + "..." if len(doc) > 500 else doc,
            full_text=doc,
            chunk_id=safe_get_str(metadata, 'chunk_id', f'chunk_{i}'),
            file_id=safe_get_str(metadata, 'file_id', 'unknown'),
            knowledge_base_id=safe_get_str(metadata, 'knowledge_base_id', knowledge_base_id),
            filename=safe_get_str(metadata, 'filename', 'unknown'),
            relevance_score=safe_get_score(score),
            page_number=page,
            chunk_type=chunk_type,
            chunk_index=safe_get_int(metadata, 'chunk_index', i),
            semantic_score=(
                safe_get_score(metadata['semantic_score'])
                if metadata.get('semantic_score') else None
            ),
            keyword_score=(
                safe_get_score(metadata['keyword_score'])
                if metadata.get('keyword_score') else None
            ),
        ))

    context = "\n\n".join(context_parts)
    return sources, context


def log_query_to_db(
    db: Session,
    user_id: str,
    kb_id: str,
    question: str,
    answer: str,
    top_k: int,
    chunks_retrieved: int,
    processing_time: float,
    confidence: float,
    search_type: str,
    conversation_id: Optional[str] = None,
) -> None:
    """Persist a query record; silently swallows DB errors."""
    try:
        query_log = QueryLog(
            user_id=user_id,
            knowledge_base_id=kb_id,
            question=question,
            answer=answer,
            top_k=top_k,
            chunks_retrieved=chunks_retrieved,
            processing_time_seconds=processing_time,
            confidence_score=confidence,
            search_type=search_type,
            queried_at=datetime.utcnow(),
            conversation_id=conversation_id,
        )
        db.add(query_log)
        db.commit()
    except Exception as exc:
        logger.warning("Failed to log query: %s", exc)
        db.rollback()


def _verify_kb_access(
    db: Session,
    knowledge_base_id: str,
    user_id: str,
) -> KnowledgeBase:
    """Return the KB object or raise 404."""
    kb = db.query(KnowledgeBase).filter(
        KnowledgeBase.knowledge_base_id == knowledge_base_id,
        KnowledgeBase.user_id == user_id,
    ).first()

    if not kb:
        raise HTTPException(
            status_code=404,
            detail=f"Knowledge base not found or access denied: {knowledge_base_id}",
        )
    return kb


# ==================== POST-FILTERING ====================

# Maps the filter key produced by _extract_filters() to the list of payload
# field names we check in the Qdrant metadata dict.
# Order matters — first match wins.
_FILTER_KEY_ALIASES: Dict[str, List[str]] = {
    # "record_id":      ["record_id", "book_id", "product_id", "item_id", "id"],
    "file_id":        ["file_id"],
    # "isbn":           ["isbn"],
    # "published_year": ["published_year", "year", "publication_year"],
    "page_number":    ["page_number", "page"],
}


def _chunk_matches_filter(
    doc_text: str,
    metadata: Dict,
    filter_key: str,
    filter_value: str,
) -> bool:
    """
    Return True if this chunk satisfies a single extracted filter.

    Strategy (in order):
    1. Check every known alias for *filter_key* in the payload metadata dict.
    2. If none matched, fall back to a substring scan of the raw document text
       (handles cases where the field was not separately indexed in the payload).
    """
    normalised_value = str(filter_value).lower().strip()

    # ── 1. Metadata dict check ────────────────────────────────────────────
    aliases = _FILTER_KEY_ALIASES.get(filter_key, [filter_key])
    for alias in aliases:
        raw = metadata.get(alias)
        if raw is not None and str(raw).lower().strip() == normalised_value:
            return True

    # ── 2. Full-text fallback ─────────────────────────────────────────────
    # Catches values embedded in the stored document text (e.g. JSON blobs)
    # but only when the value is reasonably specific (≥ 4 chars) to avoid
    # false positives on short tokens like "42".
    if len(normalised_value) >= 4 and normalised_value in doc_text.lower():
        return True

    return False


def apply_metadata_filters(
    documents: List[str],
    metadatas: List[Dict],
    scores: List[float],
    filters: dict,
    top_k: int,
) -> Tuple[List[str], List[Dict], List[float]]:
    """
    Post-filter retrieved chunks against the structured filters extracted by
    ``preprocess_query()``.

    All filter conditions must be satisfied (AND semantics).

    Fallback behaviour
    ------------------
    If *no* chunk passes the filter — which can happen when the vector store
    returned a broad candidate set that genuinely does not contain the
    requested record — the function returns the original unfiltered results
    so the LLM can still give a "not found" answer rather than receiving
    an empty context.

    Parameters
    ----------
    documents, metadatas, scores : parallel lists from ``VectorStoreService.search``
    filters : dict extracted by ``preprocess_query`` (may be empty)
    top_k : final result count cap
    """
    if not filters:
        return documents[:top_k], metadatas[:top_k], scores[:top_k]

    kept_docs, kept_metas, kept_scores = [], [], []

    for doc, meta, score in zip(documents, metadatas, scores):
        if all(
            _chunk_matches_filter(doc, meta, fk, fv)
            for fk, fv in filters.items()
        ):
            kept_docs.append(doc)
            kept_metas.append(meta)
            kept_scores.append(score)

    if not kept_docs:
        logger.warning(
            "Post-filter found 0 matches for filters=%s — "
            "returning unfiltered results so LLM can answer 'not found'",
            filters,
        )
        return documents[:top_k], metadatas[:top_k], scores[:top_k]

    logger.info(
        "Post-filter: %d/%d chunks kept for filters=%s",
        len(kept_docs), len(documents), filters,
    )
    return kept_docs[:top_k], kept_metas[:top_k], kept_scores[:top_k]

# ==================== MAIN QUERY ENDPOINTS ====================

@router.post("/", response_model=QueryResponse)
async def query_knowledge_base(
    request: QueryRequest,
    user_id: str = Depends(get_user_id),
    db: Session = Depends(get_database_session),
):
    """
    Query a specific knowledge base.

    Requires ``knowledge_base_id`` (mandatory) for strict KB isolation.
    The raw question is preprocessed before vector search:
    structured IDs become metadata filters and stop-phrases are removed,
    giving the embedding model a clean semantic signal.

    Headers
    -------
    X-User-ID : str
        User identifier (required).

    Body
    ----
    question : str
        Natural-language question.
    knowledge_base_id : str
        KB to query (required).
    top_k : int
        Number of chunks to retrieve (default 5).
    search_type : dense | sparse | hybrid
        Retrieval strategy (default hybrid).
    temperature : float
        LLM sampling temperature (default 0.7).
    """
    start_time = time.time()

    try:
        # ── 1. Ownership check ────────────────────────────────────────────
        kb = _verify_kb_access(db, request.knowledge_base_id, user_id)
        kb.last_accessed = datetime.utcnow()
        db.commit()

        search_type = request.search_type or SearchType.HYBRID

        # ── 2. Preprocess query ───────────────────────────────────────────
        parsed = preprocess_query(request.question)
        logger.info(
            "Query | KB='%s' (%s) | user=%s | "
            "original='%s' → semantic='%s' | filters=%s | "
            "search_type=%s | top_k=%d",
            kb.knowledge_base_name, kb.knowledge_base_id,
            user_id,
            request.question[:60], parsed.semantic_query,
            parsed.filters,
            search_type.value, request.top_k,
        )

        # ── 3. Cache lookup (keyed on original question for UX consistency) ─
        cached = query_cache.get(
            user_id, request.knowledge_base_id,
            request.question, request.top_k,
        )
        if cached:
            logger.info("Returning cached result")
            return QueryResponse(**cached)

        # ── 4. Vector search — clean semantic query, wider candidate pool ────
        # Fetch more candidates when filters are active so post-filtering
        # still yields enough results after narrowing.
        fetch_k = request.top_k * 6 if parsed.filters else request.top_k
        try:
            documents, metadatas, scores = get_vector_store().search(
                query=parsed.semantic_query,   # ← cleaned, not raw
                user_id=user_id,
                knowledge_base_id=request.knowledge_base_id,
                top_k=fetch_k,
                search_type=search_type.value,
            )
        except Exception as exc:
            logger.error("Vector search failed: %s", exc)
            raise SearchError(f"Search failed: {exc}") from exc

        if not documents:
            raise SearchError(
                f"No documents found in knowledge base '{kb.knowledge_base_name}'. "
                "Please upload and process documents first."
            )

        # ── 5. Post-filter by extracted metadata (IDs, years, etc.) ──────
        documents, metadatas, scores = apply_metadata_filters(
            documents, metadatas, scores,
            parsed.filters,
            top_k=request.top_k,
        )

        # ── 6. Build context ──────────────────────────────────────────────
        sources, context = build_source_documents(
            documents, metadatas, scores, request.knowledge_base_id
        )

        # ── 6. Generate answer (LLM sees the original question) ───────────
        try:
            answer = get_llm_service().generate_answer(
                question=request.question,   # ← original kept for naturalness
                context=context,
                user_id=user_id,
                db=db,
                conversation_id=None,
                include_citations=True,
                temperature=request.temperature or 0.7,
            )
        except Exception as exc:
            logger.error("LLM generation failed: %s", exc)
            raise LLMGenerationError(f"Failed to generate answer: {exc}") from exc

        # ── 7. Hallucination guard ────────────────────────────────────────
        is_hallucination, confidence = get_llm_service().detect_hallucination(
            answer, context
        )
        if is_hallucination:
            logger.warning("Low confidence detected: %.2f", confidence)
            answer += f"\n\n⚠️ Note: Lower confidence ({confidence:.2f}). Please verify."

        processing_time = time.time() - start_time

        # ── 8. Build + cache response ─────────────────────────────────────
        response_data = {
            "question": request.question,
            "answer": answer,
            "sources": sources,
            "processing_time": round(processing_time, 2),
            "search_type": search_type,
            "confidence_score": round(confidence, 3),
            "knowledge_base_id": request.knowledge_base_id,
        }

        query_cache.set(
            user_id, request.knowledge_base_id,
            request.question, request.top_k, response_data,
        )

        # ── 9. Audit log ──────────────────────────────────────────────────
        log_query_to_db(
            db, user_id, request.knowledge_base_id,
            request.question, answer,
            request.top_k, len(documents),
            processing_time, confidence,
            search_type.value, None,
        )

        logger.info(
            "✓ Query completed | KB='%s' | time=%.2fs | confidence=%.2f",
            kb.knowledge_base_name, processing_time, confidence,
        )

        return QueryResponse(**response_data)

    except SearchError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except LLMGenerationError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    except RAGException as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc))
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Unexpected query error: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {exc}",
        )


@router.post("/conversation", response_model=QueryResponse)
async def query_with_conversation(
    request: ConversationQueryRequest,
    user_id: str = Depends(get_user_id),
    db: Session = Depends(get_database_session),
):
    """
    Conversational query with KB context.

    Maintains per-conversation history within a KB context.
    Applies the same query preprocessing as the standard query endpoint.

    Headers
    -------
    X-User-ID : str
        User identifier (required).

    Body
    ----
    Same as ``POST /`` plus:
    conversation_id : str
        Unique conversation identifier used to load/persist history.
    """
    start_time = time.time()

    try:
        # ── 1. Ownership check ────────────────────────────────────────────
        kb = _verify_kb_access(db, request.knowledge_base_id, user_id)
        kb.last_accessed = datetime.utcnow()
        db.commit()

        search_type = request.search_type or SearchType.HYBRID

        # ── 2. Preprocess query ───────────────────────────────────────────
        parsed = preprocess_query(request.question)
        logger.info(
            "Conversation query | KB='%s' | conv=%s | "
            "original='%s' → semantic='%s' | filters=%s",
            kb.knowledge_base_name, request.conversation_id,
            request.question[:60], parsed.semantic_query, parsed.filters,
        )

        # ── 3. Vector search — wider candidate pool when filters present ──
        fetch_k = request.top_k * 6 if parsed.filters else request.top_k
        try:
            documents, metadatas, scores = get_vector_store().search(
                query=parsed.semantic_query,
                user_id=user_id,
                knowledge_base_id=request.knowledge_base_id,
                top_k=fetch_k,
                search_type=search_type.value,
            )
        except Exception as exc:
            logger.error("Vector search failed: %s", exc)
            raise SearchError(f"Search failed: {exc}") from exc

        if not documents:
            raise SearchError(
                f"No documents found in KB '{kb.knowledge_base_name}'."
            )

        # ── 4. Post-filter by extracted metadata (IDs, years, etc.) ──────
        documents, metadatas, scores = apply_metadata_filters(
            documents, metadatas, scores,
            parsed.filters,
            top_k=request.top_k,
        )

        # ── 5. Build context ──────────────────────────────────────────────
        sources, context = build_source_documents(
            documents, metadatas, scores, request.knowledge_base_id
        )

        # ── 6. Generate answer with conversation history ──────────────────
        try:
            answer = get_llm_service().generate_answer(
                question=request.question,
                context=context,
                user_id=user_id,
                db=db,
                conversation_id=request.conversation_id,
                include_citations=True,
                temperature=request.temperature or 0.7,
            )
        except Exception as exc:
            logger.error("LLM generation failed: %s", exc)
            raise LLMGenerationError(f"Failed to generate answer: {exc}") from exc

        # ── 7. Hallucination guard ────────────────────────────────────────
        is_hallucination, confidence = get_llm_service().detect_hallucination(
            answer, context
        )
        if is_hallucination:
            answer += f"\n\n⚠️ Note: Lower confidence ({confidence:.2f})."

        processing_time = time.time() - start_time

        # ── 8. Build response ─────────────────────────────────────────────
        response_data = {
            "question": request.question,
            "answer": answer,
            "sources": sources,
            "processing_time": round(processing_time, 2),
            "search_type": search_type,
            "confidence_score": round(confidence, 3),
            "knowledge_base_id": request.knowledge_base_id,
            "conversation_id": request.conversation_id,
        }

        # ── 9. Audit log ──────────────────────────────────────────────────
        log_query_to_db(
            db, user_id, request.knowledge_base_id,
            request.question, answer,
            request.top_k, len(documents),
            processing_time, confidence,
            search_type.value, request.conversation_id,
        )

        logger.info(
            "✓ Conversation query completed | KB='%s' | conv=%s | time=%.2fs",
            kb.knowledge_base_name, request.conversation_id, processing_time,
        )

        return QueryResponse(**response_data)

    except SearchError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except LLMGenerationError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    except RAGException as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc))
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Conversation query error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error: {exc}")


# ==================== CONVERSATION MANAGEMENT ====================

@router.post("/clear-conversation/{conversation_id}")
async def clear_conversation(
    conversation_id: str,
    user_id: str = Depends(get_user_id),
    db: Session = Depends(get_database_session),
):
    """Clear conversation history for *conversation_id*."""
    try:
        get_llm_service().clear_conversation(conversation_id, user_id, db)
        return {
            "message": f"Conversation {conversation_id} cleared",
            "conversation_id": conversation_id,
            "user_id": user_id,
        }
    except Exception as exc:
        logger.error("Error clearing conversation: %s", exc)
        raise HTTPException(status_code=500,
                            detail=f"Error clearing conversation: {exc}")


@router.get("/history/{conversation_id}")
async def get_conversation_history(
    conversation_id: str,
    user_id: str = Depends(get_user_id),
    db: Session = Depends(get_database_session),
):
    """Return the last 8 messages for *conversation_id*."""
    try:
        from app.db.models import ConversationHistory
        from sqlalchemy import desc

        messages = (
            db.query(ConversationHistory)
            .filter(
                ConversationHistory.conversation_id == conversation_id,
                ConversationHistory.user_id == user_id,
            )
            .order_by(desc(ConversationHistory.created_at))
            .limit(8)
            .all()
        )
        messages.reverse()

        history = []
        kb_id = None
        for msg in messages:
            history.append({
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.created_at.isoformat(),
            })
            if not kb_id and msg.knowledge_base_id:
                kb_id = msg.knowledge_base_id

        kb_info = None
        if kb_id:
            kb = db.query(KnowledgeBase).filter(
                KnowledgeBase.knowledge_base_id == kb_id
            ).first()
            if kb:
                kb_info = {
                    "knowledge_base_id": kb.knowledge_base_id,
                    "knowledge_base_name": kb.knowledge_base_name,
                }

        return {
            "conversation_id": conversation_id,
            "user_id": user_id,
            "knowledge_base": kb_info,
            "message_count": len(history),
            "messages": history,
        }

    except Exception as exc:
        logger.error("Error retrieving history: %s", exc)
        raise HTTPException(status_code=500,
                            detail=f"Error retrieving conversation history: {exc}")


@router.get("/my-queries")
async def get_my_queries(
    user_id: str = Depends(get_user_id),
    db: Session = Depends(get_database_session),
    knowledge_base_id: Optional[str] = None,
    limit: int = 20,
):
    """
    Return recent queries for the current user.

    Parameters
    ----------
    knowledge_base_id : str, optional
        Filter to a specific KB.
    limit : int
        Maximum number of results (1–100, default 20).
    """
    try:
        from sqlalchemy import desc

        limit = max(1, min(limit, 100))

        query = db.query(QueryLog).filter(QueryLog.user_id == user_id)

        if knowledge_base_id:
            _verify_kb_access(db, knowledge_base_id, user_id)
            query = query.filter(QueryLog.knowledge_base_id == knowledge_base_id)

        rows = query.order_by(desc(QueryLog.queried_at)).limit(limit).all()

        # Cache KB names to avoid repeated DB hits
        kb_name_cache: dict = {}

        def _kb_name(kb_id: Optional[str]) -> str:
            if not kb_id:
                return "Unknown"
            if kb_id not in kb_name_cache:
                kb = db.query(KnowledgeBase).filter(
                    KnowledgeBase.knowledge_base_id == kb_id
                ).first()
                kb_name_cache[kb_id] = kb.knowledge_base_name if kb else "Unknown"
            return kb_name_cache[kb_id]

        query_list = [
            {
                "id": q.id,
                "question": (
                    q.question[:100] + "..."
                    if len(q.question) > 100 else q.question
                ),
                "knowledge_base_id": q.knowledge_base_id,
                "knowledge_base_name": _kb_name(q.knowledge_base_id),
                "queried_at": q.queried_at.isoformat(),
                "processing_time": q.processing_time_seconds,
                "confidence_score": q.confidence_score,
                "search_type": q.search_type,
                "conversation_id": q.conversation_id,
            }
            for q in rows
        ]

        return {
            "user_id": user_id,
            "knowledge_base_id": knowledge_base_id,
            "total_queries": len(query_list),
            "queries": query_list,
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Error retrieving queries: %s", exc)
        raise HTTPException(status_code=500,
                            detail=f"Error retrieving query history: {exc}")


@router.post("/clear-cache")
async def clear_query_cache(
    user_id: str = Depends(get_user_id),
    knowledge_base_id: Optional[str] = None,
):
    """
    Clear the query result cache.

    If *knowledge_base_id* is provided, only entries for that KB are removed.
    Otherwise the entire cache is wiped.
    """
    try:
        if knowledge_base_id:
            query_cache.clear_kb(knowledge_base_id)
            return {
                "message": f"Cache cleared for KB {knowledge_base_id}",
                "knowledge_base_id": knowledge_base_id,
            }
        query_cache._cache.clear()
        return {
            "message": "All cache cleared",
            "user_id": user_id,
        }
    except Exception as exc:
        logger.error("Error clearing cache: %s", exc)
        raise HTTPException(status_code=500,
                            detail=f"Error clearing cache: {exc}")