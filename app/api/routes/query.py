"""
Query Route - MULTI-TENANT VERSION
All queries now require knowledge_base_id for strict KB isolation
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
from app.core.config import settings
from app.core.exceptions import (
    SearchError, LLMGenerationError, RAGException
)
from app.db.models import QueryLog, KnowledgeBase, FileMetadata, ProcessingStatus
import time
import logging
import uuid
from datetime import datetime, timedelta
from typing import Tuple, List, Dict
from functools import lru_cache
import hashlib
import re

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/query", tags=["Query"])

# Initialize services with lazy loading
_vector_store = None
_llm_service = None

def get_vector_store():
    """Get or initialize vector store (lazy loading)"""
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStoreService()
    return _vector_store

def get_llm_service():
    """Get or initialize LLM service (lazy loading)"""
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service


# ==================== QUERY RESULT CACHE ====================

class QueryCache:
    """KB-aware query cache"""
    
    def __init__(self, ttl_seconds: int = 300):
        self.cache = {}
        self.ttl = ttl_seconds
    
    def _make_key(
        self,
        user_id: str,
        kb_id: str,
        question: str,
        top_k: int,
        file_id: str = None,
        min_relevance_score: float = None,
    ) -> str:
        """Generate cache key with KB isolation"""
        score_key = "default" if min_relevance_score is None else f"{min_relevance_score:.3f}"
        data = f"{user_id}:{kb_id}:{question}:{top_k}:{file_id or 'all'}:{score_key}"
        return hashlib.sha256(data.encode()).hexdigest()
    
    def get(
        self,
        user_id: str,
        kb_id: str,
        question: str,
        top_k: int,
        file_id: str = None,
        min_relevance_score: float = None,
    ) -> dict:
        """Get cached result"""
        key = self._make_key(user_id, kb_id, question, top_k, file_id, min_relevance_score)
        
        if key in self.cache:
            result, timestamp = self.cache[key]
            if datetime.utcnow() - timestamp < timedelta(seconds=self.ttl):
                logger.info(f"✓ Cache hit for query in KB {kb_id}")
                return result
            else:
                del self.cache[key]
        
        return None
    
    def set(
        self,
        user_id: str,
        kb_id: str,
        question: str,
        top_k: int,
        result: dict,
        file_id: str = None,
        min_relevance_score: float = None,
    ):
        """Cache result"""
        key = self._make_key(user_id, kb_id, question, top_k, file_id, min_relevance_score)
        self.cache[key] = (result, datetime.utcnow())
    
    def clear_kb(self, kb_id: str):
        """Clear cache for specific KB"""
        keys_to_delete = [k for k in self.cache.keys() if kb_id in k]
        for key in keys_to_delete:
            del self.cache[key]


# Global cache
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


def safe_get_score(score: float) -> float:
    try:
        return max(0.0, min(1.0, float(score)))
    except (ValueError, TypeError):
        return 0.0


def is_greeting_query(question: str) -> bool:
    """Detect simple greeting-only queries that should not hit retrieval."""
    normalized = re.sub(r"\s+", " ", (question or "").strip().lower())
    greeting_patterns = {
        "hi", "hello", "hey", "hii", "helo", "hola",
        "good morning", "good afternoon", "good evening",
        "how are you", "how are you doing", "what's up", "whats up",
    }
    return normalized in greeting_patterns


def build_fallback_response(
    question: str,
    knowledge_base_id: str,
    conversation_id: str,
    answer: str,
    processing_time: float,
    search_type: SearchType,
    confidence_score: float,
) -> Dict:
    """Return a regular response shape for greeting and no-context cases."""
    return {
        "question": question,
        "answer": answer,
        "sources": [],
        "processing_time": round(processing_time, 2),
        "search_type": search_type,
        "confidence_score": round(confidence_score, 3),
        "knowledge_base_id": knowledge_base_id,
        "conversation_id": conversation_id,
    }


def extract_query_identifiers(question: str) -> List[str]:
    """Extract exact identifiers such as bk-110, wh-001-east, UUID-like tokens."""
    candidates = re.findall(r"\b[a-zA-Z0-9]+(?:[-_:][a-zA-Z0-9]+)+\b", (question or "").lower())
    identifiers = []
    for token in candidates:
        if any(ch.isdigit() for ch in token):
            identifiers.append(token)
    return list(dict.fromkeys(identifiers))


def extract_query_phrases(question: str) -> List[str]:
    """Extract likely exact title/name phrases from the user's question."""
    question = question or ""
    phrases: List[str] = []

    # Quoted phrases are the strongest signal.
    for match in re.finditer(r'"([^"]{2,})"|\'([^\']{2,})\'', question):
        phrase = next((group for group in match.groups() if group), "").strip()
        if len(phrase.split()) >= 2:
            phrases.append(phrase.lower())

    # Also capture title-cased runs such as "The Glass Castle".
    tokens = re.findall(r"[A-Za-z0-9'*_-]+", question)
    connectors = {"of", "the", "and", "a", "an", "in", "to", "for", "with", "on", "at"}

    i = 0
    while i < len(tokens):
        token = tokens[i]
        if not token or not token[0].isupper():
            i += 1
            continue

        phrase_tokens = [token]
        uppercase_count = 1
        j = i + 1
        while j < len(tokens):
            candidate = tokens[j]
            if candidate and candidate[0].isupper():
                phrase_tokens.append(candidate)
                uppercase_count += 1
                j += 1
                continue

            lower_candidate = candidate.lower()
            if (
                lower_candidate in connectors and
                j + 1 < len(tokens) and
                tokens[j + 1] and
                tokens[j + 1][0].isupper()
            ):
                phrase_tokens.append(candidate)
                j += 1
                continue

            break

        if uppercase_count >= 2:
            while phrase_tokens and phrase_tokens[-1].lower() in connectors:
                phrase_tokens.pop()
            if phrase_tokens:
                phrases.append(" ".join(phrase_tokens).lower())

        i = max(j, i + 1)

    return list(dict.fromkeys(phrases))


def _metadata_text(metadata: Dict) -> str:
    """Flatten simple metadata values for exact-match checks."""
    parts: List[str] = []
    for value in (metadata or {}).values():
        if isinstance(value, (str, int, float, bool)):
            parts.append(str(value))
        elif isinstance(value, list):
            parts.extend(str(item) for item in value if isinstance(item, (str, int, float, bool)))
    return " ".join(parts).lower()


def filter_results_by_relevance(
    documents: List[str],
    metadatas: List[Dict],
    scores: List[float],
    min_relevance_score: float,
    question: str = "",
) -> Tuple[List[str], List[Dict], List[float]]:
    """Keep only strong results, but rescue exact matches when present."""
    filtered = [
        (doc, metadata, safe_get_score(score))
        for doc, metadata, score in zip(documents, metadatas, scores)
        if safe_get_score(score) >= min_relevance_score
    ]

    if filtered:
        docs, metas, filtered_scores = zip(*filtered)
        return list(docs), list(metas), list(filtered_scores)

    identifiers = extract_query_identifiers(question)
    phrases = extract_query_phrases(question)
    exact_terms = identifiers + phrases
    if not exact_terms:
        return [], [], []

    relaxed_floor = min(min_relevance_score, 0.35)
    rescued = []
    for doc, metadata, score in zip(documents, metadatas, scores):
        normalized_score = safe_get_score(score)
        if normalized_score < relaxed_floor:
            continue

        haystacks = [str(doc or "").lower(), _metadata_text(metadata)]
        exact_hits = sum(
            1 for term in exact_terms
            if any(term in haystack for haystack in haystacks)
        )
        if exact_hits <= 0:
            continue

        rescued.append((exact_hits, normalized_score, doc, metadata))

    if not rescued:
        return [], [], []

    rescued.sort(key=lambda item: (item[0], item[1]), reverse=True)
    docs = [item[2] for item in rescued]
    metas = [item[3] for item in rescued]
    rescued_scores = [item[1] for item in rescued]
    return docs, metas, rescued_scores


def extract_query_keywords(question: str) -> List[str]:
    """Extract simple content-bearing keywords for lexical fallback."""
    stop_words = {
        "what", "is", "the", "a", "an", "of", "about", "how", "to", "in",
        "on", "for", "and", "or", "are", "was", "were", "be", "does", "do",
        "did", "tell", "me", "please", "can", "you",
        "name", "title", "book", "books", "having", "have",
        "book_id", "id",
        "which", "who", "whom", "whose", "much", "many",
    }
    words = re.findall(r"\b[a-zA-Z][a-zA-Z0-9_-]{2,}\b", (question or "").lower())
    return [word for word in words if word not in stop_words]


def build_lexical_fallback_sources(
    db: Session,
    user_id: str,
    knowledge_base_id: str,
    question: str,
    top_k: int,
) -> Tuple[List[SourceDocument], str]:
    """
    Search markdown files directly when vector retrieval misses.
    This is a fallback, not a replacement for vector search.
    """
    identifiers = extract_query_identifiers(question)
    phrases = extract_query_phrases(question)
    keywords = extract_query_keywords(question)
    if not identifiers and not phrases and not keywords:
        return [], ""

    candidate_files = db.query(FileMetadata).filter(
        FileMetadata.user_id == user_id,
        FileMetadata.knowledge_base_id == knowledge_base_id,
        FileMetadata.status == ProcessingStatus.COMPLETED,
        FileMetadata.markdown_path.isnot(None),
    ).all()

    matches: List[Tuple[float, Dict[str, str]]] = []
    for file_meta in candidate_files:
        try:
            with open(file_meta.markdown_path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception:
            continue

        lowered = content.lower()
        phrase_hits = sum(lowered.count(phrase) for phrase in phrases)
        identifier_hits = sum(lowered.count(identifier) for identifier in identifiers)
        keyword_hits = sum(lowered.count(keyword) for keyword in keywords)

        # If the user asked for an exact phrase or identifier, require it in the file.
        if phrases and phrase_hits <= 0:
            continue
        if identifiers and identifier_hits <= 0:
            continue

        hit_count = (phrase_hits * 30) + (identifier_hits * 20) + keyword_hits
        if hit_count <= 0:
            continue

        if phrase_hits > 0:
            search_terms = phrases
        elif identifier_hits > 0:
            search_terms = identifiers
        else:
            search_terms = keywords
        best_pos = min(
            (lowered.find(term) for term in search_terms if lowered.find(term) != -1),
            default=-1,
        )
        if best_pos == -1:
            continue

        snippet_start = max(0, best_pos - 250)
        snippet_end = min(len(content), best_pos + 750)
        snippet = content[snippet_start:snippet_end].strip()
        score = min(0.99, 0.35 + (0.25 * phrase_hits) + (0.2 * identifier_hits) + (0.03 * keyword_hits))

        matches.append((
            score,
            {
                "content": snippet,
                "filename": file_meta.safe_filename,
                "file_id": file_meta.file_id,
                "knowledge_base_id": knowledge_base_id,
                "chunk_id": f"lexical::{file_meta.file_id}::{best_pos}",
                "page_number": 0,
                "chunk_type": "lexical_fallback",
                "chunk_index": 0,
            },
        ))

    if not matches:
        return [], ""

    matches.sort(key=lambda item: item[0], reverse=True)
    selected = matches[:top_k]

    documents = [item[1]["content"] for item in selected]
    metadatas = [item[1] for item in selected]
    scores = [item[0] for item in selected]
    return build_source_documents(documents, metadatas, scores, knowledge_base_id)


def build_source_documents(
    documents: List[str],
    metadatas: List[Dict],
    scores: List[float],
    knowledge_base_id: str
) -> Tuple[List[SourceDocument], str]:
    """Build source documents with KB info"""
    sources = []
    context_parts = []
    
    for i, (doc, metadata, score) in enumerate(zip(documents, metadatas, scores)):
        # Build source header
        source_header = f"[Source {i+1}]"
        
        page = safe_get_int(metadata, 'page_number', 0)
        if page > 0:
            source_header += f" [Page {page}]"
        
        chunk_type = safe_get_str(metadata, 'chunk_type', 'text')
        if chunk_type != 'text':
            source_header += f" [{chunk_type.upper()}]"
        
        # Add to context
        context_parts.append(f"{source_header}:\n{doc}\n")
        
        # Create source document with KB info
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
            semantic_score=safe_get_score(metadata.get('semantic_score', 0.0)) if metadata.get('semantic_score') else None,
            keyword_score=safe_get_score(metadata.get('keyword_score', 0.0)) if metadata.get('keyword_score') else None
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
    conversation_id: str = None
):
    """Log query with KB tracking"""
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
            conversation_id=conversation_id
        )
        db.add(query_log)
        db.commit()
    except Exception as e:
        logger.warning(f"Failed to log query: {e}")
        db.rollback()


# ==================== MAIN QUERY ENDPOINTS ====================

@router.post("/", response_model=QueryResponse)
async def query_knowledge_base(
    request: QueryRequest,
    user_id: str = Depends(get_user_id),
    db: Session = Depends(get_database_session)
):
    """
    Query a specific knowledge base
    
    NEW: Requires knowledge_base_id (MANDATORY)
    Ensures strict KB isolation - no cross-KB queries
    
    Headers:
        X-User-ID: User identifier (REQUIRED)
    
    Body:
        question: Question to ask
        knowledge_base_id: KB to query (REQUIRED)
        top_k: Number of results (default: 5)
        file_id: Optional file filter
        search_type: dense/sparse/hybrid (default: hybrid)
        temperature: LLM temperature (optional)
    
    Returns:
        QueryResponse with answer, sources, and KB info
    """
    start_time = time.time()
    
    try:
        min_relevance_score = (
            request.min_relevance_score
            if request.min_relevance_score is not None
            else settings.retrieval_min_score
        )
        conversation_id = getattr(request, "conversation_id", None) or str(uuid.uuid4())

        # CRITICAL: Verify KB ownership
        kb = db.query(KnowledgeBase).filter(
            KnowledgeBase.knowledge_base_id == request.knowledge_base_id,
            KnowledgeBase.user_id == user_id
        ).first()
        
        if not kb:
            raise HTTPException(
                status_code=404,
                detail=f"Knowledge base not found or access denied: {request.knowledge_base_id}"
            )
        
        # Update last accessed
        kb.last_accessed = datetime.utcnow()
        db.commit()
        
        search_type = request.search_type or SearchType.HYBRID
        
        logger.info(
            f"Query: KB='{kb.knowledge_base_name}' ({kb.knowledge_base_id}), "
            f"user={user_id}, query='{request.question[:50]}...', "
            f"search_type={search_type.value}, top_k={request.top_k}"
        )

        if is_greeting_query(request.question):
            answer = "Hello. How can I help you with the knowledge base?"
            processing_time = time.time() - start_time
            response_data = build_fallback_response(
                question=request.question,
                knowledge_base_id=request.knowledge_base_id,
                conversation_id=conversation_id,
                answer=answer,
                processing_time=processing_time,
                search_type=search_type,
                confidence_score=1.0,
            )
            log_query_to_db(
                db, user_id, request.knowledge_base_id, request.question, answer,
                request.top_k, 0, processing_time, 1.0, search_type.value, conversation_id
            )
            return QueryResponse(**response_data)
        
        # Vector search WITH KB ISOLATION
        try:
            documents, metadatas, scores = get_vector_store().search(
                query=request.question,
                user_id=user_id,
                knowledge_base_id=request.knowledge_base_id,  # MANDATORY
                top_k=request.top_k,
                # file_id=request.file_id,
                search_type=search_type.value
            )
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            raise SearchError(f"Search failed: {str(e)}")

        documents, metadatas, scores = filter_results_by_relevance(
            documents,
            metadatas,
            scores,
            min_relevance_score,
            request.question,
        )
        
        if not documents:
            sources, context = build_lexical_fallback_sources(
                db=db,
                user_id=user_id,
                knowledge_base_id=request.knowledge_base_id,
                question=request.question,
                top_k=request.top_k,
            )

            if sources:
                try:
                    answer = get_llm_service().generate_answer(
                        question=request.question,
                        context=context,
                        user_id=user_id,
                        db=db,
                        conversation_id=conversation_id,
                        include_citations=True,
                        temperature=request.temperature or 0.7
                    )
                except Exception as e:
                    logger.error(f"LLM generation failed on lexical fallback: {e}")
                    raise LLMGenerationError(f"Failed to generate answer: {str(e)}")

                processing_time = time.time() - start_time
                response_data = {
                    "question": request.question,
                    "answer": answer,
                    "sources": sources,
                    "processing_time": round(processing_time, 2),
                    "search_type": search_type,
                    "confidence_score": 0.55,
                    "knowledge_base_id": request.knowledge_base_id,
                    "conversation_id": conversation_id
                }
                log_query_to_db(
                    db, user_id, request.knowledge_base_id, request.question, answer,
                    request.top_k, len(sources), processing_time,
                    0.55, search_type.value, conversation_id
                )
                return QueryResponse(**response_data)

            answer = (
                "I don't have enough information in the knowledge base to answer that. "
                "Please add relevant documents or ask a question covered by the indexed content."
            )
            processing_time = time.time() - start_time
            response_data = build_fallback_response(
                question=request.question,
                knowledge_base_id=request.knowledge_base_id,
                conversation_id=conversation_id,
                answer=answer,
                processing_time=processing_time,
                search_type=search_type,
                confidence_score=0.0,
            )
            log_query_to_db(
                db, user_id, request.knowledge_base_id, request.question, answer,
                request.top_k, 0, processing_time, 0.0, search_type.value, conversation_id
            )
            return QueryResponse(**response_data)
        
        # Build context and sources
        sources, context = build_source_documents(
            documents, metadatas, scores, request.knowledge_base_id
        )
        
        # Always ensure a conversation_id exists — generate one if not provided
        conversation_id = getattr(request, "conversation_id", None) or str(uuid.uuid4())

        # Generate answer
        try:
            answer = get_llm_service().generate_answer(
                question=request.question,
                context=context,
                user_id=user_id,
                db=db,
                conversation_id=conversation_id,
                include_citations=True,
                temperature=request.temperature or 0.7
            )
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise LLMGenerationError(f"Failed to generate answer: {str(e)}")
        
        # Detect hallucinations
        is_hallucination, confidence = get_llm_service().detect_hallucination(answer, context)
        
        if is_hallucination:
            logger.warning(f"Low confidence detected: {confidence:.2f}")
            answer += f"\n\n⚠️ Note: Lower confidence ({confidence:.2f}). Please verify."
        
        processing_time = time.time() - start_time
        
        # Build response
        response_data = {
            "question": request.question,
            "answer": answer,
            "sources": sources,
            "processing_time": round(processing_time, 2),
            "search_type": search_type,
            "confidence_score": round(confidence, 3),
            "knowledge_base_id": request.knowledge_base_id,
            "conversation_id": conversation_id
        }
        
        # Cache result
        query_cache.set(
            user_id,
            request.knowledge_base_id,
            request.question,
            request.top_k,
            response_data,
            # request.file_id
            min_relevance_score=min_relevance_score,
        )
        
        # Log to database
        log_query_to_db(
            db, user_id, request.knowledge_base_id, request.question, answer,
            request.top_k, len(documents), processing_time,
            confidence, search_type.value, conversation_id
        )
        
        logger.info(
            f"✓ Query completed: KB='{kb.knowledge_base_name}', "
            f"time={processing_time:.2f}s, confidence={confidence:.2f}"
        )
        
        return QueryResponse(**response_data)
    
    except SearchError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except LLMGenerationError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except RAGException as e:
        raise HTTPException(status_code=e.status_code, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected query error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )


@router.post("/conversation", response_model=QueryResponse)
async def query_with_conversation(
    request: ConversationQueryRequest,
    user_id: str = Depends(get_user_id),
    db: Session = Depends(get_database_session)
):
    """
    Conversational query with KB context
    
    Maintains conversation history within a KB context
    
    Headers:
        X-User-ID: User identifier (REQUIRED)
    
    Body:
        Same as /query plus:
        conversation_id: Unique conversation identifier
    
    Returns:
        QueryResponse with conversation context
    """
    start_time = time.time()
    
    try:
        min_relevance_score = (
            request.min_relevance_score
            if request.min_relevance_score is not None
            else settings.retrieval_min_score
        )
        conversation_id = request.conversation_id or str(uuid.uuid4())

        # Verify KB ownership
        kb = db.query(KnowledgeBase).filter(
            KnowledgeBase.knowledge_base_id == request.knowledge_base_id,
            KnowledgeBase.user_id == user_id
        ).first()
        
        if not kb:
            raise HTTPException(
                status_code=404,
                detail=f"Knowledge base not found or access denied"
            )
        
        kb.last_accessed = datetime.utcnow()
        db.commit()
        
        # Always ensure a conversation_id exists — generate one if not provided
        if not request.conversation_id:
            request.conversation_id = str(uuid.uuid4())
        request.conversation_id = conversation_id

        search_type = request.search_type or SearchType.HYBRID
        
        logger.info(
            f"Conversation query: KB='{kb.knowledge_base_name}', "
            f"conv_id={request.conversation_id}, search_type={search_type.value}"
        )

        if is_greeting_query(request.question):
            answer = "Hello. How can I help you with the knowledge base?"
            processing_time = time.time() - start_time
            response_data = build_fallback_response(
                question=request.question,
                knowledge_base_id=request.knowledge_base_id,
                conversation_id=request.conversation_id,
                answer=answer,
                processing_time=processing_time,
                search_type=search_type,
                confidence_score=1.0,
            )
            log_query_to_db(
                db, user_id, request.knowledge_base_id, request.question, answer,
                request.top_k, 0, processing_time, 1.0, search_type.value, request.conversation_id
            )
            return QueryResponse(**response_data)
        
        # Vector search with KB isolation
        try:
            documents, metadatas, scores = get_vector_store().search(
                query=request.question,
                user_id=user_id,
                knowledge_base_id=request.knowledge_base_id,
                top_k=request.top_k,
                # file_id=request.file_id,
                search_type=search_type.value
            )
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            raise SearchError(f"Search failed: {str(e)}")

        documents, metadatas, scores = filter_results_by_relevance(
            documents,
            metadatas,
            scores,
            min_relevance_score,
            request.question,
        )
        
        if not documents:
            sources, context = build_lexical_fallback_sources(
                db=db,
                user_id=user_id,
                knowledge_base_id=request.knowledge_base_id,
                question=request.question,
                top_k=request.top_k,
            )

            if sources:
                try:
                    answer = get_llm_service().generate_answer(
                        question=request.question,
                        context=context,
                        user_id=user_id,
                        db=db,
                        conversation_id=request.conversation_id,
                        include_citations=True,
                        temperature=request.temperature or 0.7
                    )
                except Exception as e:
                    logger.error(f"LLM generation failed on lexical fallback: {e}")
                    raise LLMGenerationError(f"Failed to generate answer: {str(e)}")

                processing_time = time.time() - start_time
                response_data = {
                    "question": request.question,
                    "answer": answer,
                    "sources": sources,
                    "processing_time": round(processing_time, 2),
                    "search_type": search_type,
                    "confidence_score": 0.55,
                    "knowledge_base_id": request.knowledge_base_id,
                    "conversation_id": request.conversation_id
                }
                log_query_to_db(
                    db, user_id, request.knowledge_base_id, request.question, answer,
                    request.top_k, len(sources), processing_time,
                    0.55, search_type.value, request.conversation_id
                )
                return QueryResponse(**response_data)

            answer = (
                "I don't have enough information in the knowledge base to answer that. "
                "Please add relevant documents or ask a question covered by the indexed content."
            )
            processing_time = time.time() - start_time
            response_data = build_fallback_response(
                question=request.question,
                knowledge_base_id=request.knowledge_base_id,
                conversation_id=request.conversation_id,
                answer=answer,
                processing_time=processing_time,
                search_type=search_type,
                confidence_score=0.0,
            )
            log_query_to_db(
                db, user_id, request.knowledge_base_id, request.question, answer,
                request.top_k, 0, processing_time, 0.0, search_type.value, request.conversation_id
            )
            return QueryResponse(**response_data)
        
        # Build context and sources
        sources, context = build_source_documents(
            documents, metadatas, scores, request.knowledge_base_id
        )
        
        # Generate answer with conversation history
        try:
            answer = get_llm_service().generate_answer(
                question=request.question,
                context=context,
                user_id=user_id,
                db=db,
                conversation_id=request.conversation_id,
                include_citations=True,
                temperature=request.temperature or 0.7
            )
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise LLMGenerationError(f"Failed to generate answer: {str(e)}")
        
        # Detect hallucinations
        is_hallucination, confidence = get_llm_service().detect_hallucination(answer, context)
        
        if is_hallucination:
            answer += f"\n\n⚠️ Note: Lower confidence ({confidence:.2f})."
        
        processing_time = time.time() - start_time
        
        # Build response
        response_data = {
            "question": request.question,
            "answer": answer,
            "sources": sources,
            "processing_time": round(processing_time, 2),
            "search_type": search_type,
            "confidence_score": round(confidence, 3),
            "knowledge_base_id": request.knowledge_base_id,
            "conversation_id": request.conversation_id
        }
        
        # Log to database with conversation tracking
        log_query_to_db(
            db, user_id, request.knowledge_base_id, request.question, answer,
            request.top_k, len(documents), processing_time,
            confidence, search_type.value, request.conversation_id
        )
        
        logger.info(
            f"✓ Conversation query completed: KB='{kb.knowledge_base_name}', "
            f"conv={request.conversation_id}, time={processing_time:.2f}s"
        )
        
        return QueryResponse(**response_data)
    
    except SearchError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except LLMGenerationError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except RAGException as e:
        raise HTTPException(status_code=e.status_code, detail=str(e))
    except Exception as e:
        logger.error(f"Conversation query error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error: {str(e)}"
        )


# ==================== CONVERSATION MANAGEMENT ====================

@router.post("/clear-conversation/{conversation_id}")
async def clear_conversation(
    conversation_id: str,
    user_id: str = Depends(get_user_id),
    db: Session = Depends(get_database_session)
):
    """Clear conversation history"""
    try:
        get_llm_service().clear_conversation(conversation_id, user_id, db)
        
        return {
            "message": f"Conversation {conversation_id} cleared",
            "conversation_id": conversation_id,
            "user_id": user_id
        }
    except Exception as e:
        logger.error(f"Error clearing conversation: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error clearing conversation: {str(e)}"
        )


@router.get("/history/{conversation_id}")
async def get_conversation_history(
    conversation_id: str,
    user_id: str = Depends(get_user_id),
    db: Session = Depends(get_database_session)
):
    """Get conversation history"""
    try:
        from app.db.models import ConversationHistory
        from sqlalchemy import desc
        
        messages = db.query(ConversationHistory).filter(
            ConversationHistory.conversation_id == conversation_id,
            ConversationHistory.user_id == user_id
        ).order_by(
            desc(ConversationHistory.created_at)
        ).limit(8).all()
        
        messages.reverse()
        
        history = []
        kb_id = None
        for msg in messages:
            history.append({
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.created_at.isoformat()
            })
            if not kb_id and msg.knowledge_base_id:
                kb_id = msg.knowledge_base_id
        
        # Get KB info if available
        kb_info = None
        if kb_id:
            kb = db.query(KnowledgeBase).filter(
                KnowledgeBase.knowledge_base_id == kb_id
            ).first()
            if kb:
                kb_info = {
                    "knowledge_base_id": kb.knowledge_base_id,
                    "knowledge_base_name": kb.knowledge_base_name
                }
        
        return {
            "conversation_id": conversation_id,
            "user_id": user_id,
            "knowledge_base": kb_info,
            "message_count": len(history),
            "messages": history
        }
        
    except Exception as e:
        logger.error(f"Error retrieving history: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving conversation history: {str(e)}"
        )


@router.get("/my-queries")
async def get_my_queries(
    user_id: str = Depends(get_user_id),
    db: Session = Depends(get_database_session),
    knowledge_base_id: str = None,
    limit: int = 20
):
    """
    Get recent queries for current user
    
    Optional KB filter to see queries for specific KB
    """
    try:
        from sqlalchemy import desc
        
        limit = min(max(1, limit), 100)
        
        query = db.query(QueryLog).filter(
            QueryLog.user_id == user_id
        )
        
        # Optional KB filter
        if knowledge_base_id:
            query = query.filter(QueryLog.knowledge_base_id == knowledge_base_id)
            
            # Verify KB access
            kb = db.query(KnowledgeBase).filter(
                KnowledgeBase.knowledge_base_id == knowledge_base_id,
                KnowledgeBase.user_id == user_id
            ).first()
            
            if not kb:
                raise HTTPException(
                    status_code=404,
                    detail="Knowledge base not found or access denied"
                )
        
        queries = query.order_by(
            desc(QueryLog.queried_at)
        ).limit(limit).all()
        
        query_list = []
        for q in queries:
            # Get KB name
            kb_name = "Unknown"
            if q.knowledge_base_id:
                kb = db.query(KnowledgeBase).filter(
                    KnowledgeBase.knowledge_base_id == q.knowledge_base_id
                ).first()
                if kb:
                    kb_name = kb.knowledge_base_name
            
            query_list.append({
                "id": q.id,
                "question": q.question[:100] + "..." if len(q.question) > 100 else q.question,
                "knowledge_base_id": q.knowledge_base_id,
                "knowledge_base_name": kb_name,
                "queried_at": q.queried_at.isoformat(),
                "processing_time": q.processing_time_seconds,
                "confidence_score": q.confidence_score,
                "search_type": q.search_type,
                "conversation_id": q.conversation_id
            })
        
        return {
            "user_id": user_id,
            "knowledge_base_id": knowledge_base_id,
            "total_queries": len(query_list),
            "queries": query_list
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving queries: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving query history: {str(e)}"
        )


@router.post("/clear-cache")
async def clear_query_cache(
    user_id: str = Depends(get_user_id),
    knowledge_base_id: str = None
):
    """
    Clear query cache
    
    If KB ID provided, clears cache for that KB only
    Otherwise clears all cache for user
    """
    try:
        if knowledge_base_id:
            query_cache.clear_kb(knowledge_base_id)
            return {
                "message": f"Cache cleared for KB {knowledge_base_id}",
                "knowledge_base_id": knowledge_base_id
            }
        else:
            # Clear all - would need to implement user-specific clearing
            query_cache.cache.clear()
            return {
                "message": "All cache cleared",
                "user_id": user_id
            }
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error clearing cache: {str(e)}"
        )
