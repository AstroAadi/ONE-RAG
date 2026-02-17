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
from app.core.exceptions import (
    SearchError, LLMGenerationError, RAGException
)
from app.db.models import QueryLog, KnowledgeBase
import time
import logging
from datetime import datetime, timedelta
from typing import Tuple, List, Dict
from functools import lru_cache
import hashlib

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/query", tags=["Query"])

# Initialize services
vector_store = VectorStoreService()
llm_service = LLMService()


# ==================== QUERY RESULT CACHE ====================

class QueryCache:
    """KB-aware query cache"""
    
    def __init__(self, ttl_seconds: int = 300):
        self.cache = {}
        self.ttl = ttl_seconds
    
    def _make_key(self, user_id: str, kb_id: str, question: str, top_k: int, file_id: str = None) -> str:
        """Generate cache key with KB isolation"""
        data = f"{user_id}:{kb_id}:{question}:{top_k}:{file_id or 'all'}"
        return hashlib.sha256(data.encode()).hexdigest()
    
    def get(self, user_id: str, kb_id: str, question: str, top_k: int, file_id: str = None) -> dict:
        """Get cached result"""
        key = self._make_key(user_id, kb_id, question, top_k, file_id)
        
        if key in self.cache:
            result, timestamp = self.cache[key]
            if datetime.utcnow() - timestamp < timedelta(seconds=self.ttl):
                logger.info(f"✓ Cache hit for query in KB {kb_id}")
                return result
            else:
                del self.cache[key]
        
        return None
    
    def set(self, user_id: str, kb_id: str, question: str, top_k: int, result: dict, file_id: str = None):
        """Cache result"""
        key = self._make_key(user_id, kb_id, question, top_k, file_id)
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
        
        # Check cache
        cached_result = query_cache.get(
            user_id, 
            request.knowledge_base_id,
            request.question, 
            request.top_k,
            request.file_id
        )
        
        if cached_result:
            logger.info("Returning cached result")
            return QueryResponse(**cached_result)
        
        # Vector search WITH KB ISOLATION
        try:
            documents, metadatas, scores = vector_store.search(
                query=request.question,
                user_id=user_id,
                knowledge_base_id=request.knowledge_base_id,  # MANDATORY
                top_k=request.top_k,
                file_id=request.file_id,
                search_type=search_type.value
            )
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            raise SearchError(f"Search failed: {str(e)}")
        
        if not documents:
            raise SearchError(
                f"No documents found in knowledge base '{kb.knowledge_base_name}'. "
                f"Please upload and process documents first."
            )
        
        # Build context and sources
        sources, context = build_source_documents(
            documents, metadatas, scores, request.knowledge_base_id
        )
        
        # Generate answer
        try:
            answer = llm_service.generate_answer(
                question=request.question,
                context=context,
                user_id=user_id,
                db=db,
                conversation_id=None,
                include_citations=True,
                temperature=request.temperature or 0.7
            )
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise LLMGenerationError(f"Failed to generate answer: {str(e)}")
        
        # Detect hallucinations
        is_hallucination, confidence = llm_service.detect_hallucination(answer, context)
        
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
            "knowledge_base_id": request.knowledge_base_id
        }
        
        # Cache result
        query_cache.set(
            user_id,
            request.knowledge_base_id,
            request.question,
            request.top_k,
            response_data,
            request.file_id
        )
        
        # Log to database
        log_query_to_db(
            db, user_id, request.knowledge_base_id, request.question, answer,
            request.top_k, len(documents), processing_time,
            confidence, search_type.value, None
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
        
        search_type = request.search_type or SearchType.HYBRID
        
        logger.info(
            f"Conversation query: KB='{kb.knowledge_base_name}', "
            f"conv_id={request.conversation_id}, search_type={search_type.value}"
        )
        
        # Vector search with KB isolation
        try:
            documents, metadatas, scores = vector_store.search(
                query=request.question,
                user_id=user_id,
                knowledge_base_id=request.knowledge_base_id,
                top_k=request.top_k,
                file_id=request.file_id,
                search_type=search_type.value
            )
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            raise SearchError(f"Search failed: {str(e)}")
        
        if not documents:
            raise SearchError(f"No documents found in KB '{kb.knowledge_base_name}'")
        
        # Build context and sources
        sources, context = build_source_documents(
            documents, metadatas, scores, request.knowledge_base_id
        )
        
        # Generate answer with conversation history
        try:
            answer = llm_service.generate_answer(
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
        is_hallucination, confidence = llm_service.detect_hallucination(answer, context)
        
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
        llm_service.clear_conversation(conversation_id, user_id, db)
        
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