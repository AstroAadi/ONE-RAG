"""
Query Route - CONCURRENT VERSION with Optimized I/O Operations
Uses ThreadPoolExecutor for parallel embedding and vector search
"""

from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from app.models.schemas import (
    QueryRequest, QueryResponse, SourceDocument,
    ConversationQueryRequest, SearchType
)
from app.services.vector_store_service import VectorStoreService
from app.services.llm_service import LLMService, LLMGenerationError
from app.core.config import settings
from app.core.dependencies import get_user_id, get_database_session
from app.db.models import QueryLog, KnowledgeBase
import time
import logging
from datetime import datetime
from typing import Tuple, List, Dict, Optional
from app.core.task_manager import get_task_manager, TaskType

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/query", tags=["Query"])

# Initialize services (these are thread-safe)
vector_store = VectorStoreService()
llm_service = LLMService()


def safe_get_int(metadata: dict, key: str, default: int = 0) -> int:
    """Safely get integer from metadata"""
    try:
        value = metadata.get(key, default)
        return int(value) if value is not None else default
    except (ValueError, TypeError):
        return default


def safe_get_str(metadata: dict, key: str, default: str = '') -> str:
    """Safely get string from metadata"""
    try:
        value = metadata.get(key, default)
        return str(value) if value is not None else default
    except (ValueError, TypeError):
        return default


def safe_get_score(score: float) -> float:
    """Safely clamp score to valid range [0.0, 1.0]"""
    try:
        return max(0.0, min(1.0, float(score)))
    except (ValueError, TypeError):
        return 0.0


def _execute_vector_search(
    query: str,
    top_k: int,
    search_type: str,
    user_id: Optional[str] = None,
    # The task manager only injects file_id when it is truthy, so this must
    # have a default or an unfiltered query raises a TypeError.
    file_id: Optional[str] = None,
    knowledge_base_id: Optional[str] = None,
    # Injected into every submitted task by the task manager
    progress_callback=None,
    task_id=None
) -> Tuple[List[str], List[Dict], List[float]]:
    """
    Execute vector search (runs in I/O thread pool)
    
    This function is designed to be called from ThreadPoolExecutor
    """
    return vector_store.search(
        query=query,
        top_k=top_k,
        user_id=user_id,
        file_id=file_id,
        search_type=search_type,
        knowledge_base_id=knowledge_base_id
    )


def _embedding_error_detail(error: Exception) -> Optional[str]:
    """
    Actionable message for a retrieval failure caused by the embedding
    provider, or None if the error is unrelated to embeddings.
    """
    lowered = str(error).lower()

    if not ("embedding" in lowered or "empty vectors" in lowered):
        return None

    if "402" in lowered or "payment required" in lowered or "credits" in lowered:
        return (
            "Could not search: the hosted embedding API reports no remaining "
            "credits. Set EMBEDDING_BACKEND=local in the backend .env to run "
            "the embedding model on this machine instead (no credits needed)."
        )

    if "401" in lowered or "403" in lowered:
        return (
            "Could not search: the embedding API rejected the credentials. "
            "Check HUGGINGFACE_API_TOKEN, or set EMBEDDING_BACKEND=local to "
            "embed on this machine."
        )

    return f"Could not search: {error}"


def _llm_error_detail(error: LLMGenerationError) -> Tuple[int, str]:
    """
    Map an LLM provider failure onto an HTTP status and an actionable message.
    """
    status = error.status_code
    provider_message = (error.provider_message or str(error)).strip()
    lowered = provider_message.lower()

    available = ", ".join(settings.available_llm_models)

    if status == 402 or "no remaining credits" in lowered:
        return 503, (
            f"Could not generate an answer: the HuggingFace Inference API "
            f"reports no remaining credits for '{settings.llm_model_id}'. "
            f"Add credits to the account, or set LLM_MODEL_ID to a model "
            f"served by a provider you have access to. "
            f"(Retrieval works offline, so only answer generation is blocked.)"
        )

    if status in (401, 403):
        return 503, (
            "Could not generate an answer: the HuggingFace API rejected the "
            "credentials. Check HUGGINGFACE_API_TOKEN in the backend .env."
        )

    if "not supported by any provider" in lowered or "not a chat model" in lowered:
        return 503, (
            f"Could not generate an answer: '{settings.llm_model_id}' is not "
            f"available to your HuggingFace account "
            f"({provider_message}). Pick a different chat model via "
            f"POST /api/settings/models/llm or set LLM_MODEL_ID in the "
            f"backend .env. Currently configured options: {available}."
        )

    return 502, f"Could not generate an answer: {provider_message}"


def _require_knowledge_base(
    db: Session,
    knowledge_base_id: str,
    user_id: str
) -> KnowledgeBase:
    """
    Resolve the knowledge base being queried, enforcing ownership.
    
    Returning 404 here (rather than silently searching everything) keeps a
    stale or wrong KB id from quietly producing answers sourced from an
    unrelated document set.
    """
    knowledge_base = db.query(KnowledgeBase).filter(
        KnowledgeBase.knowledge_base_id == knowledge_base_id,
        KnowledgeBase.user_id == user_id
    ).first()
    
    if not knowledge_base:
        raise HTTPException(
            status_code=404,
            detail=f"Knowledge base not found: {knowledge_base_id}"
        )
    
    return knowledge_base


def _generate_llm_answer(
    question: str,
    context: str,
    temperature: float,
    user_id: Optional[str] = None,
    conversation_id: Optional[str] = None,
    file_id: Optional[str] = None,
    # Injected into every submitted task by the task manager
    progress_callback=None,
    task_id=None
) -> str:
    """
    Generate LLM answer (runs in I/O thread pool)
    
    This function is designed to be called from ThreadPoolExecutor
    """
    from app.core.database import get_db_context
    
    with get_db_context() as db:
        return llm_service.generate_answer(
            question=question,
            context=context,
            user_id=user_id,
            db=db,
            conversation_id=conversation_id,
            include_citations=True,
            temperature=temperature
        )


@router.post("/", response_model=QueryResponse)
async def query_knowledge_base(
    request: QueryRequest,
    user_id: str = Depends(get_user_id),
    db: Session = Depends(get_database_session)
):
    """
    Query endpoint with concurrent I/O operations
    
    This endpoint uses ThreadPoolExecutor to parallelize:
    - Vector search (embedding + retrieval)
    - LLM generation (if needed)
    
    Headers:
        X-User-ID: User identifier (REQUIRED)
    """
    start_time = time.time()
    
    # Reject unknown knowledge bases before doing any work
    knowledge_base = _require_knowledge_base(db, request.knowledge_base_id, user_id)
    
    try:
        # Get search type
        search_type = request.search_type or SearchType.HYBRID
        
        logger.info(
            f"Query from user {user_id} on KB {knowledge_base.knowledge_base_id}: "
            f"search_type={search_type.value}, top_k={request.top_k}"
        )
        
        # Get task manager for I/O operations
        task_manager = get_task_manager()
        
        # Submit vector search to I/O thread pool (non-blocking)
        search_task_id = task_manager.submit_task(
            func=_execute_vector_search,
            task_type=TaskType.IO_BOUND,  # Vector search is I/O-bound
            description=f"Vector search for query",
            user_id=user_id,
            file_id=request.file_id,
            query=request.question,
            top_k=request.top_k,
            search_type=search_type.value,
            knowledge_base_id=request.knowledge_base_id
        )
        
        # Wait for search to complete (with timeout)
        try:
            documents, metadatas, scores = task_manager.get_task_result(
                search_task_id,
                timeout=30.0  # 30 second timeout
            )
        except TimeoutError:
            raise HTTPException(
                status_code=504,
                detail="Vector search timeout. Please try again."
            )
        
        if not documents:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"No indexed content found in knowledge base "
                    f"'{knowledge_base.knowledge_base_name}'. Upload a document "
                    f"and finalize its chunks first."
                )
            )
        
        # Build context for LLM
        context_parts = []
        sources = []
        
        for i, (doc, metadata, score) in enumerate(zip(documents, metadatas, scores)):
            source_header = f"[Source {i+1}]"
            
            page = safe_get_int(metadata, 'page_number', 0)
            if page > 0:
                source_header += f" [Page {page}]"
            
            chunk_type = safe_get_str(metadata, 'chunk_type', 'text')
            if chunk_type != 'text':
                source_header += f" [{chunk_type.upper()}]"
            
            context_parts.append(f"{source_header}:\n{doc}\n")
            
            sources.append(SourceDocument(
                content=doc[:500] + "..." if len(doc) > 500 else doc,
                full_text=doc,
                chunk_id=safe_get_str(metadata, 'chunk_id', f'chunk_{i}'),
                file_id=safe_get_str(metadata, 'file_id', 'unknown'),
                knowledge_base_id=safe_get_str(
                    metadata, 'knowledge_base_id', request.knowledge_base_id
                ),
                filename=safe_get_str(metadata, 'filename', 'unknown'),
                relevance_score=safe_get_score(score),
                page_number=page,
                chunk_type=chunk_type,
                chunk_index=safe_get_int(metadata, 'chunk_index', i),
            ))
        
        context = "\n\n".join(context_parts)
        
        # Generate answer (also in I/O thread pool for API calls)
        answer_task_id = task_manager.submit_task(
            func=_generate_llm_answer,
            task_type=TaskType.IO_BOUND,  # LLM API calls are I/O-bound
            description=f"Generate LLM answer",
            user_id=user_id,
            question=request.question,
            context=context,
            # user_id=user_id,
            conversation_id=None,
            temperature=request.temperature or 0.7
        )
        
        # Wait for answer (with timeout)
        try:
            answer = task_manager.get_task_result(
                answer_task_id,
                timeout=60.0  # 60 second timeout for LLM
            )
        except TimeoutError:
            raise HTTPException(
                status_code=504,
                detail="LLM generation timeout. Please try again."
            )
        
        # Detect hallucinations
        is_hallucination, confidence = llm_service.detect_hallucination(answer, context)
        
        if is_hallucination:
            logger.warning(f"Low confidence detected: {confidence}")
            answer += f"\n\n⚠️ Note: Lower confidence ({confidence:.2f}). Please verify."
        
        processing_time = time.time() - start_time
        
        # Log query to database (asynchronously)
        try:
            query_log = QueryLog(
                user_id=user_id,
                question=request.question,
                answer=answer,
                top_k=request.top_k,
                chunks_retrieved=len(documents),
                processing_time_seconds=processing_time,
                confidence_score=confidence,
                queried_at=datetime.utcnow(),
                conversation_id=None
            )
            db.add(query_log)
            db.commit()
        except Exception as e:
            logger.warning(f"Failed to log query: {e}")
        
        return QueryResponse(
            question=request.question,
            answer=answer,
            sources=sources,
            processing_time=round(processing_time, 2),
            search_type=search_type,
            confidence_score=round(confidence, 3),
            knowledge_base_id=request.knowledge_base_id
        )
    
    except HTTPException:
        raise
    except LLMGenerationError as e:
        logger.error(f"Query generation error: {e}", exc_info=True)
        status, detail = _llm_error_detail(e)
        raise HTTPException(status_code=status, detail=detail)
    except Exception as e:
        logger.error(f"Query error: {str(e)}", exc_info=True)

        embedding_detail = _embedding_error_detail(e)
        if embedding_detail:
            raise HTTPException(status_code=503, detail=embedding_detail)

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
    Conversational query with concurrent I/O operations
    
    Similar to standard query but maintains conversation history
    """
    start_time = time.time()
    
    knowledge_base = _require_knowledge_base(db, request.knowledge_base_id, user_id)
    
    try:
        search_type = request.search_type or SearchType.HYBRID
        
        logger.info(
            f"Conversation query on KB {knowledge_base.knowledge_base_id}: "
            f"search_type={search_type.value}"
        )
        
        task_manager = get_task_manager()
        
        # Vector search in background
        search_task_id = task_manager.submit_task(
            func=_execute_vector_search,
            task_type=TaskType.IO_BOUND,
            description=f"Vector search for conversation",
            user_id=user_id,
            file_id=request.file_id,
            query=request.question,
            top_k=request.top_k,
            search_type=search_type.value,
            knowledge_base_id=request.knowledge_base_id
        )
        
        # Wait for search
        try:
            documents, metadatas, scores = task_manager.get_task_result(
                search_task_id,
                timeout=30.0
            )
        except TimeoutError:
            raise HTTPException(
                status_code=504,
                detail="Vector search timeout"
            )
        
        if not documents:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"No indexed content found in knowledge base "
                    f"'{knowledge_base.knowledge_base_name}'. Upload a document "
                    f"and finalize its chunks first."
                )
            )
        
        # Build context
        context_parts = []
        sources = []
        
        for i, (doc, metadata, score) in enumerate(zip(documents, metadatas, scores)):
            source_header = f"[Source {i+1}]"
            
            page = safe_get_int(metadata, 'page_number', 0)
            if page > 0:
                source_header += f" [Page {page}]"
            
            chunk_type = safe_get_str(metadata, 'chunk_type', 'text')
            if chunk_type != 'text':
                source_header += f" [{chunk_type.upper()}]"
            
            context_parts.append(f"{source_header}:\n{doc}\n")
            
            sources.append(SourceDocument(
                content=doc[:500] + "..." if len(doc) > 500 else doc,
                full_text=doc,
                chunk_id=safe_get_str(metadata, 'chunk_id', f'chunk_{i}'),
                file_id=safe_get_str(metadata, 'file_id', 'unknown'),
                knowledge_base_id=safe_get_str(
                    metadata, 'knowledge_base_id', request.knowledge_base_id
                ),
                filename=safe_get_str(metadata, 'filename', 'unknown'),
                relevance_score=safe_get_score(score),
                page_number=page,
                chunk_type=chunk_type,
                chunk_index=safe_get_int(metadata, 'chunk_index', i)
            ))
        
        context = "\n\n".join(context_parts)
        
        # Generate answer with conversation history
        answer_task_id = task_manager.submit_task(
            func=_generate_llm_answer,
            task_type=TaskType.IO_BOUND,
            description=f"Generate conversation answer",
            user_id=user_id,
            question=request.question,
            context=context,
            # user_id=user_id,
            conversation_id=request.conversation_id,
            temperature=request.temperature or 0.7
        )
        
        try:
            answer = task_manager.get_task_result(
                answer_task_id,
                timeout=60.0
            )
        except TimeoutError:
            raise HTTPException(
                status_code=504,
                detail="LLM generation timeout"
            )
        
        # Detect hallucinations
        is_hallucination, confidence = llm_service.detect_hallucination(answer, context)
        
        if is_hallucination:
            answer += f"\n\n⚠️ Note: Lower confidence ({confidence:.2f})."
        
        processing_time = time.time() - start_time
        
        # Log query
        try:
            query_log = QueryLog(
                user_id=user_id,
                question=request.question,
                answer=answer,
                top_k=request.top_k,
                chunks_retrieved=len(documents),
                processing_time_seconds=processing_time,
                confidence_score=confidence,
                queried_at=datetime.utcnow(),
                conversation_id=request.conversation_id
            )
            db.add(query_log)
            db.commit()
        except Exception as e:
            logger.warning(f"Failed to log query: {e}")
        
        return QueryResponse(
            question=request.question,
            answer=answer,
            sources=sources,
            processing_time=round(processing_time, 2),
            search_type=search_type,
            confidence_score=round(confidence, 3),
            knowledge_base_id=request.knowledge_base_id,
            conversation_id=request.conversation_id
        )
    
    except HTTPException:
        raise
    except LLMGenerationError as e:
        logger.error(f"Conversation generation error: {e}", exc_info=True)
        status, detail = _llm_error_detail(e)
        raise HTTPException(status_code=status, detail=detail)
    except Exception as e:
        logger.error(f"Conversation query error: {str(e)}", exc_info=True)

        embedding_detail = _embedding_error_detail(e)
        if embedding_detail:
            raise HTTPException(status_code=503, detail=embedding_detail)

        raise HTTPException(
            status_code=500,
            detail=f"Error: {str(e)}"
        )


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
        for msg in messages:
            history.append({
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.created_at.isoformat()
            })
        
        return {
            "conversation_id": conversation_id,
            "user_id": user_id,
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
    limit: int = 20
):
    """Get recent queries for current user"""
    try:
        from sqlalchemy import desc
        
        limit = min(max(1, limit), 100)
        
        queries = db.query(QueryLog).filter(
            QueryLog.user_id == user_id
        ).order_by(
            desc(QueryLog.queried_at)
        ).limit(limit).all()
        
        query_list = []
        for query in queries:
            query_list.append({
                "id": query.id,
                "question": query.question[:100] + "..." if len(query.question) > 100 else query.question,
                "queried_at": query.queried_at.isoformat(),
                "processing_time": query.processing_time_seconds,
                "confidence_score": query.confidence_score,
                "conversation_id": query.conversation_id
            })
        
        return {
            "user_id": user_id,
            "total_queries": len(query_list),
            "queries": query_list
        }
        
    except Exception as e:
        logger.error(f"Error retrieving queries: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving query history: {str(e)}"
        )