"""
Query Route with MANDATORY USER AUTH and DATABASE-BACKED CONVERSATION HISTORY
- All queries require X-User-ID header
- Full user isolation in vector search
- Conversation history stored in database (FIFO)
- Query logging to database
- UPDATED: Uses search_type (dense/sparse/hybrid) instead of weights
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
from app.db.models import QueryLog
import time
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/query", tags=["Query"])

# Initialize services
vector_store = VectorStoreService()
llm_service = LLMService()


def safe_get_int(metadata: dict, key: str, default: int = 0) -> int:
    """Safely get integer from metadata (handles both str and int)"""
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


@router.post("/", response_model=QueryResponse)
async def query_knowledge_base(
    request: QueryRequest,
    user_id: str = Depends(get_user_id),
    db: Session = Depends(get_database_session)
):
    """
    ENHANCED Query endpoint with full chunk metadata for editing
    
    Headers:
        X-User-ID: User identifier (REQUIRED)
    
    Returns:
        QueryResponse where each source includes:
        - chunk_id: Unique chunk identifier (for editing)
        - file_id: Parent file identifier (for editing)
        - full_text: Complete chunk text (not truncated)
        - content: Display text (may be truncated)
        - All other metadata fields
    """
    start_time = time.time()
    
    try:
        # Get search type (default to hybrid)
        search_type = request.search_type or SearchType.HYBRID
        
        logger.info(f"Query with search_type={search_type.value}, top_k={request.top_k}")
        
        # Search with USER ISOLATION - returns full metadata
        documents, metadatas, scores = vector_store.search(
            query=request.question,
            top_k=request.top_k,
            user_id=user_id,
            file_id=getattr(request, 'file_id', None),
            search_type=search_type.value
        )
        
        if not documents:
            raise HTTPException(
                status_code=404,
                detail="No documents found. Please upload and process documents first."
            )
        
        # Build context for LLM
        context_parts = []
        sources = []
        
        for i, (doc, metadata, score) in enumerate(zip(documents, metadatas, scores)):
            source_header = f"[Source {i+1}]"
            
            # Add page number - safe conversion
            page = safe_get_int(metadata, 'page_number', 0)
            if page > 0:
                source_header += f" [Page {page}]"
            
            # Add content type - safe conversion
            chunk_type = safe_get_str(metadata, 'chunk_type', 'text')
            if chunk_type != 'text':
                source_header += f" [{chunk_type.upper()}]"
            
            context_parts.append(f"{source_header}:\n{doc}\n")
            
            # ENHANCED: Include full chunk metadata for editing
            sources.append(SourceDocument(
                # Display content (truncated for UI)
                content=doc[:500] + "..." if len(doc) > 500 else doc,
                
                # FULL text for editing (not truncated)
                full_text=doc,
                
                # CRITICAL: chunk_id and file_id for editing
                chunk_id=safe_get_str(metadata, 'chunk_id', f'chunk_{i}'),
                file_id=safe_get_str(metadata, 'file_id', 'unknown'),
                
                # Other metadata
                filename=safe_get_str(metadata, 'filename', 'unknown'),
                relevance_score=safe_get_score(score),
                page_number=page,
                chunk_type=chunk_type,
                chunk_index=safe_get_int(metadata, 'chunk_index', i),
                
            ))
        
        context = "\n\n".join(context_parts)
        
        # Generate answer
        answer = llm_service.generate_answer(
            question=request.question,
            context=context,
            user_id=user_id,
            db=db,
            conversation_id=None,
            include_citations=True,
            temperature=request.temperature
        )
        
        # Detect hallucinations
        is_hallucination, confidence = llm_service.detect_hallucination(answer, context)
        
        if is_hallucination:
            logger.warning(f"Low confidence detected: {confidence}")
            answer += f"\n\n⚠️ Note: Lower confidence ({confidence:.2f}). Please verify."
        
        processing_time = time.time() - start_time
        
        # LOG QUERY TO DATABASE
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
            sources=sources,  # Now includes chunk_id, file_id, and full_text
            processing_time=round(processing_time, 2),
            search_type=search_type,
            confidence_score=round(confidence, 3)
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query error: {str(e)}")
        import traceback
        traceback.print_exc()
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
    ENHANCED Conversational query with full chunk metadata
    
    Headers:
        X-User-ID: User identifier (REQUIRED)
    
    Returns:
        QueryResponse with sources containing chunk_id, file_id, and full_text
    """
    start_time = time.time()
    
    try:
        # Get search type (default to hybrid)
        search_type = request.search_type or SearchType.HYBRID
        
        logger.info(f"Conversation query with search_type={search_type.value}")
        
        # Retrieve chunks with full metadata
        documents, metadatas, scores = vector_store.search(
            query=request.question,
            top_k=request.top_k,
            user_id=user_id,
            file_id=getattr(request, 'file_id', None),
            search_type=search_type.value
        )
        
        if not documents:
            raise HTTPException(
                status_code=404,
                detail="No documents found. Please upload and process documents first."
            )
        
        # Build context with safe metadata access
        context_parts = []
        sources = []
        
        for i, (doc, metadata, score) in enumerate(zip(documents, metadatas, scores)):
            source_header = f"[Source {i+1}]"
            
            # Safe metadata access
            page = safe_get_int(metadata, 'page_number', 0)
            if page > 0:
                source_header += f" [Page {page}]"
            
            chunk_type = safe_get_str(metadata, 'chunk_type', 'text')
            if chunk_type != 'text':
                source_header += f" [{chunk_type.upper()}]"
            
            context_parts.append(f"{source_header}:\n{doc}\n")
            
            # ENHANCED: Full metadata for editing
            sources.append(SourceDocument(
                content=doc[:500] + "..." if len(doc) > 500 else doc,
                full_text=doc,  # Complete text
                chunk_id=safe_get_str(metadata, 'chunk_id', f'chunk_{i}'),
                file_id=safe_get_str(metadata, 'file_id', 'unknown'),
                filename=safe_get_str(metadata, 'filename', 'unknown'),
                relevance_score=safe_get_score(score),
                page_number=page,
                chunk_type=chunk_type,
                chunk_index=safe_get_int(metadata, 'chunk_index', i)
            ))
        
        context = "\n\n".join(context_parts)
        
        # Generate answer with conversation history
        answer = llm_service.generate_answer(
            question=request.question,
            context=context,
            user_id=user_id,
            db=db,
            conversation_id=request.conversation_id,
            include_citations=True,
            temperature=request.temperature
        )
        
        # Detect hallucinations
        is_hallucination, confidence = llm_service.detect_hallucination(answer, context)
        
        if is_hallucination:
            answer += f"\n\n⚠️ Note: Lower confidence ({confidence:.2f})."
        
        processing_time = time.time() - start_time
        
        # LOG QUERY TO DATABASE
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
            sources=sources,  # Includes chunk_id, file_id, full_text
            processing_time=round(processing_time, 2),
            search_type=search_type,
            confidence_score=round(confidence, 3),
            conversation_id=request.conversation_id
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Conversation query error: {str(e)}")
        import traceback
        traceback.print_exc()
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
    """
    Clear conversation history from DATABASE
    
    Headers:
        X-User-ID: User identifier (REQUIRED)
    
    Args:
        conversation_id: Conversation identifier to clear
        
    Returns:
        Success message
    """
    try:
        # Clear from database with user isolation
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
    """
    Get conversation history from DATABASE
    
    Headers:
        X-User-ID: User identifier (REQUIRED)
    
    Args:
        conversation_id: Conversation identifier
        
    Returns:
        Conversation history (last 4 Q&A pairs)
    """
    try:
        from app.db.models import ConversationHistory
        from sqlalchemy import desc
        
        # Get last 8 messages (4 Q&A pairs) with user isolation
        messages = db.query(ConversationHistory).filter(
            ConversationHistory.conversation_id == conversation_id,
            ConversationHistory.user_id == user_id  # USER ISOLATION
        ).order_by(
            desc(ConversationHistory.created_at)
        ).limit(8).all()
        
        # Reverse to chronological order
        messages.reverse()
        
        # Format history
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
    """
    Get recent queries for current user
    
    Headers:
        X-User-ID: User identifier (REQUIRED)
    
    Query Parameters:
        limit: Number of queries to return (default: 20, max: 100)
        
    Returns:
        List of recent queries
    """
    try:
        from sqlalchemy import desc
        
        # Limit to reasonable range
        limit = min(max(1, limit), 100)
        
        # Get recent queries with user isolation
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