"""
Query Route with Hybrid Search
Combines BM25 keyword and semantic search
Fixed: Safe metadata access with type conversion
"""

from fastapi import APIRouter, HTTPException
from app.models.schemas import (
    QueryRequest, QueryResponse, SourceDocument,
    ConversationQueryRequest
)
from app.services.vector_store_service import VectorStoreService
from app.services.llm_service import LLMService
import time
import logging

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
async def query_knowledge_base(request: QueryRequest):
    """
    Query with hybrid search (BM25 + Semantic)
    
    Default weights:
    - Keyword (BM25): 0.3
    - Semantic: 0.7
    
    You can adjust weights, but they must sum to 1.0
    """
    start_time = time.time()
    
    try:
        # Validate and normalize weights
        keyword_weight = request.keyword_weight if request.keyword_weight is not None else 0.3
        semantic_weight = request.semantic_weight if request.semantic_weight is not None else 0.7
        
        # Ensure they sum to 1.0
        total = keyword_weight + semantic_weight
        if abs(total - 1.0) > 0.01:
            keyword_weight = keyword_weight / total
            semantic_weight = semantic_weight / total
            logger.info(f"Normalized weights: keyword={keyword_weight:.2f}, semantic={semantic_weight:.2f}")
        
        # Hybrid search
        documents, metadatas, scores = vector_store.hybrid_search(
            query=request.question,
            top_k=request.top_k,
            file_id=getattr(request, 'file_id', None),
            keyword_weight=keyword_weight,
            semantic_weight=semantic_weight
        )
        
        if not documents:
            raise HTTPException(
                status_code=404,
                detail="No documents found. Please upload and process documents first."
            )
        
        # Build context
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
            
            # Prepare source - safe metadata access
            sources.append(SourceDocument(
                content=doc[:500] + "..." if len(doc) > 500 else doc,
                filename=safe_get_str(metadata, 'filename', 'unknown'),
                chunk_id=safe_get_str(metadata, 'chunk_id', f'chunk_{i}'),
                relevance_score=safe_get_score(score),  # Clamp to valid range
                page_number=page,
                chunk_type=chunk_type,
                semantic_score=None,
                keyword_score=None
            ))
        
        context = "\n\n".join(context_parts)
        
        # Generate answer
        answer = llm_service.generate_answer(
            question=request.question,
            context=context,
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
        
        return QueryResponse(
            question=request.question,
            answer=answer,
            sources=sources,
            processing_time=round(processing_time, 2),
            search_strategy="hybrid",
            confidence_score=round(confidence, 3),
            keyword_weight=round(keyword_weight, 2),
            semantic_weight=round(semantic_weight, 2)
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
async def query_with_conversation(request: ConversationQueryRequest):
    """Multi-turn conversational query with hybrid search"""
    start_time = time.time()
    
    try:
        # Validate and normalize weights
        keyword_weight = request.keyword_weight if request.keyword_weight is not None else 0.3
        semantic_weight = request.semantic_weight if request.semantic_weight is not None else 0.7
        
        total = keyword_weight + semantic_weight
        if abs(total - 1.0) > 0.01:
            keyword_weight = keyword_weight / total
            semantic_weight = semantic_weight / total
        
        # Retrieve chunks with hybrid search
        documents, metadatas, scores = vector_store.hybrid_search(
            query=request.question,
            top_k=request.top_k,
            file_id=request.file_id,
            keyword_weight=keyword_weight,
            semantic_weight=semantic_weight
        )
        
        if not documents:
            raise HTTPException(
                status_code=404,
                detail="No documents found."
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
            
            sources.append(SourceDocument(
                content=doc[:500] + "..." if len(doc) > 500 else doc,
                filename=safe_get_str(metadata, 'filename', 'unknown'),
                chunk_id=safe_get_str(metadata, 'chunk_id', f'chunk_{i}'),
                relevance_score=safe_get_score(score),  # Clamp to valid range
                page_number=page,
                chunk_type=chunk_type
            ))
        
        context = "\n\n".join(context_parts)
        
        # Generate answer with conversation history
        answer = llm_service.generate_answer(
            question=request.question,
            context=context,
            conversation_id=request.conversation_id,
            include_citations=True,
            temperature=request.temperature
        )
        
        # Detect hallucinations
        is_hallucination, confidence = llm_service.detect_hallucination(answer, context)
        
        if is_hallucination:
            answer += f"\n\n⚠️ Note: Lower confidence ({confidence:.2f})."
        
        processing_time = time.time() - start_time
        
        return QueryResponse(
            question=request.question,
            answer=answer,
            sources=sources,
            processing_time=round(processing_time, 2),
            search_strategy="hybrid",
            confidence_score=round(confidence, 3),
            conversation_id=request.conversation_id,
            keyword_weight=round(keyword_weight, 2),
            semantic_weight=round(semantic_weight, 2)
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
async def clear_conversation(conversation_id: str):
    """Clear conversation history"""
    try:
        llm_service.clear_conversation(conversation_id)
        return {
            "message": f"Conversation {conversation_id} cleared",
            "conversation_id": conversation_id
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error clearing conversation: {str(e)}"
        )