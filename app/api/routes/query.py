"""
Enhanced Query Route with Advanced RAG Capabilities
Supports hybrid search, conversation history, and intelligent retrieval

FIXED: 
- Removed unused /expand-query endpoint
- Removed unused generate_query_variations() functionality
"""

from fastapi import APIRouter, HTTPException
from app.models.schemas import (
    QueryRequest, QueryResponse, SourceDocument,
    EnhancedQueryRequest, ConversationQueryRequest
)
from app.services.vector_store_service import EnhancedVectorStoreService
from app.services.llm_service import EnhancedLLMService
import time
import logging
from typing import Optional

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/query", tags=["Query"])

# Initialize services
vector_store = EnhancedVectorStoreService()
llm_service = EnhancedLLMService()


@router.post("/", response_model=QueryResponse)
async def query_knowledge_base(request: QueryRequest):
    """
    Standard query with hybrid search and contextual understanding
    
    Process:
    1. Classify query intent (cached to avoid slow classification)
    2. Perform hybrid search (semantic + structural)
    3. Compress and rank retrieved context
    4. Generate answer with citations
    5. Verify answer quality
    """
    start_time = time.time()
    
    try:
        # Use hybrid search mode by default (query classification removed as requested)
        search_mode = "hybrid"
        
        # Hybrid search with intelligent retrieval
        documents, metadatas, distances = vector_store.hybrid_search(
            query=request.question,
            top_k=request.top_k,
            file_id=getattr(request, 'file_id', None),
            search_mode=search_mode
        )
        
        if not documents:
            raise HTTPException(
                status_code=404,
                detail="No documents found in knowledge base. Please upload and process documents first."
            )
        
        # Step 4: Build enhanced context
        context_parts = []
        sources = []
        
        for i, (doc, metadata, distance) in enumerate(zip(documents, metadatas, distances)):
            # Build rich context with metadata
            source_header = f"[Source {i+1}]"
            
            # Add section hierarchy if available
            section_hierarchy = metadata.get('section_hierarchy', '')
            if section_hierarchy and section_hierarchy != '[]':
                try:
                    # Handle both JSON string and comma-separated string
                    if section_hierarchy.startswith('['):
                        import json
                        hierarchy = json.loads(section_hierarchy)
                    else:
                        hierarchy = [s.strip() for s in section_hierarchy.split(',') if s.strip()]
                    
                    if hierarchy:
                        source_header += f" (Section: {' > '.join(hierarchy)})"
                except Exception as e:
                    logger.debug(f"Could not parse section hierarchy: {e}")
            
            # Add page number
            page = metadata.get('page_number', 0)
            if page > 0:
                source_header += f" [Page {page}]"
            
            # Add content type
            chunk_type = metadata.get('chunk_type', 'text')
            if chunk_type != 'text':
                source_header += f" [{chunk_type.upper()}]"
            
            context_parts.append(f"{source_header}:\n{doc}\n")
            
            # Calculate enhanced relevance score
            relevance_score = max(0.0, 1.0 - (distance / 2.0))
            
            # Prepare source for response
            sources.append(SourceDocument(
                content=doc[:500] + "..." if len(doc) > 500 else doc,
                filename=metadata.get('filename', 'unknown'),
                chunk_id=metadata.get('chunk_id', f'chunk_{i}'),
                relevance_score=round(relevance_score, 3),
                page_number=page,
                chunk_type=chunk_type,
                section=hierarchy[0] if 'hierarchy' in locals() and hierarchy else None
            ))
        
        context = "\n\n".join(context_parts)
        
        # Step 5: Generate answer with conversation awareness
        answer = llm_service.generate_answer(
            question=request.question,
            context=context,
            conversation_id=None,  # For basic query, no conversation
            include_citations=True,
            temperature=request.temperature  # Pass user-specified temperature
        )
        
        # Step 6: Detect potential hallucinations
        is_hallucination, confidence = llm_service.detect_hallucination(answer, context)
        
        if is_hallucination:
            logger.warning(f"Potential hallucination detected with confidence {confidence}")
            answer += f"\n\n⚠️ Note: This answer has lower confidence ({confidence:.2f}). Please verify with source documents."
        
        processing_time = time.time() - start_time
        
        return QueryResponse(
            question=request.question,
            answer=answer,
            sources=sources,
            processing_time=round(processing_time, 2),
            search_strategy=search_mode,
            confidence_score=round(confidence, 3)
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )


@router.post("/conversation", response_model=QueryResponse)
async def query_with_conversation(request: ConversationQueryRequest):
    """
    Multi-turn conversational query with history awareness
    
    Maintains conversation context across multiple queries
    """
    start_time = time.time()
    
    try:
        conversation_id = request.conversation_id
        
        # Use specified search mode or default to hybrid (query classification removed)
        search_mode = request.search_mode or "hybrid"
        
        # Retrieve relevant chunks
        documents, metadatas, distances = vector_store.hybrid_search(
            query=request.question,
            top_k=request.top_k,
            file_id=request.file_id,
            search_mode=search_mode
        )
        
        if not documents:
            raise HTTPException(
                status_code=404,
                detail="No documents found."
            )
        
        # Build context
        context_parts = []
        sources = []
        
        for i, (doc, metadata, distance) in enumerate(zip(documents, metadatas, distances)):
            source_header = f"[Source {i+1}]"
            
            # Add metadata
            page = metadata.get('page_number', 0)
            if page > 0:
                source_header += f" [Page {page}]"
            
            chunk_type = metadata.get('chunk_type', 'text')
            if chunk_type != 'text':
                source_header += f" [{chunk_type.upper()}]"
            
            context_parts.append(f"{source_header}:\n{doc}\n")
            
            relevance_score = max(0.0, 1.0 - (distance / 2.0))
            
            sources.append(SourceDocument(
                content=doc[:500] + "..." if len(doc) > 500 else doc,
                filename=metadata.get('filename', 'unknown'),
                chunk_id=metadata.get('chunk_id', f'chunk_{i}'),
                relevance_score=round(relevance_score, 3),
                page_number=page,
                chunk_type=chunk_type
            ))
        
        context = "\n\n".join(context_parts)
        
        # Generate answer with conversation history
        answer = llm_service.generate_answer(
            question=request.question,
            context=context,
            conversation_id=conversation_id,
            include_citations=True,
            temperature=request.temperature  # Pass user-specified temperature
        )
        
        # Detect hallucinations
        is_hallucination, confidence = llm_service.detect_hallucination(answer, context)
        
        if is_hallucination:
            answer += f"\n\n⚠️ Note: Lower confidence answer ({confidence:.2f}). Please verify."
        
        processing_time = time.time() - start_time
        
        return QueryResponse(
            question=request.question,
            answer=answer,
            sources=sources,
            processing_time=round(processing_time, 2),
            search_strategy=search_mode,
            confidence_score=round(confidence, 3),
            conversation_id=conversation_id
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in conversation query: {str(e)}")
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


@router.get("/search-by-metadata")
async def search_by_metadata(
    file_id: Optional[str] = None,
    page_number: Optional[int] = None,
    chunk_type: Optional[str] = None,
    limit: int = 10
):
    """
    Search chunks by metadata filters
    
    Useful for finding specific types of content (tables, images, etc.)
    """
    try:
        filters = {}
        
        if file_id:
            filters['file_id'] = file_id
        if page_number is not None:
            filters['page_number'] = page_number
        if chunk_type:
            filters['chunk_type'] = chunk_type
        
        if not filters:
            raise HTTPException(
                status_code=400,
                detail="At least one filter must be provided"
            )
        
        results = vector_store.search_by_metadata(filters, limit)
        
        return {
            "filters_applied": filters,
            "results_count": len(results),
            "results": results
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error in metadata search: {str(e)}"
        )