from fastapi import APIRouter, HTTPException
import time

from app.models.schemas import QueryRequest, QueryResponse, SourceDocument
from app.services.vector_store_service import VectorStoreService
from app.services.bedrock_service import BedrockService

router = APIRouter(prefix="/api/query", tags=["Query"])

# Initialize services
vector_store = VectorStoreService()
bedrock_service = BedrockService()


@router.post("/", response_model=QueryResponse)
async def query_knowledge_base(request: QueryRequest):
    """
    Query the knowledge base using RAG
    
    Process:
    1. Convert question to embedding
    2. Retrieve relevant chunks from vector store
    3. Construct context from retrieved chunks
    4. Generate answer using Claude via Bedrock
    5. Return answer with source citations
    """
    start_time = time.time()
    
    try:
        # Step 1 & 2: Search for relevant chunks
        documents, metadatas, distances = vector_store.search_similar_chunks(
            request.question,
            request.top_k
        )
        
        if not documents:
            raise HTTPException(
                status_code=404,
                detail="No documents found in knowledge base. Please upload and process documents first."
            )
        
        # Step 3: Construct context from retrieved chunks
        context_parts = []
        sources = []
        
        for i, (doc, metadata, distance) in enumerate(zip(documents, metadatas, distances)):
            # Add to context
            context_parts.append(f"[Source {i+1}]:\n{doc}\n")
            
            # Calculate relevance score (convert distance to similarity)
            # ChromaDB uses L2 distance, smaller is better
            relevance_score = max(0.0, 1.0 - (distance / 2.0))
            
            # Add to sources
            sources.append(SourceDocument(
                content=doc[:500] + "..." if len(doc) > 500 else doc,  # Truncate for response
                filename=metadata.get('filename', 'unknown'),
                chunk_id=f"{metadata.get('file_id', 'unknown')}_chunk_{metadata.get('chunk_index', i)}",
                relevance_score=round(relevance_score, 3)
            ))
        
        context = "\n\n".join(context_parts)
        
        # Step 4: Generate answer using Claude
        answer = bedrock_service.generate_response_with_claude(
            question=request.question,
            context=context,
            max_tokens=2000,
            temperature=0.7
        )
        
        processing_time = time.time() - start_time
        
        return QueryResponse(
            question=request.question,
            answer=answer,
            sources=sources,
            processing_time=round(processing_time, 2)
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )