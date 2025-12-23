from fastapi import APIRouter, HTTPException

from app.models.schemas import KnowledgeBaseStatus
from app.services.vector_store_service import VectorStoreService

router = APIRouter(prefix="/api/knowledge-base", tags=["Knowledge Base"])

# Initialize services
vector_store = VectorStoreService()


@router.get("/status", response_model=KnowledgeBaseStatus)
async def get_knowledge_base_status():
    """
    Get current status of the knowledge base
    
    Returns:
    - Total number of documents
    - Total number of chunks
    - List of documents with their IDs
    """
    try:
        stats = vector_store.get_collection_stats()
        
        return KnowledgeBaseStatus(
            total_documents=stats['total_documents'],
            total_chunks=stats['total_chunks'],
            documents=stats['documents']
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving knowledge base status: {str(e)}"
        )


@router.delete("/document/{file_id}")
async def delete_document(file_id: str):
    """
    Delete a specific document from the knowledge base
    
    Args:
        file_id: The unique identifier of the file to delete
    """
    try:
        success = vector_store.delete_document_by_file_id(file_id)
        
        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"Document with file_id '{file_id}' not found"
            )
        
        return {
            "message": f"Document {file_id} deleted successfully",
            "file_id": file_id
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting document: {str(e)}"
        )


@router.delete("/reset")
async def reset_knowledge_base():
    """
    Reset the entire knowledge base (delete all documents)
    
    Warning: This action cannot be undone!
    """
    try:
        success = vector_store.reset_collection()
        
        if not success:
            raise HTTPException(
                status_code=500,
                detail="Failed to reset knowledge base"
            )
        
        return {
            "message": "Knowledge base reset successfully. All documents have been deleted.",
            "status": "success"
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error resetting knowledge base: {str(e)}"
        )