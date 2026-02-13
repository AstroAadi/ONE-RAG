from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.llm_service import LLMService
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/settings", tags=["Settings"])

# Initialize LLM service with lazy loading
_llm_service = None

def get_llm_service():
    """Get or initialize LLM service (lazy loading)"""
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service


class ModelUpdateRequest(BaseModel):
    model_id: str


class ModelInfoResponse(BaseModel):
    current_model_id: str
    temperature: float
    max_tokens: int
    embedding_model_id: str


@router.get("/model", response_model=ModelInfoResponse)
async def get_current_model():
    """
    Get current model configuration
    
    Returns current model ID and parameters
    """
    try:
        return ModelInfoResponse(
            current_model_id=get_llm_service().get_current_model(),
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
            embedding_model_id=settings.embedding_model_id
        )
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting model info: {str(e)}"
        )


@router.post("/model")
async def update_model(request: ModelUpdateRequest):
    """
    Update the LLM model at runtime
    
    Args:
        model_id: New Bedrock model ID
    
    Example:
        {
            "model_id": "anthropic.claude-3-5-sonnet-20240620-v1:0"
        }
    
    Note: This only updates for current session. 
    To make permanent, update BEDROCK_MODEL_ID in .env
    """
    try:
        success = get_llm_service().update_model(request.model_id)
        
        if success:
            return {
                "message": f"Model updated to {request.model_id}",
                "new_model_id": request.model_id,
                "note": "This change is temporary. Update .env for permanent change."
            }
        else:
            raise HTTPException(
                status_code=500,
                detail="Failed to update model"
            )
    
    except Exception as e:
        logger.error(f"Error updating model: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error updating model: {str(e)}"
        )