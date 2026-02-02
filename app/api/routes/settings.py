"""
Settings Route - Updated for Open Source Models
Allows runtime switching of LLM, Embedding, and Vision models
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Literal
from app.services.llm_service import LLMService
from app.services.embedding_service import OpenSourceEmbeddings
from app.services.image_description_service import OpenSourceVisionService
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/settings", tags=["Settings"])

# Initialize services
llm_service = LLMService()


class ModelUpdateRequest(BaseModel):
    model_id: str


class EmbeddingUpdateRequest(BaseModel):
    model_id: str


class VisionUpdateRequest(BaseModel):
    model_id: str


class ModelInfoResponse(BaseModel):
    current_llm_model: str
    current_embedding_model: str
    current_vision_model: str
    temperature: float
    max_tokens: int
    available_llm_models: list
    available_embedding_models: list
    available_vision_models: list


@router.get("/models", response_model=ModelInfoResponse)
async def get_current_models():
    """
    Get current model configuration
    
    Returns current model IDs and available options
    """
    try:
        return ModelInfoResponse(
            current_llm_model=llm_service.get_current_model(),
            current_embedding_model=settings.embedding_model_id,
            current_vision_model=settings.vision_model_id,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
            available_llm_models=settings.available_llm_models,
            available_embedding_models=settings.available_embedding_models,
            available_vision_models=settings.available_vision_models
        )
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting model info: {str(e)}"
        )


@router.post("/models/llm")
async def update_llm_model(request: ModelUpdateRequest):
    """
    Update the LLM model at runtime
    
    Args:
        model_id: New HuggingFace model ID
    
    Available models:
        - meta-llama/Meta-Llama-3-8B-Instruct
        - mistralai/Mistral-7B-Instruct-v0.2
        - google/gemma-7b-it
    
    Example:
        {
            "model_id": "meta-llama/Meta-Llama-3-8B-Instruct"
        }
    """
    try:
        # Validate model
        if request.model_id not in settings.available_llm_models:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid model. Available models: {', '.join(settings.available_llm_models)}"
            )
        
        success = llm_service.update_model(request.model_id)
        
        if success:
            # Also update in settings for new service instances
            settings.llm_model_id = request.model_id
            
            return {
                "message": f"LLM model updated to {request.model_id}",
                "new_model_id": request.model_id,
                "note": "This change is temporary. Update .env for permanent change."
            }
        else:
            raise HTTPException(
                status_code=500,
                detail="Failed to update LLM model"
            )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating LLM model: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error updating LLM model: {str(e)}"
        )


@router.post("/models/embedding")
async def update_embedding_model(request: EmbeddingUpdateRequest):
    """
    Update the embedding model at runtime
    
    Args:
        model_id: New HuggingFace embedding model ID
    
    Available models:
        - BAAI/bge-large-en-v1.5
        - sentence-transformers/all-MiniLM-L6-v2
        - intfloat/e5-large-v2
    
    Example:
        {
            "model_id": "BAAI/bge-large-en-v1.5"
        }
    
    Note: After changing embedding model, you may need to re-index documents
    """
    try:
        # Validate model
        if request.model_id not in settings.available_embedding_models:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid model. Available models: {', '.join(settings.available_embedding_models)}"
            )
        
        # Update in settings
        settings.embedding_model_id = request.model_id
        
        # Note: Existing vector store will need to be re-initialized
        # Or documents re-indexed with new embeddings
        
        return {
            "message": f"Embedding model updated to {request.model_id}",
            "new_model_id": request.model_id,
            "warning": "Existing documents should be re-indexed with new embedding model for consistency",
            "note": "This change is temporary. Update .env for permanent change."
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating embedding model: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error updating embedding model: {str(e)}"
        )


@router.post("/models/vision")
async def update_vision_model(request: VisionUpdateRequest):
    """
    Update the vision model at runtime
    
    Args:
        model_id: New HuggingFace vision model ID
    
    Available models:
        - llava-hf/llava-1.5-7b-hf
        - microsoft/Phi-3-vision-128k-instruct
        - SkunkworksAI/BakLLaVA-1
    
    Example:
        {
            "model_id": "llava-hf/llava-1.5-7b-hf"
        }
    """
    try:
        # Validate model
        if request.model_id not in settings.available_vision_models:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid model. Available models: {', '.join(settings.available_vision_models)}"
            )
        
        # Update in settings
        settings.vision_model_id = request.model_id
        
        # Update vision service
        from app.services.vision_service import get_image_description_service
        vision_service = get_image_description_service()
        success = vision_service.update_model(request.model_id)
        
        if success:
            return {
                "message": f"Vision model updated to {request.model_id}",
                "new_model_id": request.model_id,
                "note": "This change is temporary. Update .env for permanent change."
            }
        else:
            raise HTTPException(
                status_code=500,
                detail="Failed to update vision model"
            )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating vision model: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error updating vision model: {str(e)}"
        )


@router.get("/models/available")
async def get_available_models():
    """
    Get all available models for each category
    
    Returns lists of available LLM, embedding, and vision models
    """
    try:
        return {
            "llm_models": settings.available_llm_models,
            "embedding_models": settings.available_embedding_models,
            "vision_models": settings.available_vision_models
        }
    except Exception as e:
        logger.error(f"Error getting available models: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting available models: {str(e)}"
        )