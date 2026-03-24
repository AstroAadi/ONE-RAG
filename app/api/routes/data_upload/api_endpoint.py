
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Body
from sqlalchemy.orm import Session
import uuid
from datetime import datetime
import logging

from app.models.schemas import APIConfig, APIKBRequest
from app.core.dependencies import get_user_id, get_database_session
from app.db.models import KnowledgeBase, DataSourceType as DBDataSourceType

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/upload", tags=["API Upload"])


@router.post("/api-endpoint", response_model=dict)
async def create_api_kb(
    request: APIKBRequest = Body(...),
    user_id: str = Depends(get_user_id),
    db: Session = Depends(get_database_session)
):
    """
    Create a knowledge base with API source
    
    Headers:
        X-User-ID: User identifier (REQUIRED)
    
    Request Body (JSON):
        {
            "knowledge_base_name": "External API",
            "description": "API data source",
            "api_config": {
                "endpoint_url": "https://api.example.com/data",
                "method": "GET",
                "auth_type": "bearer",
                "bearer_token": "your-token",
                "headers": {"Accept": "application/json"},
                "timeout": 30
            },
            "auto_sync_enabled": true,
            "sync_interval_hours": 12
        }
    """
    try:
        api_cfg = request.api_config
        
        # Normalize KB name
        normalized_name = request.knowledge_base_name.lower().strip()
        
        # Check if KB exists
        existing = db.query(KnowledgeBase).filter(
            KnowledgeBase.user_id == user_id,
            KnowledgeBase.knowledge_base_name_normalized == normalized_name
        ).first()
        
        if existing:
            raise HTTPException(
                status_code=400,
                detail=f"Knowledge base '{request.knowledge_base_name}' already exists"
            )
        
        # TODO: Encrypt sensitive data before storing
        data_source_config = api_cfg.model_dump()
        
        # Create KB
        kb_id = str(uuid.uuid4())
        new_kb = KnowledgeBase(
            knowledge_base_id=kb_id,
            user_id=user_id,
            knowledge_base_name=request.knowledge_base_name.strip(),
            knowledge_base_name_normalized=normalized_name,
            description=request.description,
            data_source_type=DBDataSourceType.API,
            data_source_config=data_source_config,
            auto_sync_enabled=request.auto_sync_enabled,
            sync_interval_hours=request.sync_interval_hours,
            document_count=0,
            total_chunks=0,
            total_size_bytes=0,
            created_at=datetime.utcnow()
        )
        
        db.add(new_kb)
        db.commit()
        db.refresh(new_kb)
        
        logger.info(f"✓ Created API KB: {kb_id} - '{new_kb.knowledge_base_name}' for user {user_id}")
        
        return {
            "knowledge_base_id": kb_id,
            "knowledge_base_name": new_kb.knowledge_base_name,
            "data_source_type": "api",
            "endpoint_url": api_cfg.endpoint_url,
            "method": api_cfg.method,
            "auto_sync_enabled": request.auto_sync_enabled,
            "sync_interval_hours": request.sync_interval_hours,
            "message": "API knowledge base created successfully. Sync not yet implemented.",
            "warning": "API credentials stored unencrypted - implement encryption before production!"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error creating API KB: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
