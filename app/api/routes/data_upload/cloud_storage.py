
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Body
from sqlalchemy.orm import Session
import uuid
from datetime import datetime
import logging

from app.models.schemas import CloudStorageConfig, CloudStorageKBRequest
from app.core.dependencies import get_user_id, get_database_session
from app.db.models import KnowledgeBase, DataSourceType as DBDataSourceType

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/upload", tags=["Cloud Storage Upload"])


@router.post("/cloud-storage", response_model=dict)
async def create_cloud_storage_kb(
    request: CloudStorageKBRequest = Body(...),
    user_id: str = Depends(get_user_id),
    db: Session = Depends(get_database_session)
):
    """
    Create a knowledge base with Cloud Storage source
    
    Supports: AWS S3, Azure Blob Storage, Google Cloud Storage
    
    Headers:
        X-User-ID: User identifier (REQUIRED)
    
    Request Body (JSON) - AWS S3 Example:
        {
            "knowledge_base_name": "S3 Documents",
            "description": "Documents from S3 bucket",
            "cloud_storage_config": {
                "provider": "aws_s3",
                "bucket_name": "my-documents",
                "path_prefix": "exports/",
                "aws_access_key_id": "AKIA...",
                "aws_secret_access_key": "secret...",
                "aws_region": "us-east-1"
            },
            "auto_sync_enabled": true,
            "sync_interval_hours": 6
        }
    """
    try:
        cloud_cfg = request.cloud_storage_config
        
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
        data_source_config = cloud_cfg.model_dump()
        
        # Create KB
        kb_id = str(uuid.uuid4())
        new_kb = KnowledgeBase(
            knowledge_base_id=kb_id,
            user_id=user_id,
            knowledge_base_name=request.knowledge_base_name.strip(),
            knowledge_base_name_normalized=normalized_name,
            description=request.description,
            data_source_type=DBDataSourceType.CLOUD_STORAGE,
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
        
        logger.info(f"✓ Created CLOUD_STORAGE KB: {kb_id} - '{new_kb.knowledge_base_name}' for user {user_id}")
        
        return {
            "knowledge_base_id": kb_id,
            "knowledge_base_name": new_kb.knowledge_base_name,
            "data_source_type": "cloud_storage",
            "provider": cloud_cfg.provider.value,
            "bucket_name": cloud_cfg.bucket_name,
            "path_prefix": cloud_cfg.path_prefix or "",
            "auto_sync_enabled": request.auto_sync_enabled,
            "sync_interval_hours": request.sync_interval_hours,
            "message": f"{cloud_cfg.provider.value.upper()} knowledge base created successfully. Sync not yet implemented.",
            "warning": "Cloud credentials stored unencrypted - implement encryption before production!"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error creating cloud storage KB: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

