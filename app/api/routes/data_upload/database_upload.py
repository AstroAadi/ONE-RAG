from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Body
from sqlalchemy.orm import Session
import uuid
from datetime import datetime
import logging

from app.models.schemas import DatabaseConfig, DatabaseKBRequest, DatabaseType, DatabaseKBAddDataRequest
from app.core.dependencies import get_user_id, get_database_session
from app.db.models import KnowledgeBase, DataSourceType as DBDataSourceType
from app.services.data_sources.database_sync_service import DatabaseSyncService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/upload", tags=["Database Upload"])


@router.post("/database", response_model=dict)
async def create_database_kb(
    request: DatabaseKBRequest = Body(...),
    user_id: str = Depends(get_user_id),
    db: Session = Depends(get_database_session)
):
    """
    Create a knowledge base from a user's database, fetch rows using the
    provided query, and store them (row-wise) into the vector store.

    Headers:
        X-User-ID: User identifier (REQUIRED)

    Request Body (JSON):
        {
            "knowledge_base_name": "Sales Database",
            "description": "Customer sales records",
            "database_config": {
                "db_type": "postgresql",
                "host": "db.example.com",
                "port": 5432,
                "username": "readonly_user",
                "password": "secret_password",
                "database_name": "sales_db",
                "query": "SELECT * FROM customers WHERE active=1",
                "ssl_enabled": false,
                "connection_timeout": 30
            },
            "auto_sync_enabled": true,
            "sync_interval_hours": 24
        }
    """
    try:
        db_config = request.database_config

        # Normalize KB name
        normalized_name = request.knowledge_base_name.lower().strip()

        # ------------------------------------------------------------------ #
        # 1. Check duplicate KB name (queries OUR app DB, not the user's DB) #
        # ------------------------------------------------------------------ #
        existing = db.query(KnowledgeBase).filter(
            KnowledgeBase.user_id == user_id,
            KnowledgeBase.knowledge_base_name_normalized == normalized_name
        ).first()

        if existing:
            raise HTTPException(
                status_code=400,
                detail=f"Knowledge base '{request.knowledge_base_name}' already exists"
            )

        # ------------------------------------------------------------------ #
        # 2. Fill default port if omitted                                     #
        # ------------------------------------------------------------------ #
        db_type = db_config.db_type
        if db_config.port is None:
            default_ports = {
                DatabaseType.MYSQL: 3306,
                DatabaseType.POSTGRESQL: 5432,
                DatabaseType.MSSQL: 1433,
                DatabaseType.MONGODB: 27017
            }
            db_config.port = default_ports.get(db_type, 3306)

        logger.info(f"Creating {db_type.value.upper()} KB: {request.knowledge_base_name}")

        # ------------------------------------------------------------------ #
        # 3. Persist KB record in OUR app DB                                 #
        # ------------------------------------------------------------------ #
        # TODO: Encrypt sensitive fields (password) before storing
        data_source_config = db_config.model_dump()
        # Pydantic enums → plain strings so JSON serialization works cleanly
        data_source_config['db_type'] = db_type.value

        kb_id = str(uuid.uuid4())
        new_kb = KnowledgeBase(
            knowledge_base_id=kb_id,
            user_id=user_id,
            knowledge_base_name=request.knowledge_base_name.strip(),
            knowledge_base_name_normalized=normalized_name,
            description=request.description,
            data_source_type=DBDataSourceType.DATABASE,
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

        logger.info(
            f"✓ Created DATABASE KB: {kb_id} - '{new_kb.knowledge_base_name}' "
            f"for user {user_id}"
        )

        # ------------------------------------------------------------------ #
        # 4. Immediately fetch rows from the user's DB and store them        #
        #    row-wise into the vector store (the whole point of this endpoint)#
        # ------------------------------------------------------------------ #
        sync_service = DatabaseSyncService()
        success, message, metadata = await sync_service.sync_database_kb(
            knowledge_base_id=kb_id,
            user_id=user_id,
            database_config=data_source_config,
            db_session=db
        )

        if not success:
            # KB record was created but sync failed — clean up and surface error
            db.delete(new_kb)
            db.commit()
            raise HTTPException(
                status_code=500,
                detail=f"KB created but data sync failed: {message}"
            )

        return {
            "knowledge_base_id": kb_id,
            "knowledge_base_name": new_kb.knowledge_base_name,
            "data_source_type": "database",
            "database_type": db_type.value,
            "host": db_config.host,
            "port": db_config.port,
            "database_name": db_config.database_name,
            "auto_sync_enabled": request.auto_sync_enabled,
            "sync_interval_hours": request.sync_interval_hours,
            "rows_retrieved": metadata.get("rows_retrieved"),
            "rows_embedded": metadata.get("rows_embedded"),
            "columns": metadata.get("columns"),
            "chunks_created": metadata.get("chunks_created"),
            "collection_name": metadata.get("collection_name"),
            "message": (
                f"{db_type.value.upper()} knowledge base created and "
                f"{metadata.get('rows_embedded', 0)} rows indexed successfully."
            ),
            "status": "ready_for_query",
            "warning": "Credentials stored unencrypted — implement encryption before production!"
        }

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error creating database KB: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/database/sync/{kb_id}", response_model=dict)
async def sync_database_kb(
    kb_id: str,
    user_id: str = Depends(get_user_id),
    db: Session = Depends(get_database_session)
):
    """
    Manually re-sync a database knowledge base (re-runs the stored query and
    re-indexes all rows into the vector store).

    Headers:
        X-User-ID: User identifier (REQUIRED)
    """
    try:
        kb = db.query(KnowledgeBase).filter(
            KnowledgeBase.knowledge_base_id == kb_id,
            KnowledgeBase.user_id == user_id
        ).first()

        if not kb:
            raise HTTPException(
                status_code=404,
                detail=f"Knowledge base {kb_id} not found"
            )

        if kb.data_source_type != DBDataSourceType.DATABASE:
            raise HTTPException(
                status_code=400,
                detail=f"Knowledge base is not a database source "
                       f"(type: {kb.data_source_type.value})"
            )

        if not kb.data_source_config:
            raise HTTPException(
                status_code=400,
                detail="Database configuration not found"
            )

        # TODO: Decrypt credentials here before using
        database_config = kb.data_source_config

        logger.info(f"Re-syncing database KB: {kb_id} for user {user_id}")

        sync_service = DatabaseSyncService()
        success, message, metadata = await sync_service.sync_database_kb(
            knowledge_base_id=kb_id,
            user_id=user_id,
            database_config=database_config,
            db_session=db
        )

        if not success:
            raise HTTPException(
                status_code=500,
                detail=f"Sync failed: {message}"
            )

        return {
            "knowledge_base_id": kb_id,
            "knowledge_base_name": kb.knowledge_base_name,
            "sync_status": "success",
            "message": message,
            "rows_retrieved": metadata.get("rows_retrieved"),
            "rows_embedded": metadata.get("rows_embedded"),
            "columns": metadata.get("columns"),
            "chunks_created": metadata.get("chunks_created"),
            "collection_name": metadata.get("collection_name"),
            "status": "ready_for_query",
            "next_step": "Data is now indexed. Use POST /api/query to query your data!"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Sync error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# --------------------------------------------------------------------------- #
# NEW: PUT /api/upload/database/{kb_id}                                       #
# Append rows from a new query into an existing database KB.                  #
# Credentials are used in-memory only — never stored.                         #
# --------------------------------------------------------------------------- #
@router.put("/database/{kb_id}", response_model=dict)
async def add_data_to_database_kb(
    kb_id: str,
    request: DatabaseKBAddDataRequest = Body(...),
    user_id: str = Depends(get_user_id),
    db: Session = Depends(get_database_session)
):
    """
    Append more data to an existing database knowledge base by running a
    new query against the user's database.

    - The existing indexed rows are preserved — nothing is deleted.
    - The new query's rows are embedded and added on top.
    - Credentials (username / password) are used in-memory only and are
      NEVER stored in our system.

    Headers:
        X-User-ID: User identifier (REQUIRED)

    Request Body (JSON):
        {
            "database_config": {
                "db_type": "postgresql",
                "host": "db.example.com",
                "port": 5432,
                "username": "readonly_user",
                "password": "secret_password",
                "database_name": "sales_db",
                "query": "SELECT * FROM orders WHERE year = 2024",
                "ssl_enabled": false,
                "connection_timeout": 30
            },
            "query_label": "orders_2024"
        }
    """
    try:
        # 1. Verify KB exists and belongs to this user
        kb = db.query(KnowledgeBase).filter(
            KnowledgeBase.knowledge_base_id == kb_id,
            KnowledgeBase.user_id == user_id
        ).first()

        if not kb:
            raise HTTPException(status_code=404, detail=f"Knowledge base {kb_id} not found")

        if kb.data_source_type != DBDataSourceType.DATABASE:
            raise HTTPException(
                status_code=400,
                detail=f"Knowledge base is not a database source "
                       f"(type: {kb.data_source_type.value})"
            )

        # 2. Resolve default port
        db_config = request.database_config
        db_type = db_config.db_type
        if db_config.port is None:
            default_ports = {
                DatabaseType.MYSQL: 3306,
                DatabaseType.POSTGRESQL: 5432,
                DatabaseType.MSSQL: 1433,
                DatabaseType.MONGODB: 27017,
            }
            db_config.port = default_ports.get(db_type, 3306)

        # 3. Build full config dict (stays in memory, never persisted)
        full_config = db_config.model_dump()
        full_config['db_type'] = db_type.value

        # Use provided label or auto-generate one from timestamp
        query_label = (
            request.query_label.strip()
            if request.query_label
            else f"query_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        )

        logger.info(
            f"Appending data to KB {kb_id} ('{kb.knowledge_base_name}') "
            f"label='{query_label}' db={db_type.value.upper()} "
            f"host={db_config.host}/{db_config.database_name}"
        )

        # 4. Run the new query and append rows — credentials never leave memory
        sync_service = DatabaseSyncService()
        success, message, metadata = await sync_service.append_query_to_kb(
            knowledge_base_id=kb_id,
            user_id=user_id,
            database_config=full_config,
            query_label=query_label,
            db_session=db,
        )

        if not success:
            raise HTTPException(status_code=500, detail=f"Data append failed: {message}")

        return {
            "knowledge_base_id": kb_id,
            "knowledge_base_name": kb.knowledge_base_name,
            "query_label": query_label,
            "database_type": db_type.value,
            "host": db_config.host,
            "database_name": db_config.database_name,
            "rows_retrieved": metadata.get("rows_retrieved"),
            "rows_embedded": metadata.get("rows_embedded"),
            "columns": metadata.get("columns"),
            "chunks_added": metadata.get("chunks_added"),
            "total_chunks_in_kb": metadata.get("total_chunks_in_kb"),
            "message": (
                f"Query '{query_label}' ran successfully. "
                f"{metadata.get('rows_embedded', 0)} new rows appended to "
                f"'{kb.knowledge_base_name}'. Credentials were NOT stored."
            ),
            "status": "ready_for_query",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error appending data to KB {kb_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
