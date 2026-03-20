"""
Database Sync Service
Retrieves data from the user's database, converts rows to JSON,
generates embeddings (dense + sparse), and stores them row-wise
in the single shared collection (one-rag-collection).

Points are stored with BOTH dense and sparse vectors — identical to
how file-based chunks are stored — so the collection schema stays
consistent and the recreation guard in VectorStoreService never fires.
"""

import logging
from typing import Dict, Any, Tuple, List
from datetime import datetime
import pandas as pd
import json
import uuid

from app.services.data_sources.database_connector import DatabaseConnectorFactory
from app.services.embedding_service import BedrockEmbeddingService
from app.services.splade_service import GPUSPLADEEmbedder
from app.services.vector_store_service import VectorStoreService, string_to_uuid

logger = logging.getLogger(__name__)


class DatabaseSyncService:
    """Service to sync database data directly to the vector database"""

    def __init__(self):
        self.embedding_service = BedrockEmbeddingService()
        self.vector_store = VectorStoreService()

    # ---------------------------------------------------------------------- #
    # Public: full replace sync (used by POST /database and POST /sync)      #
    # ---------------------------------------------------------------------- #
    async def sync_database_kb(
        self,
        knowledge_base_id: str,
        user_id: str,
        database_config: Dict[str, Any],
        db_session,
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Full-replace sync: connect → run query → embed rows → upsert into
        the shared collection, replacing all previously indexed rows for
        this KB (via deterministic UUIDs, existing rows are overwritten).
        KB stats are reset to reflect only this query's rows.
        """
        return await self._run_query_and_index(
            knowledge_base_id=knowledge_base_id,
            user_id=user_id,
            database_config=database_config,
            query_label="sync",
            append=False,
            db_session=db_session,
        )

    # ---------------------------------------------------------------------- #
    # Public: append sync (used by PUT /database/{kb_id})                    #
    # ---------------------------------------------------------------------- #
    async def append_query_to_kb(
        self,
        knowledge_base_id: str,
        user_id: str,
        database_config: Dict[str, Any],
        query_label: str,
        db_session,
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Append sync: connect → run query → embed rows → upsert into the
        shared collection WITHOUT deleting existing rows.

        Chunk IDs incorporate query_label so rows from different queries
        never collide even if row indices overlap.
        KB stats are incremented by the new row count.
        """
        return await self._run_query_and_index(
            knowledge_base_id=knowledge_base_id,
            user_id=user_id,
            database_config=database_config,
            query_label=query_label,
            append=True,
            db_session=db_session,
        )

    # ---------------------------------------------------------------------- #
    # Internal: shared implementation                                         #
    # ---------------------------------------------------------------------- #
    async def _run_query_and_index(
        self,
        knowledge_base_id: str,
        user_id: str,
        database_config: Dict[str, Any],
        query_label: str,
        append: bool,
        db_session,
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Core logic shared by both sync modes.

        append=False → chunk count is reset (full replace semantics)
        append=True  → chunk count is incremented (additive semantics)
        """
        try:
            from app.db.models import KnowledgeBase
            from qdrant_client.models import PointStruct

            logger.info(
                f"{'Appending' if append else 'Syncing'} KB {knowledge_base_id} "
                f"label='{query_label}'"
            )

            # ---------------------------------------------------------------- #
            # 1. Normalise db_type                                             #
            # ---------------------------------------------------------------- #
            db_type_raw = database_config.get('db_type', '')
            db_type = db_type_raw.value if hasattr(db_type_raw, 'value') else str(db_type_raw)
            db_type = db_type.lower()

            # ---------------------------------------------------------------- #
            # 2. Connect to user's database                                    #
            # ---------------------------------------------------------------- #
            connector = DatabaseConnectorFactory.get_connector(db_type, database_config)
            if not connector.connect():
                return False, f"Failed to connect to {db_type} database", {}
            logger.info(f"✓ Connected to {db_type} database")

            # ---------------------------------------------------------------- #
            # 3. Execute query                                                  #
            # ---------------------------------------------------------------- #
            query = database_config.get('query', '')
            if not query:
                connector.disconnect()
                return False, "No query provided in database_config", {}

            success, result_df, error = connector.execute_query(query)
            connector.disconnect()

            if not success:
                return False, f"Query execution failed: {error}", {}
            if result_df is None or result_df.empty:
                return False, "Query returned no data", {}

            row_count = len(result_df)
            logger.info(f"✓ Query returned {row_count} rows")

            # ---------------------------------------------------------------- #
            # 4. Build text list for batch embedding                           #
            # ---------------------------------------------------------------- #
            row_texts: List[str] = []
            row_meta: List[Dict[str, Any]] = []

            for idx, row in result_df.iterrows():
                try:
                    row_json = self._clean_json(row.to_dict())
                    text_content = self._row_to_text(row_json)

                    # Chunk ID is deterministic and scoped to (kb, label, row_index)
                    # so rows from different query_labels never overwrite each other
                    # but re-running the same label+query does upsert (idempotent).
                    chunk_id = str(uuid.uuid5(
                        uuid.NAMESPACE_URL,
                        f"{knowledge_base_id}/{query_label}/row/{idx}",
                    ))

                    row_texts.append(text_content)
                    row_meta.append({
                        'chunk_id': chunk_id,
                        'text': text_content,
                        'row_json': row_json,
                        'row_index': int(idx),
                    })
                except Exception as row_err:
                    logger.warning(f"Failed to prepare row {idx}: {row_err}")
                    continue

            if not row_texts:
                return False, "No rows were successfully prepared", {}

            # ---------------------------------------------------------------- #
            # 5. Generate dense + sparse embeddings (batch)                    #
            # ---------------------------------------------------------------- #
            max_len = 2048
            truncated = [t[:max_len] for t in row_texts]

            logger.info(f"Generating dense embeddings for {len(truncated)} rows...")
            dense_vectors = self.embedding_service.embed_documents(truncated)

            sparse_embedder = self.vector_store.sparse_embeddings
            if sparse_embedder is None:
                return False, (
                    "SPLADE sparse embeddings are not available. "
                    "Cannot store rows without sparse vectors."
                ), {}

            logger.info(f"Generating sparse embeddings for {len(truncated)} rows...")
            sparse_vectors = sparse_embedder.embed_documents(truncated)

            # ---------------------------------------------------------------- #
            # 6. Build PointStructs                                            #
            # ---------------------------------------------------------------- #
            qdrant_points: List[PointStruct] = []
            for i, meta in enumerate(row_meta):
                payload = {
                    'user_id': user_id,
                    'knowledge_base_id': knowledge_base_id,
                    'knowledge_base_name': '',
                    'file_id': f"db_{knowledge_base_id}",
                    'filename': f"{db_type}_query_result",
                    'chunk_id': meta['chunk_id'],
                    'text': meta['text'],
                    'chunk_type': 'database_row',
                    'chunk_index': meta['row_index'],
                    'page_number': 0,
                    'source_type': 'database',
                    'database_type': db_type,
                    'query_label': query_label,
                    'row_index': meta['row_index'],
                    'row_data': meta['row_json'],
                    'synced_at': datetime.utcnow().isoformat(),
                    'created_at': datetime.utcnow().isoformat(),
                }
                qdrant_points.append(
                    PointStruct(
                        id=string_to_uuid(meta['chunk_id']),
                        vector={"dense": dense_vectors[i], "sparse": sparse_vectors[i]},
                        payload=payload,
                    )
                )

            # ---------------------------------------------------------------- #
            # 7. Upsert into shared collection                                 #
            # ---------------------------------------------------------------- #
            collection_name = self.vector_store.collection_name
            try:
                batch_size = 100
                for i in range(0, len(qdrant_points), batch_size):
                    self.vector_store.client.upsert(
                        collection_name=collection_name,
                        points=qdrant_points[i: i + batch_size],
                    )
                logger.info(
                    f"✓ Upserted {len(qdrant_points)} rows into '{collection_name}'"
                )
            except Exception as vec_err:
                logger.error(f"Vector DB upsert failed: {vec_err}")
                return False, f"Failed to store in vector database: {vec_err}", {}

            # ---------------------------------------------------------------- #
            # 8. Update KB stats                                               #
            # ---------------------------------------------------------------- #
            kb = db_session.query(KnowledgeBase).filter(
                KnowledgeBase.knowledge_base_id == knowledge_base_id
            ).first()

            if kb:
                if append:
                    # Accumulate: add new rows on top of existing count
                    kb.total_chunks = (kb.total_chunks or 0) + len(qdrant_points)
                    kb.document_count = (kb.document_count or 0) + 1
                else:
                    # Full replace: reflect only this sync's rows
                    kb.total_chunks = len(qdrant_points)
                    kb.document_count = 1
                kb.last_sync_at = datetime.utcnow()
                kb.updated_at = datetime.utcnow()
                db_session.commit()

            logger.info(f"✓ Updated KB stats (append={append})")

            metadata = {
                'rows_retrieved': row_count,
                'rows_embedded': len(qdrant_points),
                'columns': list(result_df.columns),
                'collection_name': collection_name,
                # Naming differs slightly so callers can read either key
                'chunks_created': len(qdrant_points),   # for sync callers
                'chunks_added': len(qdrant_points),      # for append callers
                'total_chunks_in_kb': kb.total_chunks if kb else len(qdrant_points),
            }

            action = "appended to" if append else "synced into"
            return (
                True,
                f"Successfully {action} '{collection_name}': "
                f"{len(qdrant_points)} rows from {db_type} (label='{query_label}')",
                metadata,
            )

        except Exception as e:
            logger.error(f"Database sync failed: {e}", exc_info=True)
            return False, f"Sync failed: {str(e)}", {}

    # ---------------------------------------------------------------------- #
    # Helpers                                                                 #
    # ---------------------------------------------------------------------- #

    def _clean_json(self, row_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitise a row dict for JSON serialisation."""
        import numpy as np

        cleaned = {}
        for key, value in row_dict.items():
            if value is None:
                cleaned[key] = None
                continue
            try:
                if pd.isna(value):
                    cleaned[key] = None
                    continue
            except (ValueError, TypeError):
                pass

            if isinstance(value, (pd.Timestamp, datetime)):
                cleaned[key] = value.isoformat()
            elif isinstance(value, np.integer):
                cleaned[key] = int(value)
            elif isinstance(value, np.floating):
                cleaned[key] = float(value)
            elif isinstance(value, np.ndarray):
                cleaned[key] = value.tolist()
            else:
                try:
                    json.dumps(value)
                    cleaned[key] = value
                except (TypeError, ValueError):
                    cleaned[key] = str(value)
        return cleaned

    def _row_to_text(self, row_json: Dict[str, Any]) -> str:
        """Convert a cleaned row dict to plain text for embedding."""
        return ", ".join(
            f"{k}: {v if v is not None else 'null'}"
            for k, v in row_json.items()
        )
