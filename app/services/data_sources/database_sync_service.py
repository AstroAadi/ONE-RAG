"""
Database Sync Service
Retrieves data from the user's database, converts rows to JSON,
generates embeddings, and stores them row-wise in the vector DB.
ChunkMetadata is NOT written here because chunk_metadata has a strict
FK → file_metadata.file_id and database rows have no file_metadata entry.
All row-level tracking is done via KB stats + vector store payload.
"""

import logging
from typing import Dict, Any, Tuple, List
from datetime import datetime
import pandas as pd
import json
import uuid

from app.services.data_sources.database_connector import DatabaseConnectorFactory
from app.services.embedding_service import BedrockEmbeddingService
from app.services.vector_store_service import VectorStoreService

logger = logging.getLogger(__name__)


class DatabaseSyncService:
    """Service to sync database data directly to the vector database"""

    def __init__(self):
        self.embedding_service = BedrockEmbeddingService()
        self.vector_store = VectorStoreService()

    async def sync_database_kb(
        self,
        knowledge_base_id: str,
        user_id: str,
        database_config: Dict[str, Any],
        db_session,
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Sync data from the user's database directly to the vector database.

        Steps:
        1. Connect to the user's database using their credentials
        2. Execute the user-supplied query exactly as given
        3. Convert each result row to a text representation
        4. Generate a dense embedding for each row using embed_query()
        5. Upsert all embeddings into Qdrant
        6. Update KB stats (total_chunks, document_count, last_sync_at)

        Returns:
            (success, message, metadata_dict)
        """
        try:
            from app.db.models import KnowledgeBase

            logger.info(f"Starting database sync for KB: {knowledge_base_id}")

            # -------------------------------------------------------------- #
            # 1. Normalise db_type (may be a DatabaseType enum or a string)  #
            # -------------------------------------------------------------- #
            db_type_raw = database_config.get('db_type', '')
            db_type = db_type_raw.value if hasattr(db_type_raw, 'value') else str(db_type_raw)
            db_type = db_type.lower()

            # -------------------------------------------------------------- #
            # 2. Connect to the user's database                              #
            # -------------------------------------------------------------- #
            connector = DatabaseConnectorFactory.get_connector(db_type, database_config)
            connected = connector.connect()
            if not connected:
                return False, f"Failed to connect to {db_type} database", {}

            logger.info(f"✓ Connected to {db_type} database")

            # -------------------------------------------------------------- #
            # 3. Execute the user-supplied query                             #
            # -------------------------------------------------------------- #
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

            # -------------------------------------------------------------- #
            # 4. Build embeddings list                                       #
            # FIX: BedrockEmbeddingService exposes embed_query() and        #
            # embed_documents(), NOT generate_embedding(). Use embed_query() #
            # for single-text embedding.                                     #
            # -------------------------------------------------------------- #
            embedded_rows: List[Dict[str, Any]] = []

            for idx, row in result_df.iterrows():
                try:
                    row_dict = row.to_dict()
                    row_json = self._clean_json(row_dict)
                    text_content = self._row_to_text(row_json)

                    # Deterministic UUID — re-syncing upserts instead of
                    # creating duplicate points in Qdrant
                    chunk_id = str(uuid.uuid5(
                        uuid.NAMESPACE_URL,
                        f"{knowledge_base_id}/{db_type}/row/{idx}",
                    ))

                    # FIX: correct method name is embed_query(), not generate_embedding()
                    embedding = self.embedding_service.embed_query(text_content)

                    embedded_rows.append({
                        'id': chunk_id,
                        'embedding': embedding,
                        'text': text_content,
                        'metadata': {
                            'knowledge_base_id': knowledge_base_id,
                            'user_id': user_id,
                            'source_type': 'database',
                            'database_type': db_type,
                            'row_index': int(idx),
                            'row_data': row_json,
                            'synced_at': datetime.utcnow().isoformat(),
                        },
                    })
                except Exception as row_err:
                    logger.warning(f"Failed to process row {idx}: {row_err}")
                    continue

            if not embedded_rows:
                return False, "No rows were successfully processed", {}

            logger.info(f"✓ Generated embeddings for {len(embedded_rows)} rows")

            # -------------------------------------------------------------- #
            # 5. Upsert into vector database (Qdrant)                       #
            # -------------------------------------------------------------- #
            collection_name = f"{user_id}_{knowledge_base_id}"

            try:
                self.vector_store.create_collection_if_not_exists(collection_name)

                batch_size = 100
                for i in range(0, len(embedded_rows), batch_size):
                    batch = embedded_rows[i: i + batch_size]
                    points = [
                        {
                            'id': item['id'],
                            'vector': item['embedding'],
                            'payload': {
                                'text': item['text'],
                                'metadata': item['metadata'],
                            },
                        }
                        for item in batch
                    ]
                    self.vector_store.upsert_points(collection_name, points)

                logger.info(f"✓ Stored {len(embedded_rows)} embeddings in vector DB")

            except Exception as vec_err:
                logger.error(f"Vector DB storage failed: {vec_err}")
                return False, f"Failed to store in vector database: {vec_err}", {}

            # -------------------------------------------------------------- #
            # 6. Update KB stats                                             #
            # -------------------------------------------------------------- #
            kb = db_session.query(KnowledgeBase).filter(
                KnowledgeBase.knowledge_base_id == knowledge_base_id
            ).first()

            if kb:
                kb.total_chunks = len(embedded_rows)   # full replace on re-sync
                kb.document_count = 1                   # whole query result = 1 logical doc
                kb.last_sync_at = datetime.utcnow()
                kb.updated_at = datetime.utcnow()
                db_session.commit()

            logger.info("✓ Updated KB stats")

            metadata = {
                'rows_retrieved': row_count,
                'rows_embedded': len(embedded_rows),
                'columns': list(result_df.columns),
                'collection_name': collection_name,
                'chunks_created': len(embedded_rows),
            }

            return (
                True,
                f"Successfully synced {len(embedded_rows)} rows from "
                f"{db_type} database to vector store",
                metadata,
            )

        except Exception as e:
            logger.error(f"Database sync failed: {e}", exc_info=True)
            return False, f"Sync failed: {str(e)}", {}

    # ---------------------------------------------------------------------- #
    # Helpers                                                                 #
    # ---------------------------------------------------------------------- #

    def _clean_json(self, row_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitise a row dict for JSON serialisation.
        Handles None, NaN/NaT, datetime, numpy scalars/arrays, and any
        other non-serialisable type (coerced to str).
        """
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
        """
        Convert a cleaned row dict to a plain-text string for embedding.
        Format: "col1: val1, col2: val2, ..."
        """
        parts = []
        for key, value in row_json.items():
            parts.append(f"{key}: {value if value is not None else 'null'}")
        return ", ".join(parts)
