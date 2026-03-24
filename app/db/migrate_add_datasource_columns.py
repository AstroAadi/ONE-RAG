"""
One-time migration: add data-source columns to knowledge_bases table.

Run once:
    cd /home/aditya-gupta/Desktop/one_Rag/rag-platform
    python -m app.db.migrate_add_datasource_columns
"""

import logging
from sqlalchemy import text
from app.db.session import engine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MIGRATION_SQL = """
ALTER TABLE knowledge_bases
    ADD COLUMN IF NOT EXISTS data_source_type  VARCHAR(50)  NOT NULL DEFAULT 'file_upload',
    ADD COLUMN IF NOT EXISTS data_source_config JSON,
    ADD COLUMN IF NOT EXISTS last_sync_at       TIMESTAMP,
    ADD COLUMN IF NOT EXISTS auto_sync_enabled  BOOLEAN      NOT NULL DEFAULT FALSE,
    ADD COLUMN IF NOT EXISTS sync_interval_hours INTEGER;

CREATE INDEX IF NOT EXISTS idx_data_source_type ON knowledge_bases (data_source_type);
"""

def run():
    with engine.begin() as conn:
        conn.execute(text(MIGRATION_SQL))
    logger.info("✅ Migration complete: data-source columns added to knowledge_bases")

if __name__ == "__main__":
    run()
