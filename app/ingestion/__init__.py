"""
Ingestion Module
Core components for data ingestion pipeline

Usage:
    from app.ingestion.ingestion_service import IngestionService
    from app.ingestion.file_tracker import FileTracker
    from app.ingestion.file_upload_source import FileUploadSource
"""

__all__ = [
    "BaseDataSource",
    "FileUploadSource",
    "CloudStorageSource",
    "DatabaseSource",
    "APISource",
    "FileTracker",
    "IngestionService",
]
