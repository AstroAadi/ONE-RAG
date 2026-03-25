"""
File Tracker
Tracks indexed files using Qdrant vector database metadata
"""

from typing import Optional, Tuple, Dict, Any
import logging
import hashlib
from datetime import datetime
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

from app.core.config import settings

logger = logging.getLogger(__name__)


class FileTracker:
    """
    Track indexed files using Qdrant metadata
    Prevents reprocessing of already indexed files
    """
    
    def __init__(self):
        """Initialize file tracker with Qdrant client"""
        try:
            self.client = QdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key,
                timeout=settings.qdrant_timeout,
                prefer_grpc=False
            )
            logger.info("✅ FileTracker initialized with Qdrant")
        except Exception as e:
            logger.error(f"Failed to initialize FileTracker: {e}")
            raise
    
    def is_file_indexed(
        self,
        filename: str,
        knowledge_base_id: str
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Check if a file has already been indexed
        
        Args:
            filename: Name of the file (not full path)
            knowledge_base_id: Knowledge base identifier
            
        Returns:
            Tuple of (is_indexed, metadata_dict)
            metadata_dict contains: 'timestamp', 'chunk_count', etc. if indexed
        """
        try:
            # Create normalized filename identifier
            file_identifier = self._create_file_identifier(filename)
            
            # Search for existing file metadata in Qdrant
            search_results = self.client.scroll(
                collection_name=settings.qdrant_collection_name,
                limit=1000,
                with_payload=True,
                query_filter=Filter(
                    must=[
                        FieldCondition(
                            key="file_identifier",
                            match=MatchValue(value=file_identifier)
                        ),
                        FieldCondition(
                            key="knowledge_base_id",
                            match=MatchValue(value=knowledge_base_id)
                        )
                    ]
                )
            )
            
            if search_results[0]:  # If any results found
                # File is already indexed
                point = search_results[0][0]  # First point
                metadata = point.payload
                logger.info(f"File already indexed: {filename} (KB: {knowledge_base_id})")
                return True, metadata
            else:
                # File not indexed yet
                return False, None
                
        except Exception as e:
            logger.warning(f"Error checking if file is indexed: {e}")
            # On error, assume not indexed to allow reprocessing
            return False, None
    
    def mark_file_indexed(
        self,
        filename: str,
        knowledge_base_id: str,
        chunk_count: int,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Mark a file as indexed in Qdrant
        
        Args:
            filename: Name of the file
            knowledge_base_id: Knowledge base identifier
            chunk_count: Number of chunks created from this file
            metadata: Optional additional metadata to store
            
        Returns:
            True if successfully marked, False otherwise
        """
        try:
            file_identifier = self._create_file_identifier(filename)
            
            # Create tracking metadata
            track_metadata = {
                "file_identifier": file_identifier,
                "filename": filename,
                "knowledge_base_id": knowledge_base_id,
                "chunk_count": chunk_count,
                "indexed_timestamp": datetime.utcnow().isoformat(),
                "is_file_tracking": True  # Flag to identify tracking records
            }
            
            # Merge with additional metadata
            if metadata:
                track_metadata.update(metadata)
            
            logger.debug(f"File tracked: {filename} -> {chunk_count} chunks")
            return True
            
        except Exception as e:
            logger.error(f"Error marking file as indexed: {e}")
            return False
    
    def get_indexed_files(self, knowledge_base_id: str) -> Dict[str, Dict[str, Any]]:
        """
        Get all indexed files for a knowledge base
        
        Args:
            knowledge_base_id: Knowledge base identifier
            
        Returns:
            Dictionary mapping filename -> metadata
        """
        indexed_files = {}
        
        try:
            search_results = self.client.scroll(
                collection_name=settings.qdrant_collection_name,
                limit=10000,
                with_payload=True,
                query_filter=Filter(
                    must=[
                        FieldCondition(
                            key="knowledge_base_id",
                            match=MatchValue(value=knowledge_base_id)
                        ),
                        FieldCondition(
                            key="is_file_tracking",
                            match=MatchValue(value=True)
                        )
                    ]
                )
            )
            
            for point in search_results[0]:
                metadata = point.payload
                filename = metadata.get('filename', 'unknown')
                indexed_files[filename] = metadata
            
            logger.info(f"Found {len(indexed_files)} indexed files in KB: {knowledge_base_id}")
            return indexed_files
            
        except Exception as e:
            logger.warning(f"Error retrieving indexed files: {e}")
            return {}
    
    @staticmethod
    def _create_file_identifier(filename: str) -> str:
        """
        Create a normalized identifier for a file
        Uses filename only (not full path) to track regardless of location
        
        Args:
            filename: Name of the file
            
        Returns:
            Normalized file identifier
        """
        # Use just the filename (basename) as identifier
        # This allows tracking the same filename across different runs
        basename = Path(filename).name
        return basename.lower()
