"""
Base Data Source Abstract Class
All data source handlers inherit from this
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class DataSource:
    """Represents a data source item"""
    identifier: str  # filename, table name, api endpoint, etc.
    source_type: str  # 'file', 'database_row', 'api_response', etc.
    metadata: Dict[str, Any]  # source-specific metadata


@dataclass
class IngestionResult:
    """Result of processing a single data source"""
    success: bool
    identifier: str  # The source identifier
    chunks: List[Dict[str, Any]] = None  # Chunks created from this source
    error: str = None  # Error message if failed
    metadata: Dict[str, Any] = None  # Processing metadata


class BaseDataSource(ABC):
    """
    Abstract base class for all data source handlers
    
    Responsible for:
    1. Listing/discovering data sources
    2. Fetching raw data from source
    3. Returning data in a standardized format
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize data source handler
        
        Args:
            config: Configuration dictionary for this source
        """
        self.config = config
        logger.info(f"Initialized {self.__class__.__name__}")
    
    @abstractmethod
    def discover_sources(self) -> Tuple[List[DataSource], List[str]]:
        """
        Discover available data sources
        
        Returns:
            Tuple of (list of DataSource objects, list of error messages)
            
        Example for FileUploadSource:
            If path="/data/docs", return list of file DataSource objects
            
        Example for DatabaseSource:
            If table="users", return list of row DataSource objects
        """
        pass
    
    @abstractmethod
    def fetch_data(self, source: DataSource) -> Tuple[str, str, Dict[str, Any]]:
        """
        Fetch raw data from a specific source
        
        Args:
            source: DataSource object to fetch from
            
        Returns:
            Tuple of (content, content_type, metadata)
            - content (str): Raw content from source
            - content_type (str): Type hint ('text', 'markdown', 'html', 'json', etc.)
            - metadata (dict): Additional metadata from source
            
        Example for FileUploadSource:
            Return file content
            
        Example for DatabaseSource:
            Return row data as JSON
        """
        pass
    
    def validate_config(self) -> Tuple[bool, str]:
        """
        Validate configuration for this source
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        return True, ""
    
    def cleanup(self):
        """
        Cleanup resources used by this source
        Called after ingestion completes or fails
        """
        pass
