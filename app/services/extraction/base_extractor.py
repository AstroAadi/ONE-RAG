"""
Base Extractor Abstract Class
All document extractors inherit from this
"""

from abc import ABC, abstractmethod
from typing import Tuple, Dict, Optional, List
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class BaseExtractor(ABC):
    """
    Abstract base class for all document extractors
    
    All extractors must implement:
    - _get_supported_formats(): Return list of supported extensions
    - extract(): Main extraction logic
    - validate_file(): File validation logic
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize extractor with optional configuration
        
        Args:
            config: Dictionary of configuration options
                   Example: {'do_ocr': True, 'extract_images': True}
        """
        self.config = config or {}
        self.supported_formats = self._get_supported_formats()
        logger.debug(f"{self.__class__.__name__} initialized with formats: {self.supported_formats}")
    
    @abstractmethod
    def _get_supported_formats(self) -> List[str]:
        """
        Return list of supported file extensions (without dots)
        
        Example: ['pdf', 'docx', 'doc']
        """
        pass
    
    @abstractmethod
    def extract(self, file_path: str) -> Tuple[str, bool, Dict]:
        """
        Extract content from document
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Tuple containing:
            - markdown_content (str): Extracted content in markdown format
            - success (bool): Whether extraction was successful
            - metadata (dict): Extraction metadata including:
                - filename: Name of the file
                - extension: File extension
                - size_bytes: File size
                - total_pages: Number of pages (if applicable)
                - images_extracted: Number of images extracted
                - error: Error message if success=False
        """
        pass
    
    @abstractmethod
    def validate_file(self, file_path: str) -> Tuple[bool, str]:
        """
        Validate if file can be processed by this extractor
        
        Args:
            file_path: Path to the file
            
        Returns:
            Tuple of (is_valid, message)
        """
        pass
    
    def supports_format(self, extension: str) -> bool:
        """
        Check if this extractor supports the given format
        
        Args:
            extension: File extension (with or without dot)
            
        Returns:
            True if supported, False otherwise
        """
        clean_ext = extension.lower().lstrip('.')
        return clean_ext in self.supported_formats
    
    def get_metadata(self, file_path: str) -> Dict:
        """
        Extract basic file metadata
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary with basic metadata
        """
        path = Path(file_path)
        
        metadata = {
            'filename': path.name,
            'extension': path.suffix.lstrip('.'),
            'size_bytes': 0
        }
        
        if path.exists():
            metadata['size_bytes'] = path.stat().st_size
        
        return metadata
    
    def get_config(self, key: str, default=None):
        """
        Get configuration value with fallback to default
        
        Args:
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        return self.config.get(key, default)