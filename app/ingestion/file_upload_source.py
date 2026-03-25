"""
File Upload Data Source
Handles single files and directories
"""

from pathlib import Path
from typing import List, Tuple, Dict, Any
import logging
import os

from .base_data_source import BaseDataSource, DataSource

logger = logging.getLogger(__name__)


class FileUploadSource(BaseDataSource):
    """
    Data source for file uploads
    Supports single files and directories
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize file upload source
        
        Args:
            config: Must contain 'file_path' key
        """
        super().__init__(config)
        self.file_path = config.get('file_path')
        self.path_obj = Path(self.file_path) if self.file_path else None
    
    def validate_config(self) -> Tuple[bool, str]:
        """Validate file path configuration"""
        if not self.file_path:
            return False, "file_path is required for file_upload source"
        
        if not self.path_obj.exists():
            return False, f"Path does not exist: {self.file_path}"
        
        if self.path_obj.is_file():
            if not self.path_obj.is_file():
                return False, f"Path is not a file: {self.file_path}"
        elif self.path_obj.is_dir():
            if not os.listdir(self.path_obj):
                return False, f"Directory is empty: {self.file_path}"
        else:
            return False, f"Path is neither file nor directory: {self.file_path}"
        
        return True, ""
    
    def discover_sources(self) -> Tuple[List[DataSource], List[str]]:
        """
        Discover files in the given path
        
        If path is a file: return single file
        If path is directory: recursively find all supported files
        
        Returns:
            Tuple of (DataSource list, error list)
        """
        sources = []
        errors = []
        
        # Import extractor factory to check supported formats
        try:
            from app.services.extraction.extractor_factory import ExtractorFactory
        except ImportError:
            errors.append("Failed to import ExtractorFactory")
            return sources, errors
        
        if self.path_obj.is_file():
            # Single file
            ext = self.path_obj.suffix.lstrip('.').lower()
            if ExtractorFactory.get_extractor(ext):
                sources.append(DataSource(
                    identifier=str(self.path_obj),
                    source_type='file',
                    metadata={
                        'filename': self.path_obj.name,
                        'path': str(self.path_obj),
                        'size_bytes': self.path_obj.stat().st_size,
                        'extension': ext
                    }
                ))
            else:
                errors.append(f"Unsupported file format: {self.path_obj.name}")
        
        elif self.path_obj.is_dir():
            # Directory - recursively find all supported files
            supported_files = self._find_supported_files(self.path_obj)
            
            for file_path in supported_files:
                ext = file_path.suffix.lstrip('.').lower()
                sources.append(DataSource(
                    identifier=str(file_path),
                    source_type='file',
                    metadata={
                        'filename': file_path.name,
                        'path': str(file_path),
                        'size_bytes': file_path.stat().st_size,
                        'extension': ext,
                        'relative_path': str(file_path.relative_to(self.path_obj))
                    }
                ))
            
            if not supported_files:
                errors.append(f"No supported files found in directory: {self.file_path}")
        
        logger.info(f"Discovered {len(sources)} files, {len(errors)} errors")
        return sources, errors
    
    def _find_supported_files(self, directory: Path) -> List[Path]:
        """
        Recursively find all supported files in directory
        
        Args:
            directory: Path to search
            
        Returns:
            List of Path objects for supported files
        """
        from app.services.extraction.extractor_factory import ExtractorFactory
        
        supported_files = []
        
        try:
            for item in directory.rglob('*'):
                if item.is_file():
                    ext = item.suffix.lstrip('.').lower()
                    if ExtractorFactory.get_extractor(ext):
                        supported_files.append(item)
                    else:
                        logger.debug(f"Skipped unsupported file: {item.name}")
        except Exception as e:
            logger.error(f"Error scanning directory: {e}")
        
        return sorted(supported_files)
    
    def fetch_data(self, source: DataSource) -> Tuple[str, str, Dict[str, Any]]:
        """
        Read file content
        
        Args:
            source: DataSource representing the file
            
        Returns:
            Tuple of (file_content, content_type, metadata)
            
        Raises:
            IOError if file cannot be read
        """
        file_path = Path(source.identifier)
        
        try:
            # Read file as bytes
            with open(file_path, 'rb') as f:
                content = f.read()
            
            # Determine content type
            ext = file_path.suffix.lstrip('.').lower()
            
            # For now, return raw bytes and let extractors handle it
            metadata = {
                'filename': file_path.name,
                'size_bytes': len(content),
                'path': str(file_path),
                'extension': ext
            }
            
            return content, 'file', metadata
            
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            raise IOError(f"Failed to read file: {e}")
    
    def cleanup(self):
        """No cleanup needed for file source"""
        pass
