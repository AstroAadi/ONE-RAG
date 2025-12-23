"""
File Validator with Dynamic Format Support
Validates uploaded files and checks against supported formats
"""

from fastapi import UploadFile
from typing import Tuple, List, Set
import logging

logger = logging.getLogger(__name__)


class FileValidator:
    """
    Validate uploaded files with dynamic format support
    
    Automatically detects supported formats from ExtractorFactory
    """
    
    # These will be populated dynamically
    _allowed_extensions: Set[str] = None
    _allowed_mime_types: Set[str] = None
    
    # Static MIME type mappings
    MIME_TYPE_MAP = {
        'pdf': 'application/pdf',
        'docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        'doc': 'application/msword',
        'pptx': 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
        'ppt': 'application/vnd.ms-powerpoint',
        'xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        'xls': 'application/vnd.ms-excel',
        'txt': 'text/plain',
        'md': 'text/markdown',
        'csv': 'text/csv',
        'png': 'image/png',
        'jpg': 'image/jpeg',
        'jpeg': 'image/jpeg',
        'gif': 'image/gif',
        'bmp': 'image/bmp',
        'webp': 'image/webp',
        'tiff': 'image/tiff',
        'html': 'text/html',
        'xml': 'application/xml',
    }
    
    @classmethod
    def _initialize_formats(cls):
        """Initialize allowed formats from ExtractorFactory"""
        if cls._allowed_extensions is None:
            try:
                from app.services.extraction.extractor_factory import ExtractorFactory
                
                # Get supported formats from factory
                supported_formats = ExtractorFactory.get_supported_formats()
                cls._allowed_extensions = set(f'.{ext}' for ext in supported_formats)
                
                # Build MIME types from supported formats
                cls._allowed_mime_types = set()
                for ext in supported_formats:
                    mime_type = cls.MIME_TYPE_MAP.get(ext)
                    if mime_type:
                        cls._allowed_mime_types.add(mime_type)
                
                logger.info(f"FileValidator initialized with {len(cls._allowed_extensions)} supported formats")
                logger.debug(f"Supported extensions: {sorted(cls._allowed_extensions)}")
                
            except ImportError as e:
                logger.warning(f"Could not import ExtractorFactory: {e}")
                # Fallback to basic formats
                cls._allowed_extensions = {'.pdf', '.docx', '.doc'}
                cls._allowed_mime_types = {
                    'application/pdf',
                    'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                    'application/msword'
                }
    
    @classmethod
    def get_allowed_extensions(cls) -> Set[str]:
        """
        Get set of allowed file extensions
        
        Returns:
            Set of extensions with dots (e.g., {'.pdf', '.docx'})
        """
        if cls._allowed_extensions is None:
            cls._initialize_formats()
        return cls._allowed_extensions
    
    @classmethod
    def get_allowed_mime_types(cls) -> Set[str]:
        """
        Get set of allowed MIME types
        
        Returns:
            Set of MIME type strings
        """
        if cls._allowed_mime_types is None:
            cls._initialize_formats()
        return cls._allowed_mime_types
    
    @classmethod
    def validate_file(cls, file: UploadFile) -> Tuple[bool, str]:
        """
        Validate uploaded file
        
        Checks:
        1. File exists and has a name
        2. File extension is supported
        3. MIME type is valid (if available)
        
        Args:
            file: FastAPI UploadFile object
            
        Returns:
            Tuple of (is_valid, error_message)
            - is_valid: True if file is valid, False otherwise
            - error_message: Empty string if valid, error description if invalid
        """
        # Initialize formats if needed
        if cls._allowed_extensions is None:
            cls._initialize_formats()
        
        # Check if file exists
        if not file or not file.filename:
            return False, "No file provided"
        
        # Get file extension
        file_extension = cls.get_file_extension(file.filename, with_dot=True)
        
        # Validate extension
        if file_extension not in cls._allowed_extensions:
            supported = ', '.join(sorted(cls._allowed_extensions))
            return False, f"File type '{file_extension}' not supported. Allowed types: {supported}"
        
        # Validate MIME type if available
        if file.content_type:
            # Some browsers send generic MIME types, so we're lenient here
            if file.content_type not in cls._allowed_mime_types:
                # Check if it's a generic type that might be acceptable
                generic_types = ['application/octet-stream', 'binary/octet-stream']
                if file.content_type not in generic_types:
                    logger.warning(f"Unexpected MIME type for {file.filename}: {file.content_type}")
                    # Don't fail on MIME type mismatch, just log it
        
        return True, ""
    
    @classmethod
    def validate_file_size(cls, file_size: int, max_size_mb: int = 50) -> Tuple[bool, str]:
        """
        Validate file size
        
        Args:
            file_size: Size of file in bytes
            max_size_mb: Maximum allowed size in megabytes (default: 50)
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if file_size <= 0:
            return False, "File is empty"
        
        max_size_bytes = max_size_mb * 1024 * 1024
        
        if file_size > max_size_bytes:
            size_mb = file_size / (1024 * 1024)
            return False, f"File size ({size_mb:.2f}MB) exceeds maximum allowed size of {max_size_mb}MB"
        
        return True, ""
    
    @staticmethod
    def get_file_extension(filename: str, with_dot: bool = False) -> str:
        """
        Get file extension from filename
        
        Args:
            filename: Name of the file
            with_dot: If True, return extension with dot (e.g., '.pdf'), otherwise without (e.g., 'pdf')
            
        Returns:
            File extension string
        """
        if not filename or '.' not in filename:
            return ''
        
        extension = filename.split('.')[-1].lower()
        
        if with_dot:
            return f'.{extension}'
        return extension
    
    @classmethod
    def get_mime_type_for_extension(cls, extension: str) -> str:
        """
        Get MIME type for a given file extension
        
        Args:
            extension: File extension (with or without dot)
            
        Returns:
            MIME type string or 'application/octet-stream' if unknown
        """
        ext = extension.lower().lstrip('.')
        return cls.MIME_TYPE_MAP.get(ext, 'application/octet-stream')
    
    @classmethod
    def is_document(cls, filename: str) -> bool:
        """Check if file is a document (PDF, DOCX, etc.)"""
        ext = cls.get_file_extension(filename, with_dot=True)
        document_types = {'.pdf', '.docx', '.doc', '.txt', '.md'}
        return ext in document_types
    
    @classmethod
    def is_presentation(cls, filename: str) -> bool:
        """Check if file is a presentation (PPTX, PPT)"""
        ext = cls.get_file_extension(filename, with_dot=True)
        presentation_types = {'.pptx', '.ppt'}
        return ext in presentation_types
    
    @classmethod
    def is_spreadsheet(cls, filename: str) -> bool:
        """Check if file is a spreadsheet (XLSX, XLS, CSV)"""
        ext = cls.get_file_extension(filename, with_dot=True)
        spreadsheet_types = {'.xlsx', '.xls', '.csv'}
        return ext in spreadsheet_types
    
    @classmethod
    def is_image(cls, filename: str) -> bool:
        """Check if file is an image"""
        ext = cls.get_file_extension(filename, with_dot=True)
        image_types = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp', '.tiff'}
        return ext in image_types
    
    @classmethod
    def get_file_category(cls, filename: str) -> str:
        """
        Get category of file
        
        Returns:
            One of: 'document', 'presentation', 'spreadsheet', 'image', 'unknown'
        """
        if cls.is_document(filename):
            return 'document'
        elif cls.is_presentation(filename):
            return 'presentation'
        elif cls.is_spreadsheet(filename):
            return 'spreadsheet'
        elif cls.is_image(filename):
            return 'image'
        else:
            return 'unknown'
    
    @classmethod
    def validate_filename(cls, filename: str) -> Tuple[bool, str]:
        """
        Validate filename for security and sanity
        
        Checks for:
        - Path traversal attempts
        - Invalid characters
        - Reasonable length
        
        Args:
            filename: Filename to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not filename:
            return False, "Filename is empty"
        
        # Check length
        if len(filename) > 255:
            return False, "Filename too long (max 255 characters)"
        
        # Check for path traversal
        if '..' in filename or '/' in filename or '\\' in filename:
            return False, "Filename contains invalid path characters"
        
        # Check for invalid characters
        invalid_chars = ['<', '>', ':', '"', '|', '?', '*', '\0']
        for char in invalid_chars:
            if char in filename:
                return False, f"Filename contains invalid character: '{char}'"
        
        # Check if filename has an extension
        if '.' not in filename:
            return False, "Filename must have an extension"
        
        return True, ""
    
    @classmethod
    def get_safe_filename(cls, filename: str) -> str:
        """
        Get a safe version of the filename
        
        Removes or replaces problematic characters
        
        Args:
            filename: Original filename
            
        Returns:
            Sanitized filename
        """
        import re
        
        # Replace spaces with underscores
        filename = filename.replace(' ', '_')
        
        # Remove invalid characters
        filename = re.sub(r'[<>:"|?*\\/]', '', filename)
        
        # Remove any non-ASCII characters
        filename = filename.encode('ascii', 'ignore').decode('ascii')
        
        # Ensure it's not too long
        if len(filename) > 200:
            name, ext = filename.rsplit('.', 1)
            filename = name[:195] + '.' + ext
        
        return filename