from fastapi import UploadFile, HTTPException
from app.core.config import settings


class FileValidator:
    """Validate uploaded files"""
    
    ALLOWED_EXTENSIONS = {'.pdf', '.docx', '.doc'}
    ALLOWED_MIME_TYPES = {
        'application/pdf',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        'application/msword'
    }
    
    @staticmethod
    def validate_file(file: UploadFile) -> tuple[bool, str]:
        """
        Validate uploaded file
        
        Args:
            file: FastAPI UploadFile object
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check if file exists
        if not file or not file.filename:
            return False, "No file provided"
        
        # Get file extension
        file_extension = '.' + file.filename.split('.')[-1].lower()
        
        # Validate extension
        if file_extension not in FileValidator.ALLOWED_EXTENSIONS:
            return False, f"File type not supported. Allowed types: {', '.join(FileValidator.ALLOWED_EXTENSIONS)}"
        
        # Validate MIME type if available
        if file.content_type and file.content_type not in FileValidator.ALLOWED_MIME_TYPES:
            return False, f"Invalid file MIME type: {file.content_type}"
        
        return True, ""
    
    @staticmethod
    def validate_file_size(file_size: int) -> tuple[bool, str]:
        """
        Validate file size
        
        Args:
            file_size: Size of file in bytes
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        max_size_bytes = settings.max_file_size_mb * 1024 * 1024
        
        if file_size > max_size_bytes:
            return False, f"File size exceeds maximum allowed size of {settings.max_file_size_mb}MB"
        
        return True, ""
    
    @staticmethod
    def get_file_extension(filename: str) -> str:
        """Get file extension from filename"""
        return filename.split('.')[-1].lower()