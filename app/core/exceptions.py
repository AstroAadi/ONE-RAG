"""
Custom Exception Classes for RAG System
Backward compatible - existing code using HTTPException still works
"""

from typing import Optional
from fastapi import HTTPException


class RAGSystemError(Exception):
    """
    Base exception for all RAG system errors
    
    Can be used standalone or converted to HTTPException for FastAPI
    """
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[dict] = None,
        status_code: int = 500
    ):
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}
        self.status_code = status_code
        super().__init__(self.message)
    
    def to_dict(self) -> dict:
        """Convert exception to dictionary for API responses"""
        return {
            "error": self.error_code,
            "message": self.message,
            "details": self.details
        }
    
    def to_http_exception(self) -> HTTPException:
        """Convert to FastAPI HTTPException"""
        return HTTPException(
            status_code=self.status_code,
            detail=self.message
        )


# ==================== File Processing Errors ====================

class FileProcessingError(RAGSystemError):
    """Base class for file processing errors"""
    def __init__(self, message: str, error_code: Optional[str] = None, details: Optional[dict] = None):
        super().__init__(message, error_code, details, status_code=422)


class FileValidationError(FileProcessingError):
    """File validation failed (invalid format, size, etc.)"""
    def __init__(self, message: str, error_code: Optional[str] = None, details: Optional[dict] = None):
        super().__init__(message, error_code, details)
        self.status_code = 400


class FileNotFoundError(FileProcessingError):
    """File not found in storage"""
    def __init__(self, message: str, error_code: Optional[str] = None, details: Optional[dict] = None):
        super().__init__(message, error_code, details)
        self.status_code = 404


class FileUploadError(FileProcessingError):
    """Error during file upload"""
    pass


class FileSizeExceededError(FileValidationError):
    """File size exceeds maximum allowed"""
    pass


class UnsupportedFileTypeError(FileValidationError):
    """File type not supported"""
    def __init__(self, message: str, error_code: Optional[str] = None, details: Optional[dict] = None):
        super().__init__(message, error_code, details)
        self.status_code = 415


# ==================== Document Extraction Errors ====================

class DocumentExtractionError(RAGSystemError):
    """Base class for document extraction errors"""
    def __init__(self, message: str, error_code: Optional[str] = None, details: Optional[dict] = None):
        super().__init__(message, error_code, details, status_code=422)


class ExtractionFailedError(DocumentExtractionError):
    """Content extraction from document failed"""
    pass


class EmptyDocumentError(DocumentExtractionError):
    """No content extracted from document"""
    pass


class CorruptedDocumentError(DocumentExtractionError):
    """Document is corrupted or unreadable"""
    pass


# ==================== Chunking Errors ====================

class ChunkingError(RAGSystemError):
    """Base class for chunking errors"""
    def __init__(self, message: str, error_code: Optional[str] = None, details: Optional[dict] = None):
        super().__init__(message, error_code, details, status_code=422)


class ChunkSizeExceededError(ChunkingError):
    """Chunk exceeds maximum allowed size"""
    pass


class NoChunksCreatedError(ChunkingError):
    """No chunks created from document"""
    pass


# ==================== Vector Store Errors ====================

class VectorStoreException(RAGSystemError):
    """Base class for vector store errors"""
    pass


class EmbeddingError(VectorStoreException):
    """Error generating embeddings"""
    pass


class SearchError(VectorStoreException):
    """Error during vector search"""
    pass


class IndexingError(VectorStoreException):
    """Error indexing documents to vector store"""
    pass


class NoDocumentsFoundError(VectorStoreException):
    """No documents found in vector store"""
    def __init__(self, message: str = "No documents found. Please upload and process documents first.", 
                 error_code: Optional[str] = None, details: Optional[dict] = None):
        super().__init__(message, error_code, details, status_code=404)


# ==================== LLM Errors ====================

class LLMError(RAGSystemError):
    """Base class for LLM errors"""
    pass


class LLMResponseError(LLMError):
    """Error getting response from LLM"""
    pass


class LLMTimeoutError(LLMError):
    """LLM request timed out"""
    def __init__(self, message: str, error_code: Optional[str] = None, details: Optional[dict] = None):
        super().__init__(message, error_code, details, status_code=504)


class LLMQuotaExceededError(LLMError):
    """LLM quota or rate limit exceeded"""
    def __init__(self, message: str, error_code: Optional[str] = None, details: Optional[dict] = None):
        super().__init__(message, error_code, details, status_code=429)


# ==================== Query Errors ====================

class QueryError(RAGSystemError):
    """Base class for query errors"""
    def __init__(self, message: str, error_code: Optional[str] = None, details: Optional[dict] = None):
        super().__init__(message, error_code, details, status_code=400)


class InvalidQueryError(QueryError):
    """Query is invalid or malformed"""
    pass


class NoResultsError(QueryError):
    """No results found for query"""
    def __init__(self, message: str, error_code: Optional[str] = None, details: Optional[dict] = None):
        super().__init__(message, error_code, details, status_code=404)


# ==================== User/Authorization Errors ====================

class AuthorizationError(RAGSystemError):
    """Base class for authorization errors"""
    def __init__(self, message: str, error_code: Optional[str] = None, details: Optional[dict] = None):
        super().__init__(message, error_code, details, status_code=401)


class InvalidUserIDError(AuthorizationError):
    """User ID is invalid or missing"""
    def __init__(self, message: str, error_code: Optional[str] = None, details: Optional[dict] = None):
        super().__init__(message, error_code, details, status_code=400)


class UnauthorizedAccessError(AuthorizationError):
    """User not authorized to access resource"""
    def __init__(self, message: str, error_code: Optional[str] = None, details: Optional[dict] = None):
        super().__init__(message, error_code, details, status_code=403)


# ==================== Database Errors ====================

class DatabaseError(RAGSystemError):
    """Base class for database errors"""
    pass


class RecordNotFoundError(DatabaseError):
    """Database record not found"""
    def __init__(self, message: str, error_code: Optional[str] = None, details: Optional[dict] = None):
        super().__init__(message, error_code, details, status_code=404)


class DatabaseConnectionError(DatabaseError):
    """Cannot connect to database"""
    def __init__(self, message: str, error_code: Optional[str] = None, details: Optional[dict] = None):
        super().__init__(message, error_code, details, status_code=503)


# ==================== Configuration Errors ====================

class ConfigurationError(RAGSystemError):
    """Base class for configuration errors"""
    pass


class MissingConfigurationError(ConfigurationError):
    """Required configuration is missing"""
    pass


class InvalidConfigurationError(ConfigurationError):
    """Configuration value is invalid"""
    pass


# ==================== External Service Errors ====================

class ExternalServiceError(RAGSystemError):
    """Base class for external service errors"""
    def __init__(self, message: str, error_code: Optional[str] = None, details: Optional[dict] = None):
        super().__init__(message, error_code, details, status_code=502)


class BedrockServiceError(ExternalServiceError):
    """Error communicating with AWS Bedrock"""
    pass


class OCRServiceError(ExternalServiceError):
    """Error with OCR service"""
    pass


# ==================== Helper Functions ====================

def get_error_message(error: Exception) -> str:
    """
    Get user-friendly error message from exception
    
    Args:
        error: Exception instance
    
    Returns:
        User-friendly error message
    """
    if isinstance(error, RAGSystemError):
        return error.message
    return "An unexpected error occurred. Please try again."


def get_error_details(error: Exception) -> dict:
    """
    Get error details for logging
    
    Args:
        error: Exception instance
    
    Returns:
        Dictionary with error details
    """
    if isinstance(error, RAGSystemError):
        return {
            "error_code": error.error_code,
            "message": error.message,
            "details": error.details,
            "type": type(error).__name__,
            "status_code": error.status_code
        }
    return {
        "error_code": "UNKNOWN_ERROR",
        "message": str(error),
        "type": type(error).__name__,
        "status_code": 500
    }


def handle_exception(error: Exception) -> HTTPException:
    """
    Convert any exception to HTTPException
    
    Args:
        error: Any exception
    
    Returns:
        HTTPException suitable for FastAPI
    
    Usage:
        try:
            do_something()
        except Exception as e:
            raise handle_exception(e)
    """
    if isinstance(error, RAGSystemError):
        return error.to_http_exception()
    elif isinstance(error, HTTPException):
        return error
    else:
        # Unknown exception - return generic 500
        return HTTPException(
            status_code=500,
            detail="An internal error occurred"
        )


# ==================== Decorator for Automatic Exception Handling ====================

from functools import wraps
from typing import Callable
import logging

logger = logging.getLogger(__name__)


def handle_errors(func: Callable) -> Callable:
    """
    Decorator to automatically handle exceptions in route handlers
    
    Usage:
        @router.post("/upload")
        @handle_errors
        async def upload_file(...):
            # Your code here
            # Exceptions automatically converted to HTTPException
    """
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except HTTPException:
            # Already an HTTPException, re-raise as-is
            raise
        except RAGSystemError as e:
            # Our custom error - log and convert
            logger.error(f"{e.error_code}: {e.message}", extra=e.details)
            raise e.to_http_exception()
        except Exception as e:
            # Unexpected error - log and return generic 500
            logger.exception(f"Unexpected error in {func.__name__}: {e}")
            raise HTTPException(status_code=500, detail="An internal error occurred")
    
    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except HTTPException:
            raise
        except RAGSystemError as e:
            logger.error(f"{e.error_code}: {e.message}", extra=e.details)
            raise e.to_http_exception()
        except Exception as e:
            logger.exception(f"Unexpected error in {func.__name__}: {e}")
            raise HTTPException(status_code=500, detail="An internal error occurred")
    
    import asyncio
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    return sync_wrapper