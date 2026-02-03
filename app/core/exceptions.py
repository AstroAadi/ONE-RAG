"""
Custom Exceptions - STANDARDIZED VERSION
Provides consistent error handling across the RAG system
"""

from typing import Optional, Dict, Any


class RAGException(Exception):
    """
    Base exception for all RAG system errors
    
    All custom exceptions should inherit from this class
    This allows for consistent error handling at the API level
    """
    status_code = 500
    error_code = "INTERNAL_ERROR"
    
    def __init__(
        self, 
        message: str, 
        details: Optional[Dict[str, Any]] = None,
        status_code: Optional[int] = None
    ):
        """
        Initialize exception
        
        Args:
            message: Human-readable error message
            details: Optional additional error details
            status_code: Optional HTTP status code override
        """
        self.message = message
        self.details = details or {}
        if status_code:
            self.status_code = status_code
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for API response"""
        return {
            "error": self.error_code,
            "message": self.message,
            "details": self.details,
            "status_code": self.status_code
        }


# ==================== CLIENT ERRORS (4xx) ====================

class ValidationError(RAGException):
    """Invalid input or request parameters"""
    status_code = 400
    error_code = "VALIDATION_ERROR"


class AuthenticationError(RAGException):
    """Authentication failed"""
    status_code = 401
    error_code = "AUTHENTICATION_ERROR"


class AuthorizationError(RAGException):
    """User lacks permission for this resource"""
    status_code = 403
    error_code = "AUTHORIZATION_ERROR"


class NotFoundError(RAGException):
    """Requested resource not found"""
    status_code = 404
    error_code = "NOT_FOUND"


class ConflictError(RAGException):
    """Resource conflict (e.g., duplicate upload)"""
    status_code = 409
    error_code = "CONFLICT"


class RateLimitError(RAGException):
    """Rate limit exceeded"""
    status_code = 429
    error_code = "RATE_LIMIT_EXCEEDED"


# ==================== SERVER ERRORS (5xx) ====================

class ServerError(RAGException):
    """Generic server error"""
    status_code = 500
    error_code = "INTERNAL_SERVER_ERROR"


class ServiceUnavailableError(RAGException):
    """Service temporarily unavailable"""
    status_code = 503
    error_code = "SERVICE_UNAVAILABLE"


class TimeoutError(RAGException):
    """Operation timed out"""
    status_code = 504
    error_code = "TIMEOUT"


# ==================== DOMAIN-SPECIFIC ERRORS ====================

# File Upload Errors
class FileUploadError(ValidationError):
    """File upload failed"""
    error_code = "FILE_UPLOAD_ERROR"


class FileSizeError(ValidationError):
    """File size exceeds limit"""
    error_code = "FILE_SIZE_ERROR"


class FileTypeError(ValidationError):
    """Unsupported file type"""
    error_code = "FILE_TYPE_ERROR"


# Document Processing Errors
class DocumentProcessingError(ServerError):
    """Document processing failed"""
    error_code = "DOCUMENT_PROCESSING_ERROR"


class ExtractionError(DocumentProcessingError):
    """Content extraction failed"""
    error_code = "EXTRACTION_ERROR"


class ChunkingError(DocumentProcessingError):
    """Text chunking failed"""
    error_code = "CHUNKING_ERROR"


# Embedding & Vector Store Errors
class EmbeddingError(ServerError):
    """Embedding generation failed"""
    error_code = "EMBEDDING_ERROR"


class VectorStoreException(ServerError):
    """Vector store operation failed"""
    error_code = "VECTOR_STORE_ERROR"


class NoDocumentsFoundError(NotFoundError):
    """No documents found in knowledge base"""
    error_code = "NO_DOCUMENTS_FOUND"


# Search Errors
class SearchError(ServerError):
    """Search operation failed"""
    error_code = "SEARCH_ERROR"


class QueryError(SearchError):
    """Query processing failed"""
    error_code = "QUERY_ERROR"


# LLM Errors
class LLMError(ServerError):
    """LLM service error"""
    error_code = "LLM_ERROR"


class LLMGenerationError(LLMError):
    """Answer generation failed"""
    error_code = "LLM_GENERATION_ERROR"


class LLMTimeoutError(TimeoutError):
    """LLM request timed out"""
    error_code = "LLM_TIMEOUT"


# Database Errors
class DatabaseError(ServerError):
    """Database operation failed"""
    error_code = "DATABASE_ERROR"


class DatabaseConnectionError(ServiceUnavailableError):
    """Cannot connect to database"""
    error_code = "DATABASE_CONNECTION_ERROR"


# Task Management Errors
class TaskError(ServerError):
    """Background task error"""
    error_code = "TASK_ERROR"


class TaskNotFoundError(NotFoundError):
    """Task not found"""
    error_code = "TASK_NOT_FOUND"


class TaskCancelledError(RAGException):
    """Task was cancelled"""
    status_code = 499  # Client Closed Request
    error_code = "TASK_CANCELLED"


# Configuration Errors
class ConfigurationError(ServerError):
    """System configuration error"""
    error_code = "CONFIGURATION_ERROR"


class MissingCredentialsError(ConfigurationError):
    """Required credentials not configured"""
    error_code = "MISSING_CREDENTIALS"


# ==================== HELPER FUNCTIONS ====================

def raise_for_status(
    condition: bool,
    exception_class: type[RAGException],
    message: str,
    **details
):
    """
    Raise exception if condition is True
    
    Example:
        raise_for_status(
            len(text) > MAX_LENGTH,
            ValidationError,
            "Text too long",
            max_length=MAX_LENGTH,
            actual_length=len(text)
        )
    """
    if condition:
        raise exception_class(message, details=details)


def handle_external_error(error: Exception, context: str = "") -> RAGException:
    """
    Convert external library errors to RAG exceptions
    
    Args:
        error: Original exception
        context: Context string (e.g., "embedding generation")
        
    Returns:
        Appropriate RAG exception
    
    Example:
        try:
            result = bedrock_client.embed(text)
        except Exception as e:
            raise handle_external_error(e, "embedding generation")
    """
    error_message = f"{context}: {str(error)}" if context else str(error)
    
    # Map common errors
    if "timeout" in str(error).lower():
        return TimeoutError(error_message)
    elif "not found" in str(error).lower():
        return NotFoundError(error_message)
    elif "permission" in str(error).lower() or "unauthorized" in str(error).lower():
        return AuthorizationError(error_message)
    elif "rate limit" in str(error).lower():
        return RateLimitError(error_message)
    else:
        return ServerError(error_message)