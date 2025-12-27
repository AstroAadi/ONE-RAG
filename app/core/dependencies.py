"""
Dependency Injection for Services
Provides proper FastAPI dependency injection for all services
"""

from typing import Generator, Optional
from functools import lru_cache
import logging

from sqlalchemy.orm import Session

from app.db.session import get_db
from app.core.config import Settings, get_settings
from app.core.exceptions import InvalidUserIDError

logger = logging.getLogger(__name__)


# ==================== Configuration ====================

@lru_cache()
def get_cached_settings() -> Settings:
    """
    Get cached settings instance
    
    Uses lru_cache to ensure single instance
    """
    return get_settings()


# ==================== User Validation ====================

def get_user_id(x_user_id: Optional[str] = None) -> str:
    """
    Extract and validate user ID from request header
    
    Args:
        x_user_id: User ID from X-User-ID header
    
    Returns:
        Validated user ID
    
    Raises:
        InvalidUserIDError: If user ID is missing or invalid
    
    Usage in FastAPI:
        @router.get("/documents")
        def get_documents(user_id: str = Depends(get_user_id)):
            ...
    """
    if not x_user_id:
        raise InvalidUserIDError(
            "User ID is required. Please provide X-User-ID header."
        )
    
    # Validate user ID
    x_user_id = x_user_id.strip()
    
    if not x_user_id:
        raise InvalidUserIDError("User ID cannot be empty")
    
    settings = get_cached_settings()
    if len(x_user_id) > settings.max_user_id_length:
        raise InvalidUserIDError(
            f"User ID too long (max {settings.max_user_id_length} characters)"
        )
    
    # Basic sanitization - remove potentially dangerous characters
    dangerous_chars = ['/', '\\', '..', '<', '>', '|', '*', '?', '"', ':']
    for char in dangerous_chars:
        if char in x_user_id:
            raise InvalidUserIDError(
                f"User ID contains invalid character: {char}"
            )
    
    return x_user_id


# ==================== Service Dependencies ====================

# These will be imported and used in service files
# to avoid circular imports, actual service instances
# are created lazily in each service module


def get_vector_store_service():
    """
    Get VectorStoreService instance
    
    Import here to avoid circular dependencies
    """
    from app.services.vector_store_service import VectorStoreService
    return VectorStoreService()


def get_llm_service():
    """
    Get LLMService instance
    
    Import here to avoid circular dependencies
    """
    from app.services.llm_service import LLMService
    return LLMService()


def get_chunking_service():
    """
    Get ChunkingService instance
    
    Import here to avoid circular dependencies
    """
    from app.services.chunking_service import SemanticChunkingService
    settings = get_cached_settings()
    return SemanticChunkingService(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        max_chunk_size=settings.max_chunk_size
    )


# ==================== Database Dependencies ====================

def get_database_session() -> Generator[Session, None, None]:
    """
    Get database session with automatic cleanup
    
    Usage in FastAPI:
        @router.get("/items")
        def get_items(db: Session = Depends(get_database_session)):
            ...
    """
    db = next(get_db())
    try:
        yield db
    finally:
        db.close()


# ==================== Combined Dependencies ====================

class ServiceDependencies:
    """
    Combined dependencies for routes that need multiple services
    
    Usage in FastAPI:
        @router.post("/process")
        def process(deps: ServiceDependencies = Depends()):
            vector_store = deps.vector_store
            llm = deps.llm
            ...
    """
    
    def __init__(
        self,
        db: Session = None,
        user_id: str = None,
        settings: Settings = None
    ):
        self.db = db
        self.user_id = user_id
        self.settings = settings or get_cached_settings()
        
        # Lazy load services only when needed
        self._vector_store = None
        self._llm = None
        self._chunking = None
    
    @property
    def vector_store(self):
        """Lazy load vector store service"""
        if self._vector_store is None:
            self._vector_store = get_vector_store_service()
        return self._vector_store
    
    @property
    def llm(self):
        """Lazy load LLM service"""
        if self._llm is None:
            self._llm = get_llm_service()
        return self._llm
    
    @property
    def chunking(self):
        """Lazy load chunking service"""
        if self._chunking is None:
            self._chunking = get_chunking_service()
        return self._chunking


# ==================== Logging Utilities ====================

def log_request_info(
    endpoint: str,
    user_id: str,
    additional_info: Optional[dict] = None
):
    """
    Log request information for monitoring
    
    Args:
        endpoint: API endpoint name
        user_id: User making the request
        additional_info: Additional context to log
    """
    info = {
        "endpoint": endpoint,
        "user_id": user_id,
        **(additional_info or {})
    }
    
    logger.info(f"Request: {info}")