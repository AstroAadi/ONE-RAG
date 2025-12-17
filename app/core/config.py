"""
Enhanced Configuration with Advanced RAG Settings
FIXED: Removed Anthropic API key, uses Bedrock for all operations
"""

from pydantic_settings import BaseSettings
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()


class Settings(BaseSettings):
    # ==================== AWS Bedrock Configuration ====================
    aws_region: str = os.getenv("AWS_REGION", "us-east-1")
    aws_access_key_id: str = os.getenv("AWS_ACCESS_KEY_ID", "")
    aws_secret_access_key: str = os.getenv("AWS_SECRET_ACCESS_KEY", "")
    aws_profile_name: str = os.getenv("AWS_PROFILE_NAME", "")
    
    # ==================== Model Configuration ====================
    # Main LLM model for answer generation
    bedrock_model_id: str = os.getenv(
        "BEDROCK_MODEL_ID", 
        "anthropic.claude-3-5-sonnet-20241022"  # Updated to latest
    )
    
    # Embedding model for vector search
    embedding_model_id: str = os.getenv(
        "EMBEDDING_MODEL_ID",
        "cohere.embed-english-v3"
    )
    
    # Model Parameters
    temperature: float = float(os.getenv("TEMPERATURE", "0.7"))
    max_tokens: int = int(os.getenv("MAX_TOKENS", "4000"))  # Increased for detailed answers
    
    # ==================== Image Analysis Configuration ====================
    # Uses AWS Bedrock Claude for image analysis instead of direct Anthropic API
    enable_bedrock_image_analysis: bool = os.getenv("ENABLE_BEDROCK_IMAGE_ANALYSIS", "true").lower() == "true"
    
    # Image analysis model (Claude with vision via Bedrock)
    # Use inference profile for cross-region support
    bedrock_vision_model_id: str = os.getenv(
        "BEDROCK_VISION_MODEL_ID",
        "us.anthropic.claude-3-5-sonnet-20241022-v2:0"  # Inference profile (not direct model ID)
    )
    
    # ==================== Application Configuration ====================
    upload_dir: str = "./uploads"
    markdown_dir: str = "./markdown_output"
    max_file_size_mb: int = int(os.getenv("MAX_FILE_SIZE_MB", "50"))
    
    # ==================== ChromaDB Configuration ====================
    chroma_persist_dir: str = "./chroma_db"
    
    # ==================== Advanced Chunking Configuration ====================
    # Base chunk size for semantic chunks
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
    
    # Overlap between chunks for context continuity
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    
    # Maximum chunk size (hard limit for embeddings)
    max_chunk_size: int = int(os.getenv("MAX_CHUNK_SIZE", "1800"))
    
    # Parent chunk size multiplier (parent chunks are larger)
    parent_chunk_multiplier: int = int(os.getenv("PARENT_CHUNK_MULTIPLIER", "4"))
    
    # ==================== RAG Configuration ====================
    # Default number of chunks to retrieve
    default_top_k: int = int(os.getenv("DEFAULT_TOP_K", "5"))
    
    # Maximum top_k allowed
    max_top_k: int = int(os.getenv("MAX_TOP_K", "20"))
    
    # Default search mode: 'semantic', 'parent', 'hybrid'
    default_search_mode: str = os.getenv("DEFAULT_SEARCH_MODE", "hybrid")
    
    # Hybrid search split (percentage for child chunks)
    hybrid_child_ratio: float = float(os.getenv("HYBRID_CHILD_RATIO", "0.7"))
    
    # Context compression max length
    max_context_length: int = int(os.getenv("MAX_CONTEXT_LENGTH", "8000"))
    
    # Conversation history max turns
    max_conversation_turns: int = int(os.getenv("MAX_CONVERSATION_TURNS", "5"))
    
    # Hallucination detection threshold
    hallucination_threshold: float = float(os.getenv("HALLUCINATION_THRESHOLD", "0.6"))
    
    # ==================== Document Processing Configuration ====================
    # PDF extraction strategy priority: 'hi_res', 'fast', 'basic'
    pdf_extraction_strategy: str = os.getenv("PDF_EXTRACTION_STRATEGY", "hi_res")
    
    # Enable AI image analysis (uses Bedrock - no additional keys needed)
    enable_ai_image_analysis: bool = os.getenv("ENABLE_AI_IMAGE_ANALYSIS", "true").lower() == "true"
    
    # Extract images from PDFs
    extract_images: bool = os.getenv("EXTRACT_IMAGES", "true").lower() == "true"
    
    # Infer table structure
    infer_table_structure: bool = os.getenv("INFER_TABLE_STRUCTURE", "true").lower() == "true"
    
    # ==================== Performance Configuration ====================
    # Enable caching for embeddings
    enable_embedding_cache: bool = os.getenv("ENABLE_EMBEDDING_CACHE", "false").lower() == "true"
    
    # Batch size for embedding generation
    embedding_batch_size: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "10"))
    
    # ==================== Logging Configuration ====================
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    enable_query_logging: bool = os.getenv("ENABLE_QUERY_LOGGING", "true").lower() == "true"
    
    # ==================== Feature Flags ====================
    # Enable parent-child chunking
    enable_parent_child_chunks: bool = os.getenv("ENABLE_PARENT_CHILD_CHUNKS", "true").lower() == "true"
    
    # Enable conversation history
    enable_conversation_history: bool = os.getenv("ENABLE_CONVERSATION_HISTORY", "true").lower() == "true"
    
    # Enable hallucination detection
    enable_hallucination_detection: bool = os.getenv("ENABLE_HALLUCINATION_DETECTION", "true").lower() == "true"
    
    # Enable query classification
    enable_query_classification: bool = os.getenv("ENABLE_QUERY_CLASSIFICATION", "true").lower() == "true"
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"


# Initialize settings
settings = Settings()

# Create necessary directories
Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)
Path(settings.markdown_dir).mkdir(parents=True, exist_ok=True)
Path(settings.chroma_persist_dir).mkdir(parents=True, exist_ok=True)

# Create images directory for extracted images
Path(settings.upload_dir, "extracted_images").mkdir(parents=True, exist_ok=True)


# ==================== Helper Functions ====================

def get_chunking_config() -> dict:
    """Get chunking configuration"""
    return {
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "max_chunk_size": settings.max_chunk_size,
        "enable_parent_child": settings.enable_parent_child_chunks,
        "parent_multiplier": settings.parent_chunk_multiplier
    }


def get_rag_config() -> dict:
    """Get RAG configuration"""
    return {
        "default_top_k": settings.default_top_k,
        "max_top_k": settings.max_top_k,
        "default_search_mode": settings.default_search_mode,
        "hybrid_child_ratio": settings.hybrid_child_ratio,
        "max_context_length": settings.max_context_length,
        "enable_conversation": settings.enable_conversation_history,
        "enable_hallucination_check": settings.enable_hallucination_detection
    }


def get_extraction_config() -> dict:
    """Get document extraction configuration"""
    return {
        "strategy": settings.pdf_extraction_strategy,
        "extract_images": settings.extract_images,
        "infer_tables": settings.infer_table_structure,
        "ai_image_analysis": settings.enable_ai_image_analysis and _check_bedrock_availability()
    }


def _check_bedrock_availability() -> bool:
    """Check if Bedrock is available for image analysis"""
    try:
        import boto3
        if settings.aws_profile_name:
            session = boto3.Session(profile_name=settings.aws_profile_name, region_name=settings.aws_region)
            client = session.client('bedrock-runtime')
        else:
            client = boto3.client('bedrock-runtime', region_name=settings.aws_region)
        return True
    except Exception as e:
        print(f"Warning: Bedrock not available: {e}")
        return False


def validate_config():
    """Validate configuration and warn about missing required settings"""
    warnings = []
    
    if not settings.aws_access_key_id or not settings.aws_secret_access_key:
        if not settings.aws_profile_name:
            warnings.append("AWS credentials not configured. Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY or AWS_PROFILE_NAME")
    
    if not settings.bedrock_model_id:
        warnings.append("BEDROCK_MODEL_ID not set")
    
    if not settings.embedding_model_id:
        warnings.append("EMBEDDING_MODEL_ID not set")
    
    # Check Bedrock availability
    bedrock_available = _check_bedrock_availability()
    
    if settings.enable_ai_image_analysis and not bedrock_available:
        warnings.append("AI image analysis enabled but Bedrock is not accessible. Image analysis will be disabled.")
    
    if warnings:
        print("\n⚠️  Configuration Warnings:")
        for warning in warnings:
            print(f"  - {warning}")
        print()
    else:
        print("✅ Configuration validated successfully")
        print(f"  - LLM Model: {settings.bedrock_model_id}")
        print(f"  - Embedding Model: {settings.embedding_model_id}")
        print(f"  - Chunking: {settings.chunk_size} chars with {settings.chunk_overlap} overlap")
        print(f"  - RAG Mode: {settings.default_search_mode}")
        print(f"  - AI Image Analysis: {'Enabled (via Bedrock)' if settings.enable_ai_image_analysis and bedrock_available else 'Disabled'}")
        if bedrock_available:
            print(f"  - Vision Model: {settings.bedrock_vision_model_id}")
        print()


# Run validation on import
validate_config()