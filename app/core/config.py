"""
Clean Configuration for RAG System
Removed: parent-child chunking, hybrid search ratios, unused features
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
    bedrock_model_id: str = os.getenv(
        "BEDROCK_MODEL_ID", 
        "anthropic.claude-3-5-sonnet-20241022"
    )
    
    embedding_model_id: str = os.getenv(
        "EMBEDDING_MODEL_ID",
        "cohere.embed-english-v3"
    )
    
    # Model Parameters
    temperature: float = float(os.getenv("TEMPERATURE", "0.5"))
    max_tokens: int = int(os.getenv("MAX_TOKENS", "4000"))
    
    # ==================== Image Analysis Configuration ====================
    bedrock_vision_model_id: str = os.getenv(
        "BEDROCK_VISION_MODEL_ID",
        "us.anthropic.claude-3-5-sonnet-20241022-v2:0"
    )
    
    # ==================== Application Configuration ====================
    upload_dir: str = "./uploads"
    markdown_dir: str = "./markdown_output"
    max_file_size_mb: int = int(os.getenv("MAX_FILE_SIZE_MB", "50"))
    
    # ==================== ChromaDB Configuration ====================
    chroma_persist_dir: str = "./chroma_db"
    
    # ==================== Chunking Configuration ====================
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1500"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    max_chunk_size: int = int(os.getenv("MAX_CHUNK_SIZE", "6000"))
    
    # ==================== RAG Configuration ====================
    default_top_k: int = int(os.getenv("DEFAULT_TOP_K", "5"))
    max_top_k: int = int(os.getenv("MAX_TOP_K", "20"))
    max_context_length: int = int(os.getenv("MAX_CONTEXT_LENGTH", "8000"))
    
    # Conversation history
    max_conversation_turns: int = int(os.getenv("MAX_CONVERSATION_TURNS", "5"))
    
    # Hallucination detection
    hallucination_threshold: float = float(os.getenv("HALLUCINATION_THRESHOLD", "0.6"))
    
    # ==================== Document Processing Configuration ====================
    enable_ai_image_analysis: bool = os.getenv("ENABLE_AI_IMAGE_ANALYSIS", "true").lower() == "true"
    extract_images: bool = os.getenv("EXTRACT_IMAGES", "true").lower() == "true"
    infer_table_structure: bool = os.getenv("INFER_TABLE_STRUCTURE", "true").lower() == "true"
    
    # ==================== Logging Configuration ====================
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    enable_query_logging: bool = os.getenv("ENABLE_QUERY_LOGGING", "true").lower() == "true"
    
    # ==================== Feature Flags ====================
    enable_conversation_history: bool = os.getenv("ENABLE_CONVERSATION_HISTORY", "true").lower() == "true"
    enable_hallucination_detection: bool = os.getenv("ENABLE_HALLUCINATION_DETECTION", "true").lower() == "true"
    
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
Path(settings.upload_dir, "extracted_images").mkdir(parents=True, exist_ok=True)


def validate_config():
    """Validate configuration"""
    warnings = []
    
    if not settings.aws_access_key_id or not settings.aws_secret_access_key:
        if not settings.aws_profile_name:
            warnings.append("AWS credentials not configured")
    
    if not settings.bedrock_model_id:
        warnings.append("BEDROCK_MODEL_ID not set")
    
    if not settings.embedding_model_id:
        warnings.append("EMBEDDING_MODEL_ID not set")
    
    # Check Bedrock availability
    try:
        import boto3
        if settings.aws_profile_name:
            session = boto3.Session(profile_name=settings.aws_profile_name, region_name=settings.aws_region)
            client = session.client('bedrock-runtime')
        else:
            client = boto3.client('bedrock-runtime', region_name=settings.aws_region)
        bedrock_available = True
    except Exception as e:
        bedrock_available = False
        if settings.enable_ai_image_analysis:
            warnings.append(f"Bedrock not accessible: {e}")
    
    if warnings:
        print("\n⚠️  Configuration Warnings:")
        for warning in warnings:
            print(f"  - {warning}")
        print()
    else:
        print("✅ Configuration validated")
        print(f"  - LLM: {settings.bedrock_model_id}")
        print(f"  - Embedding: {settings.embedding_model_id}")
        print(f"  - Chunking: {settings.chunk_size} chars, {settings.chunk_overlap} overlap")
        print(f"  - AI Image Analysis: {'Enabled' if settings.enable_ai_image_analysis and bedrock_available else 'Disabled'}")
        print()


# Run validation
validate_config()