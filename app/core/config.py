"""
Centralized Configuration for RAG System - UPDATED WITH TOKEN CONSTRAINTS
All configuration in one place with proper validation and type safety
"""

from pydantic import Field, field_validator, ConfigDict
from pydantic_settings import BaseSettings
from pathlib import Path
from typing import Literal
from dotenv import load_dotenv
import os
import logging

load_dotenv()

# Get logger for validation warnings
logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Application settings with validation"""
    
    model_config = ConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore",
        validate_assignment=True
    )
    
    # ==================== AWS Bedrock Configuration ====================
    aws_region: str = Field(default="us-east-1", description="AWS region for Bedrock")
    aws_access_key_id: str = Field(default="", description="AWS access key")
    aws_secret_access_key: str = Field(default="", description="AWS secret key")
    aws_profile_name: str = Field(default="", description="AWS profile name")
    
    # ==================== Model Configuration ====================
    bedrock_model_id: str = Field(
        default="anthropic.claude-3-5-sonnet-20241022",
        description="Bedrock LLM model ID"
    )
    
    embedding_model_id: str = Field(
        default="cohere.embed-english-v3",
        description="Embedding model ID"
    )
    
    # Embedding model limits (characters, not tokens)
    embedding_max_length: int = Field(
        default=2048,
        ge=512,
        le=8192,
        description="Max characters for embedding model (Cohere v3: 2048, Amazon Titan: 8192)"
    )
    
    bedrock_vision_model_id: str = Field(
        default="us.anthropic.claude-3-5-sonnet-20241022-v2:0",
        description="Vision model for image analysis"
    )
    
    # Model Parameters (consolidated)
    temperature: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="LLM temperature"
    )
    max_tokens: int = Field(
        default=4000,
        ge=100,
        le=100000,
        description="Max tokens for LLM response"
    )
    vision_max_tokens: int = Field(
        default=2000,
        ge=100,
        le=10000,
        description="Max tokens for vision model"
    )
    vision_temperature: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Temperature for vision model (lower for factual descriptions)"
    )
    
    # ==================== Storage Configuration ====================
    upload_dir: Path = Field(default=Path("./uploads"), description="Upload directory")
    markdown_dir: Path = Field(default=Path("./markdown_output"), description="Markdown output directory")
    database_url: str = Field(
        default="sqlite:///./rag_system.db",
        description="SQLite database URL"
    )
    
    # File Upload Limits
    max_file_size_mb: int = Field(
        default=50,
        ge=1,
        le=500,
        description="Max file size in MB"
    )
    max_filename_length: int = Field(
        default=255,
        ge=1,
        le=500,
        description="Max filename length"
    )
    
    # ==================== QDRANT CLOUD CONFIGURATION ====================
    qdrant_url: str = Field(
        default="",
        description="Qdrant Cloud cluster URL (e.g., https://xxx.cloud.qdrant.io)"
    )
    qdrant_api_key: str = Field(
        default="",
        description="Qdrant Cloud API key"
    )
    qdrant_collection_name: str = Field(
        default="rag_documents",
        description="Qdrant collection name"
    )
    
    @field_validator('qdrant_url')
    @classmethod
    def validate_qdrant_url(cls, v: str) -> str:
        """Ensure Qdrant URL is properly formatted"""
        if v and not v.startswith(('http://', 'https://')):
            raise ValueError("qdrant_url must start with http:// or https://")
        return v
    
    # ==================== TOKEN-BASED CHUNKING CONSTRAINTS (NEW) ====================
    
    # Token limits (the primary constraints)
    min_chunk_tokens: int = Field(
        default=200,
        ge=50,
        le=500,
        description="Minimum chunk size in tokens"
    )
    
    target_chunk_tokens: int = Field(
        default=512,
        ge=200,
        le=2000,
        description="Target chunk size in tokens (aim for this)"
    )
    
    max_chunk_tokens: int = Field(
        default=600,
        ge=300,
        le=2000,
        description="Maximum chunk size in tokens (hard limit)"
    )
    
    chunk_overlap_tokens: int = Field(
        default=50,
        ge=0,
        le=200,
        description="Overlap between chunks in tokens (10-20% of target)"
    )
    
    # Tokenizer configuration
    tokenizer_model: str = Field(
        default="cl100k_base",  # For OpenAI models
        description="Tokenizer to use (cl100k_base, p50k_base, or gpt2)"
    )
    
    # Character-based limits (DEPRECATED - kept for backward compatibility)
    chunk_size: int = Field(
        default=1500,
        ge=100,
        le=8000,
        description="DEPRECATED: Use token-based limits instead"
    )
    chunk_overlap: int = Field(
        default=200,
        ge=0,
        le=1000,
        description="DEPRECATED: Use chunk_overlap_tokens instead"
    )
    max_chunk_size: int = Field(
        default=2000,
        ge=1000,
        le=10000,
        description="DEPRECATED: Use max_chunk_tokens instead"
    )
    
    @field_validator('max_chunk_tokens')
    @classmethod
    def validate_token_limits(cls, v: int, info) -> int:
        """Ensure token limits are consistent"""
        target = info.data.get('target_chunk_tokens', 512)
        min_tokens = info.data.get('min_chunk_tokens', 200)
        
        if v < target:
            raise ValueError(
                f"max_chunk_tokens ({v}) must be >= target_chunk_tokens ({target})"
            )
        
        if target < min_tokens:
            raise ValueError(
                f"target_chunk_tokens ({target}) must be >= min_chunk_tokens ({min_tokens})"
            )
        
        return v
    
    @field_validator('chunk_overlap_tokens')
    @classmethod
    def validate_overlap_tokens(cls, v: int, info) -> int:
        """Ensure overlap is reasonable (10-30% of target)"""
        target = info.data.get('target_chunk_tokens', 512)
        
        max_overlap = int(target * 0.3)  # 30% max
        
        if v > max_overlap:
            logger.warning(
                f"chunk_overlap_tokens ({v}) exceeds 30% of target ({max_overlap}). "
                f"Consider reducing overlap for better performance."
            )
        
        return v
    
    # Atomic element handling (NEW)
    max_atomic_element_tokens: int = Field(
        default=1000,
        ge=500,
        le=4000,
        description="Max tokens for atomic elements (tables, code blocks). Exceeding this splits them."
    )
    
    split_oversized_atomic: bool = Field(
        default=True,
        description="Split atomic elements that exceed max_atomic_element_tokens"
    )
    
    max_code_block_lines: int = Field(
        default=500,
        ge=100,
        le=2000,
        description="Maximum lines in a code block before splitting"
    )
    
    # ==================== ChromaDB Configuration (DEPRECATED) ====================
    chroma_persist_dir: Path = Field(
        default=Path("./chroma_db"),
        description="ChromaDB persistence directory (DEPRECATED)"
    )
    chroma_collection_name: str = Field(
        default="rag_documents",
        description="Default ChromaDB collection name (DEPRECATED)"
    )
    
    # ==================== RAG Configuration ====================
    default_top_k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Default number of chunks to retrieve"
    )
    max_top_k: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Maximum allowed top_k"
    )
    max_context_length: int = Field(
        default=8000,
        ge=1000,
        le=50000,
        description="Max context length for LLM"
    )
    
    # @field_validator('default_semantic_weight')
    # @classmethod
    # def validate_weights_sum(cls, v: float, info) -> float:
    #     """Ensure weights sum to 1.0"""
    #     keyword_weight = info.data.get('default_keyword_weight', 0.3)
    #     total = keyword_weight + v
    #     if abs(total - 1.0) > 0.01:
    #         raise ValueError(
    #             f"keyword_weight ({keyword_weight}) + semantic_weight ({v}) must equal 1.0"
    #         )
    #     return v
    
    # ==================== Conversation Configuration ====================
    max_conversation_turns: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Max conversation history turns to keep"
    )
    conversation_cleanup_minutes: int = Field(
        default=60,
        ge=1,
        le=1440,
        description="Minutes before cleaning up old conversations"
    )
    
    # ==================== Document Processing Configuration ====================
    enable_ai_image_analysis: bool = Field(
        default=True,
        description="Enable AI-powered image description"
    )
    extract_images: bool = Field(
        default=True,
        description="Extract images from documents"
    )
    infer_table_structure: bool = Field(
        default=True,
        description="Infer table structure during extraction"
    )
    enable_ocr: bool = Field(
        default=True,
        description="Enable OCR for image text extraction"
    )
    
    # Image Processing Settings
    image_scale: float = Field(
        default=2.0,
        ge=1.0,
        le=4.0,
        description="Image scaling factor"
    )
    image_quality: int = Field(
        default=85,
        ge=1,
        le=100,
        description="Image compression quality"
    )
    
    # ==================== Logging & Monitoring Configuration ====================
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Logging level"
    )
    enable_query_logging: bool = Field(
        default=True,
        description="Log all queries for monitoring"
    )
    enable_performance_metrics: bool = Field(
        default=True,
        description="Enable performance metric collection"
    )
    log_file_path: Path = Field(
        default=Path("./logs/rag_system.log"),
        description="Log file path"
    )
    
    # ==================== Error Handling Configuration ====================
    max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Max retries for external API calls"
    )
    retry_delay_seconds: float = Field(
        default=1.0,
        ge=0.1,
        le=60.0,
        description="Delay between retries in seconds"
    )
    
    # ==================== Feature Flags ====================
    enable_conversation_history: bool = Field(
        default=True,
        description="Enable conversation history tracking"
    )
    enable_hallucination_detection: bool = Field(
        default=False,
        description="Enable hallucination detection (experimental)"
    )
    hallucination_threshold: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Threshold for hallucination detection"
    )
    
    # ==================== Security Configuration ====================
    allowed_file_extensions: set[str] = Field(
        default={
            'pdf', 'docx', 'doc', 'pptx', 'ppt',
            'txt', 'md', 'markdown',
            'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp', 'tiff'
        },
        description="Allowed file extensions"
    )
    max_user_id_length: int = Field(
        default=100,
        ge=1,
        le=500,
        description="Maximum user ID length"
    )
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Create directories on initialization
        self._create_directories()
    
    def _create_directories(self) -> None:
        """Create necessary directories"""
        directories = [
            self.upload_dir,
            self.markdown_dir,
            self.log_file_path.parent,
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    @property
    def max_file_size_bytes(self) -> int:
        """Get max file size in bytes"""
        return self.max_file_size_mb * 1024 * 1024
    
    def get_chroma_settings(self) -> dict:
        """
        Get ChromaDB settings (DEPRECATED - keep for migration)
        
        Returns:
            Dictionary with ChromaDB configuration
        """
        # Ensure directory exists
        self.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
        
        return {
            "persist_directory": str(self.chroma_persist_dir.absolute()),
            "allow_reset": True
        }
    
    def validate_configuration(self) -> tuple[bool, list[str]]:
        """
        Validate configuration and return warnings
        
        Returns:
            Tuple of (is_valid, list_of_warnings)
        """
        warnings = []
        is_valid = True
        
        # Check AWS credentials
        if not self.aws_access_key_id or not self.aws_secret_access_key:
            if not self.aws_profile_name:
                warnings.append("AWS credentials not configured (neither keys nor profile)")
                is_valid = False
        
        # Check model IDs
        if not self.bedrock_model_id:
            warnings.append("BEDROCK_MODEL_ID not set")
            is_valid = False
        
        if not self.embedding_model_id:
            warnings.append("EMBEDDING_MODEL_ID not set")
            is_valid = False
        
        # Check Qdrant configuration
        if not self.qdrant_url:
            warnings.append("QDRANT_URL not set ")
        
        if not self.qdrant_api_key and self.qdrant_url:
            warnings.append("QDRANT_API_KEY not set but URL provided")
            is_valid = False
        
        # Check token limits
        if self.max_chunk_tokens > 2000:
            warnings.append(
                f"max_chunk_tokens ({self.max_chunk_tokens}) exceeds recommended 2000. "
                f"Very large chunks may hurt search quality."
            )
        
        # Check Bedrock availability
        if self.enable_ai_image_analysis:
            try:
                import boto3
                if self.aws_profile_name:
                    session = boto3.Session(
                        profile_name=self.aws_profile_name,
                        region_name=self.aws_region
                    )
                    session.client('bedrock-runtime')
                else:
                    boto3.client('bedrock-runtime', region_name=self.aws_region)
            except Exception as e:
                warnings.append(f"Bedrock not accessible: {e}")
        
        return is_valid, warnings


# Initialize global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Dependency injection function for FastAPI"""
    return settings


def print_configuration_summary():
    """Print configuration summary on startup"""
    is_valid, warnings = settings.validate_configuration()
    
    print("\n" + "="*60)
    print("RAG SYSTEM CONFIGURATION - TOKEN-CONSTRAINED SEMANTIC CHUNKING")
    print("="*60)
    
    print(f"\n📊 Models:")
    print(f"  - LLM: {settings.bedrock_model_id}")
    print(f"  - Embedding: {settings.embedding_model_id}")
    print(f"  - Vision: {settings.bedrock_vision_model_id}")
    print(f"  - Tokenizer: {settings.tokenizer_model}")
    
    print(f"\n📦 Chunking (Token-Based Constraints):")
    print(f"  - Min Tokens: {settings.min_chunk_tokens}")
    print(f"  - Target Tokens: {settings.target_chunk_tokens}")
    print(f"  - Max Tokens: {settings.max_chunk_tokens}")
    print(f"  - Overlap: {settings.chunk_overlap_tokens} tokens ({settings.chunk_overlap_tokens/settings.target_chunk_tokens*100:.1f}%)")
    print(f"  - Max Atomic Element: {settings.max_atomic_element_tokens} tokens")
    print(f"  - Split Oversized Atomic: {'✅' if settings.split_oversized_atomic else '❌'}")
    
    print(f"\n🔍 Search:")
    print(f"  - Default top_k: {settings.default_top_k}")
    # print(f"  - Keyword weight: {settings.default_keyword_weight}")
    # print(f"  - Semantic weight: {settings.default_semantic_weight}")
    
    print(f"\n💾 Storage:")
    print(f"  - Database: {settings.database_url}")
    print(f"  - Vector DB: {'Qdrant Cloud ✓' if settings.qdrant_url else 'ChromaDB (deprecated)'}")
    if settings.qdrant_url:
        print(f"  - Qdrant URL: {settings.qdrant_url}")
        print(f"  - Qdrant Collection: {settings.qdrant_collection_name}")
    print(f"  - Uploads: {settings.upload_dir}")
    
    print(f"\n🔧 Features:")
    print(f"  - AI Image Analysis: {'✅' if settings.enable_ai_image_analysis else '❌'}")
    print(f"  - Conversation History: {'✅' if settings.enable_conversation_history else '❌'}")
    print(f"  - Query Logging: {'✅' if settings.enable_query_logging else '❌'}")
    print(f"  - Performance Metrics: {'✅' if settings.enable_performance_metrics else '❌'}")
    
    if warnings:
        print(f"\n⚠️  Warnings ({len(warnings)}):")
        for warning in warnings:
            print(f"  - {warning}")
    else:
        print("\n✅ Configuration valid - no warnings")
    
    if not is_valid:
        print("\n❌ CRITICAL: Configuration is invalid!")
    
    print("="*60 + "\n")


# Run validation on import
print_configuration_summary()