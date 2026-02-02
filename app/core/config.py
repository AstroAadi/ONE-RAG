"""
Updated Configuration for Concurrent RAG System
Includes connection pooling and worker pool settings
"""

from pydantic import Field, field_validator, ConfigDict
from pydantic_settings import BaseSettings
from pathlib import Path
from typing import Literal
from dotenv import load_dotenv
import os
import logging

load_dotenv()

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Application settings with concurrency optimizations"""
    
    model_config = ConfigDict(
        env_file=".env.example",
        case_sensitive=False,
        extra="ignore",
        validate_assignment=True
    )
    
    # # ==================== AWS Bedrock Configuration ====================
    # aws_region: str = Field(default="us-east-1")
    # aws_access_key_id: str = Field(default="")
    # aws_secret_access_key: str = Field(default="")
    # aws_profile_name: str = Field(default="")
    

    # ==================== HuggingFace Configuration ====================
    huggingface_api_token: str = Field(
        default="",
        description="HuggingFace API token (required)"
    )
    
    # Use HuggingFace Router API (more reliable)
    huggingface_api_url: str = Field(
        default="https://router.huggingface.co/v1/chat/completions",
        description="HuggingFace Router API URL"
    )
    
    # ==================== Model Configuration ====================
    # bedrock_model_id: str = Field(
    #     default="anthropic.claude-3-5-sonnet-20241022"
    # )

    llm_model_id: str = Field(
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        description="Chat model ID from HuggingFace"
    )
    
    available_llm_models: list = Field(
        default=[
            "meta-llama/Meta-Llama-3-8B-Instruct",
            "mistralai/Mistral-7B-Instruct-v0.2",
            "HuggingFaceH4/zephyr-7b-beta"  # Works better than Gemma with API
        ],
        description="Available LLM models"
    )
    # # ==================== Embedding Model Configuration ====================
    # embedding_model_id: str = Field(
    #     default="cohere.embed-english-v3"
    # )
    
    # embedding_max_length: int = Field(default=2048, ge=512, le=8192)
    
    # bedrock_vision_model_id: str = Field(
    #     default="us.anthropic.claude-3-5-sonnet-20241022-v2:0"
    # )
    
    # temperature: float = Field(default=0.6, ge=0.0, le=1.0)
    # max_tokens: int = Field(default=4000, ge=100, le=100000)
    # vision_max_tokens: int = Field(default=2000, ge=100, le=10000)
    # vision_temperature: float = Field(default=0.6, ge=0.0, le=1.0)
    
    # ==================== Embedding Model Configuration ====================
    embedding_model_id: str = Field(
        default="BAAI/bge-large-en-v1.5",
        description="Embedding model ID from HuggingFace"
    )
    
    available_embedding_models: list = Field(
        default=[
            "BAAI/bge-large-en-v1.5",
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/paraphrase-mpnet-base-v2"
        ],
        description="Available embedding models"
    )
    
    # Embedding model limits
    embedding_max_length: int = Field(
        default=512,
        ge=128,
        le=8192,
        description="Max tokens for embedding model"
    )
    
    
    # ==================== Vision Model Configuration ====================
    vision_model_id: str = Field(
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        description="Vision model for image understanding"
    )
    
    available_vision_models: list = Field(
        default=[
            "Qwen/Qwen2.5-VL-7B-Instruct",
            "meta-llama/Llama-3.2-11B-Vision-Instruct",
            "microsoft/Phi-3.5-vision-instruct"
        ],
        description="Available vision models"
    )
    
    # ==================== Model Parameters ====================
    temperature: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="LLM temperature"
    )
    
    max_tokens: int = Field(
        default=2048,
        ge=100,
        le=4096,
        description="Max tokens for LLM response"
    )
    
    vision_max_tokens: int = Field(
        default=1024,
        ge=100,
        le=2048,
        description="Max tokens for vision model"
    )
    
    vision_temperature: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Temperature for vision model"
    )
    
    # ==================== Storage Configuration ====================
    upload_dir: Path = Field(default=Path("./uploads"))
    markdown_dir: Path = Field(default=Path("./markdown_output"))
    database_url: str = Field(default="sqlite:///./rag_system.db")
    model_cache_dir: Path = Field(
        default=Path("./model_cache"),
        description="Directory to cache downloaded models"
    )
    max_file_size_mb: int = Field(default=50, ge=1, le=500)
    max_filename_length: int = Field(default=255, ge=1, le=500)
    
    # ==================== CONCURRENCY SETTINGS (NEW) ====================
    
    # Worker Pool Settings
    max_io_workers: int = Field(
        default=20,
        ge=4,
        le=100,
        description="Maximum I/O worker threads (embeddings, API calls, database)"
    )
    
    max_cpu_workers: int = Field(
        default=4,
        ge=1,
        le=16,
        description="Maximum CPU worker processes (document extraction, chunking)"
    )
    
    # Database Connection Pool Settings
    db_pool_size: int = Field(
        default=10,
        ge=5,
        le=50,
        description="Number of database connections to keep open"
    )
    
    db_max_overflow: int = Field(
        default=20,
        ge=0,
        le=100,
        description="Additional connections when pool is exhausted"
    )
    
    db_pool_timeout: int = Field(
        default=30,
        ge=5,
        le=120,
        description="Seconds to wait for connection from pool"
    )
    
    db_pool_recycle: int = Field(
        default=3600,
        ge=300,
        le=7200,
        description="Recycle connections after N seconds (1 hour default)"
    )
    
    # Task Management
    task_cleanup_interval_hours: int = Field(
        default=24,
        ge=1,
        le=168,
        description="Clean up completed tasks older than N hours"
    )
    
    task_timeout_seconds: int = Field(
        default=600,
        ge=60,
        le=3600,
        description="Default timeout for background tasks (10 minutes)"
    )
    
    # ==================== QDRANT CLOUD CONFIGURATION ====================
    qdrant_url: str = Field(default="")
    qdrant_api_key: str = Field(default="")
    qdrant_collection_name: str = Field(default="rag_documents")
    
    # Qdrant Connection Pool
    qdrant_timeout: int = Field(
        default=30,
        ge=5,
        le=120,
        description="Qdrant request timeout in seconds"
    )
    
    qdrant_pool_size: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Qdrant connection pool size"
    )
    
    @field_validator('qdrant_url')
    @classmethod
    def validate_qdrant_url(cls, v: str) -> str:
        if v and not v.startswith(('http://', 'https://')):
            raise ValueError("qdrant_url must start with http:// or https://")
        return v
    
    # ==================== TOKEN-BASED CHUNKING CONSTRAINTS ====================
    min_chunk_tokens: int = Field(default=200, ge=50, le=500)
    target_chunk_tokens: int = Field(default=512, ge=200, le=2000)
    max_chunk_tokens: int = Field(default=600, ge=300, le=2000)
    chunk_overlap_tokens: int = Field(default=50, ge=0, le=200)
    
    tokenizer_model: str = Field(default="cl100k_base")
    
    # Deprecated character-based limits
    chunk_size: int = Field(default=1500, ge=100, le=8000)
    chunk_overlap: int = Field(default=200, ge=0, le=1000)
    max_chunk_size: int = Field(default=2000, ge=1000, le=10000)
    
    max_atomic_element_tokens: int = Field(default=1000, ge=500, le=4000)
    split_oversized_atomic: bool = Field(default=True)
    max_code_block_lines: int = Field(default=500, ge=100, le=2000)
    
    # ==================== Search Configuration ====================
    default_top_k: int = Field(default=5, ge=1, le=50)
    default_keyword_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    default_semantic_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    
    # ==================== Extraction Configuration ====================
    extract_images: bool = Field(default=True)
    infer_table_structure: bool = Field(default=True)
    enable_ocr: bool = Field(default=True)
    
    image_scale: float = Field(default=2.0, ge=1.0, le=4.0)
    image_quality: int = Field(default=85, ge=1, le=100)
    
    # ==================== AI Services Configuration ====================
    enable_ai_image_analysis: bool = Field(default=True)
    
    # ==================== Logging & Monitoring ====================
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO"
    )
    enable_query_logging: bool = Field(default=True)
    enable_performance_metrics: bool = Field(default=True)
    log_file_path: Path = Field(default=Path("./logs/rag_system.log"))
    
    # ==================== Error Handling ====================
    max_retries: int = Field(default=3, ge=0, le=10)
    retry_delay_seconds: float = Field(default=1.0, ge=0.1, le=60.0)
    
    # ==================== Feature Flags ====================
    enable_conversation_history: bool = Field(default=True)
    enable_hallucination_detection: bool = Field(default=False)
    hallucination_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    enable_conversation_history: bool = Field(default=True)
    enable_hallucination_detection: bool = Field(default=False)
    hallucination_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    
    # ==================== Security ====================
    allowed_file_extensions: set[str] = Field(
        default={
            'pdf', 'docx', 'doc', 'pptx', 'ppt',
            'txt', 'md', 'markdown',
            'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp', 'tiff'
        }
    )
    max_user_id_length: int = Field(default=100, ge=1, le=500)
    
    # Rate limiting (NEW)
    enable_rate_limiting: bool = Field(
        default=True,
        description="Enable per-user rate limiting"
    )
    
    rate_limit_requests_per_minute: int = Field(
        default=60,
        ge=10,
        le=1000,
        description="Max requests per user per minute"
    )
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._create_directories()
    
    def _create_directories(self) -> None:
        """Create necessary directories"""
        directories = [
            self.upload_dir,
            self.markdown_dir,
            self.model_cache_dir,
            self.log_file_path.parent,
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    @property
    def max_file_size_bytes(self) -> int:
        return self.max_file_size_mb * 1024 * 1024
    
    def validate_configuration(self) -> tuple[bool, list[str]]:
        """Validate configuration and return warnings"""
        warnings = []
        is_valid = True
        
        if not self.huggingface_api_token:
            warnings.append("HUGGINGFACE_API_TOKEN not set")
            is_valid = False
        
        if not self.llm_model_id:
            warnings.append("LLM_MODEL_ID not set")
            is_valid = False
        
        if not self.embedding_model_id:
            warnings.append("EMBEDDING_MODEL_ID not set")
            is_valid = False
        
        if not self.qdrant_url:
            warnings.append("QDRANT_URL not set")
        
        if not self.qdrant_api_key and self.qdrant_url:
            warnings.append("QDRANT_API_KEY not set but URL provided")
            is_valid = False
        
        return is_valid, warnings


# Initialize global settings
# Initialize global settings
settings = Settings()


def get_settings() -> Settings:
    """Dependency injection function"""
    return settings


def print_configuration_summary():
    """Print configuration summary on startup"""
    is_valid, warnings = settings.validate_configuration()
    
    print("\n" + "="*70)
    print("RAG SYSTEM CONFIGURATION - CONCURRENT PROCESSING ENABLED")
    print("="*70)
    
    print(f"\n🤖 Models:")
    print(f"  - Chat/LLM: {settings.llm_model_id}")
    print(f"  - Embedding: {settings.embedding_model_id}")
    print(f"  - Vision: {settings.vision_model_id}")
    
    print(f"\n⚡ Concurrency Settings:")
    print(f"  - I/O Workers: {settings.max_io_workers} threads")
    print(f"  - CPU Workers: {settings.max_cpu_workers} processes")
    print(f"  - DB Pool Size: {settings.db_pool_size}")
    print(f"  - DB Max Overflow: {settings.db_max_overflow}")
    print(f"  - Total DB Connections: {settings.db_pool_size + settings.db_max_overflow}")
    
    print(f"\n📦 Chunking:")
    print(f"  - Target Tokens: {settings.target_chunk_tokens}")
    print(f"  - Overlap: {settings.chunk_overlap_tokens} tokens")
    
    print(f"\n💾 Storage:")
    print(f"  - Database: {settings.database_url}")
    print(f"  - Vector DB: {'Qdrant Cloud ✓' if settings.qdrant_url else 'Not configured'}")
    
    print(f"\n🔧 Features:")
    print(f"  - AI Image Analysis: {'✅' if settings.enable_ai_image_analysis else '❌'}")
    print(f"  - Rate Limiting: {'✅' if settings.enable_rate_limiting else '❌'}")
    print(f"  - Performance Metrics: {'✅' if settings.enable_performance_metrics else '❌'}")
    
    if warnings:
        print(f"\n⚠️  Warnings ({len(warnings)}):")
        for warning in warnings:
            print(f"  - {warning}")
    else:
        print("\n✅ Configuration valid - no warnings")
    
    if not is_valid:
        print("\n❌ CRITICAL: Configuration is invalid!")
    
    print("="*70 + "\n")
    print("="*70 + "\n")


# Run validation on import
print_configuration_summary()