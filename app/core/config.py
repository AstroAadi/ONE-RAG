"""
Centralized Configuration for RAG System - OPEN SOURCE MODELS
Using HuggingFace with LangChain integration
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
    """Application settings with validation"""
    
    model_config = ConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore",
        validate_assignment=True
    )
    
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
    
    # ==================== Chat/LLM Model Configuration ====================
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
    upload_dir: Path = Field(default=Path("./uploads"), description="Upload directory")
    markdown_dir: Path = Field(default=Path("./markdown_output"), description="Markdown output directory")
    model_cache_dir: Path = Field(
        default=Path("./model_cache"),
        description="Directory to cache downloaded models"
    )
    database_url: str = Field(
        default="sqlite:///./rag_system.db",
        description="SQLite database URL"
    )
    
    # File Upload Limits
    max_file_size_mb: int = Field(default=50, ge=1, le=500)
    max_filename_length: int = Field(default=255, ge=1, le=500)
    
    # ==================== QDRANT CLOUD CONFIGURATION ====================
    qdrant_url: str = Field(
        default="",
        description="Qdrant Cloud cluster URL"
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
        if v and not v.startswith(('http://', 'https://')):
            raise ValueError("qdrant_url must start with http:// or https://")
        return v
    
    # ==================== TOKEN-BASED CHUNKING ====================
    min_chunk_tokens: int = Field(default=200, ge=50, le=500)
    target_chunk_tokens: int = Field(default=512, ge=200, le=2000)
    max_chunk_tokens: int = Field(default=600, ge=300, le=2000)
    chunk_overlap_tokens: int = Field(default=50, ge=0, le=200)
    tokenizer_model: str = Field(default="cl100k_base")
    
    # DEPRECATED character-based limits
    chunk_size: int = Field(default=1500, ge=100, le=8000)
    chunk_overlap: int = Field(default=200, ge=0, le=1000)
    max_chunk_size: int = Field(default=2000, ge=1000, le=10000)
    
    @field_validator('max_chunk_tokens')
    @classmethod
    def validate_token_limits(cls, v: int, info) -> int:
        target = info.data.get('target_chunk_tokens', 512)
        min_tokens = info.data.get('min_chunk_tokens', 200)
        
        if v < target:
            raise ValueError(f"max_chunk_tokens ({v}) must be >= target_chunk_tokens ({target})")
        if target < min_tokens:
            raise ValueError(f"target_chunk_tokens ({target}) must be >= min_chunk_tokens ({min_tokens})")
        
        return v
    
    max_atomic_element_tokens: int = Field(default=1000, ge=500, le=4000)
    split_oversized_atomic: bool = Field(default=True)
    max_code_block_lines: int = Field(default=500, ge=100, le=2000)
    
    # ==================== RAG Configuration ====================
    default_top_k: int = Field(default=5, ge=1, le=50)
    max_top_k: int = Field(default=20, ge=1, le=100)
    max_context_length: int = Field(default=4096, ge=1000, le=32000)
    
    # ==================== Conversation Configuration ====================
    max_conversation_turns: int = Field(default=5, ge=1, le=20)
    conversation_cleanup_minutes: int = Field(default=60, ge=1, le=1440)
    
    # ==================== Document Processing ====================
    enable_ai_image_analysis: bool = Field(default=True)
    extract_images: bool = Field(default=True)
    infer_table_structure: bool = Field(default=True)
    enable_ocr: bool = Field(default=True)
    
    image_scale: float = Field(default=2.0, ge=1.0, le=4.0)
    image_quality: int = Field(default=85, ge=1, le=100)
    
    # ==================== Logging & Monitoring ====================
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(default="INFO")
    enable_query_logging: bool = Field(default=True)
    enable_performance_metrics: bool = Field(default=True)
    log_file_path: Path = Field(default=Path("./logs/rag_system.log"))
    
    # ==================== Error Handling ====================
    max_retries: int = Field(default=3, ge=0, le=10)
    retry_delay_seconds: float = Field(default=2.0, ge=0.1, le=60.0)
    
    # ==================== Feature Flags ====================
    enable_conversation_history: bool = Field(default=True)
    enable_hallucination_detection: bool = Field(default=False)
    hallucination_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    
    # ==================== Security Configuration ====================
    allowed_file_extensions: set[str] = Field(
        default={
            'pdf', 'docx', 'doc', 'pptx', 'ppt',
            'txt', 'md', 'markdown',
            'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp', 'tiff'
        }
    )
    max_user_id_length: int = Field(default=100, ge=1, le=500)
    
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
settings = Settings()


def get_settings() -> Settings:
    """Dependency injection function"""
    return settings


def print_configuration_summary():
    """Print configuration summary on startup"""
    is_valid, warnings = settings.validate_configuration()
    
    print("\n" + "="*70)
    print("RAG SYSTEM CONFIGURATION - OPEN SOURCE MODELS (LANGCHAIN)")
    print("="*70)
    
    print(f"\n🤖 Models:")
    print(f"  - Chat/LLM: {settings.llm_model_id}")
    print(f"  - Embedding: {settings.embedding_model_id}")
    print(f"  - Vision: {settings.vision_model_id}")
    print(f"  - API: HuggingFace Router (LangChain compatible)")
    
    print(f"\n💾 Storage:")
    print(f"  - Database: {settings.database_url}")
    print(f"  - Vector DB: {'Qdrant Cloud ✓' if settings.qdrant_url else 'Not configured'}")
    
    if warnings:
        print(f"\n⚠️  Warnings ({len(warnings)}):")
        for warning in warnings:
            print(f"  - {warning}")
    else:
        print("\n✅ Configuration valid - no warnings")
    
    if not is_valid:
        print("\n❌ CRITICAL: Configuration is invalid!")
    
    print("="*70 + "\n")


# Run validation on import
print_configuration_summary()