"""
Fixed version with proper model separation:
- AWS Bedrock: Dense embeddings + LLM
- HuggingFace: SPLADE sparse vectors ONLY
- PostgreSQL Database Support
"""

from pydantic import Field, field_validator, ConfigDict
from pydantic_settings import BaseSettings
from pathlib import Path
from typing import Literal, Optional
from dotenv import load_dotenv
import os
import logging

load_dotenv()

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Application settings with GPU support and PostgreSQL"""
    
    model_config = ConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="allow",
        validate_assignment=True,
        arbitrary_types_allowed=True
        
    )
    
    # ==================== GPU/DEVICE CONFIGURATION ====================
    
    use_gpu: bool = Field(
        default=True,
        description="Use GPU if available (auto-detects)"
    )
    
    gpu_device_id: int = Field(
        default=0,
        ge=0,
        le=7,
        description="GPU device ID to use"
    )
    
    gpu_memory_fraction: float = Field(
        default=0.9,
        ge=0.1,
        le=1.0,
        description="Fraction of GPU memory to use"
    )
    
    # Batch sizes for GPU/CPU
    splade_batch_size_gpu: int = Field(default=64, ge=1, le=128)
    splade_batch_size_cpu: int = Field(default=4, ge=1, le=16)
    
    use_mixed_precision: bool = Field(default=True)
    
    # GPU optimizations
    enable_gpu_prefetch: bool = Field(default=True)
    pin_memory: bool = Field(default=True)
    
    # ==================== AWS Bedrock Configuration ====================
    
    aws_region: str = Field(default="us-east-1")
    aws_access_key_id: str = Field(default="")
    aws_secret_access_key: str = Field(default="")
    aws_profile_name: str = Field(default="")
    
    # ==================== Model Configuration ====================
    
    # AWS Bedrock LLM Model
    bedrock_model_id: str = Field(
        default="anthropic.claude-3-5-sonnet-20241022",
        description="AWS Bedrock LLM model ID"
    )
    
    # AWS Bedrock Dense Embedding Model
    bedrock_embedding_model_id: str = Field(
        default="cohere.embed-english-v3",
        description="AWS Bedrock embedding model ID for dense vectors"
    )
    
    # HuggingFace SPLADE Model (Sparse Vectors ONLY)
    splade_model_name: str = Field(
        default="naver/splade-cocondenser-selfdistil",
        description="HuggingFace SPLADE model for sparse vectors",
        token=os.getenv('HUGGINGFACE_API_TOKEN')
    )
    
    embedding_max_length: int = Field(default=512, ge=128, le=2048)
    
    # Bedrock Vision Model (if needed)
    bedrock_vision_model_id: str = Field(
        default="us.anthropic.claude-3-5-sonnet-20241022-v2:0"
    )
    
    # LLM Parameters
    temperature: float = Field(default=0.6, ge=0.0, le=1.0)
    max_tokens: int = Field(default=4000, ge=100, le=100000)
    vision_max_tokens: int = Field(default=2000, ge=100, le=10000)
    vision_temperature: float = Field(default=0.6, ge=0.0, le=1.0)
    
    # ==================== DATABASE CONFIGURATION ====================
    
    # Database URL - supports both SQLite and PostgreSQL
    database_url: str = Field(
        default="postgresql+psycopg2://postgres:aadi@localhost:5432/oneragdb",
        description="Database URL (SQLite or PostgreSQL)"
    )
    
    # PostgreSQL-specific settings
    db_pool_size: int = Field(default=10, ge=5, le=50)
    db_max_overflow: int = Field(default=20, ge=0, le=100)
    db_pool_timeout: int = Field(default=30, ge=5, le=120)
    db_pool_recycle: int = Field(default=3600, ge=300, le=7200)
    db_echo: bool = Field(default=False, description="Echo SQL queries (debug mode)")
    
    @field_validator('database_url')
    @classmethod
    def validate_database_url(cls, v: str) -> str:
        """Validate database URL format"""
        if not v:
            raise ValueError("database_url cannot be empty")
        
        valid_prefixes = ('sqlite:///', 'postgresql://', 'postgresql+psycopg2://')
        if not any(v.startswith(prefix) for prefix in valid_prefixes):
            raise ValueError(
                f"database_url must start with one of: {', '.join(valid_prefixes)}"
            )
        return v
    
    # ==================== Storage Configuration ====================
    
    upload_dir: Path = Field(default=Path("./uploads"))
    markdown_dir: Path = Field(default=Path("./markdown_output"))
    
    max_file_size_mb: int = Field(default=50, ge=1, le=500)
    max_filename_length: int = Field(default=255, ge=1, le=500)
    
    # ==================== CONCURRENCY SETTINGS ====================
    
    max_io_workers: int = Field(default=20, ge=4, le=100)
    max_cpu_workers: int = Field(default=4, ge=1, le=16)
    
    task_cleanup_interval_hours: int = Field(default=24, ge=1, le=168)
    task_timeout_seconds: int = Field(default=600, ge=60, le=3600)
    
    # ==================== QDRANT CONFIGURATION ====================
    
    qdrant_url: str = Field(default="")
    qdrant_api_key: str = Field(default="")
    qdrant_collection_name: str = Field(default="rag_documents")
    qdrant_timeout: int = Field(default=30, ge=5, le=120)
    qdrant_pool_size: int = Field(default=10, ge=1, le=50)
    
    @field_validator('qdrant_url')
    @classmethod
    def validate_qdrant_url(cls, v: str) -> str:
        if v and not v.startswith(('http://', 'https://')):
            raise ValueError("qdrant_url must start with http:// or https://")
        return v
    
    # ==================== CHUNKING ====================
    
    min_chunk_tokens: int = Field(default=200, ge=50, le=500)
    target_chunk_tokens: int = Field(default=512, ge=200, le=2000)
    max_chunk_tokens: int = Field(default=600, ge=300, le=2000)
    chunk_overlap_tokens: int = Field(default=50, ge=0, le=200)
    tokenizer_model: str = Field(default="cl100k_base")
    
    chunk_size: int = Field(default=1500, ge=100, le=8000)
    chunk_overlap: int = Field(default=200, ge=0, le=1000)
    max_chunk_size: int = Field(default=2000, ge=1000, le=10000)
    max_atomic_element_tokens: int = Field(default=1000, ge=500, le=4000)
    split_oversized_atomic: bool = Field(default=True)
    max_code_block_lines: int = Field(default=500, ge=100, le=2000)
    
    # ==================== SEARCH ====================
    
    default_top_k: int = Field(default=5, ge=1, le=50)
    default_keyword_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    default_semantic_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    
    # ==================== EXTRACTION ====================
    
    extract_images: bool = Field(default=True)
    infer_table_structure: bool = Field(default=True)
    enable_ocr: bool = Field(default=True)
    image_scale: float = Field(default=2.0, ge=1.0, le=4.0)
    image_quality: int = Field(default=85, ge=1, le=100)
    enable_ai_image_analysis: bool = Field(default=True)
    
    # ==================== LOGGING ====================
    
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(default="INFO")
    enable_query_logging: bool = Field(default=True)
    enable_performance_metrics: bool = Field(default=True)
    log_file_path: Path = Field(default=Path("./logs/rag_system.log"))
    
    # ==================== ERROR HANDLING ====================
    
    max_retries: int = Field(default=3, ge=0, le=10)
    retry_delay_seconds: float = Field(default=1.0, ge=0.1, le=60.0)
    
    # ==================== FEATURES ====================
    
    enable_conversation_history: bool = Field(default=True)
    enable_hallucination_detection: bool = Field(default=False)
    hallucination_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    
    # ==================== SECURITY ====================
    
    allowed_file_extensions: set[str] = Field(
        default={'pdf', 'docx', 'doc', 'pptx', 'ppt', 'txt', 'md', 'markdown',
                'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp', 'tiff'}
    )
    max_user_id_length: int = Field(default=100, ge=1, le=500)
    enable_rate_limiting: bool = Field(default=True)
    rate_limit_requests_per_minute: int = Field(default=60, ge=10, le=1000)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._create_directories()
        self._init_device()
    
    def _create_directories(self) -> None:
        """Create necessary directories"""
        for directory in [self.upload_dir, self.markdown_dir, self.log_file_path.parent]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def _init_device(self) -> None:
        """Initialize device settings"""
        try:
            import torch
            
            if self.use_gpu and torch.cuda.is_available():
                # GPU available
                self.device = f"cuda:{self.gpu_device_id}"
                self.is_gpu_available = True
                self.splade_batch_size = self.splade_batch_size_gpu
                
                gpu_name = torch.cuda.get_device_name(self.gpu_device_id)
                gpu_memory = torch.cuda.get_device_properties(self.gpu_device_id).total_memory / 1024**3
                
                logger.info(f"🎮 GPU DETECTED: {gpu_name}")
                logger.info(f"   Total Memory: {gpu_memory:.2f} GB")
                logger.info(f"   Using Device: {self.device}")
                
                if self.gpu_memory_fraction < 1.0:
                    torch.cuda.set_per_process_memory_fraction(
                        self.gpu_memory_fraction, self.gpu_device_id
                    )
                    logger.info(f"   Memory Limit: {gpu_memory * self.gpu_memory_fraction:.2f} GB")
            else:
                # CPU fallback
                self.device = "cpu"
                self.is_gpu_available = False
                self.splade_batch_size = self.splade_batch_size_cpu
                
                if self.use_gpu:
                    logger.warning("⚠️  GPU requested but not available, using CPU")
                else:
                    logger.info("💻 Using CPU for computation")
                    
        except ImportError:
            # PyTorch not installed
            logger.warning("⚠️  PyTorch not installed, using CPU")
            self.device = "cpu"
            self.is_gpu_available = False
            self.splade_batch_size = self.splade_batch_size_cpu
    
    @property
    def max_file_size_bytes(self) -> int:
        """Get max file size in bytes"""
        return self.max_file_size_mb * 1024 * 1024
    
    @property
    def is_postgresql(self) -> bool:
        """Check if database is PostgreSQL"""
        return self.database_url.startswith(('postgresql://', 'postgresql+psycopg2://'))
    
    @property
    def is_sqlite(self) -> bool:
        """Check if database is SQLite"""
        return self.database_url.startswith('sqlite:///')
    
    def get_device_info(self) -> dict:
        """Get current device information"""
        info = {
            "device": getattr(self, 'device', 'cpu'),
            "is_gpu": getattr(self, 'is_gpu_available', False),
            "splade_batch_size": getattr(self, 'splade_batch_size', self.splade_batch_size_cpu)
        }
        
        if getattr(self, 'is_gpu_available', False):
            try:
                import torch
                info.update({
                    "gpu_name": torch.cuda.get_device_name(self.gpu_device_id),
                    "gpu_memory_total_gb": torch.cuda.get_device_properties(self.gpu_device_id).total_memory / 1024**3,
                    "gpu_memory_allocated_gb": torch.cuda.memory_allocated(self.gpu_device_id) / 1024**3,
                    "mixed_precision": self.use_mixed_precision
                })
            except:
                pass
        
        return info
    
    def validate_configuration(self) -> tuple[bool, list[str]]:
        """Validate configuration and return warnings"""
        warnings = []
        is_valid = True
        
        # Database validation
        if self.is_postgresql:
            try:
                import psycopg2
            except ImportError:
                warnings.append("PostgreSQL selected but psycopg2 not installed. Run: pip install psycopg2-binary")
                is_valid = False
        
        # AWS Bedrock validation
        if not self.aws_access_key_id or not self.aws_secret_access_key:
            if not self.aws_profile_name:
                warnings.append("AWS credentials not configured")
                is_valid = False
        
        if not self.bedrock_model_id:
            warnings.append("BEDROCK_MODEL_ID not set")
            is_valid = False
        
        if not self.bedrock_embedding_model_id:
            warnings.append("BEDROCK_EMBEDDING_MODEL_ID not set")
            is_valid = False
        
        # Qdrant validation
        if not self.qdrant_url:
            warnings.append("QDRANT_URL not set")
        
        if not self.qdrant_api_key and self.qdrant_url:
            warnings.append("QDRANT_API_KEY not set but URL provided")
            is_valid = False
        
        # GPU validation
        if self.use_gpu:
            try:
                import torch
                if not torch.cuda.is_available():
                    warnings.append("GPU requested but CUDA not available")
            except ImportError:
                warnings.append("GPU requested but PyTorch not installed")
        
        # Resource validation
        if self.max_cpu_workers > os.cpu_count():
            warnings.append(f"max_cpu_workers exceeds CPU count")
        
        total_connections = self.db_pool_size + self.db_max_overflow
        if total_connections > 100:
            warnings.append(f"Total database connections may be excessive")
        
        return is_valid, warnings


# Initialize settings
settings = Settings()


def get_settings() -> Settings:
    """Dependency injection function"""
    return settings


def print_configuration_summary():
    """Print configuration summary on startup"""
    is_valid, warnings = settings.validate_configuration()
    
    print("\n" + "="*70)
    print("RAG SYSTEM CONFIGURATION")
    print("="*70)
    
    # Database info
    db_type = "PostgreSQL" if settings.is_postgresql else "SQLite"
    print(f"\n💾 Database: {db_type}")
    if settings.is_postgresql:
        # Extract database name from URL for display
        db_name = settings.database_url.split('/')[-1].split('?')[0]
        print(f"  Database: {db_name}")
        print(f"  Pool Size: {settings.db_pool_size}")
        print(f"  Max Overflow: {settings.db_max_overflow}")
    
    device_info = settings.get_device_info()
    print(f"\n🖥️  Device: {device_info['device'].upper()}")
    
    if device_info['is_gpu']:
        print(f"  GPU: {device_info.get('gpu_name', 'Unknown')}")
        print(f"  Memory: {device_info.get('gpu_memory_total_gb', 0):.2f} GB")
        print(f"  Mixed Precision: {'✅' if settings.use_mixed_precision else '❌'}")
        print(f"  SPLADE Batch: {device_info['splade_batch_size']}")
    else:
        print(f"  CPU Cores: {os.cpu_count()}")
        print(f"  SPLADE Batch: {device_info['splade_batch_size']}")
    
    print(f"\n📊 Models:")
    print(f"  LLM: {settings.bedrock_model_id} (AWS Bedrock)")
    print(f"  Dense Embedding: {settings.bedrock_embedding_model_id} (AWS Bedrock)")
    print(f"  Sparse Embedding: {settings.splade_model_name} (HuggingFace)")
    
    print(f"\n⚡ Concurrency:")
    print(f"  I/O Workers: {settings.max_io_workers}")
    print(f"  CPU Workers: {settings.max_cpu_workers}")
    
    if warnings:
        print(f"\n⚠️  Warnings:")
        for warning in warnings:
            print(f"  - {warning}")
    else:
        print("\n✅ Configuration valid")
    
    print("="*70 + "\n")


# Run on import
print_configuration_summary()