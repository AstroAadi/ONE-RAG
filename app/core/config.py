from pydantic_settings import BaseSettings
from pathlib import Path
from dotenv import load_dotenv
import os
load_dotenv()

class Settings(BaseSettings):
    # AWS Bedrock Configuration
    aws_region: str = os.getenv("AWS_REGION")

    aws_access_key_id: str = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_access_key: str = os.getenv("AWS_SECRET_ACCESS_KEY")
    
    # Model Configuration
    bedrock_llm_model_id: str = "anthropic.claude-3-5-sonnet-20240620-v1:0"
    bedrock_embedding_model_id: str = "cohere.embed-english-v3"
    
    # Application Configuration
    upload_dir: str = "./uploads"
    markdown_dir: str = "./markdown_output"
    max_file_size_mb: int = 50
    
    # ChromaDB Configuration
    chroma_persist_dir: str = "./chroma_db"
    
    # Chunking Configuration
    chunk_size: int = 1500
    chunk_overlap: int = 200
    
    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()

# Create necessary directories
Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)
Path(settings.markdown_dir).mkdir(parents=True, exist_ok=True)
Path(settings.chroma_persist_dir).mkdir(parents=True, exist_ok=True)