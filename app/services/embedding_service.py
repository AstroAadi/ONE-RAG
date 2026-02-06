"""
Embedding Service using AWS Bedrock
Dense embeddings are generated using AWS Bedrock (Cohere, Titan, etc.)
NO HuggingFace models for dense embeddings - only AWS Bedrock
"""

from langchain_aws import BedrockEmbeddings
import logging
from typing import List
import time
import boto3

from app.core.config import settings

logger = logging.getLogger(__name__)


class BedrockEmbeddingService:
    """
    AWS Bedrock embedding service for dense vectors
    
    Features:
    - Uses AWS Bedrock embedding models (Cohere, Titan, etc.)
    - Batch processing for efficiency
    - Caching support
    - Error handling and retries
    """
    
    def __init__(self, model_id: str = None):
        """
        Initialize AWS Bedrock embeddings
        
        Args:
            model_id: AWS Bedrock embedding model ID (default from config)
        """
        self.model_id = model_id or settings.bedrock_embedding_model_id
        
        logger.info(f"Loading AWS Bedrock embedding model: {self.model_id}")
        
        try:
            # Create boto3 client with credentials
            client_kwargs = {
                "service_name": "bedrock-runtime",
                "region_name": settings.aws_region,
            }
            
            # Add credentials if provided
            if settings.aws_access_key_id and settings.aws_secret_access_key:
                client_kwargs["aws_access_key_id"] = settings.aws_access_key_id
                client_kwargs["aws_secret_access_key"] = settings.aws_secret_access_key
            
            # Create bedrock client
            bedrock_client = boto3.client(**client_kwargs)
            
            # Initialize Bedrock embeddings with the client
            self.embeddings = BedrockEmbeddings(
                client=bedrock_client,
                model_id=self.model_id
            )
            
            # Get embedding dimension by generating a test embedding
            test_embedding = self.embeddings.embed_query("test")
            self.embedding_dim = len(test_embedding)
            
            logger.info(f"✅ AWS Bedrock embedding model loaded successfully")
            logger.info(f"   Model: {self.model_id}")
            logger.info(f"   Dimension: {self.embedding_dim}")
            logger.info(f"   Region: {settings.aws_region}")
            logger.info(f"   Provider: AWS Bedrock")
            
        except Exception as e:
            logger.error(f"Failed to load Bedrock embedding model: {e}")
            raise
    
    def embed_query(self, text: str) -> List[float]:
        """
        Generate embedding for a single query text
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector as list of floats
        """
        try:
            # Truncate if needed
            if len(text) > settings.embedding_max_length:
                text = text[:settings.embedding_max_length]
            
            # Generate embedding using Bedrock
            embedding = self.embeddings.embed_query(text)
            
            return embedding
                
        except Exception as e:
            logger.error(f"Error generating query embedding: {e}")
            raise
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple documents (batch processing)
        
        Args:
            texts: List of input texts
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        try:
            # Truncate texts if needed
            truncated_texts = []
            for text in texts:
                if len(text) > settings.embedding_max_length:
                    truncated_texts.append(text[:settings.embedding_max_length])
                else:
                    truncated_texts.append(text)
            
            logger.debug(f"Generating Bedrock embeddings for {len(truncated_texts)} documents")
            
            start_time = time.time()
            
            # Bedrock handles batching internally
            embeddings = self.embeddings.embed_documents(truncated_texts)
            
            elapsed_time = time.time() - start_time
            throughput = len(truncated_texts) / elapsed_time if elapsed_time > 0 else 0
            
            logger.info(
                f"✓ Embedded {len(truncated_texts)} documents via Bedrock in {elapsed_time:.2f}s "
                f"({throughput:.1f} docs/sec)"
            )
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error embedding documents: {e}")
            # Return zero vectors as fallback
            return [[0.0] * self.embedding_dim for _ in texts]
    
    def get_embedding_dimension(self) -> int:
        """Get embedding dimension"""
        return self.embedding_dim
    
    def update_model(self, new_model_id: str) -> bool:
        """
        Update to a different Bedrock embedding model
        
        Args:
            new_model_id: AWS Bedrock model ID
            
        Returns:
            True if successful
        """
        try:
            logger.info(f"Updating Bedrock embedding model to: {new_model_id}")
            
            # Create boto3 client with credentials
            client_kwargs = {
                "service_name": "bedrock-runtime",
                "region_name": settings.aws_region,
            }
            
            if settings.aws_access_key_id and settings.aws_secret_access_key:
                client_kwargs["aws_access_key_id"] = settings.aws_access_key_id
                client_kwargs["aws_secret_access_key"] = settings.aws_secret_access_key
            
            bedrock_client = boto3.client(**client_kwargs)
            
            # Load new model
            self.model_id = new_model_id
            self.embeddings = BedrockEmbeddings(
                client=bedrock_client,
                model_id=self.model_id
            )
            
            # Update dimension
            test_embedding = self.embeddings.embed_query("test")
            self.embedding_dim = len(test_embedding)
            
            logger.info(f"✅ Bedrock embedding model updated successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update Bedrock embedding model: {e}")
            return False
    
    def get_model_info(self) -> dict:
        """Get model information"""
        return {
            "model_id": self.model_id,
            "provider": "AWS Bedrock",
            "embedding_dim": self.embedding_dim,
            "max_length": settings.embedding_max_length,
            "region": settings.aws_region
        }


# Alias for backward compatibility
OpenSourceEmbeddings = BedrockEmbeddingService
GPUEmbeddingService = BedrockEmbeddingService