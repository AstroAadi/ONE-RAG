"""
Embedding Service - Using LangChain with HuggingFace
Works reliably with HuggingFace Inference Client
"""

from langchain_huggingface import HuggingFaceEndpointEmbeddings
from huggingface_hub import InferenceClient
import logging
from typing import List
import time
import numpy as np

from app.core.config import settings

logger = logging.getLogger(__name__)


class OpenSourceEmbeddings:
    """
    Open source embeddings using LangChain + HuggingFace
    
    Uses InferenceClient for reliable feature extraction
    """
    
    def __init__(self, model_id: str = None):
        """Initialize embeddings service"""
        self.model_id = model_id or settings.embedding_model_id
        self.api_token = settings.huggingface_api_token
        
        # Initialize HuggingFace InferenceClient (more reliable)
        self.client = InferenceClient(token=self.api_token)
        
        # Also initialize LangChain embeddings for compatibility
        try:
            self.langchain_embeddings = HuggingFaceEndpointEmbeddings(
                model=self.model_id,
                huggingfacehub_api_token=self.api_token
            )
        except Exception as e:
            logger.warning(f"LangChain embeddings init failed, using InferenceClient only: {e}")
            self.langchain_embeddings = None
        
        self.max_retries = 3
        self.retry_delay = 2
        
        logger.info(f"✅ Embedding service initialized: {self.model_id}")
    
    def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        try:
            # Truncate if needed
            if len(text) > settings.embedding_max_length:
                text = text[:settings.embedding_max_length]
            
            # Use InferenceClient for reliability
            embedding = self._embed_with_client(text)
            
            return embedding
                
        except Exception as e:
            logger.error(f"Error generating query embedding: {e}")
            # Try LangChain as fallback
            if self.langchain_embeddings:
                try:
                    return self.langchain_embeddings.embed_query(text)
                except:
                    pass
            raise
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        embeddings = []
        
        for i, text in enumerate(texts):
            try:
                # Truncate if needed
                if len(text) > settings.embedding_max_length:
                    text = text[:settings.embedding_max_length]
                
                embedding = self._embed_with_client(text)
                embeddings.append(embedding)
                
                # Log progress
                if (i + 1) % 10 == 0:
                    logger.debug(f"Embedded {i + 1}/{len(texts)} documents")
                
                # Small delay to avoid rate limits
                if i < len(texts) - 1:
                    time.sleep(0.1)
                    
            except Exception as e:
                logger.error(f"Error embedding document {i}: {e}")
                # Return zero vector as fallback
                embeddings.append([0.0] * self._get_embedding_dimension())
        
        return embeddings
    
    def _embed_with_client(self, text: str, retry_count: int = 0) -> List[float]:
        """Generate embedding using InferenceClient"""
        try:
            # Use feature_extraction which works reliably
            result = self.client.feature_extraction(
                text=text,
                model=self.model_id
            )
            
            # ✅ FIX: Handle numpy arrays, lists, and nested structures
            # Convert numpy array to list if needed
            if isinstance(result, np.ndarray):
                # Flatten if multi-dimensional
                if result.ndim > 1:
                    result = result.flatten()
                return result.tolist()
            
            # Handle lists
            elif isinstance(result, list):
                # If it's a nested list (batch), take first
                if result and isinstance(result[0], (list, np.ndarray)):
                    first_item = result[0]
                    # Convert numpy array to list if needed
                    if isinstance(first_item, np.ndarray):
                        return first_item.flatten().tolist()
                    return first_item
                return result
            
            else:
                raise ValueError(f"Unexpected embedding format: {type(result)}")
                
        except Exception as e:
            if retry_count < self.max_retries:
                logger.warning(f"Embedding failed, retrying ({retry_count + 1}/{self.max_retries})...")
                time.sleep(self.retry_delay * (retry_count + 1))
                return self._embed_with_client(text, retry_count + 1)
            else:
                raise Exception(f"Embedding failed after {self.max_retries} retries: {str(e)}")
    
    def get_embedding_dimension(self) -> int:
        """Public API for current embedding dimension."""
        return self._get_embedding_dimension()

    def _get_embedding_dimension(self) -> int:
        """Get embedding dimension for the current model"""
        dimensions = {
            "BAAI/bge-large-en-v1.5": 1024,
            "sentence-transformers/all-MiniLM-L6-v2": 384,
            "intfloat/e5-large-v2": 1024,
        }
        
        return dimensions.get(self.model_id, 768)
    
    def update_model(self, new_model_id: str) -> bool:
        """Update to a different embedding model"""
        try:
            self.model_id = new_model_id
            
            # Reinitialize client
            self.client = InferenceClient(token=self.api_token)
            
            # Reinitialize LangChain embeddings
            try:
                self.langchain_embeddings = HuggingFaceEndpointEmbeddings(
                    model=self.model_id,
                    huggingfacehub_api_token=self.api_token
                )
            except Exception as e:
                logger.warning(f"LangChain embeddings reinit failed: {e}")
                self.langchain_embeddings = None
            
            logger.info(f"✅ Embedding model updated to: {new_model_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to update embedding model: {e}")
            return False