"""
SPLADE Sparse Embedding Service
Uses HuggingFace SPLADE model for sparse vector generation
"""

import torch
import logging
from typing import List, Union
from transformers import AutoModelForMaskedLM, AutoTokenizer
from qdrant_client.models import SparseVector
import time

from app.core.config import settings

logger = logging.getLogger(__name__)


class GPUSPLADEEmbedder:
    """
    GPU-accelerated SPLADE embedder for sparse vectors
    
    Features:
    - Automatic GPU/CPU detection
    - Batch processing
    - Mixed precision support
    - Memory-efficient inference
    """
    
    def __init__(self, model_name: str = "naver/splade-v3"):
        """
        Initialize SPLADE model
        
        Args:
            model_name: HuggingFace SPLADE model name
        """
        self.model_name = model_name
        self.device = settings.device
        self.is_gpu = settings.is_gpu_available
        self.batch_size = settings.splade_batch_size
        self.use_mixed_precision = settings.use_mixed_precision and self.is_gpu
        
        logger.info(f"Loading SPLADE model: {self.model_name}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Batch size: {self.batch_size}")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                use_fast=True
            )
            
            # Load model
            self.model = AutoModelForMaskedLM.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            
            # Enable half precision if GPU
            if self.use_mixed_precision:
                self.model.half()
                logger.info("✓ Mixed precision (FP16) enabled for SPLADE")
            
            # Enable GPU optimizations
            if self.is_gpu:
                torch.backends.cudnn.benchmark = True
                logger.info("✓ cuDNN auto-tuner enabled for SPLADE")
            
            logger.info(f"✅ SPLADE model loaded successfully")
            logger.info(f"   Model: {self.model_name}")
            logger.info(f"   Device: {self.device}")
            logger.info(f"   Batch Size: {self.batch_size}")
            
            # Warm up
            if self.is_gpu:
                self._warmup()
                
        except Exception as e:
            logger.error(f"Failed to load SPLADE model: {e}")
            raise
    
    def _warmup(self):
        """Warm up GPU with sample batch"""
        try:
            logger.info("⚡ Warming up SPLADE GPU...")
            dummy_texts = ["SPLADE warmup text"] * min(self.batch_size, 16)
            _ = self.embed_documents(dummy_texts)
            
            if self.is_gpu:
                torch.cuda.synchronize()
            
            logger.info(f"✓ SPLADE GPU warmup complete")
        except Exception as e:
            logger.warning(f"SPLADE warmup failed: {e}")
    
    def _compute_sparse_vector(self, logits: torch.Tensor, attention_mask: torch.Tensor) -> SparseVector:
        """
        Compute SPLADE sparse vector from model logits
        
        Args:
            logits: Model output logits [batch_size, seq_len, vocab_size]
            attention_mask: Attention mask [batch_size, seq_len]
        
        Returns:
            SparseVector with non-zero indices and values
        """
        # Apply ReLU and log(1+x) activation
        relu_log = torch.log(1 + torch.relu(logits))
        
        # Apply attention mask
        weighted_log = relu_log * attention_mask.unsqueeze(-1)
        
        # Max pooling over sequence dimension
        max_val, _ = torch.max(weighted_log, dim=1)
        
        # Convert to sparse representation
        # Only keep non-zero values
        vec = max_val.squeeze()
        
        # Get indices and values for non-zero elements
        indices = torch.nonzero(vec).squeeze().cpu().tolist()
        values = vec[vec > 0].cpu().tolist()
        
        # Handle single value case
        if isinstance(indices, int):
            indices = [indices]
        if isinstance(values, float):
            values = [values]
        
        # Return as SparseVector
        return SparseVector(
            indices=indices if indices else [0],
            values=values if values else [0.0]
        )
    
    def embed_query(self, text: str) -> SparseVector:
        """
        Generate sparse vector for a single query
        
        Args:
            text: Input text
        
        Returns:
            SparseVector
        """
        return self.embed_documents([text])[0]
    
    def embed_documents(self, texts: List[str]) -> List[SparseVector]:
        """
        Generate sparse vectors for multiple documents
        
        Args:
            texts: List of input texts
        
        Returns:
            List of SparseVectors
        """
        if not texts:
            return []
        
        try:
            logger.debug(f"Generating SPLADE embeddings for {len(texts)} texts in batches of {self.batch_size}")
            
            start_time = time.time()
            all_sparse_vectors = []
            
            # Process in batches
            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i:i + self.batch_size]
                
                # Tokenize
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                )
                
                # Move to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Generate embeddings
                with torch.no_grad():
                    if self.use_mixed_precision:
                        with torch.cuda.amp.autocast():
                            outputs = self.model(**inputs)
                    else:
                        outputs = self.model(**inputs)
                
                logits = outputs.logits
                attention_mask = inputs['attention_mask']
                
                # Compute sparse vectors for each item in batch
                for j in range(len(batch_texts)):
                    sparse_vec = self._compute_sparse_vector(
                        logits[j:j+1],
                        attention_mask[j:j+1]
                    )
                    all_sparse_vectors.append(sparse_vec)
            
            # Synchronize GPU if needed
            if self.is_gpu:
                torch.cuda.synchronize()
            
            elapsed_time = time.time() - start_time
            throughput = len(texts) / elapsed_time
            
            logger.info(
                f"✓ Generated {len(texts)} SPLADE vectors in {elapsed_time:.2f}s "
                f"({throughput:.1f} docs/sec)"
            )
            
            return all_sparse_vectors
            
        except Exception as e:
            logger.error(f"Error generating SPLADE embeddings: {e}")
            # Return empty sparse vectors as fallback
            return [SparseVector(indices=[0], values=[0.0]) for _ in texts]
    
    def get_device_info(self) -> dict:
        """Get device information"""
        info = {
            "model_name": self.model_name,
            "device": self.device,
            "is_gpu": self.is_gpu,
            "batch_size": self.batch_size,
            "mixed_precision": self.use_mixed_precision
        }
        
        if self.is_gpu:
            info.update({
                "gpu_name": torch.cuda.get_device_name(),
                "gpu_memory_allocated_mb": torch.cuda.memory_allocated() / 1024**2,
                "gpu_memory_reserved_mb": torch.cuda.memory_reserved() / 1024**2
            })
        
        return info
    
    def clear_gpu_cache(self):
        """Clear GPU cache"""
        if self.is_gpu:
            torch.cuda.empty_cache()
            logger.info("SPLADE GPU cache cleared")
