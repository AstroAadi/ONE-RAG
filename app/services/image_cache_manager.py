"""
Image Cache Manager
Caches image descriptions using perceptual hashing to avoid redundant API calls
"""

import logging
from typing import Dict, Optional, Tuple
import io
from PIL import Image
import imagehash
import hashlib

logger = logging.getLogger(__name__)


class ImageCacheManager:
    """
    Manage image description caching using perceptual hashing
    
    Features:
    - Perceptual hashing for visual similarity detection
    - In-memory cache for session
    - Fallback to content hash if image processing fails
    - Cache statistics tracking
    """
    
    def __init__(self, hash_size: int = 8):
        """
        Initialize image cache
        
        Args:
            hash_size: Size of perceptual hash (8 = 64-bit hash, default)
                      Larger = more precise, smaller = more fuzzy matching
        """
        self.hash_size = hash_size
        self.cache: Dict[str, str] = {}  # hash -> description
        self.hits = 0
        self.misses = 0
        
        logger.debug(f"Image cache initialized (hash_size={hash_size})")
    
    def compute_hash(self, image_bytes: bytes) -> str:
        """
        Compute perceptual hash for image
        
        Uses phash (perceptual hash) which is resilient to:
        - Minor resizing
        - Compression artifacts
        - Small color changes
        - Slight cropping
        
        Args:
            image_bytes: Raw image bytes
            
        Returns:
            Hash string
        """
        try:
            # Try perceptual hash (best for visual similarity)
            img = Image.open(io.BytesIO(image_bytes))
            img_hash = imagehash.phash(img, hash_size=self.hash_size)
            return str(img_hash)
            
        except Exception as e:
            # Fallback to content hash (exact match only)
            logger.debug(f"Perceptual hash failed, using MD5: {e}")
            return f"md5_{hashlib.md5(image_bytes).hexdigest()}"
    
    def get(self, image_bytes: bytes) -> Optional[str]:
        """
        Get cached description for image
        
        Args:
            image_bytes: Raw image bytes
            
        Returns:
            Cached description or None if not found
        """
        img_hash = self.compute_hash(image_bytes)
        
        if img_hash in self.cache:
            self.hits += 1
            logger.debug(f"Cache HIT for hash {img_hash[:12]}... ({self.hits} total hits)")
            return self.cache[img_hash]
        
        self.misses += 1
        logger.debug(f"Cache MISS for hash {img_hash[:12]}... ({self.misses} total misses)")
        return None
    
    def put(self, image_bytes: bytes, description: str) -> str:
        """
        Store description in cache
        
        Args:
            image_bytes: Raw image bytes
            description: Image description to cache
            
        Returns:
            Image hash (for reference)
        """
        img_hash = self.compute_hash(image_bytes)
        self.cache[img_hash] = description
        
        logger.debug(f"Cached description for hash {img_hash[:12]}... (cache size: {len(self.cache)})")
        return img_hash
    
    def has(self, image_bytes: bytes) -> bool:
        """
        Check if image is in cache
        
        Args:
            image_bytes: Raw image bytes
            
        Returns:
            True if cached
        """
        img_hash = self.compute_hash(image_bytes)
        return img_hash in self.cache
    
    def clear(self):
        """Clear the cache"""
        size = len(self.cache)
        self.cache.clear()
        self.hits = 0
        self.misses = 0
        logger.info(f"Cache cleared ({size} entries removed)")
    
    def get_stats(self) -> Dict[str, any]:
        """
        Get cache statistics
        
        Returns:
            Dictionary with cache stats
        """
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0.0
        
        return {
            'cache_size': len(self.cache),
            'hits': self.hits,
            'misses': self.misses,
            'total_requests': total_requests,
            'hit_rate': round(hit_rate, 3),
            'hit_rate_percent': round(hit_rate * 100, 1)
        }
    
    def compute_similarity(self, hash1: str, hash2: str) -> float:
        """
        Compute similarity between two hashes
        
        Args:
            hash1: First hash string
            hash2: Second hash string
            
        Returns:
            Similarity score (0.0 to 1.0, 1.0 = identical)
        """
        # Skip MD5 hashes (exact match only)
        if hash1.startswith('md5_') or hash2.startswith('md5_'):
            return 1.0 if hash1 == hash2 else 0.0
        
        try:
            h1 = imagehash.hex_to_hash(hash1)
            h2 = imagehash.hex_to_hash(hash2)
            
            # Hamming distance
            distance = h1 - h2
            
            # Convert to similarity (0 = identical, max = completely different)
            max_distance = len(h1.hash.flatten())
            similarity = 1.0 - (distance / max_distance)
            
            return similarity
            
        except Exception as e:
            logger.debug(f"Similarity computation failed: {e}")
            return 1.0 if hash1 == hash2 else 0.0
    
    def find_similar(
        self,
        image_bytes: bytes,
        threshold: float = 0.95
    ) -> Optional[Tuple[str, str, float]]:
        """
        Find similar image in cache
        
        Args:
            image_bytes: Image to search for
            threshold: Minimum similarity (0.0-1.0)
            
        Returns:
            Tuple of (hash, description, similarity) or None
        """
        query_hash = self.compute_hash(image_bytes)
        
        best_match = None
        best_similarity = 0.0
        
        for cached_hash, description in self.cache.items():
            similarity = self.compute_similarity(query_hash, cached_hash)
            
            if similarity >= threshold and similarity > best_similarity:
                best_match = (cached_hash, description, similarity)
                best_similarity = similarity
        
        if best_match:
            logger.debug(
                f"Found similar image: "
                f"hash={best_match[0][:12]}..., "
                f"similarity={best_match[2]:.3f}"
            )
        
        return best_match
    
    def __len__(self) -> int:
        """Get cache size"""
        return len(self.cache)
    
    def __contains__(self, image_bytes: bytes) -> bool:
        """Check if image is cached (supports 'in' operator)"""
        return self.has(image_bytes)


# Singleton instance
_cache_manager = None


def get_image_cache() -> ImageCacheManager:
    """Get singleton instance of ImageCacheManager"""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = ImageCacheManager()
    return _cache_manager


def reset_image_cache():
    """Reset the global cache"""
    global _cache_manager
    if _cache_manager is not None:
        _cache_manager.clear()