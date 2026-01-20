"""
Simple Token-Based Chunking Service
====================================

NO SEMANTIC CHUNKING, NO TRUNCATION, NO DATA LOSS

Strategy:
- Fixed token-sized chunks
- Sliding window with exact overlap
- No smart splitting, no section detection
- Just: count tokens, split at boundaries, done

Perfect for preserving ALL content without loss.

Author: Simple Chunker
Date: 2026-01-19
"""

from typing import List, Dict, Any, Optional
import re
import hashlib
import logging

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    logging.warning("tiktoken not installed. Install with: pip install tiktoken")

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class TokenCounter:
    """Token counting utility with fallback"""
    
    def __init__(self, model: str = "cl100k_base"):
        """Initialize token counter"""
        self.encoding = None
        
        if TIKTOKEN_AVAILABLE:
            try:
                self.encoding = tiktoken.get_encoding(model)
                logger.info(f"✓ Token counter initialized: {model}")
            except Exception as e:
                logger.warning(f"Failed to load tiktoken encoding: {e}")
                self.encoding = None
        
        if not self.encoding:
            logger.warning("Using character-based token estimation (4 chars ≈ 1 token)")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        if not text:
            return 0
        
        if self.encoding:
            try:
                return len(self.encoding.encode(text))
            except Exception as e:
                logger.debug(f"Token counting failed: {e}, using fallback")
        
        # Fallback: estimate tokens
        return max(1, len(text) // 4)
    
    def get_tokens(self, text: str) -> List[int]:
        """Get token IDs"""
        if not text:
            return []
        
        if self.encoding:
            try:
                return self.encoding.encode(text)
            except Exception as e:
                logger.debug(f"Token encoding failed: {e}")
                return []
        
        return []
    
    def decode_tokens(self, tokens: List[int]) -> str:
        """Decode tokens back to text"""
        if not tokens:
            return ""
        
        if self.encoding:
            try:
                return self.encoding.decode(tokens)
            except Exception as e:
                logger.debug(f"Token decoding failed: {e}")
                return ""
        
        return ""


class SimpleChunkingService:
    """
    Simple token-based chunker with NO data loss
    
    Strategy:
    1. Convert text to tokens
    2. Slide window of fixed size
    3. Create overlapping chunks
    4. No truncation, no semantic splitting, no data loss
    """
    
    def __init__(
        self,
        target_chunk_tokens: int = 512,
        chunk_overlap_tokens: int = 50,
        tokenizer_model: str = "cl100k_base"
    ):
        """
        Initialize simple chunker
        
        Args:
            target_chunk_tokens: Exact chunk size in tokens (default: 512)
            chunk_overlap_tokens: Overlap between chunks in tokens (default: 50)
            tokenizer_model: Tokenizer model (default: cl100k_base)
        """
        self.target_chunk_tokens = target_chunk_tokens
        self.chunk_overlap_tokens = chunk_overlap_tokens
        self.tokenizer_model = tokenizer_model
        
        # Initialize token counter
        self.token_counter = TokenCounter(model=tokenizer_model)
        
        logger.info(
            f"✓ Simple chunker initialized:\n"
            f"  - Target chunk size: {target_chunk_tokens} tokens\n"
            f"  - Overlap: {chunk_overlap_tokens} tokens\n"
            f"  - Tokenizer: {tokenizer_model}"
        )
    
    def split_document_into_chunks(
        self,
        markdown_text: str,
        structured_data: Optional[Dict] = None,
        base_metadata: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """
        Split document into fixed-size token chunks
        
        Args:
            markdown_text: Document text (markdown)
            structured_data: Optional structured data (ignored in simple mode)
            base_metadata: Base metadata for all chunks
            
        Returns:
            List of chunk dictionaries with metadata
        """
        if not markdown_text or not markdown_text.strip():
            logger.warning("Empty document provided")
            return []
        
        # Initialize base metadata
        if base_metadata is None:
            base_metadata = {}
        
        logger.info(f"Chunking document ({len(markdown_text)} characters)...")
        
        # Simple strategy: tokenize and chunk
        chunks = self._create_fixed_size_chunks(markdown_text, base_metadata)
        
        if not chunks:
            logger.warning("No chunks created")
            return []
        
        logger.info(f"✅ Created {len(chunks)} chunks (no data loss)")
        
        return chunks
    
    def _create_fixed_size_chunks(
        self,
        text: str,
        base_metadata: Dict
    ) -> List[Dict[str, Any]]:
        """
        Create fixed-size chunks using sliding window
        
        Strategy:
        1. Tokenize entire text
        2. Slide window of target_chunk_tokens
        3. Step forward by (target_chunk_tokens - overlap)
        4. Continue until end
        5. NO TRUNCATION - every token is included
        """
        chunks = []
        
        # Tokenize entire text
        if self.token_counter.encoding:
            # Use actual tokens
            tokens = self.token_counter.get_tokens(text)
            
            if not tokens:
                logger.warning("Tokenization failed, falling back to character chunks")
                return self._create_character_chunks(text, base_metadata)
            
            logger.debug(f"Tokenized: {len(tokens)} tokens total")
            
            # Sliding window
            stride = self.target_chunk_tokens - self.chunk_overlap_tokens
            
            if stride <= 0:
                logger.error(f"Invalid stride: {stride}. Overlap must be < target size.")
                stride = max(1, self.target_chunk_tokens // 2)
            
            position = 0
            chunk_index = 1
            
            while position < len(tokens):
                # Extract chunk tokens
                chunk_tokens = tokens[position:position + self.target_chunk_tokens]
                
                # Decode to text
                chunk_text = self.token_counter.decode_tokens(chunk_tokens)
                
                if not chunk_text.strip():
                    position += stride
                    continue
                
                # Detect page number from chunk text
                page_number = self._extract_page_number(chunk_text)
                
                # Create chunk
                chunk = {
                    'chunk_id': self._generate_chunk_id(chunk_text, chunk_index),
                    'text': chunk_text,
                    'char_count': len(chunk_text),
                    'token_count': len(chunk_tokens),
                    'metadata': {
                        'chunk_index': chunk_index,
                        'chunk_type': 'text',
                        'page_number': page_number,
                        'token_position': position,
                        'total_tokens_in_doc': len(tokens),
                        **base_metadata
                    }
                }
                
                chunks.append(chunk)
                
                logger.debug(
                    f"Chunk {chunk_index}: {len(chunk_tokens)} tokens, "
                    f"position {position}/{len(tokens)}"
                )
                
                # Move window forward
                position += stride
                chunk_index += 1
            
        else:
            # Fallback: character-based chunking
            logger.warning("Using character-based chunking (no tokenizer)")
            chunks = self._create_character_chunks(text, base_metadata)
        
        return chunks
    
    def _create_character_chunks(
        self,
        text: str,
        base_metadata: Dict
    ) -> List[Dict[str, Any]]:
        """
        Fallback: character-based chunking
        
        Uses ~4 chars per token estimation
        """
        chunks = []
        
        # Estimate character sizes
        target_chars = self.target_chunk_tokens * 4
        overlap_chars = self.chunk_overlap_tokens * 4
        stride_chars = target_chars - overlap_chars
        
        if stride_chars <= 0:
            stride_chars = target_chars // 2
        
        position = 0
        chunk_index = 1
        
        while position < len(text):
            # Extract chunk text
            chunk_text = text[position:position + target_chars]
            
            if not chunk_text.strip():
                position += stride_chars
                continue
            
            # Count tokens
            token_count = self.token_counter.count_tokens(chunk_text)
            
            # Detect page number
            page_number = self._extract_page_number(chunk_text)
            
            # Create chunk
            chunk = {
                'chunk_id': self._generate_chunk_id(chunk_text, chunk_index),
                'text': chunk_text,
                'char_count': len(chunk_text),
                'token_count': token_count,
                'metadata': {
                    'chunk_index': chunk_index,
                    'chunk_type': 'text',
                    'page_number': page_number,
                    'char_position': position,
                    'total_chars_in_doc': len(text),
                    **base_metadata
                }
            }
            
            chunks.append(chunk)
            
            logger.debug(
                f"Chunk {chunk_index}: ~{token_count} tokens, "
                f"position {position}/{len(text)}"
            )
            
            # Move window forward
            position += stride_chars
            chunk_index += 1
        
        return chunks
    
    def _extract_page_number(self, text: str) -> int:
        """
        Extract page number from chunk text
        
        Looks for: ## Page N
        """
        # Look for page markers
        page_pattern = r'##\s*Page\s+(\d+)'
        matches = re.findall(page_pattern, text, re.IGNORECASE)
        
        if matches:
            # Return first page number found
            return int(matches[0])
        
        return 1  # Default to page 1
    
    def _generate_chunk_id(self, text: str, chunk_index: int) -> str:
        """Generate unique chunk ID"""
        # Use first 100 chars + index for ID
        content_sample = text[:100].strip()
        hash_input = f"{content_sample}_{chunk_index}"
        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]
    
    def get_chunk_stats(self, chunks: List[Dict]) -> Dict[str, Any]:
        """Get statistics about chunks"""
        if not chunks:
            return {
                'total_chunks': 0,
                'avg_tokens': 0,
                'min_tokens': 0,
                'max_tokens': 0,
                'total_tokens': 0
            }
        
        token_counts = [c['token_count'] for c in chunks]
        
        return {
            'total_chunks': len(chunks),
            'avg_tokens': sum(token_counts) / len(token_counts),
            'min_tokens': min(token_counts),
            'max_tokens': max(token_counts),
            'total_tokens': sum(token_counts),
            'target_tokens': self.target_chunk_tokens,
            'overlap_tokens': self.chunk_overlap_tokens
        }


# Backward compatibility alias
SemanticChunkingService = SimpleChunkingService


# Example usage
if __name__ == "__main__":
    # Test the chunker
    chunker = SimpleChunkingService(
        target_chunk_tokens=500,
        chunk_overlap_tokens=50
    )
    
    # Sample text
    sample_text = """
    # Document: Test
    
    ## Page 1
    
    This is a test document with multiple paragraphs.
    We want to ensure that no data is lost during chunking.
    
    ## Page 2
    
    More content here to test the chunking process.
    The chunker should create fixed-size chunks without truncation.
    
    ## Page 3
    
    Final page with some additional content.
    """ * 20  # Repeat to create longer document
    
    # Chunk it
    chunks = chunker.split_document_into_chunks(
        sample_text,
        base_metadata={'file_id': 'test123'}
    )
    
    # Print stats
    stats = chunker.get_chunk_stats(chunks)
    print(f"\n{'='*60}")
    print(f"CHUNKING STATS:")
    print(f"{'='*60}")
    print(f"Total chunks: {stats['total_chunks']}")
    print(f"Average tokens per chunk: {stats['avg_tokens']:.1f}")
    print(f"Min tokens: {stats['min_tokens']}")
    print(f"Max tokens: {stats['max_tokens']}")
    print(f"Total tokens: {stats['total_tokens']}")
    print(f"Target: {stats['target_tokens']} tokens")
    print(f"Overlap: {stats['overlap_tokens']} tokens")
    print(f"{'='*60}\n")
    
    # Show first chunk
    if chunks:
        print(f"First chunk preview:")
        print(f"Text: {chunks[0]['text'][:200]}...")
        print(f"Tokens: {chunks[0]['token_count']}")
        print(f"Page: {chunks[0]['metadata']['page_number']}")