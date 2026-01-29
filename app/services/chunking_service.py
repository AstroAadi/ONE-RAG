"""
Character-Based Multi-Strategy Chunking Service
================================================

Key Principles:
1. NO token-based limits (except document-level parent: max 10K tokens)
2. ALL chunking based on CHARACTER COUNT
3. User-defined delimiters for splitting
4. Maximum chunk size = limit (chunks can be smaller, never larger)
5. User selects embedding model

Supported Strategies:
- Fixed Character Chunking (with delimiter)
- Parent-Child Paragraph Chunking
- Parent-Child Document Chunking (10K token parent limit only)

Author: Character-Based Chunker
Date: 2026-01-28
"""

from typing import List, Dict, Any, Optional, Literal
from enum import Enum
import re
import hashlib
import logging

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    logging.warning("tiktoken not installed. Install with: pip install tiktoken")

logger = logging.getLogger(__name__)


# ==================== Chunking Strategy Enum ====================

class ChunkingStrategy(str, Enum):
    """Supported chunking strategies"""
    FIXED_CHARACTER = "fixed_character"
    PARENT_CHILD_PARAGRAPH = "parent_child_paragraph"
    PARENT_CHILD_DOCUMENT = "parent_child_document"


# ==================== Token Counter (Only for Document-Level Parent) ====================

class TokenCounter:
    """Token counting utility - ONLY used for document-level parent 10K limit"""
    
    def __init__(self, model: str = "cl100k_base"):
        """Initialize token counter"""
        self.encoding = None
        
        if TIKTOKEN_AVAILABLE:
            try:
                self.encoding = tiktoken.get_encoding(model)
                logger.info(f"✓ Token counter initialized: {model} (for document-level parent only)")
            except Exception as e:
                logger.warning(f"Failed to load tiktoken encoding: {e}")
                self.encoding = None
        
        if not self.encoding:
            logger.warning("Using character-based token estimation for document-level parent limit")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        if not text:
            return 0
        
        if self.encoding:
            try:
                return len(self.encoding.encode(text))
            except Exception as e:
                logger.debug(f"Token counting failed: {e}, using fallback")
        
        # Fallback: estimate tokens (4 chars ≈ 1 token)
        return max(1, len(text) // 4)
    
    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text to max tokens - ONLY for document-level parent"""
        if not text:
            return ""
        
        if self.encoding:
            try:
                tokens = self.encoding.encode(text)
                if len(tokens) <= max_tokens:
                    return text
                truncated_tokens = tokens[:max_tokens]
                return self.encoding.decode(truncated_tokens)
            except Exception as e:
                logger.debug(f"Token truncation failed: {e}, using fallback")
        
        # Fallback: character-based truncation
        max_chars = max_tokens * 4
        return text[:max_chars]


# ==================== Main Character-Based Chunking Service ====================

class CharacterBasedChunkingService:
    """
    Character-based chunking service
    
    Key Features:
    - NO token limits (except 10K for document-level parent)
    - ALL chunking based on CHARACTER COUNT
    - User-defined delimiters
    - Maximum chunk size enforced (chunks can be smaller)
    """
    
    def __init__(
        self,
        # General chunking parameters
        max_chunk_chars: int = 2000,
        overlap_chars: int = 200,
        delimiter: str = "\n\n",
        
        # Parent-child parameters
        max_child_chunk_chars: int = 1000,
        max_document_parent_tokens: int = 10000,
        
        # Tokenizer (only for document-level parent)
        tokenizer_model: str = "cl100k_base"
    ):
        """
        Initialize character-based chunker
        
        Args:
            max_chunk_chars: Maximum chunk size in characters (for fixed chunking)
            overlap_chars: Overlap size in characters
            delimiter: Delimiter for splitting (e.g., "\n\n", "\n", ".", etc.)
            max_child_chunk_chars: Maximum child chunk size in characters
            max_document_parent_tokens: Max tokens for document-level parent (10K limit)
            tokenizer_model: Tokenizer model (only for document parent limit)
        """
        self.max_chunk_chars = max_chunk_chars
        self.overlap_chars = overlap_chars
        self.delimiter = delimiter
        self.max_child_chunk_chars = max_child_chunk_chars
        self.max_document_parent_tokens = max_document_parent_tokens
        
        # Token counter (ONLY for document-level parent limit)
        self.token_counter = TokenCounter(model=tokenizer_model)
        
        logger.info(
            f"✓ Character-based chunker initialized:\n"
            f"  - Max chunk size: {max_chunk_chars} characters\n"
            f"  - Overlap: {overlap_chars} characters\n"
            f"  - Delimiter: {repr(delimiter)}\n"
            f"  - Max child chunk: {max_child_chunk_chars} characters\n"
            f"  - Document parent limit: {max_document_parent_tokens} tokens"
        )
    
    def split_document_into_chunks(
        self,
        markdown_text: str,
        structured_data: Optional[Dict] = None,
        base_metadata: Optional[Dict] = None,
        strategy: ChunkingStrategy = ChunkingStrategy.FIXED_CHARACTER,
        embedding_model: str = "cohere.embed-english-v3"
    ) -> List[Dict[str, Any]]:
        """
        Split document into chunks using specified strategy
        
        Args:
            markdown_text: Document text (markdown)
            structured_data: Optional structured data
            base_metadata: Base metadata for all chunks
            strategy: Chunking strategy to use
            embedding_model: Embedding model name (stored in metadata)
            
        Returns:
            List of chunk dictionaries with metadata
        """
        if not markdown_text or not markdown_text.strip():
            logger.warning("Empty document provided")
            return []
        
        if base_metadata is None:
            base_metadata = {}
        
        # Add embedding model to metadata
        base_metadata['embedding_model'] = embedding_model
        
        logger.info(
            f"Chunking document with strategy: {strategy.value}, "
            f"embedding model: {embedding_model}"
        )
        
        # Route to appropriate chunking method
        if strategy == ChunkingStrategy.FIXED_CHARACTER:
            chunks = self._chunk_fixed_character(markdown_text, base_metadata)
        elif strategy == ChunkingStrategy.PARENT_CHILD_PARAGRAPH:
            chunks = self._chunk_parent_child_paragraph(markdown_text, base_metadata)
        elif strategy == ChunkingStrategy.PARENT_CHILD_DOCUMENT:
            chunks = self._chunk_parent_child_document(markdown_text, base_metadata)
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")
        
        if not chunks:
            logger.warning("No chunks created")
            return []
        
        logger.info(f"✅ Created {len(chunks)} chunks using {strategy.value}")
        
        return chunks
    
    # ==================== Fixed Character Chunking ====================
    
    def _chunk_fixed_character(
        self,
        text: str,
        base_metadata: Dict
    ) -> List[Dict[str, Any]]:
        """
        Fixed character-based chunking with delimiter-aware splitting
        
        Strategy:
        1. Split text by delimiter
        2. Combine segments until max_chunk_chars is reached
        3. Never exceed max_chunk_chars
        4. Apply overlap between chunks
        """
        chunks = []
        
        # Split by delimiter
        segments = text.split(self.delimiter)
        
        logger.info(f"Split document into {len(segments)} segments using delimiter: {repr(self.delimiter)}")
        
        current_chunk = ""
        chunk_index = 1
        
        for segment in segments:
            segment = segment.strip()
            if not segment:
                continue
            
            # Check if adding this segment would exceed limit
            potential_chunk = current_chunk + self.delimiter + segment if current_chunk else segment
            
            if len(potential_chunk) <= self.max_chunk_chars:
                # Add segment to current chunk
                current_chunk = potential_chunk
            else:
                # Current chunk is full, save it
                if current_chunk:
                    chunk = self._create_chunk(
                        current_chunk,
                        chunk_index,
                        base_metadata,
                        chunk_type='text'
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                
                # Start new chunk with current segment
                # If segment itself exceeds limit, split it
                if len(segment) > self.max_chunk_chars:
                    # Split large segment
                    sub_chunks = self._split_large_segment(
                        segment,
                        chunk_index,
                        base_metadata
                    )
                    chunks.extend(sub_chunks)
                    chunk_index += len(sub_chunks)
                    current_chunk = ""
                else:
                    current_chunk = segment
        
        # Add final chunk
        if current_chunk:
            chunk = self._create_chunk(
                current_chunk,
                chunk_index,
                base_metadata,
                chunk_type='text'
            )
            chunks.append(chunk)
        
        # Apply overlap if requested
        if self.overlap_chars > 0:
            chunks = self._apply_overlap(chunks)
        
        return chunks
    
    # ==================== Parent-Child Paragraph Chunking ====================
    
    def _chunk_parent_child_paragraph(
        self,
        text: str,
        base_metadata: Dict
    ) -> List[Dict[str, Any]]:
        """
        Parent-child chunking with paragraph-level parents
        
        Strategy:
        1. Split document by delimiter (paragraphs)
        2. Each paragraph = parent chunk (NO size limit)
        3. Create child chunks from parent (max: max_child_chunk_chars)
        4. Children can be smaller, never larger than max
        """
        chunks = []
        chunk_index = 1
        
        # Split into paragraphs using delimiter
        paragraphs = text.split(self.delimiter)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        logger.info(f"Found {len(paragraphs)} paragraphs using delimiter: {repr(self.delimiter)}")
        
        for para_index, paragraph in enumerate(paragraphs):
            if not paragraph.strip():
                continue
            
            # Detect page number
            page_number = self._extract_page_number(paragraph)
            
            # Create parent chunk (FULL paragraph, NO truncation)
            parent_chunk_id = self._generate_chunk_id(paragraph, chunk_index)
            
            parent_chunk = {
                'chunk_id': parent_chunk_id,
                'text': paragraph,
                'char_count': len(paragraph),
                'metadata': {
                    'chunk_index': chunk_index,
                    'chunk_type': 'parent',
                    'page_number': page_number,
                    'paragraph_index': para_index,
                    'chunking_strategy': ChunkingStrategy.PARENT_CHILD_PARAGRAPH.value,
                    'is_parent': True,
                    **base_metadata
                }
            }
            
            chunks.append(parent_chunk)
            chunk_index += 1
            
            # Create child chunks from paragraph
            child_chunks = self._create_child_chunks(
                paragraph,
                parent_chunk_id,
                page_number,
                chunk_index,
                base_metadata
            )
            
            chunks.extend(child_chunks)
            chunk_index += len(child_chunks)
        
        return chunks
    
    # ==================== Parent-Child Document Chunking ====================
    
    def _chunk_parent_child_document(
        self,
        text: str,
        base_metadata: Dict
    ) -> List[Dict[str, Any]]:
        """
        Parent-child chunking with document-level parent
        
        Strategy:
        1. ENTIRE document = parent (LIMITED TO 10,000 TOKENS ONLY)
        2. Create child chunks (max: max_child_chunk_chars)
        3. Children can be smaller, never larger than max
        """
        chunks = []
        chunk_index = 1
        
        # Check document token count (ONLY token limit in entire system)
        doc_tokens = self.token_counter.count_tokens(text)
        
        parent_text = text
        if doc_tokens > self.max_document_parent_tokens:
            logger.warning(
                f"Document has {doc_tokens} tokens, truncating to "
                f"{self.max_document_parent_tokens} for parent chunk"
            )
            parent_text = self.token_counter.truncate_to_tokens(
                text, 
                self.max_document_parent_tokens
            )
            doc_tokens = self.max_document_parent_tokens
        
        parent_chunk_id = self._generate_chunk_id(parent_text, chunk_index)
        
        # Create parent chunk (full document, limited to 10K tokens)
        parent_chunk = {
            'chunk_id': parent_chunk_id,
            'text': parent_text,
            'char_count': len(parent_text),
            'metadata': {
                'chunk_index': chunk_index,
                'chunk_type': 'parent',
                'page_number': 1,
                'chunking_strategy': ChunkingStrategy.PARENT_CHILD_DOCUMENT.value,
                'is_parent': True,
                'is_document_parent': True,
                'document_tokens': doc_tokens,
                'truncated': doc_tokens == self.max_document_parent_tokens,
                **base_metadata
            }
        }
        
        chunks.append(parent_chunk)
        chunk_index += 1
        
        # Create child chunks from FULL document (not truncated parent)
        child_chunks = self._create_child_chunks(
            text,  # Use full text, not truncated parent
            parent_chunk_id,
            1,
            chunk_index,
            base_metadata
        )
        
        chunks.extend(child_chunks)
        
        return chunks
    
    # ==================== Helper Methods ====================
    
    def _create_child_chunks(
        self,
        text: str,
        parent_chunk_id: str,
        page_number: int,
        start_chunk_index: int,
        base_metadata: Dict
    ) -> List[Dict[str, Any]]:
        """
        Create child chunks from text
        
        Args:
            text: Text to chunk
            parent_chunk_id: ID of parent chunk
            page_number: Page number
            start_chunk_index: Starting chunk index
            base_metadata: Base metadata
            
        Returns:
            List of child chunks
        """
        child_chunks = []
        
        # Split by delimiter first
        segments = text.split(self.delimiter)
        segments = [s.strip() for s in segments if s.strip()]
        
        current_child = ""
        child_index = 0
        
        for segment in segments:
            # Check if adding segment would exceed max
            potential_child = current_child + self.delimiter + segment if current_child else segment
            
            if len(potential_child) <= self.max_child_chunk_chars:
                current_child = potential_child
            else:
                # Save current child
                if current_child:
                    child_chunk = self._create_chunk(
                        current_child,
                        start_chunk_index + child_index,
                        base_metadata,
                        chunk_type='child',
                        parent_chunk_id=parent_chunk_id,
                        child_index=child_index,
                        page_number=page_number
                    )
                    child_chunks.append(child_chunk)
                    child_index += 1
                
                # Handle large segment
                if len(segment) > self.max_child_chunk_chars:
                    # Split large segment
                    sub_chunks = self._split_large_segment(
                        segment,
                        start_chunk_index + child_index,
                        base_metadata,
                        parent_chunk_id=parent_chunk_id,
                        child_index_start=child_index,
                        page_number=page_number
                    )
                    child_chunks.extend(sub_chunks)
                    child_index += len(sub_chunks)
                    current_child = ""
                else:
                    current_child = segment
        
        # Add final child
        if current_child:
            child_chunk = self._create_chunk(
                current_child,
                start_chunk_index + child_index,
                base_metadata,
                chunk_type='child',
                parent_chunk_id=parent_chunk_id,
                child_index=child_index,
                page_number=page_number
            )
            child_chunks.append(child_chunk)
        
        return child_chunks
    
    def _split_large_segment(
        self,
        segment: str,
        start_index: int,
        base_metadata: Dict,
        parent_chunk_id: Optional[str] = None,
        child_index_start: int = 0,
        page_number: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Split a segment that exceeds max chunk size
        
        Uses simple character-based splitting
        """
        chunks = []
        max_size = self.max_child_chunk_chars if parent_chunk_id else self.max_chunk_chars
        
        position = 0
        local_index = 0
        
        while position < len(segment):
            chunk_text = segment[position:position + max_size]
            
            if not chunk_text.strip():
                position += max_size
                continue
            
            chunk = self._create_chunk(
                chunk_text,
                start_index + local_index,
                base_metadata,
                chunk_type='child' if parent_chunk_id else 'text',
                parent_chunk_id=parent_chunk_id,
                child_index=child_index_start + local_index if parent_chunk_id else None,
                page_number=page_number
            )
            
            chunks.append(chunk)
            position += max_size
            local_index += 1
        
        return chunks
    
    def _create_chunk(
        self,
        text: str,
        chunk_index: int,
        base_metadata: Dict,
        chunk_type: str = 'text',
        parent_chunk_id: Optional[str] = None,
        child_index: Optional[int] = None,
        page_number: Optional[int] = None
    ) -> Dict[str, Any]:
        """Create a chunk dictionary"""
        
        if page_number is None:
            page_number = self._extract_page_number(text)
        
        chunk = {
            'chunk_id': self._generate_chunk_id(text, chunk_index),
            'text': text,
            'char_count': len(text),
            'metadata': {
                'chunk_index': chunk_index,
                'chunk_type': chunk_type,
                'page_number': page_number,
                **base_metadata
            }
        }
        
        # Add parent-child metadata if applicable
        if parent_chunk_id:
            chunk['metadata']['parent_chunk_id'] = parent_chunk_id
            chunk['metadata']['is_parent'] = False
            if child_index is not None:
                chunk['metadata']['child_index'] = child_index
        elif chunk_type == 'parent':
            chunk['metadata']['is_parent'] = True
        
        return chunk
    
    def _apply_overlap(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Apply overlap to consecutive chunks
        
        Takes last overlap_chars from previous chunk and prepends to next chunk
        """
        if len(chunks) <= 1 or self.overlap_chars == 0:
            return chunks
        
        overlapped_chunks = [chunks[0]]
        
        for i in range(1, len(chunks)):
            prev_text = chunks[i-1]['text']
            curr_text = chunks[i]['text']
            
            # Get overlap from previous chunk
            overlap_text = prev_text[-self.overlap_chars:] if len(prev_text) >= self.overlap_chars else prev_text
            
            # Prepend overlap to current chunk
            new_text = overlap_text + curr_text
            
            # Update chunk
            new_chunk = chunks[i].copy()
            new_chunk['text'] = new_text
            new_chunk['char_count'] = len(new_text)
            
            overlapped_chunks.append(new_chunk)
        
        return overlapped_chunks
    
    def _extract_page_number(self, text: str) -> int:
        """Extract page number from chunk text"""
        page_pattern = r'##\s*Page\s+(\d+)'
        matches = re.findall(page_pattern, text, re.IGNORECASE)
        
        if matches:
            return int(matches[0])
        
        return 1
    
    def _generate_chunk_id(self, text: str, chunk_index: int) -> str:
        """Generate unique chunk ID"""
        content_sample = text[:100].strip()
        hash_input = f"{content_sample}_{chunk_index}"
        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]
    
    def get_chunk_stats(self, chunks: List[Dict]) -> Dict[str, Any]:
        """Get statistics about chunks"""
        if not chunks:
            return {
                'total_chunks': 0,
                'avg_chars': 0,
                'min_chars': 0,
                'max_chars': 0,
                'total_chars': 0
            }
        
        char_counts = [c['char_count'] for c in chunks]
        
        # Count parent/child if applicable
        parent_chunks = [c for c in chunks if c['metadata'].get('is_parent', False)]
        child_chunks = [c for c in chunks if not c['metadata'].get('is_parent', False)]
        
        stats = {
            'total_chunks': len(chunks),
            'avg_chars': sum(char_counts) / len(char_counts),
            'min_chars': min(char_counts),
            'max_chars': max(char_counts),
            'total_chars': sum(char_counts),
            'max_chunk_limit': self.max_chunk_chars,
            'overlap': self.overlap_chars,
            'delimiter': self.delimiter
        }
        
        if parent_chunks:
            stats['parent_chunks'] = len(parent_chunks)
            stats['child_chunks'] = len(child_chunks)
            stats['max_child_limit'] = self.max_child_chunk_chars
        
        return stats


# Backward compatibility aliases
SemanticChunkingService = CharacterBasedChunkingService
