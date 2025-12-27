"""
Enhanced Chunking Service with STRICT validation
Fixed: Proper timer duration handling
"""

from typing import List, Dict, Any, Optional
import re
import hashlib
import logging

from app.core.config import get_settings
from app.core.exceptions import (
    ChunkingError,
    ChunkSizeExceededError,
    NoChunksCreatedError
)
from app.core.monitoring import track_performance, PerformanceTimer

logger = logging.getLogger(__name__)
settings = get_settings()


class SemanticChunkingService:
    """
    Semantic chunking with strict size validation
    
    Features:
    - Early validation prevents oversized chunks
    - Respects chunk_size and chunk_overlap
    - Supports PAGE and SLIDE markers
    - Thread-safe
    """
    
    def __init__(
        self, 
        chunk_size: int = 1500,
        chunk_overlap: int = 200,
        max_chunk_size: int = 6000
    ):
        # Validate configuration
        if chunk_overlap >= chunk_size:
            raise ChunkingError(
                f"chunk_overlap ({chunk_overlap}) must be less than "
                f"chunk_size ({chunk_size})"
            )
        
        if chunk_size > max_chunk_size:
            raise ChunkingError(
                f"chunk_size ({chunk_size}) cannot exceed "
                f"max_chunk_size ({max_chunk_size})"
            )
        
        # Cohere embed-english-v3 max is ~6000-8000 chars for English
        self.max_chunk_size = min(max_chunk_size, 6000)
        self.chunk_size = min(chunk_size, self.max_chunk_size)
        self.chunk_overlap = min(chunk_overlap, self.chunk_size // 2)
        
        logger.info(
            f"Chunking configured: size={self.chunk_size}, "
            f"overlap={self.chunk_overlap}, max={self.max_chunk_size}"
        )
    
    @track_performance("split_document_into_chunks")
    def split_document_into_chunks(
        self,
        markdown_text: str,
        structured_data: Dict[str, Any],
        base_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Split markdown into semantic chunks with validation
        
        Args:
            markdown_text: Markdown text to chunk
            structured_data: Document structure metadata
            base_metadata: Base metadata to attach to chunks
        
        Returns:
            List of chunk dictionaries
        
        Raises:
            NoChunksCreatedError: If no chunks created
            ChunkSizeExceededError: If chunk exceeds max size
        """
        if not markdown_text or not markdown_text.strip():
            raise NoChunksCreatedError("Empty markdown text provided")
        
        with PerformanceTimer("chunking_process", log_result=False) as timer:
            chunks = []
            
            # Split by visual content blocks
            sections = self._split_by_visual_blocks(markdown_text)
            
            logger.debug(f"Split into {len(sections)} sections by visual blocks")
            
            chunk_index = 0
            for section in sections:
                section_chunks = self._chunk_section(
                    section, 
                    chunk_index, 
                    base_metadata
                )
                chunks.extend(section_chunks)
                chunk_index += len(section_chunks)
            
            if not chunks:
                raise NoChunksCreatedError(
                    "No chunks created. Document may be too short or improperly formatted."
                )
            
            # CRITICAL: Validate ALL chunks before returning
            self._validate_all_chunks(chunks)
        
        # Log AFTER timer exits (duration is now available)
        avg_chars = len(markdown_text) / len(chunks) if chunks else 0
        logger.info(
            f"Created {len(chunks)} chunks in {timer.duration:.2f}s "
            f"(avg {avg_chars:.0f} chars/chunk)"
        )
        
        return chunks
    
    def _validate_all_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Validate all chunks meet size requirements
        
        Args:
            chunks: List of chunk dictionaries
        
        Raises:
            ChunkSizeExceededError: If any chunk exceeds max size
        """
        oversized_chunks = []
        
        for chunk in chunks:
            text_length = len(chunk['text'])
            if text_length > self.max_chunk_size:
                oversized_chunks.append({
                    'chunk_id': chunk['chunk_id'],
                    'size': text_length,
                    'max_size': self.max_chunk_size
                })
        
        if oversized_chunks:
            error_msg = (
                f"CRITICAL: {len(oversized_chunks)} chunk(s) exceed max size. "
                f"First: {oversized_chunks[0]}"
            )
            logger.error(error_msg)
            raise ChunkSizeExceededError(
                error_msg,
                details={'oversized_chunks': oversized_chunks}
            )
    
    def _split_by_visual_blocks(self, text: str) -> List[str]:
        """
        Split text by visual content blocks (images, tables, charts)
        
        Args:
            text: Markdown text
        
        Returns:
            List of section strings
        """
        sections = []
        current_section = ""
        
        lines = text.split('\n')
        
        for line in lines:
            current_section += line + '\n'
            
            # Keep collecting context after images
            if line.strip().startswith('IMAGE_PATH:'):
                continue
            
            # Split if section is getting large (2x target size)
            if len(current_section) > self.chunk_size * 2:
                sections.append(current_section.strip())
                current_section = ""
        
        # Add remaining section
        if current_section.strip():
            sections.append(current_section.strip())
        
        return sections if sections else [text]
    
    def _chunk_section(
        self,
        section: str,
        start_index: int,
        base_metadata: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Chunk a section respecting chunk_size with EARLY validation
        
        Args:
            section: Text section to chunk
            start_index: Starting chunk index
            base_metadata: Base metadata dictionary
        
        Returns:
            List of chunk dictionaries
        """
        chunks = []
        
        has_visual = 'IMAGE_PATH:' in section
        
        # Small section - validate and return as single chunk
        if len(section) <= self.chunk_size:
            # Validate BEFORE creating chunk
            if len(section) > self.max_chunk_size:
                # Split if too large
                return self._split_large_paragraph(
                    section, start_index, base_metadata
                )
            
            chunks.append(self._create_chunk_dict(
                text=section,
                chunk_index=start_index,
                base_metadata=base_metadata,
                has_visual=has_visual
            ))
            return chunks
        
        # Split by paragraphs
        paragraphs = re.split(r'\n\s*\n', section)
        
        current_chunk = ""
        chunk_index = start_index
        
        for para in paragraphs:
            if not para.strip():
                continue
            
            potential_size = len(current_chunk) + len(para) + 2
            
            # Would exceed target chunk size?
            if potential_size > self.chunk_size:
                # Save current chunk if not empty
                if current_chunk.strip():
                    # VALIDATE before adding
                    if len(current_chunk) > self.max_chunk_size:
                        raise ChunkSizeExceededError(
                            f"Chunk would exceed max size: {len(current_chunk)} > {self.max_chunk_size}",
                            details={'chunk_size': len(current_chunk)}
                        )
                    
                    has_vis = 'IMAGE_PATH:' in current_chunk
                    chunks.append(self._create_chunk_dict(
                        text=current_chunk.strip(),
                        chunk_index=chunk_index,
                        base_metadata=base_metadata,
                        has_visual=has_vis
                    ))
                    chunk_index += 1
                    
                    # Add overlap
                    overlap = self._get_overlap_text(current_chunk)
                    current_chunk = overlap
                
                # Handle very large paragraphs
                if len(para) > self.chunk_size:
                    sentence_chunks = self._split_large_paragraph(
                        para, chunk_index, base_metadata
                    )
                    chunks.extend(sentence_chunks)
                    chunk_index += len(sentence_chunks)
                    current_chunk = ""
                else:
                    current_chunk = (
                        current_chunk + "\n\n" + para 
                        if current_chunk else para
                    )
            else:
                current_chunk = (
                    current_chunk + "\n\n" + para 
                    if current_chunk else para
                )
        
        # Add remaining chunk
        if current_chunk.strip():
            # VALIDATE before adding
            if len(current_chunk) > self.max_chunk_size:
                # Split further if needed
                extra_chunks = self._split_large_paragraph(
                    current_chunk, chunk_index, base_metadata
                )
                chunks.extend(extra_chunks)
            else:
                has_vis = 'IMAGE_PATH:' in current_chunk
                chunks.append(self._create_chunk_dict(
                    text=current_chunk.strip(),
                    chunk_index=chunk_index,
                    base_metadata=base_metadata,
                    has_visual=has_vis
                ))
        
        return chunks
    
    def _split_large_paragraph(
        self,
        paragraph: str,
        start_index: int,
        base_metadata: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Split large paragraph by sentences with STRICT size limits
        
        Args:
            paragraph: Large paragraph text
            start_index: Starting chunk index
            base_metadata: Base metadata dictionary
        
        Returns:
            List of chunk dictionaries
        """
        chunks = []
        sentences = re.split(r'(?<=[.!?])\s+', paragraph)
        
        current_chunk = ""
        chunk_index = start_index
        
        for sentence in sentences:
            potential_size = len(current_chunk) + len(sentence) + 1
            
            # Check if adding sentence would exceed max size
            if potential_size <= self.max_chunk_size:
                current_chunk = (
                    current_chunk + " " + sentence 
                    if current_chunk else sentence
                )
            else:
                # Save current chunk if not empty
                if current_chunk.strip():
                    has_vis = 'IMAGE_PATH:' in current_chunk
                    chunks.append(self._create_chunk_dict(
                        text=current_chunk.strip(),
                        chunk_index=chunk_index,
                        base_metadata=base_metadata,
                        has_visual=has_vis
                    ))
                    chunk_index += 1
                
                # Handle sentence that's too long on its own
                if len(sentence) > self.max_chunk_size:
                    logger.warning(
                        f"Sentence exceeds max size ({len(sentence)} chars), "
                        f"truncating to {self.max_chunk_size}"
                    )
                    # Truncate at word boundary
                    truncated = sentence[:self.max_chunk_size - 3]
                    last_space = truncated.rfind(' ')
                    if last_space > 0:
                        truncated = truncated[:last_space]
                    sentence = truncated + "..."
                
                current_chunk = sentence
        
        # Add final chunk
        if current_chunk.strip():
            has_vis = 'IMAGE_PATH:' in current_chunk
            chunks.append(self._create_chunk_dict(
                text=current_chunk.strip(),
                chunk_index=chunk_index,
                base_metadata=base_metadata,
                has_visual=has_vis
            ))
        
        return chunks
    
    def _get_overlap_text(self, text: str) -> str:
        """
        Get overlap text from end of previous chunk
        
        Args:
            text: Previous chunk text
        
        Returns:
            Overlap text
        """
        if len(text) <= self.chunk_overlap:
            return text
        
        overlap_text = text[-self.chunk_overlap:]
        
        # Try to start from sentence boundary
        sentence_start = re.search(r'[.!?]\s+', overlap_text)
        if sentence_start:
            overlap_text = overlap_text[sentence_start.end():]
        
        return overlap_text
    
    def _create_chunk_dict(
        self,
        text: str,
        chunk_index: int,
        base_metadata: Optional[Dict[str, Any]],
        has_visual: bool = False
    ) -> Dict[str, Any]:
        """
        Create chunk dictionary with metadata
        
        Args:
            text: Chunk text
            chunk_index: Chunk index
            base_metadata: Base metadata dictionary
            has_visual: Whether chunk contains visual elements
        
        Returns:
            Chunk dictionary
        """
        # FINAL validation - should never trigger if earlier checks work
        if len(text) > self.max_chunk_size:
            raise ChunkSizeExceededError(
                f"CRITICAL BUG: Chunk created that exceeds max size: "
                f"{len(text)} > {self.max_chunk_size}",
                details={'chunk_length': len(text), 'chunk_index': chunk_index}
            )
        
        chunk_id = self._generate_chunk_id(text, chunk_index, base_metadata)
        
        # Detect chunk type
        chunk_type = 'text'
        if has_visual:
            if '**[TABLE]**' in text or '| --- |' in text:
                chunk_type = 'table'
            elif '**[CHART]**' in text or '**[FIGURE]**' in text:
                chunk_type = 'chart'
            elif 'IMAGE_PATH:' in text:
                chunk_type = 'image'
            else:
                chunk_type = 'mixed'
        
        # Extract page/slide number - supports both PAGE and SLIDE markers
        page_number = 0
        page_match = re.search(
            r'##\s*(Page|Slide)\s+(\d+)|(?:PAGE|SLIDE):\s*(\d+)', 
            text, 
            re.IGNORECASE
        )
        
        if page_match:
            page_number = int(page_match.group(2) or page_match.group(3))
        
        # Extract section hierarchy
        section_hierarchy = []
        section_matches = re.findall(r'^(#{1,6})\s+(.+)$', text, re.MULTILINE)
        if section_matches:
            section_hierarchy = [match[1].strip() for match in section_matches[:3]]
        
        metadata = {
            'chunk_id': chunk_id,
            'chunk_index': chunk_index,
            'chunk_type': chunk_type,
            'char_count': len(text),
            'has_visual': has_visual,
            'page_number': page_number,
            'section_hierarchy': ','.join(section_hierarchy) if section_hierarchy else ''
        }
        
        # Add base metadata
        if base_metadata:
            for key, value in base_metadata.items():
                if value is not None and key not in metadata:
                    metadata[key] = value
        
        return {
            'text': text,
            'chunk_id': chunk_id,
            'chunk_index': chunk_index,
            'char_count': len(text),
            'metadata': metadata
        }
    
    def _generate_chunk_id(
        self,
        text: str,
        chunk_index: int,
        base_metadata: Optional[Dict[str, Any]]
    ) -> str:
        """
        Generate unique chunk ID
        
        Args:
            text: Chunk text
            chunk_index: Chunk index
            base_metadata: Base metadata dictionary
        
        Returns:
            Unique chunk ID string
        """
        file_id = (
            base_metadata.get('file_id', 'unknown') 
            if base_metadata else 'unknown'
        )
        content_hash = hashlib.md5(text.encode()).hexdigest()[:8]
        return f"{file_id}_chunk_{chunk_index}_{content_hash}"