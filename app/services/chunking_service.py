"""
Simplified Chunking Service - No Parent-Child Complexity
Respects configured chunk_size and chunk_overlap
Clean, maintainable, and efficient
"""

from typing import List, Dict, Any, Optional
import re
import hashlib
import logging

logger = logging.getLogger(__name__)


class SemanticChunkingService:
    """
    Clean semantic chunking without parent-child complexity
    Respects chunk_size configuration
    """
    
    def __init__(
        self, 
        chunk_size: int = 1500,
        chunk_overlap: int = 200,
        max_chunk_size: int = 6000
    ):
        # Cohere embed-english-v3 max is 2048 tokens (~6000-8000 chars for English)
        # Using 6000 chars as safe limit
        self.max_chunk_size = min(max_chunk_size, 6000)
        self.chunk_size = min(chunk_size, self.max_chunk_size)
        self.chunk_overlap = min(chunk_overlap, self.chunk_size // 2)
        
        logger.info(f"Chunking configured: size={self.chunk_size}, overlap={self.chunk_overlap}, max={self.max_chunk_size}")
    
    def split_document_into_chunks(
        self,
        markdown_text: str,
        structured_data: Dict[str, Any],
        base_metadata: Dict = None
    ) -> List[Dict]:
        """
        Split markdown into semantic chunks
        Simple and clean - no parent-child complexity
        """
        if not markdown_text or not markdown_text.strip():
            logger.warning("Empty markdown text provided")
            return []
        
        chunks = []
        
        # Split by visual content blocks first (images, tables, charts)
        sections = self._split_by_visual_blocks(markdown_text)
        
        logger.info(f"Split into {len(sections)} sections by visual blocks")
        
        chunk_index = 0
        for section in sections:
            section_chunks = self._chunk_section(section, chunk_index, base_metadata)
            chunks.extend(section_chunks)
            chunk_index += len(section_chunks)
        
        logger.info(f"Created {len(chunks)} total chunks")
        
        # Validate all chunks are under max size
        for chunk in chunks:
            if len(chunk['text']) > self.max_chunk_size:
                logger.error(f"Chunk {chunk['chunk_id']} exceeds max size: {len(chunk['text'])} chars - THIS SHOULD NOT HAPPEN")
        
        return chunks
    
    def _split_by_visual_blocks(self, text: str) -> List[str]:
        """
        Split text by visual content blocks
        Each image/table/chart stays with its surrounding text
        """
        # Pattern to match IMAGE_PATH lines and surrounding context
        sections = []
        current_section = ""
        
        lines = text.split('\n')
        
        for line in lines:
            current_section += line + '\n'
            
            # Check if this line has an image reference
            if line.strip().startswith('IMAGE_PATH:'):
                # Keep collecting a bit more context after the image
                continue
            
            # If section is getting large, split it
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
        base_metadata: Dict
    ) -> List[Dict]:
        """
        Chunk a section respecting chunk_size
        """
        chunks = []
        
        # Check if section has visual content
        has_visual = 'IMAGE_PATH:' in section
        
        # If section is small enough, return as single chunk
        if len(section) <= self.chunk_size:
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
            
            # Check if adding this paragraph exceeds chunk_size
            potential_size = len(current_chunk) + len(para) + 2  # +2 for \n\n
            
            if potential_size > self.chunk_size:
                # Save current chunk if it has content
                if current_chunk.strip():
                    has_vis = 'IMAGE_PATH:' in current_chunk
                    chunks.append(self._create_chunk_dict(
                        text=current_chunk.strip(),
                        chunk_index=chunk_index,
                        base_metadata=base_metadata,
                        has_visual=has_vis
                    ))
                    chunk_index += 1
                    
                    # Add overlap from end of previous chunk
                    overlap = self._get_overlap_text(current_chunk)
                    current_chunk = overlap
                
                # Handle very large paragraphs
                if len(para) > self.chunk_size:
                    # Split large paragraph by sentences
                    sentence_chunks = self._split_large_paragraph(
                        para,
                        chunk_index,
                        base_metadata
                    )
                    chunks.extend(sentence_chunks)
                    chunk_index += len(sentence_chunks)
                    current_chunk = ""
                else:
                    current_chunk = (current_chunk + "\n\n" + para if current_chunk else para)
            else:
                current_chunk = (current_chunk + "\n\n" + para if current_chunk else para)
        
        # Add remaining chunk
        if current_chunk.strip():
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
        base_metadata: Dict
    ) -> List[Dict]:
        """Split a large paragraph by sentences"""
        chunks = []
        
        # Split by sentences
        sentences = re.split(r'(?<=[.!?])\s+', paragraph)
        
        current_chunk = ""
        chunk_index = start_index
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 1 <= self.chunk_size:
                current_chunk = (current_chunk + " " + sentence if current_chunk else sentence)
            else:
                # Save current chunk
                if current_chunk.strip():
                    has_vis = 'IMAGE_PATH:' in current_chunk
                    chunks.append(self._create_chunk_dict(
                        text=current_chunk.strip(),
                        chunk_index=chunk_index,
                        base_metadata=base_metadata,
                        has_visual=has_vis
                    ))
                    chunk_index += 1
                
                # Start new chunk with this sentence
                # If sentence itself is too long, truncate it
                if len(sentence) > self.max_chunk_size:
                    logger.warning(f"Sentence too long ({len(sentence)} chars), truncating")
                    sentence = sentence[:self.max_chunk_size - 3] + "..."
                
                current_chunk = sentence
        
        # Add remaining chunk
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
        """Get overlap text from end of previous chunk"""
        if len(text) <= self.chunk_overlap:
            return text
        
        overlap_text = text[-self.chunk_overlap:]
        
        # Try to start from a sentence boundary
        sentence_start = re.search(r'[.!?]\s+', overlap_text)
        if sentence_start:
            overlap_text = overlap_text[sentence_start.end():]
        
        return overlap_text
    
    def _create_chunk_dict(
        self,
        text: str,
        chunk_index: int,
        base_metadata: Dict,
        has_visual: bool = False
    ) -> Dict:
        """Create chunk dictionary with metadata"""
        
        # Safety check - should never exceed max_chunk_size
        if len(text) > self.max_chunk_size:
            logger.error(f"CRITICAL: Chunk exceeds max size ({len(text)} > {self.max_chunk_size}), truncating")
            text = text[:self.max_chunk_size - 3] + "..."
        
        chunk_id = self._generate_chunk_id(text, chunk_index, base_metadata)
        
        # Detect chunk type
        chunk_type = 'text'
        if has_visual:
            if '**[TABLE]**' in text:
                chunk_type = 'table'
            elif '**[CHART]**' in text or '**[FIGURE]**' in text:
                chunk_type = 'chart'
            elif 'IMAGE_PATH:' in text:
                chunk_type = 'image'
            else:
                chunk_type = 'mixed'
        
        # Extract page number if available
        # Look for "## Page X" or "PAGE: X" patterns
        page_number = 0
        page_match = re.search(r'##\s*Page\s+(\d+)|PAGE:\s*(\d+)', text, re.IGNORECASE)
        if page_match:
            page_number = int(page_match.group(1) or page_match.group(2))
        
        # Extract section hierarchy if available
        section_hierarchy = []
        section_matches = re.findall(r'^(#{1,6})\s+(.+)$', text, re.MULTILINE)
        if section_matches:
            section_hierarchy = [match[1].strip() for match in section_matches[:3]]  # Keep top 3 headings
        
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
        base_metadata: Dict
    ) -> str:
        """Generate unique chunk ID"""
        file_id = base_metadata.get('file_id', 'unknown') if base_metadata else 'unknown'
        content_hash = hashlib.md5(text.encode()).hexdigest()[:8]
        return f"{file_id}_chunk_{chunk_index}_{content_hash}"
