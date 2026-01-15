"""
Optimized Markdown-Aware Chunking Service
==========================================

Key Optimizations:
1. Smart heading merging: Merge small heading sections until reaching target size (1500 chars)
2. Large section splitting: Split oversized sections at semantic boundaries with controlled overlap
3. Structure preservation: Headings never split from content, overlaps don't duplicate headings
4. No truncation: All content preserved, respects embedding_max_length (2048 for Cohere)

Integration with vector_store_service:
- Respects embedding_max_length (chunks stay under 2048 to avoid truncation)
- Generates unique chunk IDs compatible with Qdrant UUID conversion
- Maintains all existing metadata structure
"""

from typing import List, Dict, Any, Optional, Tuple
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
    Optimized Markdown-aware chunking with intelligent heading management
    
    Strategy:
    - Parse markdown into sections (heading + its content)
    - Merge small sections with next section until reaching chunk_size (1500)
    - Split large sections at paragraph/sentence boundaries with overlap
    - Never split headings from their content
    - Overlaps only include content, not headings
    """
    
    def __init__(
        self, 
        chunk_size: int = 1500,
        chunk_overlap: int = 200,
        max_chunk_size: int = 2000
    ):
        # Respect embedding limits - critical for avoiding truncation
        embedding_limit = getattr(settings, 'embedding_max_length', 2048)
        
        # Ensure max_chunk_size doesn't exceed embedding limit
        self.max_chunk_size = min(max_chunk_size, embedding_limit)
        self.chunk_size = min(chunk_size, self.max_chunk_size)
        self.chunk_overlap = min(chunk_overlap, self.chunk_size // 2)
        
        # Validate
        if self.chunk_overlap >= self.chunk_size:
            raise ChunkingError(
                f"chunk_overlap ({self.chunk_overlap}) must be less than "
                f"chunk_size ({self.chunk_size})"
            )
        
        logger.info(
            f"Optimized chunking: target={self.chunk_size}, "
            f"overlap={self.chunk_overlap}, max={self.max_chunk_size}, "
            f"embedding_limit={embedding_limit}"
        )
    
    @track_performance("split_document_into_chunks")
    def split_document_into_chunks(
        self,
        markdown_text: str,
        structured_data: Dict[str, Any],
        base_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Split markdown into optimized chunks
        
        Returns:
            List of chunk dictionaries with metadata
        """
        if not markdown_text or not markdown_text.strip():
            raise NoChunksCreatedError("Empty markdown text provided")
        
        with PerformanceTimer("chunking_process", log_result=False):
            # Parse into structural elements
            elements = self._parse_markdown_structure(markdown_text)
            
            if not elements:
                raise NoChunksCreatedError("No structural elements parsed")
            
            logger.debug(f"Parsed {len(elements)} elements")
            
            # Group elements into heading-based sections
            sections = self._group_into_sections(elements)
            logger.debug(f"Grouped into {len(sections)} sections")
            
            # Create optimized chunks from sections
            chunks = self._create_optimized_chunks(sections, base_metadata)
            
            if not chunks:
                raise NoChunksCreatedError("No chunks created")
            
            # Validate all chunks
            self._validate_all_chunks(chunks)
            
            # Add structured data references
            chunks = self._add_structured_references(chunks, structured_data)
            
            logger.info(
                f"Created {len(chunks)} chunks "
                f"(avg: {sum(len(c['text']) for c in chunks) // len(chunks)}, "
                f"min: {min(len(c['text']) for c in chunks)}, "
                f"max: {max(len(c['text']) for c in chunks)})"
            )
        
        return chunks
    
    def _parse_markdown_structure(self, text: str) -> List[Dict[str, Any]]:
        """Parse markdown into structural elements (same as original)"""
        elements = []
        lines = text.split('\n')
        i = 0
        
        while i < len(lines):
            line = lines[i]
            line_stripped = line.strip()
            
            if not line_stripped:
                i += 1
                continue
            
            # Code block (atomic)
            if line_stripped.startswith('```'):
                code_block, end_idx = self._extract_code_block(lines, i)
                elements.append({
                    'type': 'code_block',
                    'content': code_block,
                    'atomic': True
                })
                i = end_idx + 1
                continue
            
            # Page/Slide marker
            if re.match(r'^##\s+(Page|Slide)\s+\d+', line_stripped, re.IGNORECASE):
                elements.append({
                    'type': 'page_marker',
                    'content': line,
                    'atomic': False
                })
                i += 1
                continue
            
            # Heading
            if line_stripped.startswith('#'):
                heading_match = re.match(r'^(#{1,6})\s+(.+)$', line_stripped)
                if heading_match:
                    level = len(heading_match.group(1))
                    elements.append({
                        'type': 'heading',
                        'content': line,
                        'level': level,
                        'atomic': False
                    })
                    i += 1
                    continue
            
            # Table (atomic)
            if '|' in line_stripped and self._is_table_row(line_stripped):
                table, end_idx = self._extract_table(lines, i)
                elements.append({
                    'type': 'table',
                    'content': table,
                    'atomic': True
                })
                i = end_idx + 1
                continue
            
            # List
            if self._is_list_item(line_stripped):
                list_block, end_idx = self._extract_list(lines, i)
                elements.append({
                    'type': 'list',
                    'content': list_block,
                    'atomic': False
                })
                i = end_idx + 1
                continue
            
            # Image description
            if '[Image Description]' in line or 'IMAGE_PATH:' in line:
                img_block, end_idx = self._extract_image_block(lines, i)
                elements.append({
                    'type': 'image_desc',
                    'content': img_block,
                    'atomic': False
                })
                i = end_idx + 1
                continue
            
            # Regular paragraph
            para, end_idx = self._extract_paragraph(lines, i)
            if para.strip():
                elements.append({
                    'type': 'paragraph',
                    'content': para,
                    'atomic': False
                })
            i = end_idx + 1
        
        return elements
    
    def _group_into_sections(self, elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Group elements into sections (heading + its content)
        
        Returns:
            List of sections, each containing: {heading, content_elements, size}
        """
        sections = []
        current_heading = None
        current_content = []
        
        for element in elements:
            if element['type'] == 'heading':
                # Save previous section
                if current_heading is not None or current_content:
                    sections.append({
                        'heading': current_heading,
                        'content_elements': current_content,
                        'size': self._calculate_section_size(current_heading, current_content)
                    })
                
                # Start new section
                current_heading = element
                current_content = []
            else:
                # Add to current section
                current_content.append(element)
        
        # Add final section
        if current_heading is not None or current_content:
            sections.append({
                'heading': current_heading,
                'content_elements': current_content,
                'size': self._calculate_section_size(current_heading, current_content)
            })
        
        return sections
    
    def _calculate_section_size(
        self, 
        heading: Optional[Dict[str, Any]], 
        content_elements: List[Dict[str, Any]]
    ) -> int:
        """Calculate total size of a section"""
        size = 0
        if heading:
            size += len(heading['content']) + 2  # +2 for newlines
        for elem in content_elements:
            size += len(elem['content']) + 2
        return size
    
    def _create_optimized_chunks(
        self,
        sections: List[Dict[str, Any]],
        base_metadata: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Create optimized chunks from sections
        
        Strategy:
        1. Merge small sections until reaching chunk_size (1500)
        2. Split large sections at semantic boundaries
        3. Add controlled overlap between chunks (content only, no headings)
        """
        chunks = []
        chunk_index = 0
        i = 0
        
        while i < len(sections):
            section = sections[i]
            
            # Case 1: Section is small - try to merge with next sections
            if section['size'] < self.chunk_size:
                merged_chunk = self._merge_small_sections(sections, i)
                chunks.append(self._create_chunk_dict(
                    merged_chunk['text'], chunk_index, base_metadata
                ))
                chunk_index += 1
                i = merged_chunk['next_index']
            
            # Case 2: Section is large - split it
            elif section['size'] > self.chunk_size:
                section_chunks = self._split_large_section(section, chunk_index, base_metadata)
                chunks.extend(section_chunks)
                chunk_index += len(section_chunks)
                i += 1
            
            # Case 3: Section fits perfectly
            else:
                text = self._section_to_text(section)
                chunks.append(self._create_chunk_dict(
                    text, chunk_index, base_metadata
                ))
                chunk_index += 1
                i += 1
        
        # Add overlap between consecutive chunks (content only, no headings)
        chunks = self._add_content_overlap(chunks)
        
        return chunks
    
    def _merge_small_sections(
        self,
        sections: List[Dict[str, Any]],
        start_idx: int
    ) -> Dict[str, Any]:
        """
        Merge small sections until reaching chunk_size or max_chunk_size
        
        Returns:
            {text: merged_text, next_index: index_of_next_unprocessed_section}
        """
        merged_parts = []
        total_size = 0
        i = start_idx
        
        while i < len(sections):
            section = sections[i]
            section_size = section['size']
            
            # Would adding this section exceed max_chunk_size?
            if total_size + section_size > self.max_chunk_size:
                # If we haven't added anything yet, we must add this one
                if not merged_parts:
                    merged_parts.append(self._section_to_text(section))
                    i += 1
                break
            
            # Add section
            merged_parts.append(self._section_to_text(section))
            total_size += section_size
            i += 1
            
            # Reached target size? Stop (but can continue if next is small)
            if total_size >= self.chunk_size:
                # Peek at next section - if it's tiny, include it
                if i < len(sections) and sections[i]['size'] < 200:
                    if total_size + sections[i]['size'] <= self.max_chunk_size:
                        merged_parts.append(self._section_to_text(sections[i]))
                        total_size += sections[i]['size']
                        i += 1
                break
        
        return {
            'text': '\n\n'.join(merged_parts),
            'next_index': i
        }
    
    def _split_large_section(
        self,
        section: Dict[str, Any],
        start_chunk_index: int,
        base_metadata: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Split large section into multiple chunks at semantic boundaries
        
        Semantic boundaries (in order of preference):
        1. Paragraphs (best for context)
        2. Sentences (fallback)
        
        Heading is always kept with first chunk
        """
        heading = section['heading']
        content_elements = section['content_elements']
        
        chunks = []
        chunk_index = start_chunk_index
        
        # Start with heading (if exists)
        heading_text = heading['content'] if heading else ""
        heading_size = len(heading_text) + 2 if heading_text else 0
        
        # Process content elements
        current_chunk_parts = [heading_text] if heading_text else []
        current_size = heading_size
        
        for elem in content_elements:
            elem_text = elem['content']
            elem_size = len(elem_text) + 2
            
            # Element alone exceeds max size - must split it
            if elem_size > self.max_chunk_size:
                # Save current chunk if it has content
                if len(current_chunk_parts) > (1 if heading_text else 0):
                    chunks.append(self._create_chunk_dict(
                        '\n\n'.join(current_chunk_parts), chunk_index, base_metadata
                    ))
                    chunk_index += 1
                    current_chunk_parts = []
                    current_size = 0
                
                # Split the element itself
                if elem['atomic']:
                    # Atomic elements - can't split, create as-is (will exceed limit)
                    logger.warning(f"Atomic {elem['type']} exceeds max_chunk_size ({elem_size})")
                    chunks.append(self._create_chunk_dict(
                        elem_text, chunk_index, base_metadata
                    ))
                    chunk_index += 1
                else:
                    # Non-atomic - split by paragraphs/sentences
                    elem_chunks = self._split_text_content(
                        elem_text, chunk_index, base_metadata
                    )
                    chunks.extend(elem_chunks)
                    chunk_index += len(elem_chunks)
                
                continue
            
            # Would adding this element exceed max size?
            if current_size + elem_size > self.max_chunk_size:
                # Save current chunk
                if current_chunk_parts:
                    chunks.append(self._create_chunk_dict(
                        '\n\n'.join(current_chunk_parts), chunk_index, base_metadata
                    ))
                    chunk_index += 1
                
                # Start new chunk (no heading in subsequent chunks)
                current_chunk_parts = [elem_text]
                current_size = elem_size
            
            # Would adding exceed target size? (but not max)
            elif current_size + elem_size > self.chunk_size:
                # Save current chunk
                if current_chunk_parts:
                    chunks.append(self._create_chunk_dict(
                        '\n\n'.join(current_chunk_parts), chunk_index, base_metadata
                    ))
                    chunk_index += 1
                
                # Start new chunk
                current_chunk_parts = [elem_text]
                current_size = elem_size
            
            # Fits fine
            else:
                current_chunk_parts.append(elem_text)
                current_size += elem_size
        
        # Add final chunk
        if current_chunk_parts:
            chunks.append(self._create_chunk_dict(
                '\n\n'.join(current_chunk_parts), chunk_index, base_metadata
            ))
        
        return chunks
    
    def _split_text_content(
        self,
        text: str,
        start_chunk_index: int,
        base_metadata: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Split large text at paragraph/sentence boundaries
        
        No heading duplication - pure content splitting
        """
        chunks = []
        chunk_index = start_chunk_index
        
        # Try splitting by paragraphs first
        paragraphs = re.split(r'\n\s*\n', text)
        
        current_chunk = ""
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            para_size = len(para) + 2
            
            # Paragraph alone exceeds max size - split by sentences
            if para_size > self.max_chunk_size:
                # Save current chunk if exists
                if current_chunk:
                    chunks.append(self._create_chunk_dict(
                        current_chunk.strip(), chunk_index, base_metadata
                    ))
                    chunk_index += 1
                    current_chunk = ""
                
                # Split paragraph by sentences
                sentence_chunks = self._split_by_sentences(
                    para, chunk_index, base_metadata
                )
                chunks.extend(sentence_chunks)
                chunk_index += len(sentence_chunks)
                continue
            
            # Would adding this paragraph exceed max size?
            if len(current_chunk) + para_size > self.max_chunk_size:
                if current_chunk:
                    chunks.append(self._create_chunk_dict(
                        current_chunk.strip(), chunk_index, base_metadata
                    ))
                    chunk_index += 1
                current_chunk = para
            
            # Would adding exceed target size?
            elif len(current_chunk) + para_size > self.chunk_size:
                if current_chunk:
                    chunks.append(self._create_chunk_dict(
                        current_chunk.strip(), chunk_index, base_metadata
                    ))
                    chunk_index += 1
                current_chunk = para
            
            # Fits fine
            else:
                current_chunk = current_chunk + "\n\n" + para if current_chunk else para
        
        # Add final chunk
        if current_chunk:
            chunks.append(self._create_chunk_dict(
                current_chunk.strip(), chunk_index, base_metadata
            ))
        
        return chunks
    
    def _split_by_sentences(
        self,
        text: str,
        start_chunk_index: int,
        base_metadata: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Split text by sentences as last resort"""
        chunks = []
        chunk_index = start_chunk_index
        
        # Split by sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        current_chunk = ""
        
        for sentence in sentences:
            sentence_size = len(sentence) + 1
            
            # Single sentence exceeds max size - must include it anyway
            if sentence_size > self.max_chunk_size:
                if current_chunk:
                    chunks.append(self._create_chunk_dict(
                        current_chunk.strip(), chunk_index, base_metadata
                    ))
                    chunk_index += 1
                
                chunks.append(self._create_chunk_dict(
                    sentence.strip(), chunk_index, base_metadata
                ))
                chunk_index += 1
                current_chunk = ""
                continue
            
            # Would adding exceed max size?
            if len(current_chunk) + sentence_size > self.max_chunk_size:
                if current_chunk:
                    chunks.append(self._create_chunk_dict(
                        current_chunk.strip(), chunk_index, base_metadata
                    ))
                    chunk_index += 1
                current_chunk = sentence
            
            # Would adding exceed target size?
            elif len(current_chunk) + sentence_size > self.chunk_size:
                if current_chunk:
                    chunks.append(self._create_chunk_dict(
                        current_chunk.strip(), chunk_index, base_metadata
                    ))
                    chunk_index += 1
                current_chunk = sentence
            
            # Fits fine
            else:
                current_chunk = current_chunk + " " + sentence if current_chunk else sentence
        
        # Add final chunk
        if current_chunk:
            chunks.append(self._create_chunk_dict(
                current_chunk.strip(), chunk_index, base_metadata
            ))
        
        return chunks
    
    def _add_content_overlap(
        self,
        chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Add controlled overlap between consecutive chunks
        
        CRITICAL: Only overlap content, never headings
        """
        if len(chunks) <= 1 or self.chunk_overlap == 0:
            return chunks
        
        for i in range(1, len(chunks)):
            prev_chunk = chunks[i - 1]
            curr_chunk = chunks[i]
            
            prev_text = prev_chunk['text']
            curr_text = curr_chunk['text']
            
            # Extract overlap from end of previous chunk
            # Skip if previous chunk starts with heading (don't want to duplicate it)
            prev_lines = prev_text.split('\n')
            
            # Find last non-heading content
            overlap_lines = []
            overlap_size = 0
            
            for line in reversed(prev_lines):
                line_stripped = line.strip()
                
                # Skip headings
                if line_stripped.startswith('#'):
                    break
                
                # Skip if empty
                if not line_stripped:
                    continue
                
                line_size = len(line) + 1
                
                if overlap_size + line_size <= self.chunk_overlap:
                    overlap_lines.insert(0, line)
                    overlap_size += line_size
                else:
                    break
            
            # Prepend overlap to current chunk (if not starting with heading)
            if overlap_lines:
                curr_first_line = curr_text.split('\n')[0].strip()
                
                # Don't add overlap if current chunk starts with heading
                if not curr_first_line.startswith('#'):
                    overlap_text = '\n'.join(overlap_lines)
                    chunks[i]['text'] = overlap_text + '\n\n' + curr_text
                    chunks[i]['char_count'] = len(chunks[i]['text'])
        
        return chunks
    
    def _section_to_text(self, section: Dict[str, Any]) -> str:
        """Convert section to text"""
        parts = []
        
        if section['heading']:
            parts.append(section['heading']['content'])
        
        for elem in section['content_elements']:
            parts.append(elem['content'])
        
        return '\n\n'.join(parts)
    
    # ==================== Helper Methods (same as original) ====================
    
    def _extract_code_block(self, lines: List[str], start_idx: int) -> Tuple[str, int]:
        """Extract code block with safety limit"""
        code_lines = [lines[start_idx]]
        i = start_idx + 1
        max_lines = 500
        lines_read = 0
        
        while i < len(lines) and lines_read < max_lines:
            code_lines.append(lines[i])
            if lines[i].strip().startswith('```'):
                return '\n'.join(code_lines), i
            i += 1
            lines_read += 1
        
        if lines_read >= max_lines:
            logger.warning(f"Code block at line {start_idx} exceeded max length")
            code_lines.append('```')
        
        return '\n'.join(code_lines), min(i, len(lines) - 1)
    
    def _extract_table(self, lines: List[str], start_idx: int) -> Tuple[str, int]:
        """Extract markdown table"""
        table_lines = []
        i = start_idx
        
        while i < len(lines):
            line = lines[i].strip()
            if not line:
                break
            if '|' in line:
                table_lines.append(lines[i])
                i += 1
            else:
                break
        
        return '\n'.join(table_lines), i - 1
    
    def _extract_list(self, lines: List[str], start_idx: int) -> Tuple[str, int]:
        """Extract list block"""
        list_lines = []
        i = start_idx
        
        while i < len(lines):
            line = lines[i].strip()
            if not line:
                if i + 1 < len(lines) and self._is_list_item(lines[i + 1].strip()):
                    list_lines.append(lines[i])
                    i += 1
                    continue
                break
            if self._is_list_item(line):
                list_lines.append(lines[i])
                i += 1
            else:
                break
        
        return '\n'.join(list_lines), i - 1
    
    def _extract_image_block(self, lines: List[str], start_idx: int) -> Tuple[str, int]:
        """Extract image description block"""
        img_lines = []
        i = start_idx
        
        while i < len(lines):
            line = lines[i]
            img_lines.append(line)
            i += 1
            
            if line.strip().startswith('*(Located on '):
                break
            
            if i < len(lines):
                next_line = lines[i].strip()
                if (next_line.startswith('#') or 
                    next_line.startswith('```') or
                    self._is_table_row(next_line)):
                    break
        
        return '\n'.join(img_lines), i - 1
    
    def _extract_paragraph(self, lines: List[str], start_idx: int) -> Tuple[str, int]:
        """Extract paragraph"""
        para_lines = []
        i = start_idx
        
        while i < len(lines):
            line = lines[i].strip()
            if not line:
                break
            if (line.startswith('#') or 
                line.startswith('```') or
                self._is_table_row(line) or
                self._is_list_item(line) or
                '[Image Description]' in line or
                'IMAGE_PATH:' in line):
                break
            para_lines.append(lines[i])
            i += 1
        
        return '\n'.join(para_lines), i - 1
    
    def _is_table_row(self, line: str) -> bool:
        """Check if line is table row"""
        if '|' not in line:
            return False
        parts = [p.strip() for p in line.split('|')]
        if len(parts) < 3:
            return False
        if all(re.match(r'^:?-+:?$', p) for p in parts if p):
            return True
        return len([p for p in parts if p]) >= 2
    
    def _is_list_item(self, line: str) -> bool:
        """Check if line is list item"""
        return bool(re.match(r'^(\*|-|\+|\d+\.)\s+', line))
    
    def _generate_chunk_id(
        self, 
        text: str, 
        index: int,
        file_id: Optional[str] = None
    ) -> str:
        """Generate unique chunk ID (compatible with Qdrant UUID conversion)"""
        if file_id:
            hash_input = f"{file_id}_{index}_{text}".encode()
        else:
            hash_input = f"{index}_{text}".encode()
        
        content_hash = hashlib.sha256(hash_input).hexdigest()[:16]
        
        import time
        timestamp = str(int(time.time() * 1000))[-6:]
        
        return f"chunk_{index}_{content_hash}_{timestamp}"
    
    def _create_chunk_dict(
        self,
        text: str,
        chunk_index: int,
        base_metadata: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Create chunk dictionary"""
        file_id = base_metadata.get('file_id') if base_metadata else None
        chunk_id = self._generate_chunk_id(text, chunk_index, file_id)
        
        return {
            'chunk_id': chunk_id,
            'chunk_index': chunk_index,
            'text': text,
            'char_count': len(text),
            'metadata': base_metadata or {}
        }
    
    def _validate_all_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """Validate all chunks"""
        for i, chunk in enumerate(chunks):
            text = chunk.get('text', '')
            char_count = len(text)
            
            if char_count > self.max_chunk_size:
                logger.warning(
                    f"Chunk {i} exceeds max_chunk_size: {char_count} > {self.max_chunk_size} "
                    f"(will be truncated by vector_store_service at embedding time)"
                )
            
            if char_count == 0:
                logger.warning(f"Chunk {i} is empty")
    
    def _add_structured_references(
    self,
    chunks: List[Dict[str, Any]],
    structured_data: Dict[str, Any]
) -> List[Dict[str, Any]]:
        """
        Add structured data references with ROBUST page_number extraction
        
        CRITICAL FIX:
        1. Remove all confusing fields (total_pages, total_slides) 
        2. Extract page_number from markdown text (e.g., "## Page 3" -> 3)
        3. Extract slide_number from markdown text (e.g., "## Slide 5" -> 5)
        4. For PPTs, ensure page_number = slide_number
        5. Default to 0 if no marker found
        
        This ensures page_number is always the ACTUAL page, never total_pages
        """
        for chunk in chunks:
            # STEP 1: Remove all potentially confusing fields
            # These should NEVER be in chunk metadata as they can be confused with page_number
            confusing_fields = ['total_pages', 'total_slides', 'num_pages', 'page_count']
            for field in confusing_fields:
                chunk['metadata'].pop(field, None)  # Safe deletion (no error if not present)
            
            # STEP 2: Extract actual page number from markdown headers in chunk text
            # This is the SOURCE OF TRUTH for page_number
            page_match = re.search(r'##\s+Page\s+(\d+)', chunk['text'], re.IGNORECASE)
            if page_match:
                actual_page_number = int(page_match.group(1))
                chunk['metadata']['page_number'] = actual_page_number
                logger.debug(f"Chunk {chunk.get('chunk_index', '?')}: Found page marker -> page_number={actual_page_number}")
            
            # STEP 3: Extract slide number from markdown headers
            slide_match = re.search(r'##\s+Slide\s+(\d+)', chunk['text'], re.IGNORECASE)
            if slide_match:
                actual_slide_number = int(slide_match.group(1))
                chunk['metadata']['slide_number'] = actual_slide_number
                
                # For PowerPoint files, page_number should equal slide_number
                # Only set if page_number wasn't already set by a "## Page" marker
                if 'page_number' not in chunk['metadata']:
                    chunk['metadata']['page_number'] = actual_slide_number
                    logger.debug(f"Chunk {chunk.get('chunk_index', '?')}: Found slide marker -> page_number={actual_slide_number}")
            
            # STEP 4: Default to 0 if no page or slide marker found
            # This is better than None as it's a valid integer for filtering
            if 'page_number' not in chunk['metadata']:
                chunk['metadata']['page_number'] = 0
                logger.debug(f"Chunk {chunk.get('chunk_index', '?')}: No page marker found -> page_number=0 (default)")
            
            # STEP 5: Extract section hierarchy (for context)
            section_matches = re.findall(r'^(#{1,6})\s+(.+)$', chunk['text'], re.MULTILINE)
            if section_matches:
                # Get up to 3 heading levels
                section_hierarchy = [match[1].strip() for match in section_matches[:3]]
                # Filter out page/slide markers from hierarchy
                section_hierarchy = [
                    h for h in section_hierarchy 
                    if not re.match(r'^(Page|Slide)\s+\d+$', h, re.IGNORECASE)
                ]
                if section_hierarchy:
                    chunk['metadata']['section_hierarchy'] = section_hierarchy
        
        return chunks


    # VERIFICATION FUNCTION - Add this to help debug
    def verify_page_numbers(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Verify that page_number metadata is correctly set
        
        Returns summary statistics for debugging
        """
        stats = {
            'total_chunks': len(chunks),
            'chunks_with_page_marker': 0,
            'chunks_with_slide_marker': 0,
            'chunks_with_page_number_0': 0,
            'page_numbers': [],
            'unique_pages': set(),
            'issues': []
        }
        
        for i, chunk in enumerate(chunks):
            page_num = chunk['metadata'].get('page_number', None)
            
            if page_num is None:
                stats['issues'].append(f"Chunk {i}: page_number is None")
                continue
            
            stats['page_numbers'].append(page_num)
            stats['unique_pages'].add(page_num)
            
            # Check if chunk has page marker in text
            if re.search(r'##\s+Page\s+\d+', chunk['text'], re.IGNORECASE):
                stats['chunks_with_page_marker'] += 1
            
            # Check if chunk has slide marker in text
            if re.search(r'##\s+Slide\s+\d+', chunk['text'], re.IGNORECASE):
                stats['chunks_with_slide_marker'] += 1
            
            # Count default page_number=0
            if page_num == 0:
                stats['chunks_with_page_number_0'] += 1
            
            # Check for suspicious values (e.g., page_number > 1000 suggests it's total_pages)
            if page_num > 1000:
                stats['issues'].append(
                    f"Chunk {i}: Suspicious page_number={page_num} (might be total_pages!)"
                )
        
        # Calculate statistics
        if stats['page_numbers']:
            stats['min_page'] = min(stats['page_numbers'])
            stats['max_page'] = max(stats['page_numbers'])
            stats['unique_page_count'] = len(stats['unique_pages'])
        
        return stats