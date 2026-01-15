"""
Optimized Markdown-Aware Chunking Service
==========================================
FIXED: Page number assignment now happens during parsing, not after chunking
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
    
    FIXED: Page numbers are extracted during parsing and tracked through all elements
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
        
        FIXED: Page numbers are now tracked from parsing through to final chunks
        
        Returns:
            List of chunk dictionaries with metadata
        """
        if not markdown_text or not markdown_text.strip():
            raise NoChunksCreatedError("Empty markdown text provided")
        
        with PerformanceTimer("chunking_process", log_result=False):
            # Parse into structural elements WITH PAGE NUMBERS
            elements = self._parse_markdown_structure(markdown_text)
            
            if not elements:
                raise NoChunksCreatedError("No structural elements parsed")
            
            logger.debug(f"Parsed {len(elements)} elements")
            
            # Group elements into heading-based sections (preserves page numbers)
            sections = self._group_into_sections(elements)
            logger.debug(f"Grouped into {len(sections)} sections")
            
            # Create optimized chunks from sections (propagates page numbers)
            chunks = self._create_optimized_chunks(sections, base_metadata)
            
            if not chunks:
                raise NoChunksCreatedError("No chunks created")
            
            # Validate all chunks
            self._validate_all_chunks(chunks)
            
            # Add structured data references (NO page number extraction here)
            chunks = self._add_structured_references(chunks, structured_data)
            
            logger.info(
                f"Created {len(chunks)} chunks "
                f"(avg: {sum(len(c['text']) for c in chunks) // len(chunks)}, "
                f"min: {min(len(c['text']) for c in chunks)}, "
                f"max: {max(len(c['text']) for c in chunks)})"
            )
        
        return chunks
    
    def _parse_markdown_structure(self, text: str) -> List[Dict[str, Any]]:
        """
        Parse markdown into structural elements
        
        FIXED: Now extracts and tracks page numbers during parsing
        """
        elements = []
        lines = text.split('\n')
        i = 0
        
        # Track current page number as we parse
        current_page_number = 0
        current_slide_number = 0
        
        while i < len(lines):
            line = lines[i]
            line_stripped = line.strip()
            
            if not line_stripped:
                i += 1
                continue
            
            # CRITICAL: Page/Slide marker detection - UPDATE CURRENT PAGE
            if re.match(r'^##\s+(Page|Slide)\s+(\d+)', line_stripped, re.IGNORECASE):
                marker_match = re.match(r'^##\s+(Page|Slide)\s+(\d+)', line_stripped, re.IGNORECASE)
                marker_type = marker_match.group(1).lower()
                marker_number = int(marker_match.group(2))
                
                if marker_type == 'page':
                    current_page_number = marker_number
                elif marker_type == 'slide':
                    current_slide_number = marker_number
                    # For presentations, page_number = slide_number
                    current_page_number = marker_number
                
                elements.append({
                    'type': 'page_marker',
                    'content': line,
                    'atomic': False,
                    'page_number': current_page_number,  # FIXED: Add page number
                    'slide_number': current_slide_number if current_slide_number > 0 else None
                })
                i += 1
                continue
            
            # Code block (atomic)
            if line_stripped.startswith('```'):
                code_block, end_idx = self._extract_code_block(lines, i)
                elements.append({
                    'type': 'code_block',
                    'content': code_block,
                    'atomic': True,
                    'page_number': current_page_number  # FIXED: Add page number
                })
                i = end_idx + 1
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
                        'atomic': False,
                        'page_number': current_page_number  # FIXED: Add page number
                    })
                    i += 1
                    continue
            
            # Table (atomic)
            if '|' in line_stripped and self._is_table_row(line_stripped):
                table, end_idx = self._extract_table(lines, i)
                elements.append({
                    'type': 'table',
                    'content': table,
                    'atomic': True,
                    'page_number': current_page_number  # FIXED: Add page number
                })
                i = end_idx + 1
                continue
            
            # List
            if self._is_list_item(line_stripped):
                list_block, end_idx = self._extract_list(lines, i)
                elements.append({
                    'type': 'list',
                    'content': list_block,
                    'atomic': False,
                    'page_number': current_page_number  # FIXED: Add page number
                })
                i = end_idx + 1
                continue
            
            # Image description
            if '[Image Description]' in line or 'IMAGE_PATH:' in line:
                img_block, end_idx = self._extract_image_block(lines, i)
                elements.append({
                    'type': 'image_desc',
                    'content': img_block,
                    'atomic': False,
                    'page_number': current_page_number  # FIXED: Add page number
                })
                i = end_idx + 1
                continue
            
            # Regular paragraph
            para, end_idx = self._extract_paragraph(lines, i)
            if para.strip():
                elements.append({
                    'type': 'paragraph',
                    'content': para,
                    'atomic': False,
                    'page_number': current_page_number  # FIXED: Add page number
                })
            i = end_idx + 1
        
        return elements
    
    def _group_into_sections(self, elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Group elements into sections (heading + its content)
        
        FIXED: Now tracks page numbers and page ranges for each section
        """
        sections = []
        current_heading = None
        current_content = []
        
        for element in elements:
            if element['type'] == 'heading':
                # Save previous section
                if current_heading is not None or current_content:
                    section = self._create_section(current_heading, current_content)
                    sections.append(section)
                
                # Start new section
                current_heading = element
                current_content = []
            else:
                # Add to current section
                current_content.append(element)
        
        # Add final section
        if current_heading is not None or current_content:
            section = self._create_section(current_heading, current_content)
            sections.append(section)
        
        return sections
    
    def _create_section(
        self,
        heading: Optional[Dict[str, Any]],
        content_elements: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Create section with page number tracking
        
        FIXED: Calculates page range for the section
        """
        size = self._calculate_section_size(heading, content_elements)
        
        # Calculate page range for this section
        page_numbers = set()
        
        if heading and heading.get('page_number', 0) > 0:
            page_numbers.add(heading['page_number'])
        
        for elem in content_elements:
            if elem.get('page_number', 0) > 0:
                page_numbers.add(elem['page_number'])
        
        # Determine page range
        if page_numbers:
            min_page = min(page_numbers)
            max_page = max(page_numbers)
            page_range = [min_page, max_page]
            primary_page = min_page  # Use first page as primary
        else:
            page_range = [0, 0]
            primary_page = 0
        
        return {
            'heading': heading,
            'content_elements': content_elements,
            'size': size,
            'page_range': page_range,  # FIXED: Track page range
            'primary_page': primary_page  # FIXED: Track primary page
        }
    
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
        
        FIXED: Propagates page numbers from sections to chunks
        """
        chunks = []
        chunk_index = 0
        i = 0
        
        while i < len(sections):
            section = sections[i]
            
            # Case 1: Section is small - try to merge with next sections
            if section['size'] < self.chunk_size:
                merged_chunk = self._merge_small_sections(sections, i)
                chunk_dict = self._create_chunk_dict(
                    merged_chunk['text'], 
                    chunk_index, 
                    base_metadata,
                    page_info=merged_chunk['page_info']  # FIXED: Pass page info
                )
                chunks.append(chunk_dict)
                chunk_index += 1
                i = merged_chunk['next_index']
            
            # Case 2: Section is large - split it
            elif section['size'] > self.chunk_size:
                section_chunks = self._split_large_section(
                    section, chunk_index, base_metadata
                )
                chunks.extend(section_chunks)
                chunk_index += len(section_chunks)
                i += 1
            
            # Case 3: Section fits perfectly
            else:
                text = self._section_to_text(section)
                chunk_dict = self._create_chunk_dict(
                    text, 
                    chunk_index, 
                    base_metadata,
                    page_info={
                        'page_number': section['primary_page'],
                        'page_range': section['page_range']
                    }
                )
                chunks.append(chunk_dict)
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
        
        FIXED: Tracks page range across merged sections
        """
        merged_parts = []
        total_size = 0
        i = start_idx
        
        # Track page numbers from all merged sections
        all_page_numbers = set()
        
        while i < len(sections):
            section = sections[i]
            section_size = section['size']
            
            # Would adding this section exceed max_chunk_size?
            if total_size + section_size > self.max_chunk_size:
                # If we haven't added anything yet, we must add this one
                if not merged_parts:
                    merged_parts.append(self._section_to_text(section))
                    all_page_numbers.update(
                        p for p in section['page_range'] if p > 0
                    )
                    i += 1
                break
            
            # Add section
            merged_parts.append(self._section_to_text(section))
            all_page_numbers.update(
                p for p in section['page_range'] if p > 0
            )
            total_size += section_size
            i += 1
            
            # Reached target size? Stop (but can continue if next is small)
            if total_size >= self.chunk_size:
                # Peek at next section - if it's tiny, include it
                if i < len(sections) and sections[i]['size'] < 200:
                    if total_size + sections[i]['size'] <= self.max_chunk_size:
                        merged_parts.append(self._section_to_text(sections[i]))
                        all_page_numbers.update(
                            p for p in sections[i]['page_range'] if p > 0
                        )
                        total_size += sections[i]['size']
                        i += 1
                break
        
        # Calculate page info for merged chunk
        if all_page_numbers:
            page_list = sorted(all_page_numbers)
            page_info = {
                'page_number': page_list[0],  # Use first page as primary
                'page_range': [page_list[0], page_list[-1]]
            }
        else:
            page_info = {
                'page_number': 0,
                'page_range': [0, 0]
            }
        
        return {
            'text': '\n\n'.join(merged_parts),
            'next_index': i,
            'page_info': page_info  # FIXED: Include page info
        }
    
    def _split_large_section(
        self,
        section: Dict[str, Any],
        start_chunk_index: int,
        base_metadata: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Split large section into multiple chunks at semantic boundaries
        
        FIXED: Propagates page numbers to split chunks
        """
        heading = section['heading']
        content_elements = section['content_elements']
        section_page_range = section['page_range']
        section_primary_page = section['primary_page']
        
        chunks = []
        chunk_index = start_chunk_index
        
        # Start with heading (if exists)
        heading_text = heading['content'] if heading else ""
        heading_size = len(heading_text) + 2 if heading_text else 0
        
        # Process content elements
        current_chunk_parts = [heading_text] if heading_text else []
        current_size = heading_size
        current_chunk_pages = set()
        
        if heading and heading.get('page_number', 0) > 0:
            current_chunk_pages.add(heading['page_number'])
        
        for elem in content_elements:
            elem_text = elem['content']
            elem_size = len(elem_text) + 2
            elem_page = elem.get('page_number', 0)
            
            # Element alone exceeds max size - must split it
            if elem_size > self.max_chunk_size:
                # Save current chunk if it has content
                if len(current_chunk_parts) > (1 if heading_text else 0):
                    page_info = self._calculate_page_info(current_chunk_pages, section_primary_page)
                    chunks.append(self._create_chunk_dict(
                        '\n\n'.join(current_chunk_parts), 
                        chunk_index, 
                        base_metadata,
                        page_info=page_info
                    ))
                    chunk_index += 1
                    current_chunk_parts = []
                    current_size = 0
                    current_chunk_pages = set()
                
                # Split the element itself
                if elem['atomic']:
                    # Atomic elements - can't split, create as-is (will exceed limit)
                    logger.warning(f"Atomic {elem['type']} exceeds max_chunk_size ({elem_size})")
                    page_info = self._calculate_page_info({elem_page} if elem_page > 0 else set(), section_primary_page)
                    chunks.append(self._create_chunk_dict(
                        elem_text, 
                        chunk_index, 
                        base_metadata,
                        page_info=page_info
                    ))
                    chunk_index += 1
                else:
                    # Non-atomic - split by paragraphs/sentences
                    elem_chunks = self._split_text_content(
                        elem_text, 
                        chunk_index, 
                        base_metadata,
                        elem_page  # FIXED: Pass element's page number
                    )
                    chunks.extend(elem_chunks)
                    chunk_index += len(elem_chunks)
                
                continue
            
            # Would adding this element exceed max size?
            if current_size + elem_size > self.max_chunk_size:
                # Save current chunk
                if current_chunk_parts:
                    page_info = self._calculate_page_info(current_chunk_pages, section_primary_page)
                    chunks.append(self._create_chunk_dict(
                        '\n\n'.join(current_chunk_parts), 
                        chunk_index, 
                        base_metadata,
                        page_info=page_info
                    ))
                    chunk_index += 1
                
                # Start new chunk (no heading in subsequent chunks)
                current_chunk_parts = [elem_text]
                current_size = elem_size
                current_chunk_pages = {elem_page} if elem_page > 0 else set()
            
            # Would adding exceed target size? (but not max)
            elif current_size + elem_size > self.chunk_size:
                # Save current chunk
                if current_chunk_parts:
                    page_info = self._calculate_page_info(current_chunk_pages, section_primary_page)
                    chunks.append(self._create_chunk_dict(
                        '\n\n'.join(current_chunk_parts), 
                        chunk_index, 
                        base_metadata,
                        page_info=page_info
                    ))
                    chunk_index += 1
                
                # Start new chunk
                current_chunk_parts = [elem_text]
                current_size = elem_size
                current_chunk_pages = {elem_page} if elem_page > 0 else set()
            
            # Fits fine
            else:
                current_chunk_parts.append(elem_text)
                current_size += elem_size
                if elem_page > 0:
                    current_chunk_pages.add(elem_page)
        
        # Add final chunk
        if current_chunk_parts:
            page_info = self._calculate_page_info(current_chunk_pages, section_primary_page)
            chunks.append(self._create_chunk_dict(
                '\n\n'.join(current_chunk_parts), 
                chunk_index, 
                base_metadata,
                page_info=page_info
            ))
        
        return chunks
    
    def _split_text_content(
        self,
        text: str,
        start_chunk_index: int,
        base_metadata: Optional[Dict[str, Any]],
        element_page: int = 0  # FIXED: Add page number parameter
    ) -> List[Dict[str, Any]]:
        """
        Split large text at paragraph/sentence boundaries
        
        FIXED: Propagates page number to split text chunks
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
                    page_info = {'page_number': element_page, 'page_range': [element_page, element_page]}
                    chunks.append(self._create_chunk_dict(
                        current_chunk.strip(), 
                        chunk_index, 
                        base_metadata,
                        page_info=page_info
                    ))
                    chunk_index += 1
                    current_chunk = ""
                
                # Split paragraph by sentences
                sentence_chunks = self._split_by_sentences(
                    para, chunk_index, base_metadata, element_page
                )
                chunks.extend(sentence_chunks)
                chunk_index += len(sentence_chunks)
                continue
            
            # Would adding this paragraph exceed max size?
            if len(current_chunk) + para_size > self.max_chunk_size:
                if current_chunk:
                    page_info = {'page_number': element_page, 'page_range': [element_page, element_page]}
                    chunks.append(self._create_chunk_dict(
                        current_chunk.strip(), 
                        chunk_index, 
                        base_metadata,
                        page_info=page_info
                    ))
                    chunk_index += 1
                current_chunk = para
            
            # Would adding exceed target size?
            elif len(current_chunk) + para_size > self.chunk_size:
                if current_chunk:
                    page_info = {'page_number': element_page, 'page_range': [element_page, element_page]}
                    chunks.append(self._create_chunk_dict(
                        current_chunk.strip(), 
                        chunk_index, 
                        base_metadata,
                        page_info=page_info
                    ))
                    chunk_index += 1
                current_chunk = para
            
            # Fits fine
            else:
                current_chunk = current_chunk + "\n\n" + para if current_chunk else para
        
        # Add final chunk
        if current_chunk:
            page_info = {'page_number': element_page, 'page_range': [element_page, element_page]}
            chunks.append(self._create_chunk_dict(
                current_chunk.strip(), 
                chunk_index, 
                base_metadata,
                page_info=page_info
            ))
        
        return chunks
    
    def _split_by_sentences(
        self,
        text: str,
        start_chunk_index: int,
        base_metadata: Optional[Dict[str, Any]],
        element_page: int = 0  # FIXED: Add page number parameter
    ) -> List[Dict[str, Any]]:
        """
        Split text by sentences as last resort
        
        FIXED: Propagates page number to sentence-split chunks
        """
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
                    page_info = {'page_number': element_page, 'page_range': [element_page, element_page]}
                    chunks.append(self._create_chunk_dict(
                        current_chunk.strip(), 
                        chunk_index, 
                        base_metadata,
                        page_info=page_info
                    ))
                    chunk_index += 1
                
                page_info = {'page_number': element_page, 'page_range': [element_page, element_page]}
                chunks.append(self._create_chunk_dict(
                    sentence.strip(), 
                    chunk_index, 
                    base_metadata,
                    page_info=page_info
                ))
                chunk_index += 1
                current_chunk = ""
                continue
            
            # Would adding exceed max size?
            if len(current_chunk) + sentence_size > self.max_chunk_size:
                if current_chunk:
                    page_info = {'page_number': element_page, 'page_range': [element_page, element_page]}
                    chunks.append(self._create_chunk_dict(
                        current_chunk.strip(), 
                        chunk_index, 
                        base_metadata,
                        page_info=page_info
                    ))
                    chunk_index += 1
                current_chunk = sentence
            
            # Would adding exceed target size?
            elif len(current_chunk) + sentence_size > self.chunk_size:
                if current_chunk:
                    page_info = {'page_number': element_page, 'page_range': [element_page, element_page]}
                    chunks.append(self._create_chunk_dict(
                        current_chunk.strip(), 
                        chunk_index, 
                        base_metadata,
                        page_info=page_info
                    ))
                    chunk_index += 1
                current_chunk = sentence
            
            # Fits fine
            else:
                current_chunk = current_chunk + " " + sentence if current_chunk else sentence
        
        # Add final chunk
        if current_chunk:
            page_info = {'page_number': element_page, 'page_range': [element_page, element_page]}
            chunks.append(self._create_chunk_dict(
                current_chunk.strip(), 
                chunk_index, 
                base_metadata,
                page_info=page_info
            ))
        
        return chunks
    
    def _calculate_page_info(self, page_set: set, fallback_page: int) -> Dict[str, Any]:
        """
        Calculate page_number and page_range from a set of page numbers
        
        Args:
            page_set: Set of page numbers in the chunk
            fallback_page: Page to use if set is empty
            
        Returns:
            Dict with page_number and page_range
        """
        if page_set:
            page_list = sorted(page_set)
            return {
                'page_number': page_list[0],
                'page_range': [page_list[0], page_list[-1]]
            }
        else:
            return {
                'page_number': fallback_page,
                'page_range': [fallback_page, fallback_page]
            }
    
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
    
    # ==================== Helper Methods ====================
    
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
    base_metadata: Optional[Dict[str, Any]],
    page_info: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
        file_id = base_metadata.get('file_id') if base_metadata else None
        chunk_id = self._generate_chunk_id(text, chunk_index, file_id)
        
        metadata = dict(base_metadata) if base_metadata else {}
        
        # Add page information
        if page_info:
            metadata['page_number'] = page_info.get('page_number', 0)
            metadata['page_range'] = page_info.get('page_range', [0, 0])
        else:
            metadata['page_number'] = 0
            metadata['page_range'] = [0, 0]
        
        # CRITICAL FIX: Store chunk_index in BOTH places
        metadata['chunk_index'] = chunk_index
        
        return {
            'chunk_id': chunk_id,
            'chunk_index': chunk_index,  # ← Top level (for backward compatibility)
            'text': text,
            'char_count': len(text),
            'metadata': metadata  # ← Also in metadata (for vector store)
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
        Add structured data references
        
        This function now only adds section hierarchy for context
        """
        for chunk in chunks:
            # CRITICAL: DO NOT touch page_number - it's already correct
            # Just verify it exists and log for debugging
            page_num = chunk['metadata'].get('page_number', -1)
            page_range = chunk['metadata'].get('page_range', [-1, -1])
            
            if page_num == -1:
                logger.warning(
                    f"Chunk {chunk.get('chunk_index', '?')} missing page_number - "
                    f"this should not happen!"
                )
            else:
                logger.debug(
                    f"Chunk {chunk.get('chunk_index', '?')}: "
                    f"page_number={page_num}, page_range={page_range}"
                )
            
            # Extract section hierarchy for context (optional enhancement)
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
