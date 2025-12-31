"""
Markdown-Aware Chunking Service - FIXED VERSION
Fixed critical bug in code block extraction that could cause infinite loop
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
    Markdown-aware semantic chunking with structure preservation
    
    FIXED: Code block extraction bug that could cause infinite loop
    
    Features:
    - Splits by headings (#, ##, ###) before paragraphs
    - Preserves code blocks (```...```) as atomic units
    - Keeps tables intact (| --- |)
    - Doesn't break lists mid-item
    - Respects page/slide markers
    - Handles image descriptions
    - Strict size validation
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
            f"Markdown-aware chunking configured: size={self.chunk_size}, "
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
        Split markdown into semantic chunks with structure awareness
        
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
            # Parse markdown into structural elements
            elements = self._parse_markdown_structure(markdown_text)
            
            logger.debug(f"Parsed {len(elements)} structural elements")
            
            if not elements:
                raise NoChunksCreatedError("No structural elements parsed from markdown")
            
            # Group elements into chunks
            chunks = self._group_elements_into_chunks(elements, base_metadata)
            
            if not chunks:
                raise NoChunksCreatedError(
                    "No chunks created. Document may be too short or improperly formatted."
                )
            
            # CRITICAL: Validate ALL chunks before returning
            self._validate_all_chunks(chunks)
            
            # Add structured data references
            chunks = self._add_structured_references(chunks, structured_data)
            
            logger.info(
                f"Created {len(chunks)} chunks "
                f"(avg size: {sum(len(c['text']) for c in chunks) // len(chunks)} chars)"
            )
        
        return chunks
    
    def _parse_markdown_structure(self, text: str) -> List[Dict[str, Any]]:
        """
        FIXED: Parse markdown into structural elements
        
        Fix: Added safety limits to prevent infinite loops in code block extraction
        
        Args:
            text: Markdown text
            
        Returns:
            List of element dictionaries
        """
        elements = []
        lines = text.split('\n')
        i = 0
        
        while i < len(lines):
            line = lines[i]
            line_stripped = line.strip()
            
            # Empty line - skip
            if not line_stripped:
                i += 1
                continue
            
            # Code block (atomic) - FIXED: with safety limit
            if line_stripped.startswith('```'):
                code_block, end_idx = self._extract_code_block(lines, i)
                elements.append({
                    'type': 'code_block',
                    'content': code_block,
                    'atomic': True  # Never split
                })
                i = end_idx + 1
                continue
            
            # Page/Slide marker (keep with following content)
            if re.match(r'^##\s+(Page|Slide)\s+\d+', line_stripped, re.IGNORECASE):
                elements.append({
                    'type': 'page_marker',
                    'content': line,
                    'atomic': False
                })
                i += 1
                continue
            
            # Heading (split point)
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
            
            # Table (atomic - keep intact)
            if '|' in line_stripped and self._is_table_row(line_stripped):
                table, end_idx = self._extract_table(lines, i)
                elements.append({
                    'type': 'table',
                    'content': table,
                    'atomic': True  # Never split tables
                })
                i = end_idx + 1
                continue
            
            # List item (group with other list items)
            if self._is_list_item(line_stripped):
                list_block, end_idx = self._extract_list(lines, i)
                elements.append({
                    'type': 'list',
                    'content': list_block,
                    'atomic': False  # Can split between items if needed
                })
                i = end_idx + 1
                continue
            
            # Image description block (keep with context)
            if '[Image Description]' in line or 'IMAGE_PATH:' in line:
                img_block, end_idx = self._extract_image_block(lines, i)
                elements.append({
                    'type': 'image_desc',
                    'content': img_block,
                    'atomic': False
                })
                i = end_idx + 1
                continue
            
            # Regular paragraph (accumulate until blank line)
            para, end_idx = self._extract_paragraph(lines, i)
            if para.strip():
                elements.append({
                    'type': 'paragraph',
                    'content': para,
                    'atomic': False
                })
            i = end_idx + 1
        
        return elements
    
    def _extract_code_block(self, lines: List[str], start_idx: int) -> Tuple[str, int]:
        """
        FIXED: Extract complete code block with safety limit
        
        Fix: Added max_lines limit to prevent infinite loop if closing ``` is missing
        
        Args:
            lines: All lines
            start_idx: Starting line index (opening ```)
            
        Returns:
            Tuple of (code_block_text, end_index)
        """
        code_lines = [lines[start_idx]]
        i = start_idx + 1
        max_lines = 500  # SAFETY LIMIT: Max 500 lines in a code block
        lines_read = 0
        
        while i < len(lines) and lines_read < max_lines:
            code_lines.append(lines[i])
            
            # Found closing ```
            if lines[i].strip().startswith('```'):
                return '\n'.join(code_lines), i
            
            i += 1
            lines_read += 1
        
        # FIXED: If we exit loop without finding closing ```, still return what we have
        if lines_read >= max_lines:
            logger.warning(
                f"Code block starting at line {start_idx} exceeded max length "
                f"({max_lines} lines). Treating as unclosed block."
            )
            # Add closing ``` to make it valid
            code_lines.append('```')
        
        return '\n'.join(code_lines), min(i, len(lines) - 1)
    
    def _extract_table(self, lines: List[str], start_idx: int) -> Tuple[str, int]:
        """Extract complete markdown table"""
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
        """Extract list block (consecutive list items)"""
        list_lines = []
        i = start_idx
        
        while i < len(lines):
            line = lines[i].strip()
            if not line:
                # Check if next line is also a list item (allow one blank line)
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
        """Extract image description block (caption + description + location)"""
        img_lines = []
        i = start_idx
        
        # Collect all lines related to this image
        while i < len(lines):
            line = lines[i]
            img_lines.append(line)
            i += 1
            
            # Stop after location marker
            if line.strip().startswith('*(Located on '):
                break
            
            # Or if we hit next structural element
            if i < len(lines):
                next_line = lines[i].strip()
                if (next_line.startswith('#') or 
                    next_line.startswith('```') or
                    self._is_table_row(next_line)):
                    break
        
        return '\n'.join(img_lines), i - 1
    
    def _extract_paragraph(self, lines: List[str], start_idx: int) -> Tuple[str, int]:
        """Extract paragraph until blank line"""
        para_lines = []
        i = start_idx
        
        while i < len(lines):
            line = lines[i].strip()
            if not line:
                break
            # Stop at structural elements
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
        """Check if line is a table row"""
        line = line.strip()
        # Must have pipes and at least one on each side
        if not line.startswith('|') or not line.endswith('|'):
            return False
        # Table separator row: | --- | --- |
        if re.match(r'^\|[\s\-:]+\|', line):
            return True
        # Regular table row with content
        if line.count('|') >= 2:
            return True
        return False
    
    def _is_list_item(self, line: str) -> bool:
        """Check if line is a list item"""
        line = line.strip()
        # Unordered: -, *, +
        if re.match(r'^[-*+]\s', line):
            return True
        # Ordered: 1., 2., etc.
        if re.match(r'^\d+\.\s', line):
            return True
        return False
    
    def _group_elements_into_chunks(
        self,
        elements: List[Dict[str, Any]],
        base_metadata: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Group elements into chunks respecting structure and size limits
        
        Strategy:
        1. Start new chunk at headings (preferred split point)
        2. Keep atomic elements intact (code blocks, tables)
        3. Merge small sections
        4. Split large sections by paragraphs if needed
        
        Args:
            elements: List of parsed elements
            base_metadata: Base metadata
            
        Returns:
            List of chunk dictionaries
        """
        chunks = []
        current_chunk_elements = []
        current_size = 0
        chunk_index = 0
        
        for i, element in enumerate(elements):
            element_size = len(element['content'])
            
            # Check if element alone exceeds max size
            if element_size > self.max_chunk_size:
                # Save current chunk if exists
                if current_chunk_elements:
                    chunk_text = self._combine_elements(current_chunk_elements)
                    chunks.append(self._create_chunk_dict(
                        chunk_text, chunk_index, base_metadata
                    ))
                    chunk_index += 1
                    current_chunk_elements = []
                    current_size = 0
                
                # Handle oversized element
                if element.get('atomic'):
                    # Atomic elements (code blocks, tables) - truncate
                    logger.warning(
                        f"Atomic {element['type']} exceeds max size "
                        f"({element_size} > {self.max_chunk_size}), truncating"
                    )
                    truncated = element['content'][:self.max_chunk_size - 50] + "\n...[truncated]"
                    chunks.append(self._create_chunk_dict(
                        truncated, chunk_index, base_metadata
                    ))
                    chunk_index += 1
                else:
                    # Non-atomic elements (paragraphs, lists) - split intelligently
                    logger.warning(
                        f"Large {element['type']} exceeds max size "
                        f"({element_size} > {self.max_chunk_size}), splitting"
                    )
                    sub_chunks = self._split_large_element(
                        element, chunk_index, base_metadata
                    )
                    chunks.extend(sub_chunks)
                    chunk_index += len(sub_chunks)
                
                continue
            
            # Check if adding this element would exceed target size
            potential_size = current_size + element_size + 2  # +2 for newlines
            
            # Decision point: start new chunk?
            should_start_new = False
            
            if potential_size > self.chunk_size:
                # Only start new chunk at good boundaries
                if element['type'] == 'heading':
                    should_start_new = True
                elif element['type'] == 'page_marker':
                    should_start_new = True
                elif current_size > self.chunk_size * 0.7:  # 70% full, can split
                    should_start_new = True
            
            # Would exceed max size - must split
            if potential_size > self.max_chunk_size:
                should_start_new = True
            
            if should_start_new and current_chunk_elements:
                # Save current chunk
                chunk_text = self._combine_elements(current_chunk_elements)
                chunks.append(self._create_chunk_dict(
                    chunk_text, chunk_index, base_metadata
                ))
                chunk_index += 1
                
                # Start new chunk with overlap
                if self.chunk_overlap > 0:
                    overlap_elements = self._get_overlap_elements(
                        current_chunk_elements, self.chunk_overlap
                    )
                    current_chunk_elements = overlap_elements
                    current_size = sum(len(e['content']) for e in overlap_elements)
                else:
                    current_chunk_elements = []
                    current_size = 0
            
            # Add element to current chunk
            current_chunk_elements.append(element)
            current_size += element_size + 2
        
        # Add final chunk
        if current_chunk_elements:
            chunk_text = self._combine_elements(current_chunk_elements)
            chunks.append(self._create_chunk_dict(
                chunk_text, chunk_index, base_metadata
            ))
        
        return chunks
    
    def _combine_elements(self, elements: List[Dict[str, Any]]) -> str:
        """Combine elements into text with proper spacing"""
        if not elements:
            return ""
        
        parts = []
        for element in elements:
            content = element['content']
            if element['type'] == 'heading':
                if parts:
                    parts.append('\n')  # Extra space before headings
                parts.append(content)
            elif element['type'] in ['code_block', 'table', 'image_desc']:
                if parts:
                    parts.append('\n')
                parts.append(content)
            else:
                parts.append(content)
        
        return '\n'.join(parts)
    
    def _get_overlap_elements(
        self,
        elements: List[Dict[str, Any]],
        target_overlap: int
    ) -> List[Dict[str, Any]]:
        """Get elements for overlap (from end of chunk)"""
        if not elements:
            return []
        
        overlap_elements = []
        overlap_size = 0
        
        # Work backwards from end
        for element in reversed(elements):
            element_size = len(element['content'])
            
            if overlap_size + element_size <= target_overlap:
                overlap_elements.insert(0, element)
                overlap_size += element_size
            else:
                break
        
        return overlap_elements
    
    def _split_large_element(
        self,
        element: Dict[str, Any],
        start_chunk_index: int,
        base_metadata: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Split a large non-atomic element into multiple chunks"""
        content = element['content']
        element_type = element['type']
        chunks = []
        chunk_index = start_chunk_index
        
        if element_type == 'list':
            chunks = self._split_large_list(content, chunk_index, base_metadata)
        elif element_type == 'paragraph':
            chunks = self._split_large_paragraph(content, chunk_index, base_metadata)
        elif element_type == 'heading':
            chunks = self._split_heading_section(content, chunk_index, base_metadata)
        else:
            chunks = self._split_large_paragraph(content, chunk_index, base_metadata)
        
        return chunks
    
    def _split_large_list(
        self,
        list_content: str,
        start_chunk_index: int,
        base_metadata: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Split large list by items"""
        chunks = []
        lines = list_content.split('\n')
        current_chunk_lines = []
        current_size = 0
        chunk_index = start_chunk_index
        
        for line in lines:
            line_size = len(line) + 1
            
            if current_size + line_size > self.max_chunk_size:
                if current_chunk_lines:
                    chunk_text = '\n'.join(current_chunk_lines)
                    chunks.append(self._create_chunk_dict(
                        chunk_text, chunk_index, base_metadata
                    ))
                    chunk_index += 1
                    current_chunk_lines = []
                    current_size = 0
            
            current_chunk_lines.append(line)
            current_size += line_size
        
        if current_chunk_lines:
            chunk_text = '\n'.join(current_chunk_lines)
            chunks.append(self._create_chunk_dict(
                chunk_text, chunk_index, base_metadata
            ))
        
        return chunks
    
    def _split_large_paragraph(
        self,
        paragraph: str,
        start_chunk_index: int,
        base_metadata: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Split large paragraph by sentences"""
        chunks = []
        sentences = re.split(r'(?<=[.!?])\s+', paragraph)
        
        current_chunk = ""
        chunk_index = start_chunk_index
        
        for sentence in sentences:
            potential_size = len(current_chunk) + len(sentence) + 1
            
            if potential_size <= self.max_chunk_size:
                current_chunk = (
                    current_chunk + " " + sentence 
                    if current_chunk else sentence
                )
            else:
                if current_chunk.strip():
                    chunks.append(self._create_chunk_dict(
                        current_chunk.strip(), chunk_index, base_metadata
                    ))
                    chunk_index += 1
                
                if len(sentence) > self.max_chunk_size:
                    logger.warning(
                        f"Single sentence exceeds max size ({len(sentence)} chars), "
                        f"truncating to {self.max_chunk_size}"
                    )
                    truncated = sentence[:self.max_chunk_size - 50]
                    last_space = truncated.rfind(' ')
                    if last_space > 0:
                        truncated = truncated[:last_space]
                    sentence = truncated + "..."
                
                current_chunk = sentence
        
        if current_chunk.strip():
            chunks.append(self._create_chunk_dict(
                current_chunk.strip(), chunk_index, base_metadata
            ))
        
        return chunks
    
    def _split_heading_section(
        self,
        section_content: str,
        start_chunk_index: int,
        base_metadata: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Split heading section by paragraphs"""
        chunks = []
        parts = re.split(r'\n\s*\n', section_content)
        
        current_chunk_parts = []
        current_size = 0
        chunk_index = start_chunk_index
        
        for part in parts:
            part = part.strip()
            if not part:
                continue
            
            part_size = len(part) + 2
            
            if current_size + part_size > self.max_chunk_size:
                if current_chunk_parts:
                    chunk_text = '\n\n'.join(current_chunk_parts)
                    chunks.append(self._create_chunk_dict(
                        chunk_text, chunk_index, base_metadata
                    ))
                    chunk_index += 1
                    current_chunk_parts = []
                    current_size = 0
                
                if part_size > self.max_chunk_size:
                    sentence_chunks = self._split_large_paragraph(
                        part, chunk_index, base_metadata
                    )
                    chunks.extend(sentence_chunks)
                    chunk_index += len(sentence_chunks)
                else:
                    current_chunk_parts.append(part)
                    current_size = part_size
            else:
                current_chunk_parts.append(part)
                current_size += part_size
        
        if current_chunk_parts:
            chunk_text = '\n\n'.join(current_chunk_parts)
            chunks.append(self._create_chunk_dict(
                chunk_text, chunk_index, base_metadata
            ))
        
        return chunks
    
    def _generate_chunk_id(
    self, 
    text: str, 
    index: int,
    file_id: Optional[str] = None
) -> str:
        """Generate unique chunk ID with file_id"""
        if file_id:
            hash_input = f"{file_id}_{index}_{text}".encode()
        else:
            hash_input = f"{index}_{text}".encode()
        
        content_hash = hashlib.sha256(hash_input).hexdigest()[:16]
        
        # Add timestamp for extra uniqueness
        import time
        timestamp = str(int(time.time() * 1000))[-6:]
        
        return f"chunk_{index}_{content_hash}_{timestamp}"

    def _create_chunk_dict(
        self,
        text: str,
        chunk_index: int,
        base_metadata: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Create chunk with file_id"""
        file_id = base_metadata.get('file_id') if base_metadata else None
        chunk_id = self._generate_chunk_id(text, chunk_index, file_id)  # Pass file_id
        
        chunk = {
            'chunk_id': chunk_id,
            'chunk_index': chunk_index,
            'text': text,
            'char_count': len(text),
            'metadata': base_metadata or {}
        }
        
        return chunk
    
    def _validate_all_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """Validate all chunks meet size requirements"""
        for i, chunk in enumerate(chunks):
            text = chunk.get('text', '')
            char_count = len(text)
            
            if char_count > self.max_chunk_size:
                raise ChunkSizeExceededError(
                    f"Chunk {i} exceeds max size: {char_count} > {self.max_chunk_size}"
                )
            
            if char_count == 0:
                logger.warning(f"Chunk {i} is empty")
    
    def _add_structured_references(
        self,
        chunks: List[Dict[str, Any]],
        structured_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Add structured data references to chunks"""
        for chunk in chunks:
            # Extract page/slide numbers
            page_match = re.search(r'##\s+Page\s+(\d+)', chunk['text'], re.IGNORECASE)
            if page_match:
                chunk['metadata']['page_number'] = int(page_match.group(1))
            
            slide_match = re.search(r'##\s+Slide\s+(\d+)', chunk['text'], re.IGNORECASE)
            if slide_match:
                chunk['metadata']['slide_number'] = int(slide_match.group(1))
            
            # Extract section hierarchy
            section_matches = re.findall(r'^(#{1,6})\s+(.+)$', chunk['text'], re.MULTILINE)
            if section_matches:
                section_hierarchy = [match[1].strip() for match in section_matches[:3]]
                chunk['metadata']['section_hierarchy'] = section_hierarchy
        
        return chunks