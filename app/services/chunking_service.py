"""
FIXED Optimized Markdown-Aware Chunking Service with Token Constraints
========================================================================

FIXES APPLIED:
1. ✅ BUG #1: Page numbers now start at 1 (not 0)
2. ✅ BUG #2: Code blocks have configurable max lines, no artificial closing markers
3. ✅ BUG #3: Page range calculation tracks actual pages correctly
4. ✅ BUG #4: Atomic elements validated and split if exceeding token limits
5. ✅ BUG #5: Empty chunks are filtered out
6. ✅ BUG #6: Chunk index stored in single location (metadata only)
7. ✅ BUG #7: Clear logging for embedding truncation

NEW FEATURES:
- Token-based constraints (200-2000 tokens)
- Smart splitting of oversized atomic elements
- Token counting for all chunks
- Better error messages
"""

from typing import List, Dict, Any, Optional, Tuple
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
from app.core.exceptions import (
    ChunkingError,
    ChunkSizeExceededError,
    NoChunksCreatedError
)
from app.core.monitoring import track_performance, PerformanceTimer

logger = logging.getLogger(__name__)
settings = get_settings()


class TokenCounter:
    """
    Token counting utility with fallback
    """
    
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
        
        # Fallback: estimate tokens (rough approximation)
        # English: ~4 characters per token on average
        return max(1, len(text) // 4)
    
    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text to max tokens"""
        if not text:
            return text
        
        if self.encoding:
            try:
                tokens = self.encoding.encode(text)
                if len(tokens) <= max_tokens:
                    return text
                truncated_tokens = tokens[:max_tokens]
                return self.encoding.decode(truncated_tokens)
            except Exception as e:
                logger.debug(f"Token truncation failed: {e}, using fallback")
        
        # Fallback: truncate by characters
        max_chars = max_tokens * 4
        return text[:max_chars] if len(text) > max_chars else text


class SemanticChunkingService:
    """
    FIXED: Optimized Markdown-aware chunking with token constraints
    
    All 7 critical bugs have been fixed:
    1. Page numbers start at 1
    2. Code blocks have configurable limits
    3. Page ranges calculated correctly
    4. Atomic elements validated for size
    5. Empty chunks filtered
    6. Chunk index in metadata only
    7. Clear truncation logging
    """
    
    def __init__(
        self,
        min_chunk_tokens: Optional[int] = None,
        target_chunk_tokens: Optional[int] = None,
        max_chunk_tokens: Optional[int] = None,
        overlap_tokens: Optional[int] = None
    ):
        """
        Initialize with token-based constraints
        
        Args:
            min_chunk_tokens: Minimum chunk size (default: from config)
            target_chunk_tokens: Target chunk size (default: from config)
            max_chunk_tokens: Maximum chunk size (default: from config)
            overlap_tokens: Overlap in tokens (default: from config)
        """
        # Token constraints (primary)
        self.min_chunk_tokens = min_chunk_tokens or settings.min_chunk_tokens
        self.target_chunk_tokens = target_chunk_tokens or settings.target_chunk_tokens
        self.max_chunk_tokens = max_chunk_tokens or settings.max_chunk_tokens
        self.overlap_tokens = overlap_tokens or settings.chunk_overlap_tokens
        
        # Character estimates (for backward compatibility)
        # Assume 1 token ≈ 4 characters
        self.chunk_size = self.target_chunk_tokens * 4
        self.max_chunk_size = self.max_chunk_tokens * 4
        self.chunk_overlap = self.overlap_tokens * 4
        
        # Atomic element limits
        self.max_atomic_element_tokens = settings.max_atomic_element_tokens
        self.split_oversized_atomic = settings.split_oversized_atomic
        self.max_code_block_lines = settings.max_code_block_lines
        
        # Initialize token counter
        self.token_counter = TokenCounter(settings.tokenizer_model)
        
        # Validate configuration
        if self.overlap_tokens >= self.target_chunk_tokens:
            raise ChunkingError(
                f"overlap_tokens ({self.overlap_tokens}) must be less than "
                f"target_chunk_tokens ({self.target_chunk_tokens})"
            )
        
        logger.info(
            f"✓ Token-constrained chunking initialized:\n"
            f"  - Min: {self.min_chunk_tokens} tokens\n"
            f"  - Target: {self.target_chunk_tokens} tokens\n"
            f"  - Max: {self.max_chunk_tokens} tokens\n"
            f"  - Overlap: {self.overlap_tokens} tokens ({self.overlap_tokens/self.target_chunk_tokens*100:.1f}%)\n"
            f"  - Max atomic: {self.max_atomic_element_tokens} tokens"
        )
    
    @track_performance("split_document_into_chunks")
    def split_document_into_chunks(
        self,
        markdown_text: str,
        structured_data: Dict[str, Any],
        base_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Split markdown into optimized chunks with token constraints
        
        Returns:
            List of chunk dictionaries with metadata and token counts
        """
        if not markdown_text or not markdown_text.strip():
            raise NoChunksCreatedError("Empty markdown text provided")
        
        with PerformanceTimer("chunking_process", log_result=False):
            # Parse into structural elements WITH PAGE NUMBERS (starts at 1)
            elements = self._parse_markdown_structure(markdown_text)
            
            if not elements:
                raise NoChunksCreatedError("No structural elements parsed")
            
            logger.debug(f"Parsed {len(elements)} elements")
            
            # Validate and split oversized atomic elements (FIX #4)
            elements = self._validate_and_split_atomic_elements(elements)
            
            # Group elements into heading-based sections (preserves page numbers)
            sections = self._group_into_sections(elements)
            logger.debug(f"Grouped into {len(sections)} sections")
            
            # Create optimized chunks from sections (propagates page numbers)
            chunks = self._create_optimized_chunks(sections, base_metadata)
            
            if not chunks:
                raise NoChunksCreatedError("No chunks created")
            
            # FIX #5: Filter empty chunks
            chunks = self._filter_empty_chunks(chunks)
            
            if not chunks:
                raise NoChunksCreatedError("All chunks were empty after filtering")
            
            # Validate all chunks against token limits
            chunks = self._validate_and_enforce_token_limits(chunks)
            
            # Add structured data references
            chunks = self._add_structured_references(chunks, structured_data)
            
            # Log statistics
            token_counts = [c.get('token_count', 0) for c in chunks]
            logger.info(
                f"✓ Created {len(chunks)} chunks:\n"
                f"  - Avg tokens: {sum(token_counts) // len(chunks) if chunks else 0}\n"
                f"  - Min tokens: {min(token_counts) if token_counts else 0}\n"
                f"  - Max tokens: {max(token_counts) if token_counts else 0}\n"
                f"  - Target: {self.target_chunk_tokens} tokens"
            )
        
        return chunks
    
    def _parse_markdown_structure(self, text: str) -> List[Dict[str, Any]]:
        """
        Parse markdown into structural elements
        
        FIX #1: Page numbers now start at 1 (not 0)
        FIX #2: Code blocks have configurable max lines
        """
        elements = []
        lines = text.split('\n')
        i = 0
        
        # FIX #1: Track current page number starting at 1
        current_page_number = 1  # ← FIXED: Start at 1 instead of 0
        current_slide_number = 1  # ← FIXED: Start at 1 instead of 0
        
        while i < len(lines):
            line = lines[i]
            line_stripped = line.strip()
            
            if not line_stripped:
                i += 1
                continue
            
            # Page/Slide marker detection - UPDATE CURRENT PAGE
            if re.match(r'^##\s+(Page|Slide)\s+(\d+)', line_stripped, re.IGNORECASE):
                marker_match = re.match(r'^##\s+(Page|Slide)\s+(\d+)', line_stripped, re.IGNORECASE)
                marker_type = marker_match.group(1).lower()
                marker_number = int(marker_match.group(2))
                
                if marker_type == 'page':
                    current_page_number = marker_number
                elif marker_type == 'slide':
                    current_slide_number = marker_number
                    current_page_number = marker_number  # For presentations, page = slide
                
                elements.append({
                    'type': 'page_marker',
                    'content': line,
                    'atomic': False,
                    'page_number': current_page_number,
                    'slide_number': current_slide_number if current_slide_number > 1 else None
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
                    'page_number': current_page_number
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
                        'page_number': current_page_number
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
                    'page_number': current_page_number
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
                    'page_number': current_page_number
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
                    'page_number': current_page_number
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
                    'page_number': current_page_number
                })
            i = end_idx + 1
        
        return elements
    
    def _extract_code_block(self, lines: List[str], start_idx: int) -> Tuple[str, int]:
        """
        Extract code block with FIX #2: Configurable max lines, no artificial closing
        """
        code_lines = [lines[start_idx]]
        i = start_idx + 1
        lines_read = 0
        
        # FIX #2: Use configurable max_code_block_lines
        max_lines = self.max_code_block_lines
        
        while i < len(lines) and lines_read < max_lines:
            code_lines.append(lines[i])
            if lines[i].strip().startswith('```'):
                return '\n'.join(code_lines), i
            i += 1
            lines_read += 1
        
        # FIX #2: Don't add artificial closing marker, log warning
        if lines_read >= max_lines:
            logger.warning(
                f"Code block at line {start_idx} exceeded {max_lines} lines. "
                f"Consider increasing max_code_block_lines setting. "
                f"Code block may be incomplete."
            )
            # Return as-is without artificial marker
        
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
    
    def _validate_and_split_atomic_elements(
        self,
        elements: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        FIX #4: Validate atomic element sizes and split if necessary
        """
        validated_elements = []
        
        for element in elements:
            if not element.get('atomic', False):
                # Non-atomic elements are fine
                validated_elements.append(element)
                continue
            
            content = element['content']
            token_count = self.token_counter.count_tokens(content)
            
            # Check if atomic element exceeds limit
            if token_count <= self.max_atomic_element_tokens:
                # Within limits
                validated_elements.append(element)
                continue
            
            # Exceeds limit
            logger.warning(
                f"Atomic element ({element['type']}) exceeds token limit: "
                f"{token_count} > {self.max_atomic_element_tokens} tokens"
            )
            
            if not self.split_oversized_atomic:
                # Keep as-is, will be truncated later
                logger.warning(
                    f"split_oversized_atomic=False, keeping oversized {element['type']} "
                    f"(will exceed max_chunk_tokens)"
                )
                validated_elements.append(element)
                continue
            
            # Split the atomic element
            logger.info(f"Splitting oversized {element['type']} into smaller chunks")
            
            if element['type'] == 'table':
                # Split table by rows
                split_elements = self._split_table_by_rows(element)
            elif element['type'] == 'code_block':
                # Split code by lines
                split_elements = self._split_code_by_lines(element)
            else:
                # Generic splitting
                split_elements = self._split_element_by_tokens(element)
            
            validated_elements.extend(split_elements)
        
        return validated_elements
    
    def _split_table_by_rows(self, table_element: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Split table into smaller tables by rows"""
        lines = table_element['content'].split('\n')
        
        # Find header and separator
        header_lines = []
        data_lines = []
        in_header = True
        
        for line in lines:
            if not line.strip():
                continue
            
            # Check if separator row
            parts = [p.strip() for p in line.split('|')]
            is_separator = all(re.match(r'^:?-+:?$', p) for p in parts if p)
            
            if is_separator:
                header_lines.append(line)
                in_header = False
            elif in_header:
                header_lines.append(line)
            else:
                data_lines.append(line)
        
        if not header_lines or not data_lines:
            # Can't split properly, return as-is
            return [table_element]
        
        # Create chunks with header + subset of rows
        split_elements = []
        rows_per_chunk = max(1, len(data_lines) // 3)  # Split into ~3 parts
        
        for i in range(0, len(data_lines), rows_per_chunk):
            chunk_rows = data_lines[i:i+rows_per_chunk]
            chunk_lines = header_lines + chunk_rows
            chunk_content = '\n'.join(chunk_lines)
            
            split_elements.append({
                'type': 'table',
                'content': chunk_content,
                'atomic': True,  # Still atomic, just smaller
                'page_number': table_element['page_number']
            })
        
        logger.debug(f"Split table into {len(split_elements)} smaller tables")
        return split_elements
    
    def _split_code_by_lines(self, code_element: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Split code block into smaller blocks"""
        lines = code_element['content'].split('\n')
        
        # Extract language if present
        first_line = lines[0] if lines else ""
        language = ""
        if first_line.startswith('```'):
            language = first_line[3:].strip()
            lines = lines[1:]  # Remove opening
        
        # Remove closing if present
        if lines and lines[-1].strip() == '```':
            lines = lines[:-1]
        
        # Split into chunks
        split_elements = []
        lines_per_chunk = max(10, len(lines) // 3)  # At least 10 lines per chunk
        
        for i in range(0, len(lines), lines_per_chunk):
            chunk_lines = lines[i:i+lines_per_chunk]
            chunk_content = f"```{language}\n" + '\n'.join(chunk_lines) + "\n```"
            
            split_elements.append({
                'type': 'code_block',
                'content': chunk_content,
                'atomic': True,
                'page_number': code_element['page_number']
            })
        
        logger.debug(f"Split code block into {len(split_elements)} smaller blocks")
        return split_elements
    
    def _split_element_by_tokens(self, element: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generic token-based splitting for any element"""
        content = element['content']
        target_tokens = self.max_atomic_element_tokens // 2  # Aim for half the limit
        
        # Split by paragraphs or lines
        if '\n\n' in content:
            parts = content.split('\n\n')
        else:
            lines = content.split('\n')
            # Group lines into parts
            parts = []
            current_part = []
            for line in lines:
                current_part.append(line)
                if len(current_part) >= 10:
                    parts.append('\n'.join(current_part))
                    current_part = []
            if current_part:
                parts.append('\n'.join(current_part))
        
        # Combine parts into chunks
        split_elements = []
        current_chunk = []
        current_tokens = 0
        
        for part in parts:
            part_tokens = self.token_counter.count_tokens(part)
            
            if current_tokens + part_tokens > target_tokens and current_chunk:
                # Save current chunk
                chunk_content = '\n\n'.join(current_chunk)
                split_elements.append({
                    'type': element['type'],
                    'content': chunk_content,
                    'atomic': False,  # No longer atomic after splitting
                    'page_number': element['page_number']
                })
                current_chunk = []
                current_tokens = 0
            
            current_chunk.append(part)
            current_tokens += part_tokens
        
        # Add final chunk
        if current_chunk:
            chunk_content = '\n\n'.join(current_chunk)
            split_elements.append({
                'type': element['type'],
                'content': chunk_content,
                'atomic': False,
                'page_number': element['page_number']
            })
        
        logger.debug(f"Split {element['type']} into {len(split_elements)} parts")
        return split_elements if split_elements else [element]
    
    def _group_into_sections(self, elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Group elements into sections (heading + its content)
        
        FIX #3: Correctly tracks page numbers and page ranges
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
        Create section with FIX #3: Accurate page range calculation
        """
        size = self._calculate_section_size(heading, content_elements)
        
        # FIX #3: Calculate page range correctly
        page_numbers = []  # Use list to preserve order
        
        if heading and heading.get('page_number', 0) > 0:
            page_numbers.append(heading['page_number'])
        
        for elem in content_elements:
            page_num = elem.get('page_number', 0)
            if page_num > 0:
                page_numbers.append(page_num)
        
        # Determine page range
        if page_numbers:
            # Use actual first and last pages (not min/max of set)
            first_page = page_numbers[0]
            last_page = page_numbers[-1]
            page_range = [first_page, last_page]
            primary_page = first_page
        else:
            # FIX #1: Default to page 1 (not 0)
            page_range = [1, 1]
            primary_page = 1
        
        return {
            'heading': heading,
            'content_elements': content_elements,
            'size': size,
            'page_range': page_range,
            'primary_page': primary_page
        }
    
    def _calculate_section_size(
        self, 
        heading: Optional[Dict[str, Any]], 
        content_elements: List[Dict[str, Any]]
    ) -> int:
        """Calculate total size of a section"""
        size = 0
        if heading:
            size += len(heading['content']) + 2
        for elem in content_elements:
            size += len(elem['content']) + 2
        return size
    
    def _create_optimized_chunks(
        self,
        sections: List[Dict[str, Any]],
        base_metadata: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Create optimized chunks from sections with token enforcement
        """
        chunks = []
        chunk_index = 0
        i = 0
        
        while i < len(sections):
            section = sections[i]
            
            # Case 1: Section is small - try to merge
            if section['size'] < self.chunk_size:
                merged_chunk = self._merge_small_sections(sections, i)
                chunk_dict = self._create_chunk_dict(
                    merged_chunk['text'], 
                    chunk_index, 
                    base_metadata,
                    page_info=merged_chunk['page_info']
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
        
        # Add overlap between consecutive chunks
        chunks = self._add_content_overlap(chunks)
        
        return chunks
    
    def _merge_small_sections(
        self,
        sections: List[Dict[str, Any]],
        start_idx: int
    ) -> Dict[str, Any]:
        """Merge small sections with page tracking"""
        merged_parts = []
        total_size = 0
        i = start_idx
        
        all_page_numbers = []
        
        while i < len(sections):
            section = sections[i]
            section_size = section['size']
            
            if total_size + section_size > self.max_chunk_size:
                if not merged_parts:
                    merged_parts.append(self._section_to_text(section))
                    all_page_numbers.extend(section['page_range'])
                    i += 1
                break
            
            merged_parts.append(self._section_to_text(section))
            all_page_numbers.extend(section['page_range'])
            total_size += section_size
            i += 1
            
            if total_size >= self.chunk_size:
                if i < len(sections) and sections[i]['size'] < 200:
                    if total_size + sections[i]['size'] <= self.max_chunk_size:
                        merged_parts.append(self._section_to_text(sections[i]))
                        all_page_numbers.extend(sections[i]['page_range'])
                        total_size += sections[i]['size']
                        i += 1
                break
        
        # Calculate page info
        if all_page_numbers:
            unique_pages = sorted(set(p for p in all_page_numbers if p > 0))
            page_info = {
                'page_number': unique_pages[0] if unique_pages else 1,
                'page_range': [unique_pages[0], unique_pages[-1]] if unique_pages else [1, 1]
            }
        else:
            page_info = {
                'page_number': 1,
                'page_range': [1, 1]
            }
        
        return {
            'text': '\n\n'.join(merged_parts),
            'next_index': i,
            'page_info': page_info
        }
    
    def _split_large_section(
        self,
        section: Dict[str, Any],
        start_chunk_index: int,
        base_metadata: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Split large section with page tracking"""
        heading = section['heading']
        content_elements = section['content_elements']
        
        chunks = []
        chunk_index = start_chunk_index
        
        heading_text = heading['content'] if heading else ""
        heading_size = len(heading_text) + 2 if heading_text else 0
        
        current_chunk_parts = [heading_text] if heading_text else []
        current_size = heading_size
        current_chunk_pages = set()
        
        if heading and heading.get('page_number', 0) > 0:
            current_chunk_pages.add(heading['page_number'])
        
        for elem in content_elements:
            elem_text = elem['content']
            elem_size = len(elem_text) + 2
            elem_page = elem.get('page_number', 0)
            
            if current_size + elem_size > self.max_chunk_size:
                if current_chunk_parts:
                    page_info = self._calculate_page_info(
                        current_chunk_pages, 
                        section['primary_page']
                    )
                    chunks.append(self._create_chunk_dict(
                        '\n\n'.join(current_chunk_parts), 
                        chunk_index, 
                        base_metadata,
                        page_info=page_info
                    ))
                    chunk_index += 1
                
                current_chunk_parts = [elem_text]
                current_size = elem_size
                current_chunk_pages = {elem_page} if elem_page > 0 else set()
            else:
                current_chunk_parts.append(elem_text)
                current_size += elem_size
                if elem_page > 0:
                    current_chunk_pages.add(elem_page)
        
        if current_chunk_parts:
            page_info = self._calculate_page_info(
                current_chunk_pages,
                section['primary_page']
            )
            chunks.append(self._create_chunk_dict(
                '\n\n'.join(current_chunk_parts), 
                chunk_index, 
                base_metadata,
                page_info=page_info
            ))
        
        return chunks
    
    def _calculate_page_info(
        self,
        page_set: set,
        default_page: int
    ) -> Dict[str, Any]:
        """Calculate page info from set of pages"""
        if page_set:
            page_list = sorted(p for p in page_set if p > 0)
            if page_list:
                return {
                    'page_number': page_list[0],
                    'page_range': [page_list[0], page_list[-1]]
                }
        
        # FIX #1: Default to 1 (not 0)
        return {
            'page_number': default_page if default_page > 0 else 1,
            'page_range': [default_page if default_page > 0 else 1, 
                           default_page if default_page > 0 else 1]
        }
    
    def _section_to_text(self, section: Dict[str, Any]) -> str:
        """Convert section to text"""
        parts = []
        
        if section['heading']:
            parts.append(section['heading']['content'])
        
        for elem in section['content_elements']:
            parts.append(elem['content'])
        
        return '\n\n'.join(parts)
    
    def _add_content_overlap(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Add overlap between chunks based on token count"""
        if len(chunks) <= 1:
            return chunks
        
        for i in range(len(chunks) - 1):
            current_text = chunks[i]['text']
            next_text = chunks[i + 1]['text']
            
            # Calculate overlap in tokens
            current_tokens = self.token_counter.count_tokens(current_text)
            
            if current_tokens < self.overlap_tokens:
                continue
            
            # Extract last portion for overlap
            overlap_text = self._extract_last_n_tokens(
                current_text,
                self.overlap_tokens
            )
            
            # Prepend to next chunk
            if overlap_text:
                chunks[i + 1]['text'] = overlap_text + '\n\n' + next_text
                # Update char count
                chunks[i + 1]['char_count'] = len(chunks[i + 1]['text'])
        
        return chunks
    
    def _extract_last_n_tokens(self, text: str, n_tokens: int) -> str:
        """Extract last n tokens from text"""
        if not self.token_counter.encoding:
            # Fallback: estimate by characters
            chars = n_tokens * 4
            return text[-chars:] if len(text) > chars else ""
        
        try:
            tokens = self.token_counter.encoding.encode(text)
            if len(tokens) <= n_tokens:
                return ""
            
            overlap_tokens = tokens[-n_tokens:]
            return self.token_counter.encoding.decode(overlap_tokens)
        except Exception:
            # Fallback
            chars = n_tokens * 4
            return text[-chars:] if len(text) > chars else ""
    
    def _filter_empty_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        FIX #5: Filter out empty chunks
        """
        filtered_chunks = []
        
        for i, chunk in enumerate(chunks):
            text = chunk.get('text', '').strip()
            
            if not text:
                logger.warning(f"Removing empty chunk at index {i}")
                continue
            
            if len(text) < 10:  # Very short chunks probably not useful
                logger.warning(
                    f"Chunk {i} is very short ({len(text)} chars), "
                    f"but keeping it"
                )
            
            filtered_chunks.append(chunk)
        
        if len(filtered_chunks) < len(chunks):
            logger.info(
                f"Filtered out {len(chunks) - len(filtered_chunks)} empty chunks"
            )
        
        return filtered_chunks
    
    def _validate_and_enforce_token_limits(
        self,
        chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Validate chunks against token limits and enforce constraints
        
        FIX #7: Clear logging for truncation
        """
        validated_chunks = []
        
        for i, chunk in enumerate(chunks):
            text = chunk['text']
            token_count = self.token_counter.count_tokens(text)
            
            # Add token count to chunk
            chunk['token_count'] = token_count
            chunk['metadata'] = chunk.get('metadata', {})
            chunk['metadata']['token_count'] = token_count
            
            # Check if within limits
            if token_count <= self.max_chunk_tokens:
                # Within limits - good
                if token_count < self.min_chunk_tokens:
                    logger.debug(
                        f"Chunk {i} below min_chunk_tokens: "
                        f"{token_count} < {self.min_chunk_tokens}"
                    )
                validated_chunks.append(chunk)
            else:
                # Exceeds limit - truncate
                logger.warning(
                    f"Chunk {i} exceeds max_chunk_tokens: "
                    f"{token_count} > {self.max_chunk_tokens} tokens\n"
                    f"  Truncating to {self.max_chunk_tokens} tokens.\n"
                    f"  Original length: {len(text)} chars\n"
                    f"  This may result in incomplete content."
                )
                
                # FIX #7: Truncate with clear message
                truncated_text = self.token_counter.truncate_to_tokens(
                    text,
                    self.max_chunk_tokens
                )
                
                chunk['text'] = truncated_text
                chunk['char_count'] = len(truncated_text)
                chunk['token_count'] = self.max_chunk_tokens
                chunk['metadata']['token_count'] = self.max_chunk_tokens
                chunk['metadata']['truncated'] = True
                chunk['metadata']['original_token_count'] = token_count
                
                validated_chunks.append(chunk)
        
        return validated_chunks
    
    def _create_chunk_dict(
        self,
        text: str,
        chunk_index: int,
        base_metadata: Optional[Dict[str, Any]],
        page_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create chunk dictionary
        
        FIX #6: Chunk index stored in metadata only (single source of truth)
        """
        file_id = base_metadata.get('file_id') if base_metadata else None
        chunk_id = self._generate_chunk_id(text, chunk_index, file_id)
        
        metadata = dict(base_metadata) if base_metadata else {}
        
        # Add page information
        if page_info:
            metadata['page_number'] = page_info.get('page_number', 1)
            metadata['page_range'] = page_info.get('page_range', [1, 1])
        else:
            # FIX #1: Default to 1 (not 0)
            metadata['page_number'] = 1
            metadata['page_range'] = [1, 1]
        
        # FIX #6: Store chunk_index ONLY in metadata (single source of truth)
        metadata['chunk_index'] = chunk_index
        
        # Count tokens
        token_count = self.token_counter.count_tokens(text)
        metadata['token_count'] = token_count
        
        return {
            'chunk_id': chunk_id,
            'text': text,
            'char_count': len(text),
            'token_count': token_count,
            'metadata': metadata
        }
    
    def _generate_chunk_id(
        self, 
        text: str, 
        index: int,
        file_id: Optional[str] = None
    ) -> str:
        """Generate unique chunk ID"""
        if file_id:
            hash_input = f"{file_id}_{index}_{text}".encode()
        else:
            hash_input = f"{index}_{text}".encode()
        
        content_hash = hashlib.sha256(hash_input).hexdigest()[:16]
        
        import time
        timestamp = str(int(time.time() * 1000))[-6:]
        
        return f"chunk_{index}_{content_hash}_{timestamp}"
    
    def _add_structured_references(
        self,
        chunks: List[Dict[str, Any]],
        structured_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Add structured data references
        
        Page numbers are already correct - just add section hierarchy
        """
        for chunk in chunks:
            # Verify page_number exists
            page_num = chunk['metadata'].get('page_number', -1)
            
            if page_num == -1 or page_num == 0:
                logger.warning(
                    f"Chunk {chunk.get('chunk_id', '?')} has invalid page_number: {page_num}. "
                    f"Setting to 1."
                )
                chunk['metadata']['page_number'] = 1
                if chunk['metadata'].get('page_range', [0, 0])[0] == 0:
                    chunk['metadata']['page_range'] = [1, 1]
            
            # Extract section hierarchy
            section_matches = re.findall(r'^(#{1,6})\s+(.+)$', chunk['text'], re.MULTILINE)
            if section_matches:
                section_hierarchy = [match[1].strip() for match in section_matches[:3]]
                # Filter out page/slide markers
                section_hierarchy = [
                    h for h in section_hierarchy 
                    if not re.match(r'^(Page|Slide)\s+\d+$', h, re.IGNORECASE)
                ]
                if section_hierarchy:
                    chunk['metadata']['section_hierarchy'] = section_hierarchy
        
        return chunks