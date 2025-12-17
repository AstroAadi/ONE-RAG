"""
Streamlined Chunking Service
Works with simple markdown text (no complex structured data needed)
Preserves image references in chunks
FIXED: Ensures chunks never exceed max_chunk_size for embeddings
"""

from typing import List, Dict, Any, Optional
import re
import hashlib


class SemanticChunkingService:
    """
    Simple semantic chunking that preserves image references
    """
    
    def __init__(
        self, 
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        max_chunk_size: int = 1800
    ):
        # CRITICAL: Cohere embedding model max is 2048 chars
        # Set max_chunk_size to safe limit with buffer
        self.max_chunk_size = min(max_chunk_size, 1800)
        self.chunk_size = min(chunk_size, self.max_chunk_size)
        self.chunk_overlap = min(chunk_overlap, self.chunk_size // 2)
        
        print(f"Chunking config: chunk_size={self.chunk_size}, max={self.max_chunk_size}, overlap={self.chunk_overlap}")
    
    def split_document_into_chunks(
        self,
        markdown_text: str,
        structured_data: Dict[str, Any],
        base_metadata: Dict = None
    ) -> List[Dict]:
        """
        Split markdown into semantic chunks
        Preserves image references within chunks
        """
        if not markdown_text or not markdown_text.strip():
            return []
        
        chunks = []
        
        # Split by visual content blocks (images/tables) first
        sections = self._split_by_visual_blocks(markdown_text)
        
        print(f"Split into {len(sections)} sections by visual blocks")
        
        chunk_index = 0
        for i, section in enumerate(sections):
            section_chunks = self._chunk_section(section, chunk_index, base_metadata)
            
            # CRITICAL: Validate all chunks are under max size
            for chunk in section_chunks:
                if len(chunk['text']) > self.max_chunk_size:
                    print(f"WARNING: Chunk {chunk['chunk_id']} exceeds max size: {len(chunk['text'])} chars")
                    # Force split this chunk
                    sub_chunks = self._force_split_chunk(chunk, chunk_index, base_metadata)
                    chunks.extend(sub_chunks)
                    chunk_index += len(sub_chunks)
                else:
                    chunks.append(chunk)
                    chunk_index += 1
        
        print(f"Created {len(chunks)} total chunks")
        
        # Add parent-child relationships
        chunks = self._create_parent_child_relationships(chunks)
        
        return chunks
    
    def _split_by_visual_blocks(self, text: str) -> List[str]:
        """
        Split text by visual content blocks (---...---)
        Each visual block stays with its surrounding text
        """
        # Split by visual markers
        pattern = r'(---\n\*\*\[.*?\]\*\*.*?---\n)'
        parts = re.split(pattern, text, flags=re.DOTALL)
        
        sections = []
        current_section = ""
        
        for part in parts:
            if not part.strip():
                continue
            
            # Check if this is a visual block
            if part.startswith('---') and part.endswith('---\n'):
                # Add visual block to current section
                current_section += part
                
                # If section is getting large, save it
                if len(current_section) > self.chunk_size * 1.5:
                    sections.append(current_section)
                    current_section = ""
            else:
                # Regular text
                if len(current_section) + len(part) > self.chunk_size * 2:
                    # Save current section if it's getting large
                    if current_section.strip():
                        sections.append(current_section)
                    current_section = part
                else:
                    current_section += part
        
        # Add remaining text
        if current_section.strip():
            sections.append(current_section)
        
        return sections if sections else [text]
    
    def _chunk_section(
        self,
        section: str,
        start_index: int,
        base_metadata: Dict
    ) -> List[Dict]:
        """
        Chunk a section (may contain text + visual blocks)
        """
        chunks = []
        
        # Check if section contains visual content
        has_visual = 'IMAGE_PATH:' in section or '**[TABLE]**' in section
        
        # If section is small enough, keep as one chunk
        if len(section) <= self.chunk_size:
            chunk = self._create_chunk_dict(
                text=section,
                chunk_index=start_index,
                base_metadata=base_metadata,
                has_visual=has_visual
            )
            chunks.append(chunk)
        
        # If section has visual content, try to keep it together
        elif has_visual:
            # Try to split around visual blocks
            visual_chunks = self._split_visual_section(section, start_index, base_metadata)
            chunks.extend(visual_chunks)
        
        # Otherwise, split by paragraphs/sentences
        else:
            text_chunks = self._split_text_section(section, start_index, base_metadata)
            chunks.extend(text_chunks)
        
        return chunks
    
    def _split_visual_section(
        self,
        section: str,
        start_index: int,
        base_metadata: Dict
    ) -> List[Dict]:
        """
        Split section containing visual content
        Tries to keep visual blocks with surrounding context
        """
        chunks = []
        chunk_index = start_index
        
        # Split into visual and text parts
        pattern = r'(---\n\*\*\[.*?\]\*\*.*?---\n)'
        parts = re.split(pattern, section, flags=re.DOTALL)
        
        current_chunk = ""
        
        for part in parts:
            if not part.strip():
                continue
            
            # Check if adding this part would exceed chunk size
            if len(current_chunk) + len(part) > self.chunk_size:
                if current_chunk:
                    # Save current chunk
                    has_vis = 'IMAGE_PATH:' in current_chunk
                    chunks.append(self._create_chunk_dict(
                        text=current_chunk.strip(),
                        chunk_index=chunk_index,
                        base_metadata=base_metadata,
                        has_visual=has_vis
                    ))
                    chunk_index += 1
                
                # If part itself is too large, force split it
                if len(part) > self.max_chunk_size:
                    sub_parts = self._split_large_text(part, self.chunk_size)
                    for sub_part in sub_parts:
                        has_vis = 'IMAGE_PATH:' in sub_part
                        chunks.append(self._create_chunk_dict(
                            text=sub_part.strip(),
                            chunk_index=chunk_index,
                            base_metadata=base_metadata,
                            has_visual=has_vis
                        ))
                        chunk_index += 1
                    current_chunk = ""
                else:
                    current_chunk = part
            else:
                current_chunk += part
        
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
    
    def _split_text_section(
        self,
        section: str,
        start_index: int,
        base_metadata: Dict
    ) -> List[Dict]:
        """
        Split regular text section by paragraphs/sentences
        """
        chunks = []
        chunk_index = start_index
        
        # Split by paragraphs
        paragraphs = re.split(r'\n\s*\n', section)
        
        current_chunk = ""
        
        for para in paragraphs:
            if not para.strip():
                continue
            
            # Check size
            if len(current_chunk) + len(para) > self.chunk_size:
                if current_chunk:
                    # Save current chunk
                    chunks.append(self._create_chunk_dict(
                        text=current_chunk.strip(),
                        chunk_index=chunk_index,
                        base_metadata=base_metadata,
                        has_visual=False
                    ))
                    chunk_index += 1
                    
                    # Add overlap
                    overlap = self._get_overlap_text(current_chunk)
                    current_chunk = overlap
                
                # If paragraph too large, split by sentences
                if len(para) > self.max_chunk_size:
                    sentence_chunks = self._split_by_sentences(
                        para, 
                        chunk_index, 
                        base_metadata
                    )
                    chunks.extend(sentence_chunks)
                    chunk_index += len(sentence_chunks)
                    current_chunk = ""
                else:
                    if current_chunk:
                        current_chunk += "\n\n" + para
                    else:
                        current_chunk = para
            else:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
        
        # Add remaining chunk
        if current_chunk.strip():
            chunks.append(self._create_chunk_dict(
                text=current_chunk.strip(),
                chunk_index=chunk_index,
                base_metadata=base_metadata,
                has_visual=False
            ))
        
        return chunks
    
    def _split_by_sentences(
        self,
        text: str,
        start_index: int,
        base_metadata: Dict
    ) -> List[Dict]:
        """Split large text by sentences"""
        chunks = []
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        current_chunk = ""
        chunk_index = start_index
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= self.chunk_size:
                current_chunk += (" " if current_chunk else "") + sentence
            else:
                if current_chunk:
                    chunks.append(self._create_chunk_dict(
                        text=current_chunk.strip(),
                        chunk_index=chunk_index,
                        base_metadata=base_metadata,
                        has_visual=False
                    ))
                    chunk_index += 1
                
                # Truncate if sentence still too long
                if len(sentence) > self.max_chunk_size:
                    sentence = sentence[:self.max_chunk_size - 3] + "..."
                
                current_chunk = sentence
        
        if current_chunk.strip():
            chunks.append(self._create_chunk_dict(
                text=current_chunk.strip(),
                chunk_index=chunk_index,
                base_metadata=base_metadata,
                has_visual=False
            ))
        
        return chunks
    
    def _split_large_text(self, text: str, target_size: int) -> List[str]:
        """Force split large text into smaller pieces"""
        parts = []
        while len(text) > target_size:
            # Try to find a good break point
            break_point = text.rfind('\n', 0, target_size)
            if break_point == -1 or break_point < target_size // 2:
                break_point = text.rfind('. ', 0, target_size)
            if break_point == -1 or break_point < target_size // 2:
                break_point = text.rfind(' ', 0, target_size)
            if break_point == -1:
                break_point = target_size
            
            parts.append(text[:break_point].strip())
            text = text[break_point:].strip()
        
        if text:
            parts.append(text)
        
        return parts
    
    def _force_split_chunk(
        self,
        chunk: Dict,
        start_index: int,
        base_metadata: Dict
    ) -> List[Dict]:
        """Force split a chunk that's too large"""
        text = chunk['text']
        sub_chunks = []
        
        parts = self._split_large_text(text, self.chunk_size)
        
        for i, part in enumerate(parts):
            sub_chunk = self._create_chunk_dict(
                text=part,
                chunk_index=start_index + i,
                base_metadata=base_metadata,
                has_visual='IMAGE_PATH:' in part
            )
            sub_chunks.append(sub_chunk)
        
        return sub_chunks
    
    def _get_overlap_text(self, text: str) -> str:
        """Get overlap text from end of chunk"""
        if len(text) <= self.chunk_overlap:
            return text
        
        overlap_text = text[-self.chunk_overlap:]
        
        # Find sentence boundary
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
        
        # CRITICAL: Enforce max size
        if len(text) > self.max_chunk_size:
            print(f"WARNING: Truncating chunk from {len(text)} to {self.max_chunk_size} chars")
            text = text[:self.max_chunk_size - 3] + "..."
        
        chunk_id = self._generate_chunk_id(text, chunk_index, base_metadata)
        
        # Detect chunk type
        chunk_type = 'text'
        if has_visual:
            if '**[TABLE]**' in text:
                chunk_type = 'table'
            elif '**[CHART]**' in text or '**[FIGURE]**' in text:
                chunk_type = 'chart'
            elif '**[IMAGE]**' in text:
                chunk_type = 'image'
            else:
                chunk_type = 'mixed'
        
        # Extract page number if available
        page_match = re.search(r'PAGE:\s*(\d+)', text)
        page_number = int(page_match.group(1)) if page_match else 0
        
        # Extract section if available
        section_match = re.search(r'SECTION:\s*([^\n]+)', text)
        section = section_match.group(1).strip() if section_match else ''
        
        metadata = {
            'chunk_id': chunk_id,
            'chunk_index': chunk_index,
            'chunk_type': chunk_type,
            'char_count': len(text),
            'has_visual': has_visual,
            'page_number': page_number,
            'section_hierarchy': section
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
    
    def _create_parent_child_relationships(
        self,
        chunks: List[Dict]
    ) -> List[Dict]:
        """Create parent-child relationships"""
        parent_size = 4
        enhanced_chunks = []
        
        for i in range(0, len(chunks), parent_size):
            batch = chunks[i:i + parent_size]
            
            if len(batch) > 1:
                # Create parent chunk from children
                parent_texts = [c['text'] for c in batch]
                parent_text = "\n\n".join(parent_texts)
                
                # CRITICAL: Ensure parent doesn't exceed max size
                if len(parent_text) > self.max_chunk_size:
                    parent_text = parent_text[:self.max_chunk_size - 3] + "..."
                
                parent_id = f"parent_{batch[0]['chunk_id']}"
                
                parent_metadata = {
                    **batch[0].get('metadata', {}),
                    'is_parent': True,
                    'child_count': len(batch)
                }
                
                parent_chunk = {
                    'text': parent_text,
                    'chunk_id': parent_id,
                    'chunk_index': batch[0]['chunk_index'],
                    'char_count': len(parent_text),
                    'metadata': parent_metadata
                }
                
                enhanced_chunks.append(parent_chunk)
            
            # Add child chunks with parent reference
            for chunk in batch:
                if 'metadata' not in chunk:
                    chunk['metadata'] = {}
                
                chunk['metadata']['parent_chunk_id'] = parent_id if len(batch) > 1 else ''
                chunk['metadata']['is_parent'] = False
                
                enhanced_chunks.append(chunk)
        
        return enhanced_chunks