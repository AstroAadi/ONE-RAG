from typing import List, Dict
import re


class TextChunkingService:
    """Service for splitting text into manageable chunks"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize chunking service
        
        Args:
            chunk_size: Maximum characters per chunk (should be < 2048 for Cohere)
            chunk_overlap: Number of overlapping characters between chunks
        """
        # Ensure chunk_size respects Cohere's 2048 char limit
        # Keep it under 1800 to be safe with overlap
        self.chunk_size = min(chunk_size, 1800)
        self.chunk_overlap = chunk_overlap
    
    def split_text_into_chunks(self, text: str, metadata: Dict = None) -> List[Dict]:
        """
        Split text into overlapping chunks
        
        Args:
            text: Text to split
            metadata: Additional metadata to attach to chunks
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        if not text or not text.strip():
            return []
        
        chunks = []
        
        # Split by paragraphs first to maintain context
        paragraphs = self._split_by_paragraphs(text)
        
        current_chunk = ""
        chunk_index = 0
        
        for paragraph in paragraphs:
            # If adding this paragraph exceeds chunk size
            if len(current_chunk) + len(paragraph) > self.chunk_size:
                if current_chunk:
                    # Save current chunk
                    chunks.append(self._create_chunk_dict(
                        current_chunk, 
                        chunk_index, 
                        metadata
                    ))
                    chunk_index += 1
                    
                    # Start new chunk with overlap
                    current_chunk = self._get_overlap_text(current_chunk) + paragraph
                else:
                    # Paragraph itself is larger than chunk_size
                    sub_chunks = self._split_large_paragraph(paragraph)
                    for sub_chunk in sub_chunks:
                        chunks.append(self._create_chunk_dict(
                            sub_chunk,
                            chunk_index,
                            metadata
                        ))
                        chunk_index += 1
                    current_chunk = ""
            else:
                current_chunk += ("\n\n" if current_chunk else "") + paragraph
        
        # Add remaining chunk
        if current_chunk:
            chunks.append(self._create_chunk_dict(
                current_chunk,
                chunk_index,
                metadata
            ))
        
        return chunks
    
    def _split_by_paragraphs(self, text: str) -> List[str]:
        """Split text by paragraphs"""
        # Split by double newlines or more
        paragraphs = re.split(r'\n\s*\n', text)
        return [p.strip() for p in paragraphs if p.strip()]
    
    def _split_large_paragraph(self, paragraph: str) -> List[str]:
        """Split a large paragraph into smaller chunks"""
        chunks = []
        sentences = re.split(r'(?<=[.!?])\s+', paragraph)
        
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= self.chunk_size:
                current_chunk += (" " if current_chunk else "") + sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _get_overlap_text(self, text: str) -> str:
        """Get the last overlap_size characters for context"""
        if len(text) <= self.chunk_overlap:
            return text
        return text[-self.chunk_overlap:]
    
    def _create_chunk_dict(self, text: str, index: int, metadata: Dict = None) -> Dict:
        """Create a chunk dictionary with metadata"""
        # Ensure chunk doesn't exceed Cohere's limit (2048 chars)
        max_chunk_size = 1900  # Safety margin
        
        if len(text) > max_chunk_size:
            # Truncate if somehow a chunk is too large
            text = text[:max_chunk_size] + "..."
            print(f"Warning: Chunk {index} truncated from {len(text)} to {max_chunk_size} characters")
        
        chunk_data = {
            "text": text.strip(),
            "chunk_index": index,
            "char_count": len(text.strip())
        }
        
        if metadata:
            chunk_data.update(metadata)
        
        return chunk_data