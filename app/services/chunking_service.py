"""
Enhanced Chunking Service with 6 Strategies
=============================================

Chunking Strategies:
1. Fixed-size (token-based) - Current implementation
2. Sentence chunking - Split by sentence boundaries
3. Paragraph chunking - Split by paragraph boundaries
4. Sliding window - Overlapping fixed-size windows
5. Recursive chunking - Hierarchical split by separators
6. Semantic chunking - Group semantically similar sentences

Author: Enhanced Chunker
Date: 2026-01-27
"""

from typing import List, Dict, Any, Optional
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

try:
    import nltk
    NLTK_AVAILABLE = True
    # Download required NLTK data
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
except ImportError:
    NLTK_AVAILABLE = False
    logging.warning("nltk not installed. Install with: pip install nltk")

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class ChunkingStrategy(str, Enum):
    """Available chunking strategies"""
    DEFAULT="default_size"
    FIXED_SIZE = "fixed_size"
    HIERARCHICAL = "hierarchical"
    SEMANTIC = "semantic"


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


class EnhancedChunkingService:
    """
    Enhanced chunking service with 6 different strategies
    
    Strategies:
    1. Default size (legacy)
    2. Fixed-size (token-based)
    3. Hierarchical chunking
    4. Semantic chunking
    """
    
    def __init__(
        self,
        target_chunk_tokens: int = 300,
        chunk_overlap_tokens: int = 20,
        tokenizer_model: str = "cl100k_base",
        min_chunk_tokens: int = 100,
        max_chunk_tokens: int = 1024
    ):
        """
        Initialize enhanced chunker
        
        Args:
            target_chunk_tokens: Target chunk size in tokens (default: 300)
            chunk_overlap_tokens: Overlap between chunks in tokens (default: 20)
            tokenizer_model: Tokenizer model (default: cl100k_base)
            min_chunk_tokens: Minimum chunk size (default: 100)
            max_chunk_tokens: Maximum chunk size (default: 1024)
        """
        self.target_chunk_tokens = target_chunk_tokens
        self.chunk_overlap_tokens = chunk_overlap_tokens
        self.tokenizer_model = tokenizer_model
        self.min_chunk_tokens = min_chunk_tokens
        self.max_chunk_tokens = max_chunk_tokens
        
        # Initialize token counter
        self.token_counter = TokenCounter(model=tokenizer_model)
        
        logger.info(
            f"✓ Enhanced chunker initialized:\n"
            f"  - Target chunk size: {target_chunk_tokens} tokens\n"
            f"  - Overlap: {chunk_overlap_tokens} tokens\n"
            f"  - Min/Max: {min_chunk_tokens}/{max_chunk_tokens} tokens\n"
            f"  - Tokenizer: {tokenizer_model}"
        )
    
    def split_document_into_chunks(
        self,
        markdown_text: str,
        structured_data: Optional[Dict] = None,
        base_metadata: Optional[Dict] = None,
        strategy: ChunkingStrategy = ChunkingStrategy.FIXED_SIZE
    ) -> List[Dict[str, Any]]:
        """
        Split document into chunks using specified strategy
        
        Args:
            markdown_text: Document text (markdown)
            structured_data: Optional structured data
            base_metadata: Base metadata for all chunks
            strategy: Chunking strategy to use
            
        Returns:
            List of chunk dictionaries with metadata
        """
        if not markdown_text or not markdown_text.strip():
            logger.warning("Empty document provided")
            return []
        
        # Initialize base metadata
        if base_metadata is None:
            base_metadata = {}
        
        logger.info(f"Chunking document using {strategy.value} strategy ({len(markdown_text)} characters)...")
        
        # Route to appropriate chunking strategy
        if strategy == ChunkingStrategy.DEFAULT:
            chunks = self._chunk_default_size(markdown_text, base_metadata)
        elif strategy == ChunkingStrategy.FIXED_SIZE:
            chunks = self._chunk_fixed_size(markdown_text, base_metadata)
        elif strategy == ChunkingStrategy.HIERARCHICAL:
            chunks = self._chunk_hierarchical(markdown_text, base_metadata)
        # elif strategy == ChunkingStrategy.SENTENCE:
        #     chunks = self._chunk_by_sentence(markdown_text, base_metadata)
        # elif strategy == ChunkingStrategy.PARAGRAPH:
        #     chunks = self._chunk_by_paragraph(markdown_text, base_metadata)
        # elif strategy == ChunkingStrategy.SLIDING_WINDOW:
        #     chunks = self._chunk_sliding_window(markdown_text, base_metadata)
        # elif strategy == ChunkingStrategy.RECURSIVE:
        #     chunks = self._chunk_recursive(markdown_text, base_metadata)
        elif strategy == ChunkingStrategy.SEMANTIC:
            chunks = self._chunk_semantic(markdown_text, base_metadata)
        else:
            logger.warning(f"Unknown strategy {strategy}, falling back to fixed_size")
            chunks = self._chunk_default_size(markdown_text, base_metadata)
        
        if not chunks:
            logger.warning("No chunks created")
            return []
        
        logger.info(f"✅ Created {len(chunks)} chunks using {strategy.value}")
        
        return chunks
    
    # ==================== STRATEGY 0: Default Size (Legacy) ====================
    def _chunk_default_size(
        self,
        text: str,
        base_metadata: Dict
    ) -> List[Dict[str, Any]]:
        """
        Default chunking strategy (legacy)
        
        Creates chunks of approximately target_chunk_tokens size
        No overlap, simple sequential split
        """
        return self._chunk_fixed_size(text, base_metadata)
    
    # ==================== STRATEGY 1: Fixed-size (Token-based) ====================
    
    def _chunk_fixed_size(
        self,
        text: str,
        base_metadata: Dict
    ) -> List[Dict[str, Any]]:
        """
        Strategy 1: Fixed-size token-based chunking
        
        Creates chunks of exactly target_chunk_tokens size
        No overlap, simple sequential split
        """
        chunks = []
        
        if self.token_counter.encoding:
            tokens = self.token_counter.get_tokens(text)
            
            if not tokens:
                return self._chunk_fixed_size_chars(text, base_metadata)
            
            chunk_index = 1
            position = 0
            
            while position < len(tokens):
                chunk_tokens = tokens[position:position + self.target_chunk_tokens]
                chunk_text = self.token_counter.decode_tokens(chunk_tokens)
                
                if chunk_text.strip():
                    chunks.append(self._create_chunk(
                        chunk_text, chunk_index, base_metadata, 
                        position=position, total_tokens=len(tokens)
                    ))
                    chunk_index += 1
                
                position += self.target_chunk_tokens
        else:
            chunks = self._chunk_fixed_size_chars(text, base_metadata)
        
        return chunks
    
    
    
    def _chunk_by_sentence(
        self,
        text: str,
        base_metadata: Dict
    ) -> List[Dict[str, Any]]:
        """
        Strategy 2: Sentence-based chunking
        
        Groups sentences until target token count is reached
        Preserves sentence boundaries
        """
        chunks = []
        
        # Split into sentences
        sentences = self._split_into_sentences(text)
        
        if not sentences:
            logger.warning("No sentences detected, falling back to fixed-size")
            return self._chunk_fixed_size(text, base_metadata)
        
        chunk_index = 1
        current_chunk = []
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = self.token_counter.count_tokens(sentence)
            
            # If single sentence exceeds max, split it
            if sentence_tokens > self.max_chunk_tokens:
                # Save current chunk if exists
                if current_chunk:
                    chunk_text = " ".join(current_chunk)
                    chunks.append(self._create_chunk(chunk_text, chunk_index, base_metadata))
                    chunk_index += 1
                    current_chunk = []
                    current_tokens = 0
                
                # Split long sentence using fixed-size
                sub_chunks = self._chunk_fixed_size(sentence, base_metadata)
                for sub_chunk in sub_chunks:
                    sub_chunk['metadata']['chunk_index'] = chunk_index
                    chunks.append(sub_chunk)
                    chunk_index += 1
                continue
            
            # Check if adding sentence exceeds target
            if current_tokens + sentence_tokens > self.target_chunk_tokens and current_chunk:
                # Save current chunk
                chunk_text = " ".join(current_chunk)
                chunks.append(self._create_chunk(chunk_text, chunk_index, base_metadata))
                chunk_index += 1
                current_chunk = []
                current_tokens = 0
            
            # Add sentence to current chunk
            current_chunk.append(sentence)
            current_tokens += sentence_tokens
        
        # Save remaining chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append(self._create_chunk(chunk_text, chunk_index, base_metadata))
        
        return chunks
    
    # # ==================== STRATEGY 3: Paragraph Chunking ====================
    
    # def _chunk_by_paragraph(
    #     self,
    #     text: str,
    #     base_metadata: Dict
    # ) -> List[Dict[str, Any]]:
    #     """
    #     Strategy 3: Paragraph-based chunking
        
    #     Groups paragraphs until target token count is reached
    #     Preserves paragraph boundaries
    #     """
    #     chunks = []
        
    #     # Split into paragraphs (double newline or more)
    #     paragraphs = re.split(r'\n\s*\n+', text)
    #     paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
    #     if not paragraphs:
    #         logger.warning("No paragraphs detected, falling back to sentence chunking")
    #         return self._chunk_by_sentence(text, base_metadata)
        
    #     chunk_index = 1
    #     current_chunk = []
    #     current_tokens = 0
        
    #     for paragraph in paragraphs:
    #         para_tokens = self.token_counter.count_tokens(paragraph)
            
    #         # If single paragraph exceeds max, split it by sentences
    #         if para_tokens > self.max_chunk_tokens:
    #             # Save current chunk if exists
    #             if current_chunk:
    #                 chunk_text = "\n\n".join(current_chunk)
    #                 chunks.append(self._create_chunk(chunk_text, chunk_index, base_metadata))
    #                 chunk_index += 1
    #                 current_chunk = []
    #                 current_tokens = 0
                
    #             # Split long paragraph by sentences
    #             sub_chunks = self._chunk_by_sentence(paragraph, base_metadata)
    #             for sub_chunk in sub_chunks:
    #                 sub_chunk['metadata']['chunk_index'] = chunk_index
    #                 chunks.append(sub_chunk)
    #                 chunk_index += 1
    #             continue
            
    #         # Check if adding paragraph exceeds target
    #         if current_tokens + para_tokens > self.target_chunk_tokens and current_chunk:
    #             # Save current chunk
    #             chunk_text = "\n\n".join(current_chunk)
    #             chunks.append(self._create_chunk(chunk_text, chunk_index, base_metadata))
    #             chunk_index += 1
    #             current_chunk = []
    #             current_tokens = 0
            
    #         # Add paragraph to current chunk
    #         current_chunk.append(paragraph)
    #         current_tokens += para_tokens
        
    #     # Save remaining chunk
    #     if current_chunk:
    #         chunk_text = "\n\n".join(current_chunk)
    #         chunks.append(self._create_chunk(chunk_text, chunk_index, base_metadata))
        
    #     return chunks
    
    # # ==================== STRATEGY 4: Sliding Window ====================
    
    # def _chunk_sliding_window(
    #     self,
    #     text: str,
    #     base_metadata: Dict
    # ) -> List[Dict[str, Any]]:
    #     """
    #     Strategy 4: Sliding window chunking
        
    #     Creates overlapping chunks with specified overlap
    #     Ensures continuity across chunks
    #     """
    #     chunks = []
        
    #     if self.token_counter.encoding:
    #         tokens = self.token_counter.get_tokens(text)
            
    #         if not tokens:
    #             return self._chunk_sliding_window_chars(text, base_metadata)
            
    #         stride = self.target_chunk_tokens - self.chunk_overlap_tokens
            
    #         if stride <= 0:
    #             stride = max(1, self.target_chunk_tokens // 2)
            
    #         chunk_index = 1
    #         position = 0
            
    #         while position < len(tokens):
    #             chunk_tokens = tokens[position:position + self.target_chunk_tokens]
    #             chunk_text = self.token_counter.decode_tokens(chunk_tokens)
                
    #             if chunk_text.strip():
    #                 chunks.append(self._create_chunk(
    #                     chunk_text, chunk_index, base_metadata,
    #                     position=position, total_tokens=len(tokens)
    #                 ))
    #                 chunk_index += 1
                
    #             position += stride
    #     else:
    #         chunks = self._chunk_sliding_window_chars(text, base_metadata)
        
    #     return chunks
    
    # # ==================== STRATEGY 5: Recursive Chunking ====================
    
    # def _chunk_recursive(
    #     self,
    #     text: str,
    #     base_metadata: Dict
    # ) -> List[Dict[str, Any]]:
    #     """
    #     Strategy 5: Recursive chunking
        
    #     Hierarchically splits text by separators:
    #     1. Paragraphs (\\n\\n)
    #     2. Sentences (. ! ?)
    #     3. Words
        
    #     Tries to keep chunks at target size while preserving structure
    #     """
    #     separators = [
    #         "\n\n\n",  # Section breaks
    #         "\n\n",    # Paragraph breaks
    #         "\n",      # Line breaks
    #         ". ",      # Sentences
    #         "! ",
    #         "? ",
    #         "; ",      # Clauses
    #         ", ",      # Phrases
    #         " "        # Words
    #     ]
        
    #     chunks = self._recursive_split(text, separators, base_metadata)
        
    #     # Add chunk indices
    #     for i, chunk in enumerate(chunks, 1):
    #         chunk['metadata']['chunk_index'] = i
        
    #     return chunks
    
    # def _recursive_split(
    #     self,
    #     text: str,
    #     separators: List[str],
    #     base_metadata: Dict,
    #     depth: int = 0
    # ) -> List[Dict[str, Any]]:
    #     """Recursively split text by separators"""
        
    #     # Base case: text is small enough
    #     tokens = self.token_counter.count_tokens(text)
    #     if tokens <= self.target_chunk_tokens:
    #         if tokens >= self.min_chunk_tokens or depth > len(separators):
    #             return [self._create_chunk(text, 0, base_metadata)]
    #         else:
    #             return []
        
    #     # No more separators, force split
    #     if not separators:
    #         return self._chunk_fixed_size(text, base_metadata)
        
    #     # Try current separator
    #     separator = separators[0]
    #     remaining_separators = separators[1:]
        
    #     parts = text.split(separator)
        
    #     # If split didn't help, try next separator
    #     if len(parts) == 1:
    #         return self._recursive_split(text, remaining_separators, base_metadata, depth + 1)
        
    #     # Recursively split parts
    #     chunks = []
    #     for part in parts:
    #         if not part.strip():
    #             continue
            
    #         sub_chunks = self._recursive_split(
    #             part, remaining_separators, base_metadata, depth + 1
    #         )
    #         chunks.extend(sub_chunks)
        
    #     return chunks

    #==================== STRATEGY 2: Hierarchical Chunking ====================
    def _chunk_hierarchical(
    self,
    text: str,
    base_metadata: Dict,
    max_parent_tokens: Optional[int] = None,
    max_child_tokens: Optional[int] = None,
    parent_overlap_tokens: Optional[int] = None,
) -> List[Dict[str, Any]]:
        """
        Hierarchical chunking with configurable parent / child sizes and parent overlap.

        Default behavior (when parameters are None):
        - max_parent_tokens   = 4 × target_chunk_tokens (default 1200 if target=300)
        - max_child_tokens    = target_chunk_tokens     (default 300)
        - parent_overlap_tokens = 20% of max_parent_tokens (default ~240)

        Returns chunks where:
        - Each chunk is a child chunk
        - Child chunks have 'parent_chunk_id' in metadata
        - Parent chunks overlap to preserve context continuity
        """
        if not text.strip():
            return []

        # ── Apply defaults if not provided ─────────────────────────────────────
        if max_parent_tokens is None:
            max_parent_tokens = max(600, int(self.target_chunk_tokens * 4))   # 4× child size

        if max_child_tokens is None:
            max_child_tokens = self.target_chunk_tokens

        if parent_overlap_tokens is None:
            parent_overlap_tokens = int(max_parent_tokens * 0.20)  # 20% of parent

        # Safety: overlap should not exceed parent size
        parent_overlap_tokens = min(parent_overlap_tokens, max_parent_tokens // 2)

        # ── Phase 1: Create overlapping parent chunks ───────────────────────────
        parent_chunks = []
        parent_index = 1

        tokens = self.token_counter.get_tokens(text)
        if not tokens:
            # Fallback to character-based
            return self._chunk_fixed_size_chars(text, base_metadata)

        stride = max_parent_tokens - parent_overlap_tokens
        if stride <= 0:
            stride = max_parent_tokens // 2

        position = 0
        while position < len(tokens):
            end = min(position + max_parent_tokens, len(tokens))
            parent_tokens_slice = tokens[position:end]
            parent_text = self.token_counter.decode_tokens(parent_tokens_slice)

            if not parent_text.strip():
                position += stride
                continue

            parent = self._create_chunk(
                parent_text,
                parent_index,
                base_metadata,
                position=position,
                total_tokens=len(tokens),
                chunk_type="parent"
            )
            parent["chunk_id"] = f"P{parent_index:03d}"
            parent_chunks.append(parent)

            position += stride
            parent_index += 1

        # ── Phase 2: Split each parent into child chunks ───────────────────────
        final_chunks = []

        child_index = 1

        for parent in parent_chunks:
            parent_id = parent["chunk_id"]
            parent_text = parent["text"]

            # Now split this parent into children
            child_chunks = self._recursive_split_children(
                parent_text,
                max_child_tokens,
                base_metadata,
                parent_id=parent_id,
                child_index_start=child_index
            )

            final_chunks.extend(child_chunks)
            child_index += len(child_chunks)

        logger.info(
            f"Hierarchical chunking → "
            f"{len(parent_chunks)} parents, "
            f"{len(final_chunks)} child chunks "
            f"(parent overlap: {parent_overlap_tokens} tokens)"
        )

        return final_chunks


    def _recursive_split_children(
        self,
        text: str,
        max_tokens: int,
        base_metadata: Dict,
        parent_id: str,
        child_index_start: int = 1,
        depth: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Recursively split a parent text block into child chunks.
        """
        tokens_count = self.token_counter.count_tokens(text)

        if tokens_count <= max_tokens:
            if tokens_count >= self.min_chunk_tokens or depth >= 6:
                child = self._create_chunk(
                    text,
                    child_index_start,
                    base_metadata,
                    chunk_type="child"
                )
                child["metadata"]["parent_chunk_id"] = parent_id
                return [child]
            return []

        # Separators — coarse to fine
        separators = [
            "\n\n\n",        # sections
            "\n\n",          # paragraphs
            ". ", "! ", "? ",  # sentences
            "; ", ", ",      # clauses/phrases
            " "              # words
        ]

        if depth >= len(separators):
            # Deepest level — force fixed-size
            return self._chunk_fixed_size(text, base_metadata)

        sep = separators[depth]
        parts = [p.strip() for p in text.split(sep) if p.strip()]

        if len(parts) <= 1:
            # No useful split — go deeper
            return self._recursive_split_children(
                text, max_tokens, base_metadata, parent_id,
                child_index_start, depth + 1
            )

        child_chunks = []
        current_child_index = child_index_start

        for part in parts:
            sub_chunks = self._recursive_split_children(
                part, max_tokens, base_metadata, parent_id,
                current_child_index, depth + 1
            )
            child_chunks.extend(sub_chunks)
            current_child_index += len(sub_chunks)

        return child_chunks
        
    
    # ==================== STRATEGY 3: Semantic Chunking ====================
    def _chunk_semantic(
    self,
    text: str,
    base_metadata: Dict
    ) -> List[Dict[str, Any]]:
        """
        Improved semantic chunking:
        - Groups semantically similar sentences (keyword + cosine similarity fallback)
        - Respects paragraph boundaries, lists, code blocks
        - Enforces minimum chunk size
        - Merges very small chunks with neighbors
        - Preserves structure (headings, code, lists)
        """
        if not text.strip():
            return []

        chunks = []
        chunk_index = 1

        # ── 1. Split into meaningful units (paragraphs + preserve structure) ─────
        units = self._split_into_structured_units(text)

        if not units:
            # Fallback if no structure detected
            return self._chunk_by_sentence(text, base_metadata)

        current_chunk_sentences = []
        current_chunk_text = ""
        current_tokens = 0

        for unit in units:
            unit_type = unit["type"]
            unit_text = unit["text"]
            unit_tokens = self.token_counter.count_tokens(unit_text)

            # Special handling for structural elements
            if unit_type in ("heading", "code_block", "list_item"):
                # Structural units often start/force new chunks
                if current_chunk_sentences:
                    chunk_text = " ".join(current_chunk_sentences)
                    chunks.append(self._create_chunk(
                        chunk_text, chunk_index, base_metadata,
                        chunk_type="semantic"
                    ))
                    chunk_index += 1
                    current_chunk_sentences = []
                    current_chunk_text = ""
                    current_tokens = 0

                # Add the structural unit as its own chunk if large
                if unit_tokens > self.min_chunk_tokens:
                    chunks.append(self._create_chunk(
                        unit_text, chunk_index, base_metadata,
                        chunk_type=unit_type
                    ))
                    chunk_index += 1
                    continue

            # Normal text unit (paragraph / sentence group)
            if current_tokens + unit_tokens <= self.target_chunk_tokens:
                # Try to add to current chunk
                if not current_chunk_sentences or self._is_semantically_similar(
                    current_chunk_text, unit_text, threshold=0.25
                ):
                    current_chunk_sentences.append(unit_text)
                    current_chunk_text += " " + unit_text
                    current_tokens += unit_tokens
                else:
                    # Not similar → save current and start new
                    if current_chunk_sentences:
                        chunk_text = " ".join(current_chunk_sentences)
                        chunks.append(self._create_chunk(
                            chunk_text, chunk_index, base_metadata,
                            chunk_type="semantic"
                        ))
                        chunk_index += 1
                    current_chunk_sentences = [unit_text]
                    current_chunk_text = unit_text
                    current_tokens = unit_tokens
            else:
                # Would exceed size → save current chunk
                if current_chunk_sentences:
                    chunk_text = " ".join(current_chunk_sentences)
                    chunks.append(self._create_chunk(
                        chunk_text, chunk_index, base_metadata,
                        chunk_type="semantic"
                    ))
                    chunk_index += 1
                # Start new chunk with current unit
                current_chunk_sentences = [unit_text]
                current_chunk_text = unit_text
                current_tokens = unit_tokens

        # Save the last chunk
        if current_chunk_sentences:
            chunk_text = " ".join(current_chunk_sentences)
            chunks.append(self._create_chunk(
                chunk_text, chunk_index, base_metadata,
                chunk_type="semantic"
            ))

        # ── Post-processing: merge very small chunks ────────────────────────────
        merged_chunks = []
        i = 0
        while i < len(chunks):
            current = chunks[i]
            current_tok = current["metadata"]["token_count"]

            # If chunk is too small → try to merge with next
            if current_tok < self.min_chunk_tokens and i + 1 < len(chunks):
                next_chunk = chunks[i + 1]
                combined_text = current["text"] + " " + next_chunk["text"]
                combined_tok = current_tok + next_chunk["metadata"]["token_count"]

                if combined_tok <= self.target_chunk_tokens:
                    merged = self._create_chunk(
                        combined_text,
                        len(merged_chunks) + 1,
                        base_metadata,
                        chunk_type="semantic_merged"
                    )
                    merged_chunks.append(merged)
                    i += 2  # skip next one
                    continue

            merged_chunks.append(current)
            i += 1

        logger.info(f"Semantic chunking created {len(merged_chunks)} chunks "
                    f"(after merging small fragments)")

        return merged_chunks


    def _split_into_structured_units(self, text: str) -> List[Dict]:
        """
        Split text into meaningful units while preserving structure:
        - Headings
        - Code blocks
        - Ordered/unordered lists
        - Normal paragraphs
        """
        units = []

        # Regex patterns for structure detection
        heading_pattern = r'^(#{1,6}\s+.+?)$'
        code_block_pattern = r'```[\s\S]*?```'
        list_pattern = r'^\s*([-*+]|\d+\.)\s+.+'

        lines = text.split('\n')
        current_unit = ""
        current_type = "paragraph"

        i = 0
        while i < len(lines):
            line = lines[i].rstrip()

            # Heading
            if re.match(heading_pattern, line):
                if current_unit:
                    units.append({"type": current_type, "text": current_unit.strip()})
                units.append({"type": "heading", "text": line})
                current_unit = ""
                current_type = "paragraph"
                i += 1
                continue

            # Code block
            if line.strip().startswith("```"):
                if current_unit:
                    units.append({"type": current_type, "text": current_unit.strip()})
                code_start = i
                i += 1
                while i < len(lines) and not lines[i].strip().startswith("```"):
                    i += 1
                if i < len(lines):
                    i += 1
                code_text = "\n".join(lines[code_start:i])
                units.append({"type": "code_block", "text": code_text})
                current_unit = ""
                continue

            # List item
            if re.match(list_pattern, line):
                if current_unit and current_type != "list":
                    units.append({"type": current_type, "text": current_unit.strip()})
                current_type = "list"
                current_unit += line + "\n"
                i += 1
                continue

            # Normal paragraph line
            if line.strip():
                if current_type != "paragraph":
                    if current_unit:
                        units.append({"type": current_type, "text": current_unit.strip()})
                    current_type = "paragraph"
                    current_unit = ""
                current_unit += line + " "
            else:
                # Blank line — end current paragraph
                if current_unit.strip():
                    units.append({"type": current_type, "text": current_unit.strip()})
                    current_unit = ""
                    current_type = "paragraph"

            i += 1

        # Don't forget last unit
        if current_unit.strip():
            units.append({"type": current_type, "text": current_unit.strip()})

        return units
    
    # ==================== Helper Methods ====================
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        if NLTK_AVAILABLE:
            try:
                sentences = nltk.sent_tokenize(text)
                return [s.strip() for s in sentences if s.strip()]
            except Exception as e:
                logger.warning(f"NLTK sentence tokenization failed: {e}")
        
        # Fallback: regex-based sentence splitting
        pattern = r'(?<=[.!?])\s+(?=[A-Z])'
        sentences = re.split(pattern, text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _is_semantically_similar(self, text1: str, text2: str, threshold: float = 0.3) -> bool:
        """
        Simple semantic similarity using keyword overlap
        
        Returns True if texts share significant keywords
        """
        # Extract keywords (words longer than 3 chars)
        words1 = set(re.findall(r'\b\w{4,}\b', text1.lower()))
        words2 = set(re.findall(r'\b\w{4,}\b', text2.lower()))
        
        if not words1 or not words2:
            return False
        
        # Calculate Jaccard similarity
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        similarity = intersection / union if union > 0 else 0
        
        return similarity >= threshold
    
    def _create_chunk(
        self,
        text: str,
        chunk_index: int,
        base_metadata: Dict,
        position: Optional[int] = None,
        total_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """Create a chunk dictionary"""
        
        page_number = self._extract_page_number(text)
        token_count = self.token_counter.count_tokens(text)
        
        chunk = {
            'chunk_id': self._generate_chunk_id(text, chunk_index),
            'text': text,
            'char_count': len(text),
            'token_count': token_count,
            'metadata': {
                'chunk_index': chunk_index,
                'chunk_type': 'text',
                'page_number': page_number,
                **base_metadata
            }
        }
        
        if position is not None and total_tokens is not None:
            chunk['metadata']['token_position'] = position
            chunk['metadata']['total_tokens_in_doc'] = total_tokens
        
        return chunk
    
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
    
    def _chunk_fixed_size_chars(self, text: str, base_metadata: Dict) -> List[Dict[str, Any]]:
        """Fallback: character-based fixed-size chunking"""
        chunks = []
        target_chars = self.target_chunk_tokens * 4
        
        chunk_index = 1
        position = 0
        
        while position < len(text):
            chunk_text = text[position:position + target_chars]
            
            if chunk_text.strip():
                chunks.append(self._create_chunk(chunk_text, chunk_index, base_metadata))
                chunk_index += 1
            
            position += target_chars
        
        return chunks
    
    def _chunk_sliding_window_chars(self, text: str, base_metadata: Dict) -> List[Dict[str, Any]]:
        """Fallback: character-based sliding window"""
        chunks = []
        target_chars = self.target_chunk_tokens * 4
        overlap_chars = self.chunk_overlap_tokens * 4
        stride_chars = target_chars - overlap_chars
        
        chunk_index = 1
        position = 0
        
        while position < len(text):
            chunk_text = text[position:position + target_chars]
            
            if chunk_text.strip():
                chunks.append(self._create_chunk(chunk_text, chunk_index, base_metadata))
                chunk_index += 1
            
            position += stride_chars
        
        return chunks
    
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


# Backward compatibility
SemanticChunkingService = EnhancedChunkingService
SimpleChunkingService = EnhancedChunkingService