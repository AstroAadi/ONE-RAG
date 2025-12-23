"""
Text Processor
Shared utilities for text processing and cleaning
"""

import re
from typing import List, Dict, Optional, Tuple
import logging
import unicodedata

logger = logging.getLogger(__name__)


class TextProcessor:
    """
    Utilities for text processing and normalization
    
    Handles:
    - Text cleaning and normalization
    - Whitespace handling
    - Special character processing
    - Text encoding fixes
    - Sentence and paragraph detection
    """
    
    @staticmethod
    def clean_text(text: str, preserve_newlines: bool = True) -> str:
        """
        Clean and normalize text
        
        Args:
            text: Input text
            preserve_newlines: Whether to preserve newline characters
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Normalize unicode characters
        text = unicodedata.normalize('NFKC', text)
        
        # Remove null bytes
        text = text.replace('\x00', '')
        
        # Fix common encoding issues
        text = TextProcessor._fix_encoding_issues(text)
        
        # Normalize whitespace
        if preserve_newlines:
            # Replace multiple spaces with single space, but keep newlines
            lines = text.split('\n')
            lines = [re.sub(r' +', ' ', line.strip()) for line in lines]
            text = '\n'.join(lines)
        else:
            # Replace all whitespace with single space
            text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    @staticmethod
    def _fix_encoding_issues(text: str) -> str:
        """Fix common encoding issues"""
        # Common replacements
        replacements = {
            'â€™': "'",
            'â€œ': '"',
            'â€': '"',
            'â€"': '—',
            'â€"': '–',
            'Â': '',
            'â€¦': '...',
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        return text
    
    @staticmethod
    def remove_extra_whitespace(text: str, max_consecutive: int = 2) -> str:
        """
        Remove excessive whitespace while preserving structure
        
        Args:
            text: Input text
            max_consecutive: Maximum consecutive newlines to allow
            
        Returns:
            Text with controlled whitespace
        """
        if not text:
            return ""
        
        # Remove trailing whitespace from each line
        lines = [line.rstrip() for line in text.split('\n')]
        
        # Remove excessive blank lines
        cleaned_lines = []
        blank_count = 0
        
        for line in lines:
            if not line.strip():
                blank_count += 1
                if blank_count <= max_consecutive:
                    cleaned_lines.append(line)
            else:
                blank_count = 0
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    @staticmethod
    def normalize_punctuation(text: str) -> str:
        """
        Normalize punctuation spacing
        
        Args:
            text: Input text
            
        Returns:
            Text with normalized punctuation
        """
        if not text:
            return ""
        
        # Add space after punctuation if missing
        text = re.sub(r'([.!?;,])([A-Z])', r'\1 \2', text)
        
        # Remove space before punctuation
        text = re.sub(r'\s+([.!?;,])', r'\1', text)
        
        # Fix multiple punctuation
        text = re.sub(r'\.{4,}', '...', text)
        text = re.sub(r'!{2,}', '!', text)
        text = re.sub(r'\?{2,}', '?', text)
        
        return text
    
    @staticmethod
    def extract_sentences(text: str) -> List[str]:
        """
        Split text into sentences
        
        Args:
            text: Input text
            
        Returns:
            List of sentences
        """
        if not text:
            return []
        
        # Simple sentence splitter
        # Handles: . ! ? followed by space and capital letter
        sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])'
        sentences = re.split(sentence_pattern, text)
        
        # Clean sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    @staticmethod
    def extract_paragraphs(text: str, min_length: int = 10) -> List[str]:
        """
        Split text into paragraphs
        
        Args:
            text: Input text
            min_length: Minimum paragraph length in characters
            
        Returns:
            List of paragraphs
        """
        if not text:
            return []
        
        # Split by double newlines
        paragraphs = re.split(r'\n\s*\n', text)
        
        # Filter short paragraphs
        paragraphs = [p.strip() for p in paragraphs if len(p.strip()) >= min_length]
        
        return paragraphs
    
    @staticmethod
    def remove_urls(text: str, replacement: str = '[URL]') -> str:
        """
        Remove or replace URLs in text
        
        Args:
            text: Input text
            replacement: Replacement text for URLs
            
        Returns:
            Text with URLs removed/replaced
        """
        if not text:
            return ""
        
        # URL pattern
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        text = re.sub(url_pattern, replacement, text)
        
        return text
    
    @staticmethod
    def remove_emails(text: str, replacement: str = '[EMAIL]') -> str:
        """
        Remove or replace email addresses
        
        Args:
            text: Input text
            replacement: Replacement text for emails
            
        Returns:
            Text with emails removed/replaced
        """
        if not text:
            return ""
        
        # Email pattern
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        text = re.sub(email_pattern, replacement, text)
        
        return text
    
    @staticmethod
    def detect_language(text: str) -> Optional[str]:
        """
        Detect language of text (basic detection)
        
        Args:
            text: Input text
            
        Returns:
            Language code or None
        """
        # This is a very basic implementation
        # For production, use langdetect or similar
        
        if not text or len(text) < 20:
            return None
        
        # Count character ranges
        ascii_count = sum(1 for c in text if ord(c) < 128)
        total_chars = len(text)
        
        ascii_ratio = ascii_count / total_chars if total_chars > 0 else 0
        
        if ascii_ratio > 0.9:
            return 'en'  # Likely English
        else:
            return 'unknown'
    
    @staticmethod
    def count_words(text: str) -> int:
        """
        Count words in text
        
        Args:
            text: Input text
            
        Returns:
            Word count
        """
        if not text:
            return 0
        
        # Simple word count
        words = re.findall(r'\b\w+\b', text)
        return len(words)
    
    @staticmethod
    def truncate_text(
        text: str, 
        max_length: int, 
        suffix: str = "...",
        preserve_words: bool = True
    ) -> str:
        """
        Truncate text to maximum length
        
        Args:
            text: Input text
            max_length: Maximum length
            suffix: Suffix to add when truncated
            preserve_words: If True, truncate at word boundary
            
        Returns:
            Truncated text
        """
        if not text or len(text) <= max_length:
            return text
        
        if preserve_words:
            # Find last space before max_length
            truncated = text[:max_length - len(suffix)]
            last_space = truncated.rfind(' ')
            if last_space > 0:
                truncated = truncated[:last_space]
            return truncated + suffix
        else:
            return text[:max_length - len(suffix)] + suffix
    
    @staticmethod
    def extract_headings(text: str) -> List[Dict[str, any]]:
        """
        Extract markdown headings from text
        
        Args:
            text: Input text (markdown)
            
        Returns:
            List of heading dictionaries with level and text
        """
        headings = []
        
        for line in text.split('\n'):
            line = line.strip()
            if line.startswith('#'):
                # Count heading level
                level = 0
                for char in line:
                    if char == '#':
                        level += 1
                    else:
                        break
                
                # Extract heading text
                heading_text = line[level:].strip()
                
                if heading_text:
                    headings.append({
                        'level': level,
                        'text': heading_text
                    })
        
        return headings
    
    @staticmethod
    def normalize_line_endings(text: str, target: str = '\n') -> str:
        """
        Normalize line endings
        
        Args:
            text: Input text
            target: Target line ending ('\n', '\r\n', or '\r')
            
        Returns:
            Text with normalized line endings
        """
        if not text:
            return ""
        
        # Replace all line endings with target
        text = text.replace('\r\n', '\n')
        text = text.replace('\r', '\n')
        
        if target != '\n':
            text = text.replace('\n', target)
        
        return text
    
    @staticmethod
    def remove_code_blocks(text: str) -> str:
        """
        Remove code blocks from markdown text
        
        Args:
            text: Input markdown text
            
        Returns:
            Text without code blocks
        """
        if not text:
            return ""
        
        # Remove fenced code blocks
        text = re.sub(r'```[\s\S]*?```', '', text)
        
        # Remove inline code
        text = re.sub(r'`[^`]+`', '', text)
        
        return text
    
    @staticmethod
    def extract_numbers(text: str) -> List[float]:
        """
        Extract all numbers from text
        
        Args:
            text: Input text
            
        Returns:
            List of numbers found
        """
        if not text:
            return []
        
        # Pattern for integers and decimals
        number_pattern = r'-?\d+\.?\d*'
        matches = re.findall(number_pattern, text)
        
        numbers = []
        for match in matches:
            try:
                num = float(match)
                numbers.append(num)
            except ValueError:
                continue
        
        return numbers
    
    @staticmethod
    def contains_table(text: str) -> bool:
        """
        Check if text contains a markdown table
        
        Args:
            text: Input text
            
        Returns:
            True if table found
        """
        if not text:
            return False
        
        # Look for table separator line
        table_pattern = r'\|[\s\-:]+\|'
        return bool(re.search(table_pattern, text))
    
    @staticmethod
    def split_by_section(text: str) -> Dict[str, str]:
        """
        Split text by markdown sections (headings)
        
        Args:
            text: Input markdown text
            
        Returns:
            Dictionary mapping section titles to content
        """
        sections = {}
        current_section = "Introduction"
        current_content = []
        
        for line in text.split('\n'):
            if line.strip().startswith('#'):
                # Save previous section
                if current_content:
                    sections[current_section] = '\n'.join(current_content).strip()
                
                # Start new section
                current_section = line.strip().lstrip('#').strip()
                current_content = []
            else:
                current_content.append(line)
        
        # Save last section
        if current_content:
            sections[current_section] = '\n'.join(current_content).strip()
        
        return sections
    
    @staticmethod
    def highlight_keywords(
        text: str, 
        keywords: List[str], 
        prefix: str = '**', 
        suffix: str = '**'
    ) -> str:
        """
        Highlight keywords in text
        
        Args:
            text: Input text
            keywords: List of keywords to highlight
            prefix: Prefix for highlighting
            suffix: Suffix for highlighting
            
        Returns:
            Text with highlighted keywords
        """
        if not text or not keywords:
            return text
        
        for keyword in keywords:
            # Case-insensitive replacement
            pattern = re.compile(re.escape(keyword), re.IGNORECASE)
            text = pattern.sub(f"{prefix}{keyword}{suffix}", text)
        
        return text
    
    @staticmethod
    def calculate_reading_time(text: str, words_per_minute: int = 200) -> float:
        """
        Calculate estimated reading time
        
        Args:
            text: Input text
            words_per_minute: Average reading speed
            
        Returns:
            Reading time in minutes
        """
        word_count = TextProcessor.count_words(text)
        return word_count / words_per_minute
    
    @staticmethod
    def get_text_statistics(text: str) -> Dict[str, any]:
        """
        Get comprehensive text statistics
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with various statistics
        """
        if not text:
            return {
                'char_count': 0,
                'word_count': 0,
                'sentence_count': 0,
                'paragraph_count': 0,
                'reading_time_minutes': 0.0
            }
        
        sentences = TextProcessor.extract_sentences(text)
        paragraphs = TextProcessor.extract_paragraphs(text)
        word_count = TextProcessor.count_words(text)
        
        return {
            'char_count': len(text),
            'word_count': word_count,
            'sentence_count': len(sentences),
            'paragraph_count': len(paragraphs),
            'reading_time_minutes': round(TextProcessor.calculate_reading_time(text), 1),
            'avg_sentence_length': round(word_count / len(sentences), 1) if sentences else 0,
            'avg_word_length': round(len(text) / word_count, 1) if word_count > 0 else 0
        }