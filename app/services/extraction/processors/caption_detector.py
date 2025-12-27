"""
Caption Detector with Missing Helper Methods
Added find_caption_in_range, find_caption_before, and format_caption_as_heading
"""

import re
from typing import Optional, Tuple, Dict, List
import logging

logger = logging.getLogger(__name__)


class CaptionDetector:
    """
    Caption detection for figures and tables
    
    Includes helper methods for searching around image placeholders
    """
    
    # Simple patterns
    CAPTION_PATTERN = re.compile(
        r'^(Figure|Fig\.?|Table|Tbl\.?|Chart|Graph)\s*(\d+\.?\d*)\s*[:\-\.]?\s*(.*)$',
        re.IGNORECASE
    )
    
    @classmethod
    def detect_caption(cls, text: str) -> Optional[Tuple[str, str, str]]:
        """
        Detect caption in text line
        
        Args:
            text: Text line to check
        
        Returns:
            Tuple of (type, number, description) or None
            Example: ('figure', '1', 'Revenue Chart')
        """
        text = text.strip()
        if not text:
            return None
        
        match = cls.CAPTION_PATTERN.match(text)
        if not match:
            return None
        
        label = match.group(1).lower()
        number = match.group(2)
        description = match.group(3).strip()
        
        # Normalize label
        if 'fig' in label:
            caption_type = 'figure'
        elif 'tabl' in label or 'tbl' in label:
            caption_type = 'table'
        elif 'chart' in label or 'graph' in label:
            caption_type = 'chart'
        else:
            caption_type = label
        
        return (caption_type, number, description)
    
    @classmethod
    def find_caption_in_range(
        cls,
        lines: List[str],
        start_idx: int,
        lookahead: int = 10
    ) -> Optional[Dict]:
        """
        Search for caption in lines after start_idx
        
        Args:
            lines: List of text lines
            start_idx: Starting line index
            lookahead: Number of lines to search ahead
            
        Returns:
            Dictionary with caption info or None
        """
        end_idx = min(start_idx + lookahead, len(lines))
        
        for i in range(start_idx, end_idx):
            if i >= len(lines):
                break
                
            line = lines[i].strip()
            
            # Skip empty lines
            if not line:
                continue
            
            # Stop at structural elements
            if line.startswith('#') or '[IMAGE' in line.upper():
                break
            
            # Try to detect caption
            result = cls.detect_caption(line)
            if result:
                caption_type, number, description = result
                return {
                    'type': caption_type,
                    'number': number,
                    'description': description,
                    'line_idx': i,
                    'full_text': line
                }
        
        return None
    
    @classmethod
    def find_caption_before(
        cls,
        lines: List[str],
        image_idx: int,
        lookback: int = 10
    ) -> Optional[Dict]:
        """
        Search for caption in lines before image_idx
        
        Args:
            lines: List of text lines
            image_idx: Image line index
            lookback: Number of lines to search backward
            
        Returns:
            Dictionary with caption info or None
        """
        start_idx = max(0, image_idx - lookback)
        
        # Search backward from image_idx
        for i in range(image_idx - 1, start_idx - 1, -1):
            if i < 0:
                break
                
            line = lines[i].strip()
            
            # Skip empty lines
            if not line:
                continue
            
            # Stop at structural elements
            if line.startswith('#'):
                break
            
            # Try to detect caption
            result = cls.detect_caption(line)
            if result:
                caption_type, number, description = result
                return {
                    'type': caption_type,
                    'number': number,
                    'description': description,
                    'line_idx': i,
                    'full_text': line
                }
        
        return None
    
    @classmethod
    def format_as_heading(
        cls,
        caption_type: str,
        number: str,
        description: str,
        level: int = 3
    ) -> str:
        """
        Format caption as markdown heading
        
        Args:
            caption_type: 'figure', 'table', or 'chart'
            number: Caption number
            description: Caption description
            level: Heading level (default 3 for ###)
        
        Returns:
            Formatted markdown heading
        """
        heading_prefix = '#' * level
        cap_type = caption_type.capitalize()
        
        if description:
            return f"{heading_prefix} {cap_type} {number}: {description}"
        return f"{heading_prefix} {cap_type} {number}"
    
    @classmethod
    def format_caption_as_heading(
        cls,
        caption_type: str,
        number: str,
        description: str,
        level: int = 3
    ) -> str:
        """
        Alias for format_as_heading for backward compatibility
        
        Args:
            caption_type: 'figure', 'table', or 'chart'
            number: Caption number
            description: Caption description
            level: Heading level (default 3 for ###)
        
        Returns:
            Formatted markdown heading
        """
        return cls.format_as_heading(caption_type, number, description, level)


# Simple helper functions for backward compatibility
def detect_figure_caption(text: str) -> Optional[Tuple[str, str]]:
    """Detect figure caption - returns (number, description)"""
    result = CaptionDetector.detect_caption(text)
    if result and result[0] == 'figure':
        return (result[1], result[2])
    return None


def detect_table_caption(text: str) -> Optional[Tuple[str, str]]:
    """Detect table caption - returns (number, description)"""
    result = CaptionDetector.detect_caption(text)
    if result and result[0] == 'table':
        return (result[1], result[2])
    return None