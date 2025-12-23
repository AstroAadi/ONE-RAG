"""
Caption Detector Processor
Detects and extracts figure and table captions from text
Shared across all extractors
"""

import re
from typing import Tuple, Optional, Dict, List
import logging

logger = logging.getLogger(__name__)


class CaptionDetector:
    """
    Utility class for detecting and extracting captions
    
    Supports:
    - Figure captions: "Figure 1:", "Fig. 2:", "FIGURE 3.1 -"
    - Table captions: "Table 1:", "Tbl. 2:", "TABLE 3 -"
    - Chart captions: "Chart 1:", "Graph 2:"
    """
    
    # Regex patterns for different caption types
    FIGURE_PATTERN = r'^(Figure|Fig\.?|FIGURE|FIG\.?)\s*(\d+\.?\d*)\s*[:\-\.]?\s*(.*)$'
    TABLE_PATTERN = r'^(Table|TABLE|Tbl\.?|TBL\.?)\s*(\d+\.?\d*)\s*[:\-\.]?\s*(.*)$'
    CHART_PATTERN = r'^(Chart|CHART|Graph|GRAPH)\s*(\d+\.?\d*)\s*[:\-\.]?\s*(.*)$'
    
    # Combined pattern for any caption type
    COMBINED_PATTERN = r'^(Figure|Fig\.?|FIGURE|FIG\.?|Table|TABLE|Tbl\.?|TBL\.?|Chart|CHART|Graph|GRAPH)\s*(\d+\.?\d*)\s*[:\-\.]?\s*(.*)$'
    
    @classmethod
    def detect_caption(
        cls, 
        text: str, 
        caption_type: Optional[str] = None
    ) -> Optional[Tuple[str, str, str]]:
        """
        Detect caption in a text line
        
        Args:
            text: Text line to check for caption
            caption_type: Type of caption to detect:
                         'figure', 'table', 'chart', or None for any
            
        Returns:
            Tuple of (caption_type, number, description) or None
            Example: ('figure', '1', 'Revenue Analysis Chart')
        """
        text = text.strip()
        
        if not text:
            return None
        
        # Select pattern based on caption type
        if caption_type == 'figure':
            pattern = cls.FIGURE_PATTERN
        elif caption_type == 'table':
            pattern = cls.TABLE_PATTERN
        elif caption_type == 'chart':
            pattern = cls.CHART_PATTERN
        else:
            pattern = cls.COMBINED_PATTERN
        
        match = re.match(pattern, text, re.IGNORECASE)
        
        if match:
            label = match.group(1).lower()  # figure, fig, table, etc.
            number = match.group(2)
            description = match.group(3).strip()
            
            # Normalize label
            if 'fig' in label:
                normalized_type = 'figure'
            elif 'tabl' in label or 'tbl' in label:
                normalized_type = 'table'
            elif 'chart' in label or 'graph' in label:
                normalized_type = 'chart'
            else:
                normalized_type = label
            
            return (normalized_type, number, description)
        
        return None
    
    @classmethod
    def find_caption_in_range(
        cls,
        lines: List[str],
        start_idx: int,
        lookahead: int = 5,
        caption_type: Optional[str] = None
    ) -> Optional[Dict]:
        """
        Search for caption in a range of lines after a given position
        
        Args:
            lines: List of text lines
            start_idx: Index to start searching from
            lookahead: Number of lines to search (default: 5)
            caption_type: Type of caption to search for (None = any)
            
        Returns:
            Dictionary with caption information or None:
            {
                'type': 'figure',
                'number': '1',
                'description': 'Revenue Analysis',
                'line_idx': 42,
                'full_text': 'Figure 1: Revenue Analysis'
            }
        """
        for i in range(start_idx, min(start_idx + lookahead, len(lines))):
            line = lines[i].strip()
            
            # Skip empty lines
            if not line:
                continue
            
            # Try to detect caption
            result = cls.detect_caption(line, caption_type)
            
            if result:
                cap_type, number, description = result
                return {
                    'type': cap_type,
                    'number': number,
                    'description': description,
                    'line_idx': i,
                    'full_text': line
                }
            
            # Stop search if we hit structural elements
            if cls._is_structural_element(line):
                break
        
        return None
    
    @classmethod
    def find_caption_before(
        cls,
        lines: List[str],
        current_idx: int,
        lookback: int = 3,
        caption_type: Optional[str] = None
    ) -> Optional[Dict]:
        """
        Search for caption in lines before current position
        Useful when caption appears above the image
        
        Args:
            lines: List of text lines
            current_idx: Current position
            lookback: Number of lines to search backwards (default: 3)
            caption_type: Type of caption to search for
            
        Returns:
            Dictionary with caption information or None
        """
        start_idx = max(0, current_idx - lookback)
        
        for i in range(current_idx - 1, start_idx - 1, -1):
            line = lines[i].strip()
            
            if not line:
                continue
            
            result = cls.detect_caption(line, caption_type)
            
            if result:
                cap_type, number, description = result
                return {
                    'type': cap_type,
                    'number': number,
                    'description': description,
                    'line_idx': i,
                    'full_text': line
                }
            
            # Stop if we hit structural elements
            if cls._is_structural_element(line):
                break
        
        return None
    
    @classmethod
    def format_caption_as_heading(
        cls,
        caption_type: str,
        number: str,
        description: str,
        heading_level: int = 3
    ) -> str:
        """
        Format caption as a markdown heading
        
        Args:
            caption_type: 'figure', 'table', or 'chart'
            number: Caption number
            description: Caption description
            heading_level: Markdown heading level (default: 3 for ###)
            
        Returns:
            Formatted markdown heading
            
        Example:
            format_caption_as_heading('figure', '1', 'Revenue Chart', 3)
            Returns: "### Figure 1: Revenue Chart"
        """
        # Capitalize caption type
        cap_type_formatted = caption_type.capitalize()
        
        # Build heading
        heading_prefix = '#' * heading_level
        
        if description:
            return f"{heading_prefix} {cap_type_formatted} {number}: {description}"
        else:
            return f"{heading_prefix} {cap_type_formatted} {number}"
    
    @classmethod
    def extract_all_captions(cls, text: str) -> List[Dict]:
        """
        Extract all captions from text
        
        Args:
            text: Full text to search
            
        Returns:
            List of caption dictionaries
        """
        lines = text.split('\n')
        captions = []
        
        for idx, line in enumerate(lines):
            result = cls.detect_caption(line)
            if result:
                cap_type, number, description = result
                captions.append({
                    'type': cap_type,
                    'number': number,
                    'description': description,
                    'line_idx': idx,
                    'full_text': line.strip()
                })
        
        return captions
    
    @staticmethod
    def _is_structural_element(line: str) -> bool:
        """
        Check if line is a structural element that should stop caption search
        
        Args:
            line: Text line to check
            
        Returns:
            True if structural element, False otherwise
        """
        line = line.strip()
        
        # Markdown headings
        if line.startswith('#'):
            return True
        
        # Image placeholders
        if re.match(r'^\[IMAGE:', line, re.IGNORECASE):
            return True
        
        # Table markers
        if line.startswith('|') and line.endswith('|'):
            return True
        
        # Horizontal rules
        if re.match(r'^[-*_]{3,}$', line):
            return True
        
        return False


# Convenience functions for backward compatibility
def detect_figure_caption(text: str) -> Optional[Tuple[str, str]]:
    """
    Detect figure caption (legacy interface)
    
    Returns:
        Tuple of (number, description) or None
    """
    result = CaptionDetector.detect_caption(text, caption_type='figure')
    if result:
        _, number, description = result
        return (number, description)
    return None


def detect_table_caption(text: str) -> Optional[Tuple[str, str]]:
    """
    Detect table caption (legacy interface)
    
    Returns:
        Tuple of (number, description) or None
    """
    result = CaptionDetector.detect_caption(text, caption_type='table')
    if result:
        _, number, description = result
        return (number, description)
    return None