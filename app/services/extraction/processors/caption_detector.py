"""
Caption Detector - COMPLETE FIX
Handles all issues found in PDF extraction:
1. Standalone figure numbers (Figure 1.3.2 with no caption)
2. Body text mixed in captions
3. Chart titles as false captions
4. Multi-segment numbers
"""

import re
from typing import Optional, Tuple, Dict, List
import logging

logger = logging.getLogger(__name__)


class CaptionDetector:
    """
    Enhanced caption detection with comprehensive fixes
    
    Features:
    1. Detects standalone figure numbers: "Figure 1.3.2"
    2. Filters body text from captions
    3. Handles multi-segment numbers: 1.1.10, 1.2.3
    4. Reconstructs from references when needed
    5. Ignores chart titles
    """
    
    # Main pattern for captions with description
    CAPTION_PATTERN = re.compile(
        r'^(Figure|Fig\.?|Table|Tbl\.?|Chart|Graph)\s*'  # Label
        r'(\d+(?:\.\d+)*)'                               # Number
        r'\s*[:\-\.]?\s*'                                # Separator
        r'(.*)$',                                        # Description
        re.IGNORECASE
    )
    
    # Pattern for standalone figure numbers (no caption text)
    STANDALONE_PATTERN = re.compile(
        r'^(Figure|Fig\.?|Table|Tbl\.?|Chart|Graph)\s+'
        r'(\d+(?:\.\d+)*)\s*\.?\s*$',  # Optional period at end
        re.IGNORECASE
    )
    
    # Pattern to find figure references in body text
    REFERENCE_PATTERN = re.compile(
        r'[\(\[]?\s*(Figure|Fig\.?|Table|Tbl\.?|Chart|Graph)\s+'
        r'(\d+(?:\.\d+)+)'  # Must have at least one dot
        r'\s*[\)\]]?',
        re.IGNORECASE
    )
    
    # Body text indicators - when these appear, it's likely body text not caption
    BODY_TEXT_INDICATORS = [
        'illustrate', 'illustrates',
        'show', 'shows', 'showing',
        'highlight', 'highlights', 'highlighting',
        'display', 'displays', 'displaying',
        'visualize', 'visualizes',
        'present', 'presents',
        'demonstrate', 'demonstrates',
        'and figure', 'see figure', 'in figure',
        'as shown', 'as illustrated',
        'over the', 'during the',  # Time phrases
    ]
    
    @classmethod
    def detect_caption(cls, text: str) -> Optional[Tuple[str, str, str]]:
        """
        Detect caption with description
        
        Args:
            text: Text line to check
        
        Returns:
            Tuple of (type, number, description) or None
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
    def detect_standalone_figure(cls, text: str) -> Optional[Tuple[str, str]]:
        """
        NEW: Detect standalone figure number (no caption text)
        
        Examples:
            "Figure 1.3.2" → ('figure', '1.3.2')
            "Figure 1.3.2." → ('figure', '1.3.2')
            "Table 5" → ('table', '5')
        
        Args:
            text: Text line to check
            
        Returns:
            Tuple of (type, number) or None
        """
        text = text.strip()
        if not text:
            return None
        
        match = cls.STANDALONE_PATTERN.match(text)
        if not match:
            return None
        
        label = match.group(1).lower()
        number = match.group(2)
        
        # Normalize label
        if 'fig' in label:
            caption_type = 'figure'
        elif 'tabl' in label or 'tbl' in label:
            caption_type = 'table'
        elif 'chart' in label or 'graph' in label:
            caption_type = 'chart'
        else:
            caption_type = label
        
        return (caption_type, number)
    
    @classmethod
    def is_body_text(cls, description: str) -> bool:
        """
        NEW: Check if description is actually body text
        
        Indicators:
        - Long text (> 50 chars)
        - Contains verbs like "shows", "illustrates"
        - References other figures
        - Contains time phrases
        
        Args:
            description: Caption description text
            
        Returns:
            True if likely body text, False if likely caption
        """
        if not description or len(description) < 30:
            return False  # Short text, probably caption
        
        desc_lower = description.lower()
        
        # Check for body text indicators
        if any(indicator in desc_lower for indicator in cls.BODY_TEXT_INDICATORS):
            return True
        
        # Check if mentions other figures (e.g., "and Figure 1.3.7")
        other_figure_pattern = r'(?:and|see|in)\s+figure\s+\d+(?:\.\d+)+'
        if re.search(other_figure_pattern, desc_lower):
            return True
        
        # Check if very long (> 100 chars = likely paragraph)
        if len(description) > 100:
            return True
        
        return False
    
    @classmethod
    def is_chart_title(cls, text: str) -> bool:
        """
        NEW: Detect if line is likely a chart title (not a caption)
        
        Chart title indicators:
        - Short (< 100 chars)
        - Ends with year: 2024, 2003-24
        - Contains keywords: "number of", "by", "per"
        - No "Figure X.Y" at start
        
        Args:
            text: Text line to check
            
        Returns:
            True if likely chart title, False otherwise
        """
        text = text.strip()
        
        # Don't confuse with figure captions
        if re.match(r'^(Figure|Fig|Table)', text, re.IGNORECASE):
            return False
        
        # Chart titles are relatively short
        if len(text) > 100:
            return False
        
        # Often end with year
        year_pattern = r'\d{4}(?:[-–]\d{2,4})?$'
        has_year = bool(re.search(year_pattern, text))
        
        # Contains chart-related keywords
        chart_keywords = [
            'number of', 'percentage of', 'distribution of',
            ' by ', ' per ', ' vs ', ' versus ',
            'growth', 'change', 'trend'
        ]
        has_keyword = any(kw in text.lower() for kw in chart_keywords)
        
        # Both indicators present = likely chart title
        return has_year and has_keyword
    
    @classmethod
    def find_caption_in_range(
        cls,
        lines: List[str],
        start_idx: int,
        lookahead: int = 10
    ) -> Optional[Dict]:
        """
        Search for caption with all enhancements
        
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
            
            # NEW: Skip chart titles
            if cls.is_chart_title(line):
                logger.debug(f"Skipping chart title: {line[:50]}...")
                continue
            
            # Stop at structural elements
            if line.startswith('#') or '[IMAGE' in line.upper():
                break
            
            # Try to detect full caption first
            result = cls.detect_caption(line)
            if result:
                caption_type, number, description = result
                
                # NEW: Check if description is actually body text
                if cls.is_body_text(description):
                    logger.info(
                        f"Filtering body text from caption '{caption_type.capitalize()} {number}': "
                        f"{description[:50]}..."
                    )
                    description = ''  # Clear body text
                
                # Check if caption might be incomplete (for reconstruction)
                is_suspicious = description and description[0].isdigit()
                
                if is_suspicious:
                    # Try reconstruction
                    corrected_number = cls._find_complete_number_in_context(
                        lines, i, caption_type, number
                    )
                    if corrected_number:
                        logger.info(
                            f"Caption reconstruction: '{caption_type.capitalize()} {number}' "
                            f"→ '{caption_type.capitalize()} {corrected_number}'"
                        )
                        number = corrected_number
                        description = re.sub(r'^\d+\s*', '', description).strip()
                
                return {
                    'type': caption_type,
                    'number': number,
                    'description': description,
                    'line_idx': i,
                    'full_text': line
                }
            
            # NEW: Try standalone figure detection
            standalone = cls.detect_standalone_figure(line)
            if standalone:
                caption_type, number = standalone
                logger.debug(f"Found standalone figure: {caption_type.capitalize()} {number}")
                
                return {
                    'type': caption_type,
                    'number': number,
                    'description': '',  # No caption text
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
        Search for caption before image with all enhancements
        
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
            
            # NEW: Skip chart titles
            if cls.is_chart_title(line):
                logger.debug(f"Skipping chart title: {line[:50]}...")
                continue
            
            # Stop at structural elements
            if line.startswith('#'):
                break
            
            # Try full caption first
            result = cls.detect_caption(line)
            if result:
                caption_type, number, description = result
                
                # NEW: Check if body text
                if cls.is_body_text(description):
                    logger.info(
                        f"Filtering body text from caption '{caption_type.capitalize()} {number}': "
                        f"{description[:50]}..."
                    )
                    description = ''
                
                # Check for reconstruction
                is_suspicious = description and description[0].isdigit()
                
                if is_suspicious:
                    corrected_number = cls._find_complete_number_in_context(
                        lines, i, caption_type, number
                    )
                    if corrected_number:
                        logger.info(
                            f"Caption reconstruction: '{caption_type.capitalize()} {number}' "
                            f"→ '{caption_type.capitalize()} {corrected_number}'"
                        )
                        number = corrected_number
                        description = re.sub(r'^\d+\s*', '', description).strip()
                
                return {
                    'type': caption_type,
                    'number': number,
                    'description': description,
                    'line_idx': i,
                    'full_text': line
                }
            
            # NEW: Try standalone detection
            standalone = cls.detect_standalone_figure(line)
            if standalone:
                caption_type, number = standalone
                logger.debug(f"Found standalone figure: {caption_type.capitalize()} {number}")
                
                return {
                    'type': caption_type,
                    'number': number,
                    'description': '',
                    'line_idx': i,
                    'full_text': line
                }
        
        return None
    
    @classmethod
    def _find_complete_number_in_context(
        cls,
        lines: List[str],
        caption_idx: int,
        caption_type: str,
        partial_number: str
    ) -> Optional[str]:
        """
        Reconstruct full figure number from references
        
        Args:
            lines: All document lines
            caption_idx: Index of caption line
            caption_type: Type ('figure', 'table', etc.)
            partial_number: Incomplete number
            
        Returns:
            Complete number or None
        """
        search_start = max(0, caption_idx - 15)
        search_end = min(len(lines), caption_idx + 15)
        
        candidates = []
        
        for i in range(search_start, search_end):
            if i >= len(lines):
                break
            
            line = lines[i]
            
            for match in cls.REFERENCE_PATTERN.finditer(line):
                ref_label = match.group(1).lower()
                ref_number = match.group(2)
                
                # Normalize type
                if 'fig' in ref_label:
                    ref_type = 'figure'
                elif 'tabl' in ref_label or 'tbl' in ref_label:
                    ref_type = 'table'
                elif 'chart' in ref_label or 'graph' in ref_label:
                    ref_type = 'chart'
                else:
                    ref_type = ref_label
                
                # Must match type and start with partial number
                if ref_type != caption_type:
                    continue
                
                if not ref_number.startswith(partial_number):
                    continue
                
                # Must be more complete
                if ref_number.count('.') <= partial_number.count('.'):
                    continue
                
                distance = abs(i - caption_idx)
                candidates.append({
                    'number': ref_number,
                    'distance': distance,
                    'line_idx': i
                })
        
        if not candidates:
            return None
        
        # Return closest reference
        candidates.sort(key=lambda x: x['distance'])
        best_match = candidates[0]
        logger.debug(
            f"Found reference at line {best_match['line_idx']} "
            f"(distance: {best_match['distance']}): {best_match['number']}"
        )
        return best_match['number']
    
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
            description: Caption description (can be empty)
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
        """Alias for backward compatibility"""
        return cls.format_as_heading(caption_type, number, description, level)


# Backward compatibility helpers
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