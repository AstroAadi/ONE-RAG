"""
Markdown Converter - Simplified Version
Extractors already produce good markdown, so just do minimal cleanup
"""

from datetime import datetime
from pathlib import Path
import re
import logging

logger = logging.getLogger(__name__)


class MarkdownConverter:
    """
    Light cleanup of markdown from extractors
    
    Extractors already produce well-formatted markdown with:
    - Document headers
    - Page markers
    - Image paths
    - Tables
    - Captions
    
    This class just does final cleanup and adds processing metadata
    """
    
    @staticmethod
    def convert_text_to_markdown(text: str, filename: str, file_type: str) -> str:
        """
        Light cleanup of extractor output
        
        Args:
            text: Markdown text from extractor (already well-formatted)
            filename: Original filename (for logging)
            file_type: File extension (for logging)
            
        Returns:
            Cleaned markdown with processing timestamp
        """
        if not text or not text.strip():
            logger.warning(f"Empty text provided for {filename}")
            return f"# {filename}\n\n*No content extracted*"
        
        # Clean the markdown (preserve tables and code blocks)
        cleaned_text = MarkdownConverter._clean_markdown_text(text)
        
        # Add processing timestamp at end
        timestamp = f"\n\n---\n*Processed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n"
        
        return cleaned_text + timestamp
    
    @staticmethod
    def _clean_markdown_text(text: str) -> str:
        """
        Minimal cleanup - preserve structure
        
        - Remove excessive blank lines (max 2 consecutive)
        - Preserve table formatting
        - Preserve code blocks
        - Preserve indentation
        """
        if not text:
            return ""
        
        lines = text.split('\n')
        cleaned_lines = []
        blank_counter = 0
        
        for line in lines:
            line_stripped = line.strip()
            
            # Check if this is a table row (don't mess with table formatting)
            is_table_row = line_stripped.startswith('|') and line_stripped.endswith('|')
            
            # Check if this is a code block marker
            is_code_block = line_stripped.startswith('```')
            
            if not line_stripped:
                # Blank line
                blank_counter += 1
                if blank_counter <= 2:  # Allow max 2 consecutive blank lines
                    cleaned_lines.append("")
            else:
                # Non-blank line
                blank_counter = 0
                
                if is_table_row or is_code_block:
                    # Preserve exact formatting for tables and code
                    cleaned_lines.append(line)
                else:
                    # Keep original line with indentation
                    cleaned_lines.append(line)
        
        # Join lines
        result = '\n'.join(cleaned_lines)
        
        # Final cleanup: Remove 3+ consecutive newlines
        result = re.sub(r'\n{3,}', '\n\n', result)
        
        return result.strip()
    
    @staticmethod
    def save_markdown_file(markdown_content: str, output_path: str) -> bool:
        """
        Save markdown to file
        
        Args:
            markdown_content: Markdown text to save
            output_path: Path to save file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            p = Path(output_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            
            with open(p, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            
            logger.info(f"Saved markdown: {p.name}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving markdown to {output_path}: {e}")
            return False
    
    @staticmethod
    def count_visual_elements(text: str) -> dict:
        """
        Count visual elements in markdown
        
        Helper function for statistics
        
        Args:
            text: Markdown text
            
        Returns:
            Dictionary with counts of images, tables, charts
        """
        return {
            'images': text.count('IMAGE_PATH:'),
            'tables': text.count('**[TABLE]**') + text.count('| --- |'),
            'charts': text.count('**[CHART]**') + text.count('**[FIGURE]**'),
            'pages': text.count('## Page '),
            'slides': text.count('## Slide ')
        }