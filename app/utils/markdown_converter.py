from datetime import datetime
from pathlib import Path
import re

class MarkdownConverter:
    """Convert extracted text to well-formatted markdown"""
    
    @staticmethod
    def convert_text_to_markdown(text: str, filename: str, file_type: str) -> str:
        """
        Convert extracted text to markdown format with metadata.
        """
        markdown_content = []
        
        # 1. Metadata Header
        markdown_content.append(f"# {filename}\n")
        markdown_content.append(f"**Type:** `{file_type.upper()}` | **Date:** {datetime.now().strftime('%Y-%m-%d')}\n")
        markdown_content.append("---\n")
        
        # 2. Content Cleaning
        # The text_extractor now produces good markdown, so we just do light cleanup
        cleaned_text = MarkdownConverter._clean_markdown_text(text)
        markdown_content.append(cleaned_text)
        
        # 3. Footer
        markdown_content.append("\n\n---")
        markdown_content.append(f"\n*End of Document*")
        
        return '\n'.join(markdown_content)
    
    @staticmethod
    def _clean_markdown_text(text: str) -> str:
        """
        Clean up text but preserve table structures and code blocks.
        """
        if not text: return ""

        lines = text.split('\n')
        cleaned_lines = []
        
        blank_counter = 0
        
        for line in lines:
            line_stripped = line.strip()
            
            # Detect if this is a table row (don't mess with spacing inside tables)
            is_table_row = line_stripped.startswith('|') and line_stripped.endswith('|')
            
            if not line_stripped:
                blank_counter += 1
                if blank_counter < 3: # Allow max 2 blank lines
                    cleaned_lines.append("")
            else:
                blank_counter = 0
                cleaned_lines.append(line) # Keep original indentation
        
        # Join
        result = '\n'.join(cleaned_lines)
        
        # Fix multiple spaces (but not inside tables)
        # Note: This regex avoids matching text inside | ... | roughly
        # For simplicity, we just fix 3+ newlines globally
        result = re.sub(r'\n{3,}', '\n\n', result)
        
        return result
    
    @staticmethod
    def save_markdown_file(markdown_content: str, output_path: str) -> bool:
        try:
            p = Path(output_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            return True
        except Exception as e:
            print(f"Save Error: {e}")
            return False