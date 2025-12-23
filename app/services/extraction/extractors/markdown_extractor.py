"""
Markdown File Extractor
Processes markdown files for RAG pipeline
"""

import os
from pathlib import Path
from typing import Tuple, Dict, List
import logging
import re

try:
    from ..base_extractor import BaseExtractor
except ImportError:
    from base_extractor import BaseExtractor

logger = logging.getLogger(__name__)


class MarkdownExtractor(BaseExtractor):
    """
    Extractor for Markdown files
    
    Features:
    - Preserves markdown formatting
    - Estimates pages based on content
    - Adds page markers
    - Extracts headings for structure
    """
    
    def _get_supported_formats(self) -> List[str]:
        return ['md', 'markdown', 'mdown', 'mkd']
    
    def validate_file(self, file_path: str) -> Tuple[bool, str]:
        """Validate markdown file"""
        path = Path(file_path)
        
        if not path.exists():
            return False, "File does not exist"
        
        ext = path.suffix.lower().lstrip('.')
        if ext not in self.supported_formats:
            return False, "Not a markdown file"
        
        return True, "Valid markdown file"
    
    def extract(self, file_path: str) -> Tuple[str, bool, Dict]:
        """Extract and process markdown file"""
        try:
            is_valid, msg = self.validate_file(file_path)
            if not is_valid:
                return "", False, {'error': msg}
            
            logger.info(f"Extracting markdown: {file_path}")
            
            # Read file
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if not content.strip():
                return "", False, {'error': 'Empty markdown file'}
            
            md_stem = Path(file_path).stem
            
            # Estimate pages (roughly 500 words per page)
            word_count = len(content.split())
            estimated_pages = max(1, (word_count // 500) + 1)
            
            # Build markdown with page markers
            markdown_content = self._build_markdown_with_pages(
                content, md_stem, estimated_pages
            )
            
            # Extract statistics
            stats = self._extract_statistics(content)
            
            metadata = self.get_metadata(file_path)
            metadata.update({
                'total_pages': estimated_pages,
                'word_count': word_count,
                'char_count': len(content),
                'line_count': content.count('\n'),
                'heading_count': stats['headings'],
                'code_block_count': stats['code_blocks'],
                'extraction_method': 'markdown'
            })
            
            logger.info(f"✓ Markdown extraction complete: {estimated_pages} pages, {word_count} words")
            
            return markdown_content, True, metadata
            
        except Exception as e:
            logger.error(f"Markdown extraction error: {str(e)}")
            return "", False, {'error': str(e)}
    
    def _build_markdown_with_pages(
        self,
        content: str,
        md_stem: str,
        total_pages: int
    ) -> str:
        """Add page markers to markdown content"""
        lines = content.split('\n')
        result_lines = []
        
        # Add header
        result_lines.append(f"# Document: {md_stem}\n")
        result_lines.append(f"**Total Pages**: {total_pages}")
        result_lines.append(f"**Format**: Markdown\n")
        result_lines.append("---\n")
        
        # Add page markers based on content distribution
        total_lines = len(lines)
        lines_per_page = max(1, total_lines // total_pages)
        
        current_page = 1
        result_lines.append(f"\n## Page {current_page}\n")
        
        for i, line in enumerate(lines):
            result_lines.append(line)
            
            # Add page marker at intervals
            if i > 0 and (i % lines_per_page == 0) and current_page < total_pages:
                current_page += 1
                result_lines.append(f"\n## Page {current_page}\n")
        
        return '\n'.join(result_lines)
    
    def _extract_statistics(self, content: str) -> Dict:
        """Extract statistics from markdown"""
        stats = {
            'headings': len(re.findall(r'^#+\s', content, re.MULTILINE)),
            'code_blocks': len(re.findall(r'```', content)) // 2,
            'links': len(re.findall(r'\[([^\]]+)\]\(([^\)]+)\)', content)),
            'images': len(re.findall(r'!\[([^\]]*)\]\(([^\)]+)\)', content))
        }
        return stats