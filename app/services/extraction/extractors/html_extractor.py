"""
HTML/HTM File Extractor
Processes HTML web pages for RAG pipeline
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


class HTMLExtractor(BaseExtractor):
    """
    Extractor for HTML/HTM files
    
    Features:
    - Extracts text content from HTML
    - Preserves heading hierarchy
    - Extracts links and images
    - Removes scripts and styles
    - Converts to clean markdown
    """
    
    def _get_supported_formats(self) -> List[str]:
        return ['html', 'htm']
    
    def validate_file(self, file_path: str) -> Tuple[bool, str]:
        """Validate HTML file"""
        path = Path(file_path)
        
        if not path.exists():
            return False, "File does not exist"
        
        ext = path.suffix.lower().lstrip('.')
        if ext not in self.supported_formats:
            return False, "Not an HTML file"
        
        return True, "Valid HTML file"
    
    def extract(self, file_path: str) -> Tuple[str, bool, Dict]:
        """Extract HTML file"""
        try:
            is_valid, msg = self.validate_file(file_path)
            if not is_valid:
                return "", False, {'error': msg}
            
            logger.info(f"Extracting HTML: {file_path}")
            
            # Read file
            html_content = self._read_with_encoding(file_path)
            
            if not html_content:
                return "", False, {'error': 'Empty HTML file'}
            
            html_stem = Path(file_path).stem
            
            # Parse HTML and extract content
            parsed_data = self._parse_html(html_content)
            
            if not parsed_data['text']:
                return "", False, {'error': 'No text content in HTML'}
            
            # Build markdown
            markdown_content = self._build_markdown(html_stem, parsed_data)
            
            metadata = self.get_metadata(file_path)
            metadata.update({
                'total_pages': 1,
                'title': parsed_data.get('title', html_stem),
                'link_count': len(parsed_data.get('links', [])),
                'image_count': len(parsed_data.get('images', [])),
                'heading_count': len(parsed_data.get('headings', [])),
                'extraction_method': 'html'
            })
            
            logger.info(f"✓ HTML extraction complete")
            
            return markdown_content, True, metadata
            
        except Exception as e:
            logger.error(f"HTML extraction error: {str(e)}")
            import traceback
            traceback.print_exc()
            return "", False, {'error': str(e)}
    
    def _read_with_encoding(self, file_path: str) -> str:
        """Read HTML file with encoding detection"""
        encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252', 'iso-8859-1']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
        
        # Fallback
        with open(file_path, 'rb') as f:
            return f.read().decode('utf-8', errors='ignore')
    
    def _parse_html(self, html_content: str) -> Dict:
        """Parse HTML and extract structured data"""
        try:
            from bs4 import BeautifulSoup
            return self._parse_with_beautifulsoup(html_content)
        except ImportError:
            logger.warning("BeautifulSoup not available, using basic parsing")
            return self._parse_basic(html_content)
    
    def _parse_with_beautifulsoup(self, html_content: str) -> Dict:
        """Parse HTML using BeautifulSoup"""
        from bs4 import BeautifulSoup
        
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove script and style elements
        for element in soup(['script', 'style', 'noscript']):
            element.decompose()
        
        # Extract title
        title = ""
        title_tag = soup.find('title')
        if title_tag:
            title = title_tag.get_text().strip()
        
        # Extract headings
        headings = []
        for level in range(1, 7):
            for heading in soup.find_all(f'h{level}'):
                headings.append({
                    'level': level,
                    'text': heading.get_text().strip()
                })
        
        # Extract links
        links = []
        for link in soup.find_all('a', href=True):
            link_text = link.get_text().strip()
            href = link['href']
            if link_text or href:
                links.append({
                    'text': link_text or href,
                    'url': href
                })
        
        # Extract images
        images = []
        for img in soup.find_all('img'):
            src = img.get('src', '')
            alt = img.get('alt', '')
            if src:
                images.append({
                    'src': src,
                    'alt': alt
                })
        
        # Extract main text
        # Try to find main content area
        main_content = soup.find('main') or soup.find('article') or soup.find('body') or soup
        
        # Get text with structure
        text_parts = []
        for element in main_content.descendants:
            if element.name in ['p', 'div', 'li', 'td', 'th']:
                text = element.get_text().strip()
                if text and text not in text_parts:
                    text_parts.append(text)
        
        # If no structured text found, get all text
        if not text_parts:
            text_parts = [main_content.get_text(separator='\n').strip()]
        
        return {
            'title': title,
            'headings': headings,
            'links': links,
            'images': images,
            'text': '\n\n'.join(text_parts)
        }
    
    def _parse_basic(self, html_content: str) -> Dict:
        """Basic HTML parsing without BeautifulSoup"""
        # Extract title
        title_match = re.search(r'<title[^>]*>(.*?)</title>', html_content, re.IGNORECASE | re.DOTALL)
        title = title_match.group(1).strip() if title_match else ""
        
        # Remove scripts and styles
        html_content = re.sub(r'<script[^>]*>.*?</script>', '', html_content, flags=re.IGNORECASE | re.DOTALL)
        html_content = re.sub(r'<style[^>]*>.*?</style>', '', html_content, flags=re.IGNORECASE | re.DOTALL)
        
        # Extract headings
        headings = []
        for level in range(1, 7):
            for match in re.finditer(rf'<h{level}[^>]*>(.*?)</h{level}>', html_content, re.IGNORECASE | re.DOTALL):
                text = re.sub(r'<[^>]+>', '', match.group(1)).strip()
                if text:
                    headings.append({'level': level, 'text': text})
        
        # Extract links
        links = []
        for match in re.finditer(r'<a[^>]+href=["\']([^"\']+)["\'][^>]*>(.*?)</a>', html_content, re.IGNORECASE | re.DOTALL):
            url = match.group(1)
            text = re.sub(r'<[^>]+>', '', match.group(2)).strip()
            links.append({'url': url, 'text': text or url})
        
        # Extract images
        images = []
        for match in re.finditer(r'<img[^>]+src=["\']([^"\']+)["\'][^>]*>', html_content, re.IGNORECASE):
            src = match.group(1)
            alt_match = re.search(r'alt=["\']([^"\']+)["\']', match.group(0), re.IGNORECASE)
            alt = alt_match.group(1) if alt_match else ""
            images.append({'src': src, 'alt': alt})
        
        # Remove all tags and get text
        text = re.sub(r'<[^>]+>', '\n', html_content)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return {
            'title': title,
            'headings': headings,
            'links': links,
            'images': images,
            'text': text
        }
    
    def _build_markdown(self, html_stem: str, parsed_data: Dict) -> str:
        """Build markdown from parsed HTML"""
        lines = []
        
        # Header
        title = parsed_data.get('title', html_stem)
        lines.append(f"# {title}")
        lines.append(f"**Format**: HTML Document")
        lines.append("")
        lines.append("---\n")
        
        # Main content with headings preserved
        lines.append("## Content\n")
        
        # Add text content
        text = parsed_data.get('text', '')
        if text:
            lines.append(text)
        
        lines.append("")
        
        # Links section
        links = parsed_data.get('links', [])
        if links:
            lines.append("## Links\n")
            for i, link in enumerate(links[:50], 1):  # Limit to 50 links
                lines.append(f"{i}. [{link['text']}]({link['url']})")
            
            if len(links) > 50:
                lines.append(f"\n*(... and {len(links) - 50} more links)*")
            
            lines.append("")
        
        # Images section
        images = parsed_data.get('images', [])
        if images:
            lines.append("## Images\n")
            for i, img in enumerate(images[:20], 1):  # Limit to 20 images
                if img['alt']:
                    lines.append(f"{i}. ![{img['alt']}]({img['src']})")
                else:
                    lines.append(f"{i}. Image: {img['src']}")
            
            if len(images) > 20:
                lines.append(f"\n*(... and {len(images) - 20} more images)*")
            
            lines.append("")
        
        return '\n'.join(lines)
