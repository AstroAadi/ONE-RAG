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

# """
# Improved HTML / HTM Extractor for RAG pipelines
# - Clean, structured Markdown output
# - Smart main content detection
# - Headings, links, images, tables, code blocks handled properly
# - Robust fallback when BeautifulSoup is missing
# - No resource leaks, modern pathlib usage
# - Safe parser selection (no deprecated .features attribute)
# """

# import re
# from pathlib import Path
# from typing import Tuple, Dict, List, Optional
# from datetime import datetime
# import logging

# try:
#     from bs4 import BeautifulSoup, Tag, NavigableString
#     HAS_BS4 = True
# except ImportError:
#     HAS_BS4 = False
#     logging.warning("BeautifulSoup not installed — using regex fallback")

# logger = logging.getLogger(__name__)


# class HTMLExtractor:
#     """
#     High-quality HTML extractor optimized for RAG and human readability.
#     """

#     SUPPORTED_FORMATS = {'html', 'htm'}

#     def validate_file(self, file_path: str) -> Tuple[bool, str]:
#         path = Path(file_path)
#         if not path.exists():
#             return False, "File does not exist"
#         ext = path.suffix.lower().lstrip('.')
#         if ext not in self.SUPPORTED_FORMATS:
#             return False, f"Unsupported extension: .{ext}"
#         return True, "Valid HTML file"

#     def extract(self, file_path: str) -> Tuple[str, bool, Dict]:
#         try:
#             valid, msg = self.validate_file(file_path)
#             if not valid:
#                 logger.warning(msg)
#                 return "", False, {"error": msg}

#             logger.info(f"Extracting HTML: {file_path}")

#             html = self._read_safe(file_path)
#             if not html.strip():
#                 return "", False, {"error": "Empty file"}

#             parsed = self._parse_html(html)
#             markdown = self._build_markdown(Path(file_path).stem, parsed)

#             metadata = {
#                 "title": parsed.get("title", ""),
#                 "headings": len(parsed["headings"]),
#                 "links": len(parsed["links"]),
#                 "images": len(parsed["images"]),
#                 "tables": len(parsed["tables"]),
#                 "code_blocks": len(parsed["code_blocks"]),
#                 "content_chars": len(parsed["text"]),
#                 "method": "beautifulsoup" if HAS_BS4 else "regex",
#                 "extracted_at": datetime.now().isoformat()
#             }

#             if len(parsed["text"]) < 200:
#                 logger.warning(f"Low content length: {len(parsed['text'])} chars")

#             logger.info("HTML extraction complete")
#             return markdown, True, metadata

#         except Exception as e:
#             logger.exception(f"HTML extraction failed: {file_path}")
#             return "", False, {"error": str(e)}

#     def _read_safe(self, path_str: str) -> str:
#         """Read file with encoding fallback – no resource leak"""
#         path = Path(path_str)
#         encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252", "iso-8859-1"]

#         for enc in encodings:
#             try:
#                 return path.read_text(encoding=enc)
#             except UnicodeDecodeError:
#                 continue

#         # Binary fallback – replace errors
#         return path.read_bytes().decode("utf-8", errors="replace")

#     def _parse_html(self, html: str) -> Dict:
#         """Core parsing – prefers BeautifulSoup, falls back to regex"""
#         if HAS_BS4:
#             return self._parse_bs4(html)
#         return self._parse_regex(html)

#     def _parse_bs4(self, html: str) -> Dict:
#         """Parse using BeautifulSoup – safe parser selection"""
#         # Use built-in html.parser (no external lxml required)
#         soup = BeautifulSoup(html, "html.parser")

#         # Remove unwanted tags
#         for tag in soup(["script", "style", "noscript", "iframe", "form", "footer", "header"]):
#             tag.decompose()

#         # Title
#         title = soup.title.string.strip() if soup.title else ""

#         # Headings
#         headings = []
#         for lv in range(1, 7):
#             for h in soup.find_all(f"h{lv}"):
#                 txt = h.get_text(strip=True)
#                 if txt:
#                     headings.append({"level": lv, "text": txt})

#         # Smart main content detection
#         main = (
#             soup.find("main")
#             or soup.find("article")
#             or soup.find(id=re.compile(r"(content|main|article|post|entry|body)", re.I))
#             or soup.find(class_=re.compile(r"(content|main|article|post|entry)", re.I))
#             or soup.body
#             or soup
#         )

#         # Extract clean text (paragraph-aware)
#         text_parts = []
#         for node in main.descendants:
#             if isinstance(node, NavigableString) and (node := node.strip()):
#                 tag = getattr(node.parent, "name", "")
#                 if tag in ("p", "div", "li", "td", "th", "blockquote", "dd", "dt"):
#                     text_parts.append(node)

#         text = "\n\n".join(text_parts).strip()

#         # Links – filter junk
#         links = []
#         for a in soup.find_all("a", href=True):
#             href = a["href"].strip()
#             txt = a.get_text(strip=True)
#             if href and not href.startswith(("#", "javascript:", "mailto:", "tel:", "data:", "file:")):
#                 links.append({"text": txt or href, "url": href})

#         # Images
#         images = []
#         for img in soup.find_all("img"):
#             src = img.get("src", "").strip()
#             alt = img.get("alt", "").strip()
#             if src:
#                 images.append({"src": src, "alt": alt or "Image"})

#         # Tables → Markdown
#         tables = []
#         for tbl in soup.find_all("table"):
#             md = self._table_to_markdown(tbl)
#             if md:
#                 tables.append(md)

#         # Code blocks
#         code_blocks = []
#         for pre in soup.find_all("pre"):
#             code = pre.get_text(strip=True)
#             lang = ""
#             if pre.code and pre.code.get("class"):
#                 lang = " ".join(pre.code["class"]).replace("language-", "")
#             code_blocks.append({"lang": lang or "text", "code": code})

#         return {
#             "title": title,
#             "headings": headings,
#             "text": text,
#             "links": links,
#             "images": images,
#             "tables": tables,
#             "code_blocks": code_blocks,
#         }

#     def _table_to_markdown(self, table: Tag) -> Optional[str]:
#         """Convert HTML table to Markdown table"""
#         rows = []
#         for tr in table.find_all("tr"):
#             cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
#             if any(cells):
#                 rows.append(cells)

#         if len(rows) < 1:
#             return None

#         header = rows[0]
#         body = rows[1:]

#         md = ["| " + " | ".join(header) + " |"]
#         md.append("|" + " --- |" * len(header))
#         for row in body:
#             md.append("| " + " | ".join(row) + " |")

#         return "\n".join(md)

#     def _parse_regex(self, html: str) -> Dict:
#         """Minimal regex fallback when BeautifulSoup is unavailable"""
#         title = re.search(r"<title[^>]*>(.*?)</title>", html, re.I | re.S)
#         title = title.group(1).strip() if title else ""

#         text = re.sub(r"<script.*?</script>", "", html, flags=re.I | re.S)
#         text = re.sub(r"<style.*?</style>", "", text, flags=re.I | re.S)
#         text = re.sub(r"<[^>]+>", "\n", text)
#         text = re.sub(r"\s*\n\s*", "\n", text).strip()

#         return {
#             "title": title,
#             "headings": [],
#             "text": text,
#             "links": [],
#             "images": [],
#             "tables": [],
#             "code_blocks": [],
#         }

#     def _build_markdown(self, filename_stem: str, parsed: Dict) -> str:
#         """Generate clean, modern, RAG-friendly Markdown"""
#         lines = []

#         title = parsed.get("title", filename_stem.replace("_", " ").title())

#         # Frontmatter
#         lines.extend([
#             "---",
#             f"title: {title}",
#             "source: HTML",
#             f"extracted: {datetime.now().isoformat()}",
#             f"headings: {len(parsed.get('headings', []))}",
#             f"links: {len(parsed.get('links', []))}",
#             f"images: {len(parsed.get('images', []))}",
#             f"tables: {len(parsed.get('tables', []))}",
#             f"code_blocks: {len(parsed.get('code_blocks', []))}",
#             "---",
#             ""
#         ])

#         # Header
#         lines.extend([
#             f"# {title}",
#             f"**File**: {filename_stem}",
#             f"**Processed**: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
#             ""
#         ])

#         # Summary
#         lines.extend([
#             "## Overview",
#             f"- Title: {title or '—'}",
#             f"- Headings: {len(parsed.get('headings', []))}",
#             f"- Links: {len(parsed.get('links', []))}",
#             f"- Images: {len(parsed.get('images', []))}",
#             f"- Tables: {len(parsed.get('tables', []))}",
#             f"- Code blocks: {len(parsed.get('code_blocks', []))}",
#             ""
#         ])

#         # Main Content
#         lines.append("## Main Content")
#         lines.append("")
#         text = parsed.get("text", "").strip()
#         if text:
#             paragraphs = [p.strip() for p in re.split(r'\n{2,}', text) if p.strip()]
#             for p in paragraphs:
#                 lines.append(p)
#                 lines.append("")
#         else:
#             lines.append("*No readable text content could be extracted.*\n")

#         # Headings Outline
#         if parsed.get("headings"):
#             lines.append("## Headings Outline")
#             lines.append("")
#             for h in parsed["headings"]:
#                 prefix = "#" * h["level"]
#                 lines.append(f"{prefix} {h['text']}")
#             lines.append("")

#         # Code Blocks
#         if parsed.get("code_blocks"):
#             lines.append("## Code Blocks")
#             lines.append("")
#             for i, cb in enumerate(parsed["code_blocks"], 1):
#                 lang = cb["lang"] or "text"
#                 lines.append(f"**Block {i}** ({lang})")
#                 lines.append(f"```{lang}")
#                 lines.append(cb["code"])
#                 lines.append("```")
#                 lines.append("")

#         # Tables
#         if parsed.get("tables"):
#             lines.append("## Tables")
#             lines.append("")
#             for i, tbl in enumerate(parsed["tables"], 1):
#                 lines.append(f"**Table {i}**")
#                 lines.append(tbl)
#                 lines.append("")

#         # Links
#         links = parsed.get("links", [])
#         if links:
#             lines.append("## Links")
#             lines.append(f"Total: {len(links)}\n")

#             external = [l for l in links if l["url"].startswith(("http://", "https://"))]
#             internal = [l for l in links if l not in external]

#             if external:
#                 lines.append("### External Links")
#                 for l in external[:20]:
#                     txt = l["text"] or l["url"]
#                     lines.append(f"- [{txt}]({l['url']})")
#                 if len(external) > 20:
#                     lines.append(f"- … and {len(external)-20} more")
#                 lines.append("")

#             if internal:
#                 lines.append("### Internal / Anchors")
#                 for l in internal[:15]:
#                     txt = l["text"] or l["url"]
#                     lines.append(f"- [{txt}]({l['url']})")
#                 if len(internal) > 15:
#                     lines.append(f"- … and {len(internal)-15} more")

#         # Images
#         images = parsed.get("images", [])
#         if images:
#             lines.append("## Images")
#             lines.append(f"Found {len(images)}\n")
#             for i, img in enumerate(images[:10], 1):
#                 alt = img.get("alt", "Image").strip() or "Unnamed image"
#                 src = img.get("src", "#")
#                 lines.extend([
#                     f"**Image {i}**: {alt}",
#                     f"![{alt}]({src})",
#                     f"Source: {src}",
#                     ""
#                 ])
#             if len(images) > 10:
#                 lines.append(f"… and {len(images)-10} more")

#         # Footer
#         lines.extend([
#             "---",
#             f"*Processed from HTML on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
#             f"*Original file*: {filename_stem}"
#         ])

#         return "\n".join(lines)