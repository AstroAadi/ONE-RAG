"""
EPUB (Electronic Publication) File Extractor
Processes EPUB eBook files for RAG pipeline
"""

import os
from pathlib import Path
from typing import Tuple, Dict, List
import logging
import zipfile
import xml.etree.ElementTree as ET

try:
    from ..base_extractor import BaseExtractor
except ImportError:
    from base_extractor import BaseExtractor

logger = logging.getLogger(__name__)


class EPUBExtractor(BaseExtractor):
    """
    Extractor for EPUB eBook files
    
    Features:
    - Extracts book metadata (title, author, etc.)
    - Extracts chapter structure
    - Converts XHTML to text
    - Preserves reading order
    
    EPUB is a ZIP archive containing XHTML files
    """
    
    def _get_supported_formats(self) -> List[str]:
        return ['epub']
    
    def validate_file(self, file_path: str) -> Tuple[bool, str]:
        """Validate EPUB file"""
        path = Path(file_path)
        
        if not path.exists():
            return False, "File does not exist"
        
        ext = path.suffix.lower().lstrip('.')
        if ext not in self.supported_formats:
            return False, "Not an EPUB file"
        
        # Check if it's a valid ZIP file
        if not zipfile.is_zipfile(file_path):
            return False, "Not a valid EPUB (not a ZIP archive)"
        
        # Check for mimetype file
        try:
            with zipfile.ZipFile(file_path, 'r') as epub_zip:
                if 'mimetype' not in epub_zip.namelist():
                    return False, "Not a valid EPUB (missing mimetype)"
                
                mimetype = epub_zip.read('mimetype').decode('utf-8').strip()
                if mimetype != 'application/epub+zip':
                    return False, f"Invalid EPUB mimetype: {mimetype}"
        except Exception as e:
            return False, f"Could not validate EPUB: {e}"
        
        return True, "Valid EPUB file"
    
    def extract(self, file_path: str) -> Tuple[str, bool, Dict]:
        """Extract EPUB eBook file"""
        try:
            is_valid, msg = self.validate_file(file_path)
            if not is_valid:
                return "", False, {'error': msg}
            
            logger.info(f"Extracting EPUB: {file_path}")
            
            epub_stem = Path(file_path).stem
            
            with zipfile.ZipFile(file_path, 'r') as epub_zip:
                # Extract metadata
                metadata_info = self._extract_metadata(epub_zip)
                
                # Extract content
                chapters = self._extract_chapters(epub_zip, metadata_info)
                
                if not chapters:
                    return "", False, {'error': 'No content found in EPUB'}
                
                # Build markdown
                markdown_content = self._build_markdown(
                    epub_stem, metadata_info, chapters
                )
                
                metadata = self.get_metadata(file_path)
                metadata.update({
                    'total_pages': len(chapters),
                    'title': metadata_info.get('title', epub_stem),
                    'author': metadata_info.get('author', 'Unknown'),
                    'publisher': metadata_info.get('publisher', ''),
                    'language': metadata_info.get('language', ''),
                    'chapter_count': len(chapters),
                    'extraction_method': 'epub'
                })
                
                logger.info(f"✓ EPUB extraction complete: {len(chapters)} chapters")
                
                return markdown_content, True, metadata
            
        except Exception as e:
            logger.error(f"EPUB extraction error: {str(e)}")
            import traceback
            traceback.print_exc()
            return "", False, {'error': str(e)}
    
    def _extract_metadata(self, epub_zip: zipfile.ZipFile) -> Dict:
        """Extract EPUB metadata from OPF file"""
        metadata = {}
        
        try:
            # Find OPF file
            container_path = 'META-INF/container.xml'
            if container_path in epub_zip.namelist():
                container_content = epub_zip.read(container_path)
                container_root = ET.fromstring(container_content)
                
                # Get OPF path
                ns = {'cnt': 'urn:oasis:names:tc:opendocument:xmlns:container'}
                rootfile = container_root.find('.//cnt:rootfile', ns)
                
                if rootfile is not None:
                    opf_path = rootfile.get('full-path')
                    
                    if opf_path and opf_path in epub_zip.namelist():
                        opf_content = epub_zip.read(opf_path)
                        opf_root = ET.fromstring(opf_content)
                        
                        # Extract metadata
                        ns_dc = {'dc': 'http://purl.org/dc/elements/1.1/'}
                        ns_opf = {'opf': 'http://www.idpf.org/2007/opf'}
                        
                        title_elem = opf_root.find('.//dc:title', ns_dc)
                        if title_elem is not None:
                            metadata['title'] = title_elem.text or ''
                        
                        creator_elem = opf_root.find('.//dc:creator', ns_dc)
                        if creator_elem is not None:
                            metadata['author'] = creator_elem.text or ''
                        
                        publisher_elem = opf_root.find('.//dc:publisher', ns_dc)
                        if publisher_elem is not None:
                            metadata['publisher'] = publisher_elem.text or ''
                        
                        lang_elem = opf_root.find('.//dc:language', ns_dc)
                        if lang_elem is not None:
                            metadata['language'] = lang_elem.text or ''
                        
                        # Get spine (reading order)
                        manifest = {}
                        for item in opf_root.findall('.//opf:item', ns_opf):
                            item_id = item.get('id')
                            href = item.get('href')
                            if item_id and href:
                                manifest[item_id] = href
                        
                        spine = []
                        for itemref in opf_root.findall('.//opf:itemref', ns_opf):
                            idref = itemref.get('idref')
                            if idref and idref in manifest:
                                spine.append(manifest[idref])
                        
                        metadata['spine'] = spine
                        metadata['opf_dir'] = str(Path(opf_path).parent)
        
        except Exception as e:
            logger.warning(f"Could not extract EPUB metadata: {e}")
        
        return metadata
    
    def _extract_chapters(self, epub_zip: zipfile.ZipFile, metadata: Dict) -> List[Dict]:
        """Extract chapters from EPUB"""
        chapters = []
        
        spine = metadata.get('spine', [])
        opf_dir = metadata.get('opf_dir', '')
        
        if not spine:
            # Fallback: find all XHTML files
            for filename in epub_zip.namelist():
                if filename.endswith(('.xhtml', '.html', '.htm')):
                    spine.append(filename)
        
        for i, href in enumerate(spine):
            try:
                # Construct full path
                if opf_dir:
                    full_path = str(Path(opf_dir) / href)
                else:
                    full_path = href
                
                # Read content
                if full_path in epub_zip.namelist():
                    content = epub_zip.read(full_path).decode('utf-8', errors='ignore')
                    
                    # Convert XHTML to text
                    text = self._xhtml_to_text(content)
                    
                    if text.strip():
                        chapters.append({
                            'index': i + 1,
                            'title': self._extract_chapter_title(content) or f"Chapter {i + 1}",
                            'text': text
                        })
            
            except Exception as e:
                logger.warning(f"Could not extract chapter {href}: {e}")
                continue
        
        return chapters
    
    def _xhtml_to_text(self, xhtml: str) -> str:
        """Convert XHTML to plain text"""
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(xhtml, 'html.parser')
            
            # Remove script and style elements
            for element in soup(['script', 'style', 'head']):
                element.decompose()
            
            # Get text
            text = soup.get_text(separator='\n')
            
            # Clean up whitespace
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            return '\n\n'.join(lines)
        
        except ImportError:
            # Fallback: basic tag removal
            import re
            text = re.sub(r'<[^>]+>', '', xhtml)
            text = re.sub(r'\s+', ' ', text)
            return text.strip()
    
    def _extract_chapter_title(self, xhtml: str) -> str:
        """Extract chapter title from XHTML"""
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(xhtml, 'html.parser')
            
            # Try to find title in h1, h2, or title tag
            for tag in ['h1', 'h2', 'title']:
                element = soup.find(tag)
                if element and element.get_text().strip():
                    return element.get_text().strip()
        
        except:
            pass
        
        return ""
    
    def _build_markdown(
        self,
        epub_stem: str,
        metadata: Dict,
        chapters: List[Dict]
    ) -> str:
        """Build markdown from EPUB content"""
        lines = []
        
        # Header
        title = metadata.get('title', epub_stem)
        author = metadata.get('author', 'Unknown')
        
        lines.append(f"# {title}")
        lines.append(f"**Author**: {author}")
        lines.append(f"**Format**: EPUB eBook")
        
        if metadata.get('publisher'):
            lines.append(f"**Publisher**: {metadata['publisher']}")
        
        if metadata.get('language'):
            lines.append(f"**Language**: {metadata['language']}")
        
        lines.append(f"**Total Chapters**: {len(chapters)}")
        lines.append("")
        lines.append("---\n")
        
        # Chapters
        for chapter in chapters:
            lines.append(f"## Chapter {chapter['index']}: {chapter['title']}\n")
            lines.append(chapter['text'])
            lines.append("")
            lines.append("---\n")
        
        return '\n'.join(lines)
