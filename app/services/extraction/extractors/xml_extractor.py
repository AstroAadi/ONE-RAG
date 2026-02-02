"""
XML (Extensible Markup Language) File Extractor
Processes XML files for RAG pipeline
"""

import os
from pathlib import Path
from typing import Tuple, Dict, List
import logging
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom

try:
    from ..base_extractor import BaseExtractor
except ImportError:
    from base_extractor import BaseExtractor

logger = logging.getLogger(__name__)


class XMLExtractor(BaseExtractor):
    """
    Extractor for XML files
    
    Features:
    - Parses XML structure
    - Extracts text content from elements
    - Preserves hierarchical structure
    - Handles attributes
    - Pretty-prints XML
    """
    
    def _get_supported_formats(self) -> List[str]:
        return ['xml']
    
    def validate_file(self, file_path: str) -> Tuple[bool, str]:
        """Validate XML file"""
        path = Path(file_path)
        
        if not path.exists():
            return False, "File does not exist"
        
        ext = path.suffix.lower().lstrip('.')
        if ext not in self.supported_formats:
            return False, "Not an XML file"
        
        # Try to parse XML
        try:
            ET.parse(file_path)
            return True, "Valid XML file"
        except ET.ParseError as e:
            return False, f"Invalid XML: {e}"
        except Exception as e:
            return False, f"Could not read XML: {e}"
    
    def extract(self, file_path: str) -> Tuple[str, bool, Dict]:
        """Extract XML file"""
        try:
            is_valid, msg = self.validate_file(file_path)
            if not is_valid:
                return "", False, {'error': msg}
            
            logger.info(f"Extracting XML: {file_path}")
            
            xml_stem = Path(file_path).stem
            
            # Parse XML
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            # Extract structured content
            text_content = self._extract_text_content(root)
            element_count = self._count_elements(root)
            
            if not text_content.strip():
                # If no text, provide structure
                text_content = self._xml_structure_to_text(root)
            
            # Build markdown
            markdown_content = self._build_markdown(
                xml_stem, root, text_content, element_count
            )
            
            metadata = self.get_metadata(file_path)
            metadata.update({
                'total_pages': 1,
                'root_element': root.tag,
                'element_count': element_count,
                'has_namespace': '{' in root.tag,
                'extraction_method': 'xml'
            })
            
            logger.info(f"✓ XML extraction complete: {element_count} elements")
            
            return markdown_content, True, metadata
            
        except Exception as e:
            logger.error(f"XML extraction error: {str(e)}")
            import traceback
            traceback.print_exc()
            return "", False, {'error': str(e)}
    
    def _extract_text_content(self, element: ET.Element, depth: int = 0) -> str:
        """Recursively extract text content from XML"""
        lines = []
        
        # Get element text
        if element.text and element.text.strip():
            indent = "  " * depth
            # Remove namespace from tag
            tag = element.tag.split('}')[-1] if '}' in element.tag else element.tag
            lines.append(f"{indent}[{tag}]: {element.text.strip()}")
        
        # Process children
        for child in element:
            child_text = self._extract_text_content(child, depth + 1)
            if child_text:
                lines.append(child_text)
            
            # Get tail text (text after closing tag)
            if child.tail and child.tail.strip():
                indent = "  " * depth
                lines.append(f"{indent}{child.tail.strip()}")
        
        return '\n'.join(lines)
    
    def _xml_structure_to_text(self, element: ET.Element, depth: int = 0) -> str:
        """Convert XML structure to readable text"""
        lines = []
        indent = "  " * depth
        
        # Remove namespace from tag
        tag = element.tag.split('}')[-1] if '}' in element.tag else element.tag
        
        # Element opening
        attr_str = ""
        if element.attrib:
            attrs = [f"{k}='{v}'" for k, v in element.attrib.items()]
            attr_str = " " + " ".join(attrs)
        
        lines.append(f"{indent}<{tag}{attr_str}>")
        
        # Element text
        if element.text and element.text.strip():
            lines.append(f"{indent}  {element.text.strip()}")
        
        # Children
        for child in element:
            lines.append(self._xml_structure_to_text(child, depth + 1))
        
        # Element closing
        lines.append(f"{indent}</{tag}>")
        
        return '\n'.join(lines)
    
    def _count_elements(self, element: ET.Element) -> int:
        """Count total elements in XML tree"""
        count = 1
        for child in element:
            count += self._count_elements(child)
        return count
    
    def _build_markdown(
        self,
        xml_stem: str,
        root: ET.Element,
        text_content: str,
        element_count: int
    ) -> str:
        """Build markdown from XML content"""
        lines = []
        
        # Header
        root_tag = root.tag.split('}')[-1] if '}' in root.tag else root.tag
        lines.append(f"# XML Document: {xml_stem}")
        lines.append(f"**Format**: XML")
        lines.append(f"**Root Element**: {root_tag}")
        lines.append(f"**Total Elements**: {element_count}")
        lines.append("")
        
        # Namespace info
        if '{' in root.tag:
            namespace = root.tag.split('}')[0].lstrip('{')
            lines.append(f"**Namespace**: {namespace}")
            lines.append("")
        
        lines.append("---\n")
        
        # Content
        lines.append("## XML Content\n")
        
        if text_content:
            lines.append("```")
            lines.append(text_content)
            lines.append("```")
        else:
            lines.append("*(No text content extracted)*")
        
        lines.append("")
        
        # Root attributes
        if root.attrib:
            lines.append("## Root Attributes\n")
            for key, value in root.attrib.items():
                lines.append(f"- **{key}**: {value}")
            lines.append("")
        
        return '\n'.join(lines)
