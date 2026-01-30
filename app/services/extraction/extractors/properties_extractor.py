"""
PROPERTIES File Extractor
Processes Java properties and configuration files for RAG pipeline
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


class PropertiesExtractor(BaseExtractor):
    """
    Extractor for .properties configuration files
    
    Features:
    - Parses key-value pairs
    - Handles comments
    - Supports multi-line values
    - Groups related properties
    
    Common in Java applications and configuration files
    """
    
    def _get_supported_formats(self) -> List[str]:
        return ['properties', 'props', 'config', 'conf', 'ini']
    
    def validate_file(self, file_path: str) -> Tuple[bool, str]:
        """Validate properties file"""
        path = Path(file_path)
        
        if not path.exists():
            return False, "File does not exist"
        
        ext = path.suffix.lower().lstrip('.')
        if ext not in self.supported_formats:
            return False, "Not a properties file"
        
        return True, "Valid properties file"
    
    def extract(self, file_path: str) -> Tuple[str, bool, Dict]:
        """Extract properties file"""
        try:
            is_valid, msg = self.validate_file(file_path)
            if not is_valid:
                return "", False, {'error': msg}
            
            logger.info(f"Extracting properties: {file_path}")
            
            # Read file
            content = self._read_with_encoding(file_path)
            
            if not content:
                return "", False, {'error': 'Empty properties file'}
            
            props_stem = Path(file_path).stem
            file_ext = Path(file_path).suffix.lower().lstrip('.')
            
            # Parse properties
            properties = self._parse_properties(content)
            
            if not properties:
                return "", False, {'error': 'No properties found'}
            
            # Build markdown
            markdown_content = self._build_markdown(
                props_stem, file_ext, properties
            )
            
            metadata = self.get_metadata(file_path)
            metadata.update({
                'total_pages': 1,
                'property_count': len(properties),
                'has_comments': any(p.get('comment') for p in properties),
                'extraction_method': 'properties'
            })
            
            logger.info(f"✓ Properties extraction complete: {len(properties)} properties")
            
            return markdown_content, True, metadata
            
        except Exception as e:
            logger.error(f"Properties extraction error: {str(e)}")
            import traceback
            traceback.print_exc()
            return "", False, {'error': str(e)}
    
    def _read_with_encoding(self, file_path: str) -> str:
        """Read properties file with encoding detection"""
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
    
    def _parse_properties(self, content: str) -> List[Dict]:
        """Parse properties file"""
        properties = []
        lines = content.split('\n')
        
        i = 0
        current_comment = []
        
        while i < len(lines):
            line = lines[i].strip()
            
            # Skip empty lines
            if not line:
                current_comment = []
                i += 1
                continue
            
            # Collect comments
            if line.startswith('#') or line.startswith('!'):
                current_comment.append(line.lstrip('#!').strip())
                i += 1
                continue
            
            # Parse property
            if '=' in line or ':' in line:
                # Find separator
                sep_idx_eq = line.find('=')
                sep_idx_colon = line.find(':')
                
                if sep_idx_eq == -1:
                    sep_idx = sep_idx_colon
                elif sep_idx_colon == -1:
                    sep_idx = sep_idx_eq
                else:
                    sep_idx = min(sep_idx_eq, sep_idx_colon)
                
                key = line[:sep_idx].strip()
                value = line[sep_idx + 1:].strip()
                
                # Handle multi-line values (ending with \)
                while value.endswith('\\') and i + 1 < len(lines):
                    value = value[:-1].strip()
                    i += 1
                    value += ' ' + lines[i].strip()
                
                # Unescape special characters
                value = self._unescape_value(value)
                
                properties.append({
                    'key': key,
                    'value': value,
                    'comment': '\n'.join(current_comment) if current_comment else None
                })
                
                current_comment = []
            
            i += 1
        
        return properties
    
    def _unescape_value(self, value: str) -> str:
        """Unescape special characters in property values"""
        replacements = {
            r'\n': '\n',
            r'\r': '\r',
            r'\t': '\t',
            r'\\': '\\',
            r'\:': ':',
            r'\=': '=',
            r'\ ': ' '
        }
        
        for escape, char in replacements.items():
            value = value.replace(escape, char)
        
        return value
    
    def _group_properties(self, properties: List[Dict]) -> Dict[str, List[Dict]]:
        """Group properties by prefix"""
        groups = {}
        
        for prop in properties:
            key = prop['key']
            
            # Find prefix (e.g., "database" from "database.url")
            if '.' in key:
                prefix = key.split('.')[0]
            elif '_' in key:
                prefix = key.split('_')[0]
            else:
                prefix = 'general'
            
            if prefix not in groups:
                groups[prefix] = []
            
            groups[prefix].append(prop)
        
        return groups
    
    def _build_markdown(
        self,
        props_stem: str,
        file_ext: str,
        properties: List[Dict]
    ) -> str:
        """Build markdown from properties"""
        lines = []
        
        # Header
        lines.append(f"# Configuration: {props_stem}")
        lines.append(f"**Format**: {file_ext.upper()} Properties")
        lines.append(f"**Total Properties**: {len(properties)}")
        lines.append("")
        lines.append("---\n")
        
        # Group properties
        groups = self._group_properties(properties)
        
        # Output by groups
        for group_name, group_props in sorted(groups.items()):
            lines.append(f"## {group_name.title()} Configuration\n")
            
            for prop in group_props:
                # Add comment if present
                if prop.get('comment'):
                    lines.append(f"```")
                    lines.append(f"# {prop['comment']}")
                    lines.append(f"```")
                
                # Add property
                lines.append(f"**`{prop['key']}`** = `{prop['value']}`\n")
            
            lines.append("")
        
        return '\n'.join(lines)
