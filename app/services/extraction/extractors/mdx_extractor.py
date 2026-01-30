"""
MDX (Markdown + JSX) File Extractor
Processes MDX files (Markdown with embedded React components) for RAG pipeline
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


class MDXExtractor(BaseExtractor):
    """
    Extractor for MDX files (Markdown with JSX/React components)
    
    Features:
    - Extracts markdown content
    - Handles JSX components
    - Preserves code blocks
    - Extracts frontmatter metadata
    
    MDX is commonly used in documentation sites and blogs
    """
    
    def _get_supported_formats(self) -> List[str]:
        return ['mdx']
    
    def validate_file(self, file_path: str) -> Tuple[bool, str]:
        """Validate MDX file"""
        path = Path(file_path)
        
        if not path.exists():
            return False, "File does not exist"
        
        ext = path.suffix.lower().lstrip('.')
        if ext not in self.supported_formats:
            return False, "Not an MDX file"
        
        return True, "Valid MDX file"
    
    def extract(self, file_path: str) -> Tuple[str, bool, Dict]:
        """Extract MDX file"""
        try:
            is_valid, msg = self.validate_file(file_path)
            if not is_valid:
                return "", False, {'error': msg}
            
            logger.info(f"Extracting MDX: {file_path}")
            
            # Read file
            content = self._read_with_encoding(file_path)
            
            if not content:
                return "", False, {'error': 'Empty MDX file'}
            
            mdx_stem = Path(file_path).stem
            
            # Parse MDX components
            parsed = self._parse_mdx(content)
            
            # Build markdown
            markdown_content = self._build_markdown(
                mdx_stem, parsed
            )
            
            metadata = self.get_metadata(file_path)
            metadata.update({
                'total_pages': 1,
                'has_frontmatter': bool(parsed.get('frontmatter')),
                'jsx_component_count': len(parsed.get('jsx_components', [])),
                'code_block_count': len(parsed.get('code_blocks', [])),
                'extraction_method': 'mdx'
            })
            
            # Add frontmatter to metadata if present
            if parsed.get('frontmatter'):
                metadata['frontmatter'] = parsed['frontmatter']
            
            logger.info(f"✓ MDX extraction complete")
            
            return markdown_content, True, metadata
            
        except Exception as e:
            logger.error(f"MDX extraction error: {str(e)}")
            import traceback
            traceback.print_exc()
            return "", False, {'error': str(e)}
    
    def _read_with_encoding(self, file_path: str) -> str:
        """Read MDX file with encoding detection"""
        encodings = ['utf-8', 'utf-8-sig', 'latin-1']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
        
        # Fallback
        with open(file_path, 'rb') as f:
            return f.read().decode('utf-8', errors='ignore')
    
    def _parse_mdx(self, content: str) -> Dict:
        """Parse MDX content"""
        parsed = {
            'frontmatter': {},
            'jsx_components': [],
            'imports': [],
            'exports': [],
            'code_blocks': [],
            'markdown_content': ''
        }
        
        # Extract frontmatter (YAML between --- markers)
        frontmatter_match = re.match(r'^---\s*\n(.*?)\n---\s*\n', content, re.DOTALL)
        if frontmatter_match:
            frontmatter_text = frontmatter_match.group(1)
            parsed['frontmatter'] = self._parse_frontmatter(frontmatter_text)
            content = content[frontmatter_match.end():]
        
        # Extract imports
        import_pattern = r'^import\s+.*?from\s+["\'].*?["\'];?\s*$'
        imports = re.findall(import_pattern, content, re.MULTILINE)
        parsed['imports'] = imports
        
        # Extract exports
        export_pattern = r'^export\s+(?:default|const)\s+.*?$'
        exports = re.findall(export_pattern, content, re.MULTILINE)
        parsed['exports'] = exports
        
        # Find JSX components (simplified detection)
        jsx_pattern = r'<([A-Z][a-zA-Z0-9]*)[^>]*(?:/>|>.*?</\1>)'
        jsx_components = re.findall(jsx_pattern, content, re.DOTALL)
        parsed['jsx_components'] = list(set(jsx_components))  # Unique components
        
        # Extract code blocks
        code_block_pattern = r'```(\w+)?\n(.*?)```'
        code_blocks = re.findall(code_block_pattern, content, re.DOTALL)
        parsed['code_blocks'] = [{'language': lang, 'code': code.strip()} for lang, code in code_blocks]
        
        # Convert JSX to markdown placeholders for cleaner output
        content_cleaned = re.sub(
            r'<([A-Z][a-zA-Z0-9]*)[^>]*(?:/>|>.*?</\1>)',
            lambda m: f'\n\n[React Component: {m.group(1)}]\n\n',
            content,
            flags=re.DOTALL
        )
        
        # Remove import/export statements from main content
        content_cleaned = re.sub(import_pattern, '', content_cleaned, flags=re.MULTILINE)
        content_cleaned = re.sub(export_pattern, '', content_cleaned, flags=re.MULTILINE)
        
        parsed['markdown_content'] = content_cleaned.strip()
        
        return parsed
    
    def _parse_frontmatter(self, frontmatter_text: str) -> Dict:
        """Parse YAML frontmatter"""
        frontmatter = {}
        
        try:
            # Try using PyYAML if available
            import yaml
            frontmatter = yaml.safe_load(frontmatter_text) or {}
        except ImportError:
            # Fallback: basic key-value parsing
            for line in frontmatter_text.split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    frontmatter[key.strip()] = value.strip().strip('"\'')
        except Exception as e:
            logger.warning(f"Could not parse frontmatter: {e}")
        
        return frontmatter
    
    def _build_markdown(self, mdx_stem: str, parsed: Dict) -> str:
        """Build markdown from MDX content"""
        lines = []
        
        # Use frontmatter title if available
        title = parsed['frontmatter'].get('title', mdx_stem)
        
        # Header
        lines.append(f"# {title}")
        lines.append(f"**Format**: MDX (Markdown + JSX)")
        lines.append("")
        
        # Frontmatter metadata
        if parsed['frontmatter']:
            lines.append("## Metadata\n")
            for key, value in parsed['frontmatter'].items():
                if key != 'title':  # Already shown in header
                    lines.append(f"**{key.title()}**: {value}")
            lines.append("")
        
        # React components used
        if parsed['jsx_components']:
            lines.append("## React Components Used\n")
            for component in parsed['jsx_components']:
                lines.append(f"- `<{component}>`")
            lines.append("")
        
        lines.append("---\n")
        
        # Main content
        lines.append("## Content\n")
        lines.append(parsed['markdown_content'])
        
        return '\n'.join(lines)
