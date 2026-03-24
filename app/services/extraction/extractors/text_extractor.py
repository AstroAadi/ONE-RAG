"""
Plain Text File Extractor
Processes .txt, .text, and .log files for the RAG pipeline.

CSV and TSV files are handled by csv_extractor.py.
"""

from pathlib import Path
from typing import Tuple, Dict, List
import logging

try:
    from ..base_extractor import BaseExtractor
except ImportError:
    from base_extractor import BaseExtractor

logger = logging.getLogger(__name__)


class PlainTextExtractor(BaseExtractor):
    """
    Extractor for plain text files (.txt, .text, .log).

    Features:
    - Handles various encodings (UTF-8, Latin-1, etc.)
    - Estimates pages based on content
    - Adds page markers
    - Preserves line structure
    """

    def _get_supported_formats(self) -> List[str]:
        return ['txt', 'text', 'log']

    def validate_file(self, file_path: str) -> Tuple[bool, str]:
        """Validate text file"""
        path = Path(file_path)

        if not path.exists():
            return False, "File does not exist"

        ext = path.suffix.lower().lstrip('.')
        if ext not in self.supported_formats:
            return False, f"Not a supported text file. Supported: {', '.join(self.supported_formats)}"

        return True, "Valid text file"

    def extract(self, file_path: str) -> Tuple[str, bool, Dict]:
        """Extract plain text file"""
        try:
            is_valid, msg = self.validate_file(file_path)
            if not is_valid:
                return "", False, {'error': msg}

            logger.info(f"Extracting text: {file_path}")

            # Try different encodings
            content = self._read_with_encoding(file_path)

            if not content or not content.strip():
                return "", False, {'error': 'Empty text file'}

            txt_stem = Path(file_path).stem
            file_ext = Path(file_path).suffix.lower().lstrip('.')

            # Estimate pages
            word_count = len(content.split())
            estimated_pages = max(1, (word_count // 500) + 1)

            # Build markdown
            markdown_content = self._build_markdown_with_pages(
                content, txt_stem, file_ext, estimated_pages
            )

            metadata = self.get_metadata(file_path)
            metadata.update({
                'total_pages': estimated_pages,
                'word_count': word_count,
                'char_count': len(content),
                'line_count': content.count('\n'),
                'extraction_method': 'plain_text'
            })

            logger.info(f"✓ Text extraction complete: {estimated_pages} pages, {word_count} words")

            return markdown_content, True, metadata

        except Exception as e:
            logger.error(f"Text extraction error: {str(e)}")
            return "", False, {'error': str(e)}

    def _read_with_encoding(self, file_path: str) -> str:
        """Try reading file with different encodings"""
        encodings = ['utf-8', 'utf-16', 'latin-1', 'cp1252', 'iso-8859-1']

        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read()
                logger.info(f"Successfully read with {encoding} encoding")
                return content
            except (UnicodeDecodeError, UnicodeError):
                continue
            except Exception as e:
                logger.warning(f"Error reading with {encoding}: {e}")
                continue

        # Last resort: binary read and decode with errors='ignore'
        try:
            with open(file_path, 'rb') as f:
                content = f.read().decode('utf-8', errors='ignore')
            logger.warning("Read with error handling (some characters may be lost)")
            return content
        except Exception as e:
            raise Exception(f"Could not read file with any encoding: {e}")

    def _build_markdown_with_pages(
        self,
        content: str,
        txt_stem: str,
        file_ext: str,
        total_pages: int
    ) -> str:
        """Build markdown with page markers"""
        lines = content.split('\n')
        result_lines = []

        # Add header
        result_lines.append(f"# Document: {txt_stem}")
        result_lines.append(f"**Format**: {file_ext.upper()}")
        result_lines.append(f"**Total Pages**: {total_pages}\n")
        result_lines.append("---\n")

        # Add page markers
        total_lines = len(lines)
        lines_per_page = max(1, total_lines // total_pages)

        current_page = 1
        result_lines.append(f"\n## Page {current_page}\n")

        for i, line in enumerate(lines):
            result_lines.append(line)

            # Add page markers at appropriate intervals
            if i > 0 and (i % lines_per_page == 0) and current_page < total_pages:
                current_page += 1
                result_lines.append(f"\n## Page {current_page}\n")

        return '\n'.join(result_lines)
