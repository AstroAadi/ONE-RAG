"""
Enhanced DOCX/DOC Extractor with Page Number Tracking
Matches PDF extractor behavior with page markers and image page tracking
"""

import os
from pathlib import Path
from typing import Tuple, Dict, List
import logging
import re
from docx import Document

try:
    from ..base_extractor import BaseExtractor
    from ..processors.caption_detector import CaptionDetector
    from ..processors.image_processor import ImageProcessor
except ImportError:
    # Standalone mode
    from base_extractor import BaseExtractor
    from caption_detector import CaptionDetector
    from image_processor import ImageProcessor

logger = logging.getLogger(__name__)


class DocxExtractor(BaseExtractor):
    """
    Enhanced extractor for Word documents with page tracking
    
    Features:
    - Extracts images with page numbers
    - Outputs page markers (## Page X)
    - Tracks PAGE: tags for each image
    - Caption detection and association
    """
    
    def _get_supported_formats(self) -> List[str]:
        return ['docx', 'doc']
    
    def validate_file(self, file_path: str) -> Tuple[bool, str]:
        """Validate DOCX file"""
        path = Path(file_path)
        
        if not path.exists():
            return False, "File does not exist"
        
        if path.suffix.lower() not in ['.docx', '.doc']:
            return False, "Not a Word document"
        
        if path.stat().st_size == 0:
            return False, "File is empty"
        
        return True, "Valid Word document"
    
    def extract(self, file_path: str) -> Tuple[str, bool, Dict]:
        """Extract DOCX with page tracking"""
        try:
            is_valid, msg = self.validate_file(file_path)
            if not is_valid:
                return "", False, {'error': msg}
            
            logger.info(f"Extracting DOCX: {file_path}")
            
            # Try Docling first
            docling_result = self._try_docling_extraction(file_path)
            if docling_result[1]:  # success
                logger.info("✓ Docling extraction successful")
                return docling_result
            
            # Fallback to python-docx
            logger.info("Using python-docx with page tracking")
            return self._extract_with_page_tracking(file_path)
            
        except Exception as e:
            logger.error(f"DOCX extraction error: {str(e)}")
            import traceback
            traceback.print_exc()
            return "", False, {'error': str(e)}
    
    def _try_docling_extraction(self, file_path: str) -> Tuple[str, bool, Dict]:
        """Try Docling extraction with page tracking"""
        try:
            from docling.document_converter import DocumentConverter
            
            doc_stem = Path(file_path).stem
            image_output_dir = os.path.join(
                os.path.dirname(file_path),
                "extracted_images",
                doc_stem
            )
            os.makedirs(image_output_dir, exist_ok=True)
            
            # Convert with Docling
            converter = DocumentConverter()
            result = converter.convert(file_path)
            
            # Get total pages
            total_pages = len(result.document.pages) if hasattr(result.document, 'pages') else 1
            
            # Export with embedded images
            markdown_with_base64 = result.document.export_to_markdown(image_mode="embedded")
            
            # Extract and save images with page hints
            image_mapping = self._extract_images_with_docling(
                markdown_with_base64,
                image_output_dir,
                result.document
            )
            
            # Export with placeholders
            markdown_content = result.document.export_to_markdown(
                image_mode="placeholder",
                image_placeholder="[IMAGE:{image_name}]"
            )
            
            # Build final markdown with pages
            final_markdown = self._build_markdown_with_pages(
                markdown_content,
                image_mapping,
                doc_stem,
                total_pages
            )
            
            metadata = self.get_metadata(file_path)
            metadata.update({
                'total_pages': total_pages,
                'images_extracted': len(image_mapping),
                'image_directory': str(image_output_dir),
                'extraction_method': 'docling'
            })
            
            logger.info(f"✓ Docling: {total_pages} pages, {len(image_mapping)} images")
            
            return final_markdown, True, metadata
            
        except ImportError:
            logger.debug("Docling not available")
            return "", False, {'error': 'Docling not installed'}
        except Exception as e:
            logger.warning(f"Docling extraction failed: {str(e)}")
            return "", False, {'error': str(e)}
    
    def _extract_with_page_tracking(self, file_path: str) -> Tuple[str, bool, Dict]:
        """
        Extract with python-docx including page tracking
        
        Note: DOCX doesn't have native page concept, so we estimate pages
        based on content length and page breaks
        """
        try:
            doc = Document(file_path)
            doc_stem = Path(file_path).stem
            
            # Create image directory
            image_output_dir = os.path.join(
                os.path.dirname(file_path),
                "extracted_images",
                doc_stem
            )
            os.makedirs(image_output_dir, exist_ok=True)
            
            logger.info(f"Image directory: {image_output_dir}")
            
            # Analyze document structure
            page_breaks = self._find_page_breaks(doc)
            total_pages = self._estimate_total_pages(doc, page_breaks)
            
            logger.info(f"Estimated pages: {total_pages}")
            
            # Extract images with page tracking
            image_mapping = self._extract_images_with_pages(
                doc, image_output_dir, doc_stem, page_breaks, total_pages
            )
            
            logger.info(f"Extracted {len(image_mapping)} images")
            
            # Build markdown with page tracking
            markdown_content = self._build_content_with_pages(
                doc, image_mapping, doc_stem, total_pages, page_breaks
            )
            
            # Apply caption detection
            final_markdown = self._apply_caption_detection(
                markdown_content, image_mapping, doc_stem
            )
            
            metadata = self.get_metadata(file_path)
            metadata.update({
                'total_pages': total_pages,
                'images_extracted': len(image_mapping),
                'image_directory': str(image_output_dir),
                'tables_found': len(doc.tables),
                'extraction_method': 'python-docx'
            })
            
            logger.info("✓ Extraction complete with page tracking")
            
            return final_markdown, True, metadata
            
        except Exception as e:
            logger.error(f"Extraction failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return "", False, {'error': str(e)}
    
    def _find_page_breaks(self, doc: Document) -> List[int]:
        """
        Find page break positions in document
        Returns list of paragraph indices where page breaks occur
        """
        page_breaks = [0]  # Start of document is page 1
        
        for i, para in enumerate(doc.paragraphs):
            # Check for explicit page break
            if self._has_page_break(para):
                page_breaks.append(i)
        
        return page_breaks
    
    def _has_page_break(self, paragraph) -> bool:
        """Check if paragraph contains a page break"""
        try:
            # Check for page break in paragraph XML
            xml = paragraph._element.xml
            if 'w:br' in xml and 'w:type="page"' in xml:
                return True
            
            # Check runs for page breaks
            for run in paragraph.runs:
                if 'lastRenderedPageBreak' in run._element.xml:
                    return True
                if '<w:br w:type="page"' in run._element.xml:
                    return True
        except:
            pass
        
        return False
    
    def _estimate_total_pages(self, doc: Document, page_breaks: List[int]) -> int:
        """
        Estimate total pages based on content and page breaks
        """
        # If we found page breaks, use that
        if len(page_breaks) > 1:
            estimated = len(page_breaks)
        else:
            # Estimate based on content
            # Rough estimate: 500 words per page
            total_words = sum(len(para.text.split()) for para in doc.paragraphs)
            estimated = max(1, (total_words // 500) + 1)
        
        logger.debug(f"Page estimation: {estimated} pages")
        return estimated
    
    def _estimate_paragraph_page(
        self, 
        para_index: int, 
        total_paras: int, 
        page_breaks: List[int],
        total_pages: int
    ) -> int:
        """
        Estimate which page a paragraph is on
        """
        if len(page_breaks) > 1:
            # We have explicit page breaks
            for i, break_pos in enumerate(page_breaks):
                if para_index <= break_pos:
                    return max(1, i)
                if i < len(page_breaks) - 1 and para_index < page_breaks[i + 1]:
                    return i + 1
            return len(page_breaks)
        else:
            # Linear distribution
            if total_paras == 0:
                return 1
            page = int((para_index / total_paras) * total_pages) + 1
            return max(1, min(page, total_pages))
    
    def _extract_images_with_pages(
        self,
        doc: Document,
        output_dir: str,
        doc_stem: str,
        page_breaks: List[int],
        total_pages: int
    ) -> Dict[int, Dict]:
        """
        Extract images and track their page numbers
        """
        image_mapping = {}
        image_counter = 0
        total_paras = len(doc.paragraphs)
        
        try:
            # Get all image relationships
            image_rels = {}
            for rel in doc.part.rels.values():
                if "image" in rel.target_ref:
                    image_rels[rel.target_part] = rel
            
            logger.info(f"Found {len(image_rels)} image relationships")
            
            # Track which images we've seen
            seen_images = set()
            
            # Process paragraphs and find images
            for para_idx, para in enumerate(doc.paragraphs):
                if self._paragraph_has_image(para):
                    # Get image from this paragraph
                    for rel in doc.part.rels.values():
                        if "image" not in rel.target_ref:
                            continue
                        
                        image_part = rel.target_part
                        
                        # Skip if already processed
                        if id(image_part) in seen_images:
                            continue
                        
                        # Check if this image is in this paragraph
                        if self._image_in_paragraph(para, rel):
                            image_counter += 1
                            seen_images.add(id(image_part))
                            
                            # Estimate page number
                            page_number = self._estimate_paragraph_page(
                                para_idx, total_paras, page_breaks, total_pages
                            )
                            
                            # Save image
                            try:
                                image_data = image_part.blob
                                content_type = image_part.content_type
                                ext = self._get_extension_from_content_type(content_type)
                                
                                filename = f"page_{page_number:02d}_image_{image_counter:03d}.{ext}"
                                output_path = os.path.join(output_dir, filename)
                                
                                with open(output_path, 'wb') as f:
                                    f.write(image_data)
                                
                                image_mapping[image_counter] = {
                                    'filename': filename,
                                    'page_number': page_number,
                                    'format': ext,
                                    'content_type': content_type,
                                    'para_index': para_idx
                                }
                                
                                logger.info(f"  ✓ Saved: {filename} (page {page_number})")
                                
                            except Exception as e:
                                logger.warning(f"  ⚠ Could not save image {image_counter}: {e}")
                            
                            break  # Move to next paragraph
            
        except Exception as e:
            logger.error(f"Error extracting images: {str(e)}")
        
        return image_mapping
    
    def _paragraph_has_image(self, paragraph) -> bool:
        """Check if paragraph contains an image"""
        xml = paragraph._element.xml
        return 'pic:pic' in xml or 'Graphic' in xml or 'w:drawing' in xml
    
    def _image_in_paragraph(self, paragraph, rel) -> bool:
        """Check if specific image relationship is in this paragraph"""
        # Simple check: if paragraph has image and we haven't seen this rel
        return self._paragraph_has_image(paragraph)
    
    def _build_content_with_pages(
        self,
        doc: Document,
        image_mapping: Dict[int, Dict],
        doc_stem: str,
        total_pages: int,
        page_breaks: List[int]
    ) -> str:
        """Build markdown content with page markers"""
        markdown_parts = []
        image_counter = 0
        total_paras = len(doc.paragraphs)
        current_page = 1
        
        # Add header
        markdown_parts.append(f"# Document: {doc_stem}\n")
        markdown_parts.append(f"**Total Pages**: {total_pages}\n")
        markdown_parts.append("---\n")
        
        # Add first page marker
        markdown_parts.append(f"\n## Page 1\n")
        
        # Process paragraphs
        for para_idx, para in enumerate(doc.paragraphs):
            # Check if we've crossed a page boundary
            para_page = self._estimate_paragraph_page(
                para_idx, total_paras, page_breaks, total_pages
            )
            
            if para_page > current_page:
                current_page = para_page
                markdown_parts.append(f"\n## Page {current_page}\n")
            
            # Check if paragraph contains an image
            if self._paragraph_has_image(para):
                image_counter += 1
                
                if image_counter in image_mapping:
                    image_info = image_mapping[image_counter]
                    filename = image_info['filename']
                    page_number = image_info['page_number']
                    rel_path = os.path.join("extracted_images", doc_stem, filename)
                    
                    # Add image placeholder
                    markdown_parts.append(f"\n[IMAGE_PLACEHOLDER_{image_counter}]")
                    markdown_parts.append(f"IMAGE_PATH: {rel_path}")
                    markdown_parts.append(f"PAGE: {page_number}")
                    markdown_parts.append("")
            
            # Process text
            if para.text.strip():
                text = para.text.strip()
                style = para.style.name.lower() if para.style else ""
                
                # Check if it's a heading
                if 'heading' in style:
                    level = 2
                    try:
                        level = int(''.join(filter(str.isdigit, style))) + 1
                        level = min(level, 6)
                    except:
                        pass
                    markdown_parts.append(f"\n{'#' * level} {text}\n")
                else:
                    markdown_parts.append(text)
        
        # Process tables
        for table_idx, table in enumerate(doc.tables, start=1):
            markdown_parts.append(f"\n**[TABLE {table_idx}]**\n")
            
            rows = [[cell.text.strip() for cell in row.cells] for row in table.rows]
            
            if rows:
                max_cols = max(len(r) for r in rows)
                rows = [r + [''] * (max_cols - len(r)) for r in rows]
                
                md_table = []
                md_table.append("| " + " | ".join(rows[0]) + " |")
                md_table.append("| " + " | ".join(["---"] * max_cols) + " |")
                for row in rows[1:]:
                    md_table.append("| " + " | ".join(row) + " |")
                
                markdown_parts.append("\n".join(md_table))
                markdown_parts.append("")
        
        return "\n\n".join(markdown_parts)
    
    def _get_extension_from_content_type(self, content_type: str) -> str:
        """Get file extension from MIME type"""
        mime_to_ext = {
            'image/png': 'png',
            'image/jpeg': 'jpg',
            'image/jpg': 'jpg',
            'image/gif': 'gif',
            'image/bmp': 'bmp',
            'image/tiff': 'tiff',
            'image/webp': 'webp',
            'image/svg+xml': 'svg',
        }
        return mime_to_ext.get(content_type, 'png')
    
    def _extract_images_with_docling(
        self,
        markdown_with_base64: str,
        image_output_dir: str,
        document
    ) -> Dict[int, Dict]:
        """Extract images from Docling output with page tracking"""
        pattern = r'!\[([^\]]*)\]\(data:image/([^;]+);base64,([^)]+)\)'
        matches = re.findall(pattern, markdown_with_base64)
        image_mapping = {}
        
        # Get page hints from Docling
        page_hints = {}
        try:
            if hasattr(document, 'pictures') and document.pictures:
                for idx, picture in enumerate(document.pictures, start=1):
                    if hasattr(picture, 'prov'):
                        for prov in picture.prov:
                            if hasattr(prov, 'page_no'):
                                page_hints[idx] = prov.page_no + 1
                                break
        except Exception as e:
            logger.debug(f"Could not extract page hints: {e}")
        
        total_pages = len(document.pages) if hasattr(document, 'pages') else 1
        
        for idx, match in enumerate(matches, start=1):
            try:
                image_format = match[1]
                base64_data = match[2]
                image_data = base64.b64decode(base64_data)
                
                ext = image_format.lower()
                if ext == 'jpeg':
                    ext = 'jpg'
                
                # Get page number
                page_number = page_hints.get(idx)
                if page_number is None:
                    page_number = int((idx / len(matches)) * total_pages) + 1 if len(matches) > 0 else 1
                    page_number = max(1, min(page_number, total_pages))
                
                filename = f"page_{page_number:02d}_image_{idx:03d}.{ext}"
                output_path = os.path.join(image_output_dir, filename)
                
                with open(output_path, 'wb') as f:
                    f.write(image_data)
                
                image_mapping[idx] = {
                    'filename': filename,
                    'page_number': page_number,
                    'format': ext
                }
                
                logger.info(f"  ✓ Saved: {filename} (page {page_number})")
                
            except Exception as e:
                logger.warning(f"  ⚠ Could not save image {idx}: {e}")
        
        return image_mapping
    
    def _build_markdown_with_pages(
        self,
        markdown_content: str,
        image_mapping: Dict[int, Dict],
        doc_stem: str,
        total_pages: int
    ) -> str:
        """Build markdown with page markers (like PDF extractor)"""
        lines = markdown_content.split('\n')
        result_lines = []
        
        # Add header
        result_lines.append(f"# Document: {doc_stem}")
        result_lines.append(f"**Total Pages**: {total_pages}\n")
        
        placeholder_pattern = r'\[IMAGE:\{[^\}]+\}\]'
        image_counter = [1]
        processed_caption_indices = set()
        image_caption_map = {}
        
        # First pass: find captions
        for i, line in enumerate(lines):
            if re.search(placeholder_pattern, line):
                idx = image_counter[0]
                
                caption_info = CaptionDetector.find_caption_in_range(
                    lines, i + 1, lookahead=5
                )
                
                if caption_info:
                    image_caption_map[idx] = caption_info
                    processed_caption_indices.add(caption_info['line_idx'])
                
                if idx not in image_caption_map:
                    caption_before = CaptionDetector.find_caption_before(
                        lines, i, lookback=5
                    )
                    if caption_before:
                        image_caption_map[idx] = caption_before
                        processed_caption_indices.add(caption_before['line_idx'])
                
                image_counter[0] += 1
        
        # Second pass: build output
        image_counter[0] = 1
        
        for i, line in enumerate(lines):
            if i in processed_caption_indices:
                continue
            
            if line.startswith('#'):
                result_lines.append(line)
                continue
            
            if re.search(placeholder_pattern, line):
                idx = image_counter[0]
                
                if idx in image_mapping:
                    image_info = image_mapping[idx]
                    filename = image_info['filename']
                    page_number = image_info['page_number']
                    rel_path = os.path.join("extracted_images", doc_stem, filename)
                    
                    # Add page marker
                    if not result_lines or not result_lines[-1].startswith(f"## Page {page_number}"):
                        result_lines.append(f"\n## Page {page_number}\n")
                    
                    # Add caption if found
                    if idx in image_caption_map:
                        caption = image_caption_map[idx]
                        heading = CaptionDetector.format_caption_as_heading(
                            caption['type'], caption['number'], caption['description']
                        )
                        result_lines.append(f"{heading}\n")
                    
                    # Add image path and page tag
                    result_lines.append(f"IMAGE_PATH: {rel_path}")
                    result_lines.append(f"PAGE: {page_number}")
                    result_lines.append("")
                    
                    image_counter[0] += 1
                else:
                    result_lines.append(line)
                    image_counter[0] += 1
            else:
                result_lines.append(line)
        
        return '\n'.join(result_lines)
    
    def _apply_caption_detection(
        self,
        markdown_content: str,
        image_mapping: Dict[int, Dict],
        doc_stem: str
    ) -> str:
        """Apply caption detection (for python-docx path)"""
        lines = markdown_content.split('\n')
        result_lines = []
        
        placeholder_pattern = r'^(IMAGE_PATH:\s*(.+)|.*\[IMAGE_PLACEHOLDER_\d+\].*)$'
        image_counter = [0]
        processed_caption_indices = set()
        image_caption_map = {}
        
        # First pass: find captions
        for i, line in enumerate(lines):
            if re.match(placeholder_pattern, line):
                image_counter[0] += 1
                idx = image_counter[0]
                
                caption_info = CaptionDetector.find_caption_in_range(
                    lines, i + 1, lookahead=5
                )
                
                if caption_info:
                    image_caption_map[idx] = caption_info
                    processed_caption_indices.add(caption_info['line_idx'])
                
                if idx not in image_caption_map:
                    caption_before = CaptionDetector.find_caption_before(
                        lines, i, lookback=5
                    )
                    if caption_before:
                        image_caption_map[idx] = caption_before
                        processed_caption_indices.add(caption_before['line_idx'])
        
        # Second pass: build output
        image_counter[0] = 0
        
        for i, line in enumerate(lines):
            if i in processed_caption_indices:
                continue
            
            if line.startswith('#'):
                result_lines.append(line)
                continue
            
            if re.match(placeholder_pattern, line):
                image_counter[0] += 1
                idx = image_counter[0]
                
                # Add caption as heading if found
                if idx in image_caption_map:
                    caption = image_caption_map[idx]
                    heading = CaptionDetector.format_caption_as_heading(
                        caption['type'], caption['number'], caption['description']
                    )
                    result_lines.append(f"\n{heading}\n")
                
                # Add image lines
                if 'IMAGE_PATH:' in line:
                    result_lines.append(line)
                elif '[IMAGE_PLACEHOLDER_' in line:
                    pass  # Skip placeholder line
                
                result_lines.append("")
            else:
                result_lines.append(line)
        
        return '\n'.join(result_lines)