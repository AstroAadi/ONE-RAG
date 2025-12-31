"""
Enhanced DOCX/DOC Extractor with AI Image Descriptions
Images are described using vision LLM at ingestion time instead of storing paths
"""

import os
from pathlib import Path
from typing import Tuple, Dict, List
import logging
import re
import base64
from docx import Document

try:
    from ..base_extractor import BaseExtractor
    from ..processors.caption_detector import CaptionDetector
    from ..processors.image_processor import ImageProcessor
except ImportError:
    from base_extractor import BaseExtractor
    from caption_detector import CaptionDetector
    from image_processor import ImageProcessor

logger = logging.getLogger(__name__)


class DocxExtractor(BaseExtractor):
    """
    Enhanced extractor for Word documents with AI image descriptions
    
    Key Change: Images are converted to AI descriptions at ingestion time
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
        """Extract DOCX with AI image descriptions"""
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
            logger.info("Using python-docx with AI descriptions")
            return self._extract_with_descriptions(file_path)
            
        except Exception as e:
            logger.error(f"DOCX extraction error: {str(e)}")
            import traceback
            traceback.print_exc()
            return "", False, {'error': str(e)}
    
    def _try_docling_extraction(self, file_path: str) -> Tuple[str, bool, Dict]:
        """Try Docling extraction with AI image descriptions"""
        try:
            from docling.document_converter import DocumentConverter
            from app.services.image_description_service import get_image_description_service
            
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
            
            total_pages = len(result.document.pages) if hasattr(result.document, 'pages') else 1
            
            # Export with embedded images
            markdown_with_base64 = result.document.export_to_markdown(image_mode="embedded")
            
            # NEW: Extract and describe images
            image_data = self._extract_and_describe_images_docling(
                markdown_with_base64,
                image_output_dir,
                result.document,
                doc_stem
            )
            
            # Export with placeholders
            markdown_content = result.document.export_to_markdown(
                image_mode="placeholder",
                image_placeholder="[IMAGE:{image_name}]"
            )
            
            # NEW: Build markdown with descriptions
            final_markdown = self._build_markdown_with_descriptions(
                markdown_content,
                image_data,
                doc_stem,
                total_pages
            )
            
            metadata = self.get_metadata(file_path)
            metadata.update({
                'total_pages': total_pages,
                'images_extracted': len(image_data),
                'images_described': sum(1 for img in image_data.values() if img.get('description_success')),
                'image_directory': str(image_output_dir),
                'extraction_method': 'docling'
            })
            
            logger.info(f"✓ Docling: {total_pages} pages, {len(image_data)} images described")
            
            return final_markdown, True, metadata
            
        except ImportError:
            logger.debug("Docling not available")
            return "", False, {'error': 'Docling not installed'}
        except Exception as e:
            logger.warning(f"Docling extraction failed: {str(e)}")
            return "", False, {'error': str(e)}
    
    def _extract_and_describe_images_docling(
        self,
        markdown_with_base64: str,
        image_output_dir: str,
        document,
        doc_stem: str
    ) -> Dict[int, Dict]:
        """Extract images from Docling and generate AI descriptions"""
        from app.services.image_description_service import get_image_description_service
        
        pattern = r'!\[([^\]]*)\]\(data:image/([^;]+);base64,([^)]+)\)'
        matches = re.findall(pattern, markdown_with_base64)
        image_data = {}
        
        # Get page hints
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
        description_service = get_image_description_service()
        doc_context = f"Word document: {doc_stem}"
        
        for idx, match in enumerate(matches, start=1):
            try:
                image_format = match[1]
                base64_image = match[2]
                image_bytes = base64.b64decode(base64_image)
                
                ext = image_format.lower()
                if ext == 'jpeg':
                    ext = 'jpg'
                
                page_number = page_hints.get(idx)
                if page_number is None:
                    page_number = int((idx / len(matches)) * total_pages) + 1 if len(matches) > 0 else 1
                    page_number = max(1, min(page_number, total_pages))
                
                filename = f"page_{page_number:02d}_image_{idx:03d}.{ext}"
                output_path = os.path.join(image_output_dir, filename)
                
                # Save image
                with open(output_path, 'wb') as f:
                    f.write(image_bytes)
                
                logger.info(f"  ✓ Saved: {filename}")
                
                # Generate AI description
                if description_service.is_available():
                    media_type = f"image/{image_format}"
                    success, description = description_service.describe_image_from_base64(
                        base64_image,
                        media_type,
                        context=doc_context,
                        page_number=page_number,
                        filename=filename
                    )
                    
                    if success:
                        logger.info(f"  ✓ Described: {filename}")
                    else:
                        logger.warning(f"  ⚠ Description failed: {filename}")
                else:
                    success = False
                    description = description_service.generate_fallback_description(
                        filename, page_number=page_number, image_type="figure"
                    )
                
                image_data[idx] = {
                    'filename': filename,
                    'page_number': page_number,
                    'format': ext,
                    'description': description,
                    'description_success': success
                }
                
            except Exception as e:
                logger.warning(f"  ⚠ Could not process image {idx}: {e}")
        
        return image_data
    
    def _extract_with_descriptions(self, file_path: str) -> Tuple[str, bool, Dict]:
        """Extract with python-docx and AI image descriptions"""
        try:
            from app.services.image_description_service import get_image_description_service
            
            doc = Document(file_path)
            doc_stem = Path(file_path).stem
            
            image_output_dir = os.path.join(
                os.path.dirname(file_path),
                "extracted_images",
                doc_stem
            )
            os.makedirs(image_output_dir, exist_ok=True)
            
            # Estimate pages
            page_breaks = self._find_page_breaks(doc)
            total_pages = self._estimate_total_pages(doc, page_breaks)
            
            logger.info(f"Estimated pages: {total_pages}")
            
            # Extract images with descriptions
            image_data = self._extract_images_with_descriptions(
                doc, image_output_dir, doc_stem, page_breaks, total_pages
            )
            
            logger.info(f"Extracted {len(image_data)} images")
            
            # Build markdown with descriptions
            markdown_content = self._build_content_with_descriptions(
                doc, image_data, doc_stem, total_pages, page_breaks
            )
            
            metadata = self.get_metadata(file_path)
            metadata.update({
                'total_pages': total_pages,
                'images_extracted': len(image_data),
                'images_described': sum(1 for img in image_data.values() if img.get('description_success')),
                'image_directory': str(image_output_dir),
                'tables_found': len(doc.tables),
                'extraction_method': 'python-docx'
            })
            
            logger.info("✓ Extraction complete with AI descriptions")
            
            return markdown_content, True, metadata
            
        except Exception as e:
            logger.error(f"Extraction failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return "", False, {'error': str(e)}
    
    def _find_page_breaks(self, doc: Document) -> List[int]:
        """Find page break positions"""
        page_breaks = [0]
        for i, para in enumerate(doc.paragraphs):
            if '\\f' in para._element.xml or 'w:br w:type="page"' in para._element.xml:
                page_breaks.append(i)
        return page_breaks
    
    def _estimate_total_pages(self, doc: Document, page_breaks: List[int]) -> int:
        """Estimate total pages"""
        if len(page_breaks) > 1:
            return len(page_breaks)
        total_chars = sum(len(p.text) for p in doc.paragraphs)
        estimated_pages = max(1, (total_chars // 2000) + 1)
        return estimated_pages
    
    def _extract_images_with_descriptions(
        self,
        doc: Document,
        output_dir: str,
        doc_stem: str,
        page_breaks: List[int],
        total_pages: int
    ) -> Dict[int, Dict]:
        """Extract images and generate AI descriptions"""
        from app.services.image_description_service import get_image_description_service
        
        image_data = {}
        image_counter = 0
        
        description_service = get_image_description_service()
        doc_context = f"Word document: {doc_stem}"
        
        for para_idx, para in enumerate(doc.paragraphs):
            # Estimate page
            page_number = self._estimate_paragraph_page(para_idx, page_breaks, total_pages)
            
            # Check for images in paragraph
            if para._element.xpath('.//pic:pic'):
                for pic in para._element.xpath('.//pic:pic'):
                    image_counter += 1
                    
                    try:
                        # Extract image
                        blip = pic.xpath('.//a:blip', namespaces={'a': 'http://schemas.openxmlformats.org/drawingml/2006/main'})[0]
                        rId = blip.get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed')
                        image_part = doc.part.related_parts[rId]
                        image_bytes = image_part.blob
                        
                        # Determine format
                        content_type = image_part.content_type
                        ext = content_type.split('/')[-1]
                        if ext == 'jpeg':
                            ext = 'jpg'
                        
                        filename = f"page_{page_number:02d}_image_{image_counter:03d}.{ext}"
                        output_path = os.path.join(output_dir, filename)
                        
                        # Save image
                        with open(output_path, 'wb') as f:
                            f.write(image_bytes)
                        
                        logger.info(f"  ✓ Saved: {filename}")
                        
                        # Generate AI description
                        if description_service.is_available():
                            base64_image = base64.b64encode(image_bytes).decode('utf-8')
                            success, description = description_service.describe_image_from_base64(
                                base64_image,
                                content_type,
                                context=doc_context,
                                page_number=page_number,
                                filename=filename
                            )
                            
                            if success:
                                logger.info(f"  ✓ Described: {filename}")
                        else:
                            success = False
                            description = description_service.generate_fallback_description(
                                filename, page_number=page_number
                            )
                        
                        image_data[image_counter] = {
                            'filename': filename,
                            'page_number': page_number,
                            'paragraph_index': para_idx,
                            'format': ext,
                            'description': description,
                            'description_success': success
                        }
                        
                    except Exception as e:
                        logger.warning(f"Could not extract image {image_counter}: {e}")
        
        return image_data
    
    def _estimate_paragraph_page(self, para_idx: int, page_breaks: List[int], total_pages: int) -> int:
        """Estimate which page a paragraph is on"""
        for i, break_idx in enumerate(page_breaks):
            if para_idx < break_idx:
                return max(1, i)
        return total_pages
    
    def _build_content_with_descriptions(
        self,
        doc: Document,
        image_data: Dict[int, Dict],
        doc_stem: str,
        total_pages: int,
        page_breaks: List[int]
    ) -> str:
        """
        Build markdown with AI descriptions instead of image paths (python-docx path)
        
        Includes caption detection to associate figure labels with images
        """
        lines = []
        lines.append(f"# Document: {doc_stem}")
        lines.append(f"**Total Pages**: {total_pages}\n")
        
        current_page = 1
        lines.append(f"## Page {current_page}\n")
        
        image_counter = 0
        
        # First pass: Build content with image placeholders
        temp_lines = []
        for para_idx, para in enumerate(doc.paragraphs):
            # Check for page break
            page_number = self._estimate_paragraph_page(para_idx, page_breaks, total_pages)
            if page_number != current_page:
                current_page = page_number
                temp_lines.append(f"\n## Page {current_page}\n")
            
            # Check if paragraph has images
            has_images = bool(para._element.xpath('.//pic:pic'))
            
            if has_images:
                # Add paragraph text if any
                if para.text.strip():
                    temp_lines.append(para.text.strip())
                    temp_lines.append("")
                
                # Add image placeholders
                for pic in para._element.xpath('.//pic:pic'):
                    image_counter += 1
                    if image_counter in image_data:
                        temp_lines.append(f"[IMAGE_PLACEHOLDER_{image_counter}]")
            else:
                # Regular paragraph
                if para.text.strip():
                    temp_lines.append(para.text.strip())
                    temp_lines.append("")
        
        # Add tables
        for table in doc.tables:
            temp_lines.append(self._extract_table(table))
            temp_lines.append("")
        
        # Second pass: Detect captions and associate with images
        processed_caption_indices = set()
        image_caption_map = {}
        placeholder_pattern = r'\[IMAGE_PLACEHOLDER_(\d+)\]'
        
        for i, line in enumerate(temp_lines):
            match = re.match(placeholder_pattern, line)
            if match:
                img_idx = int(match.group(1))
                
                # Look ahead for caption
                caption_info = None
                for j in range(i + 1, min(i + 5, len(temp_lines))):
                    check_line = temp_lines[j].strip()
                    if not check_line:
                        continue
                    
                    result = CaptionDetector.detect_caption(check_line)
                    if result:
                        caption_type, number, description = result
                        if caption_type in ['figure', 'table', 'chart']:
                            caption_info = {
                                'type': caption_type,
                                'number': number,
                                'description': description,
                                'line_idx': j
                            }
                            processed_caption_indices.add(j)
                            break
                
                # Look backward if not found ahead
                if not caption_info:
                    for j in range(max(0, i - 5), i):
                        check_line = temp_lines[j].strip()
                        if not check_line:
                            continue
                        
                        result = CaptionDetector.detect_caption(check_line)
                        if result:
                            caption_type, number, description = result
                            if caption_type in ['figure', 'table', 'chart']:
                                if j not in processed_caption_indices:
                                    caption_info = {
                                        'type': caption_type,
                                        'number': number,
                                        'description': description,
                                        'line_idx': j
                                    }
                                    processed_caption_indices.add(j)
                                    break
                
                if caption_info:
                    image_caption_map[img_idx] = caption_info
        
        # Third pass: Build final output with captions and descriptions
        final_lines = []
        for i, line in enumerate(temp_lines):
            # Skip processed captions
            if i in processed_caption_indices:
                continue
            
            # Check if this is an image placeholder
            match = re.match(placeholder_pattern, line)
            if match:
                img_idx = int(match.group(1))
                if img_idx in image_data:
                    img_info = image_data[img_idx]
                    
                    # Add caption as heading if found (e.g., "### Figure 1.3.4: Revenue Trends")
                    if img_idx in image_caption_map:
                        caption = image_caption_map[img_idx]
                        heading = CaptionDetector.format_as_heading(
                            caption['type'], caption['number'], caption['description']
                        )
                        final_lines.append(f"{heading}\n")
                    
                    # Add AI description
                    final_lines.append(f"**[Image Description]**: {img_info['description']}")
                    final_lines.append(f"*(Located on page {img_info['page_number']})*")
                    final_lines.append("")
            else:
                final_lines.append(line)
        
        return '\n'.join(final_lines)
    
    def _extract_table(self, table) -> str:
        """Extract table as markdown"""
        try:
            rows = []
            for row in table.rows:
                cells = [cell.text.strip() for cell in row.cells]
                rows.append(cells)
            
            if not rows:
                return ""
            
            md_lines = []
            md_lines.append("| " + " | ".join(rows[0]) + " |")
            md_lines.append("| " + " | ".join(["---"] * len(rows[0])) + " |")
            
            for row in rows[1:]:
                while len(row) < len(rows[0]):
                    row.append("")
                md_lines.append("| " + " | ".join(row) + " |")
            
            return "\n".join(md_lines)
        except:
            return ""
    
    def _build_markdown_with_descriptions(
        self,
        markdown_content: str,
        image_data: Dict[int, Dict],
        doc_stem: str,
        total_pages: int
    ) -> str:
        """
        Build markdown with captions AND AI descriptions (Docling path)
        
        Format:
        ### Figure 1.3.4: Revenue Trends  <-- Caption from document
        **[Image Description]**: A bar chart... <-- AI description
        """
        lines = markdown_content.split('\n')
        result_lines = []
        
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
                
                # Look ahead for caption
                caption_info = CaptionDetector.find_caption_in_range(lines, i + 1, lookahead=5)
                
                if caption_info:
                    image_caption_map[idx] = caption_info
                    processed_caption_indices.add(caption_info['line_idx'])
                
                # Look backward if not found ahead
                if idx not in image_caption_map:
                    caption_before = CaptionDetector.find_caption_before(lines, i, lookback=5)
                    if caption_before:
                        # Check if not already processed
                        if caption_before['line_idx'] not in processed_caption_indices:
                            image_caption_map[idx] = caption_before
                            processed_caption_indices.add(caption_before['line_idx'])
                
                image_counter[0] += 1
        
        # Second pass: build output with captions AND descriptions
        image_counter[0] = 1
        
        for i, line in enumerate(lines):
            if i in processed_caption_indices:
                continue
            
            if line.startswith('#'):
                result_lines.append(line)
                continue
            
            if re.search(placeholder_pattern, line):
                idx = image_counter[0]
                
                if idx in image_data:
                    img_info = image_data[idx]
                    page_number = img_info['page_number']
                    
                    if not result_lines or not result_lines[-1].startswith(f"## Page {page_number}"):
                        result_lines.append(f"\n## Page {page_number}\n")
                    
                    # Add caption as heading if found (e.g., "### Figure 1.3.4: Revenue Trends")
                    if idx in image_caption_map:
                        caption = image_caption_map[idx]
                        heading = CaptionDetector.format_caption_as_heading(
                            caption['type'], caption['number'], caption['description']
                        )
                        result_lines.append(f"{heading}\n")
                    
                    # Add AI description
                    result_lines.append(f"**[Image Description]**: {img_info['description']}")
                    result_lines.append(f"*(Located on page {page_number})*")
                    result_lines.append("")
                    
                    image_counter[0] += 1
                else:
                    result_lines.append(line)
                    image_counter[0] += 1
            else:
                result_lines.append(line)
        
        return '\n'.join(result_lines)