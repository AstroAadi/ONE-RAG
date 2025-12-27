"""
PDF Extractor using Docling
Refactored from text_extractor.py
"""

import os
from pathlib import Path
from typing import Tuple, Dict, List
import base64
import re
import logging

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat

from ..base_extractor import BaseExtractor
from ..processors.caption_detector import CaptionDetector

logger = logging.getLogger(__name__)


class PDFExtractor(BaseExtractor):
    """Extractor for PDF documents using Docling"""
    
    def _get_supported_formats(self) -> List[str]:
        return ['pdf']
    
    def validate_file(self, file_path: str) -> Tuple[bool, str]:
        """Validate PDF file"""
        path = Path(file_path)
        
        if not path.exists():
            return False, "File does not exist"
        
        if path.suffix.lower() != '.pdf':
            return False, "Not a PDF file"
        
        if path.stat().st_size == 0:
            return False, "PDF file is empty"
        
        return True, "Valid PDF"
    
    def extract(self, file_path: str) -> Tuple[str, bool, Dict]:
        """Extract PDF with Docling and proper caption handling"""
        try:
            is_valid, msg = self.validate_file(file_path)
            if not is_valid:
                return "", False, {'error': msg}
            
            pdf_stem = Path(file_path).stem
            image_output_dir = os.path.join(
                os.path.dirname(file_path),
                "extracted_images",
                pdf_stem
            )
            os.makedirs(image_output_dir, exist_ok=True)
            
            logger.info(f"Extracting: {file_path}")
            
            # Configure Docling
            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_ocr = self.get_config('do_ocr', True)
            pipeline_options.do_table_structure = self.get_config('do_table_structure', True)
            pipeline_options.images_scale = self.get_config('images_scale', 2.0)
            pipeline_options.generate_picture_images = self.get_config('generate_images', True)
            
            converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
                }
            )
            
            # Convert
            result = converter.convert(file_path)
            total_pages = len(result.document.pages)
            
            # Extract images
            markdown_with_base64 = result.document.export_to_markdown(image_mode="embedded")
            image_mapping = self._extract_and_save_images(
                markdown_with_base64, image_output_dir, pdf_stem, result.document
            )
            
            # Generate markdown with placeholders
            markdown_content = result.document.export_to_markdown(
                image_mode="placeholder",
                image_placeholder="[IMAGE:{image_name}]"
            )
            
            # Build final markdown with captions
            final_markdown = self._build_markdown_with_captions(
                markdown_content, image_mapping, pdf_stem, total_pages
            )
            
            metadata = self.get_metadata(file_path)
            metadata.update({
                'total_pages': total_pages,
                'images_extracted': len(image_mapping),
                'image_directory': str(image_output_dir)
            })
            
            logger.info(f"✓ PDF extraction complete: {total_pages} pages, {len(image_mapping)} images")
            
            return final_markdown, True, metadata
            
        except Exception as e:
            logger.error(f"PDF extraction error: {str(e)}")
            import traceback
            traceback.print_exc()
            return "", False, {'error': str(e)}
    
    def _extract_and_save_images(
        self, markdown_with_base64: str, image_output_dir: str, 
        pdf_stem: str, document
    ) -> Dict[int, Dict]:
        """Extract and save images with page numbers"""
        pattern = r'!\[([^\]]*)\]\(data:image/([^;]+);base64,([^)]+)\)'
        matches = re.findall(pattern, markdown_with_base64)
        image_mapping = {}
        
        page_hints = self._get_page_hints(document)
        
        for idx, match in enumerate(matches, start=1):
            try:
                image_format = match[1]
                base64_data = match[2]
                image_data = base64.b64decode(base64_data)
                
                ext = image_format.lower()
                if ext == 'jpeg':
                    ext = 'jpg'
                
                page_number = page_hints.get(idx, self._estimate_page(idx, len(matches), len(document.pages)))
                image_filename = f"page_{page_number:02d}_image_{idx:03d}.{ext}"
                image_path = os.path.join(image_output_dir, image_filename)
                
                with open(image_path, 'wb') as f:
                    f.write(image_data)
                
                image_mapping[idx] = {
                    'filename': image_filename,
                    'page_number': page_number,
                    'format': ext
                }
                
            except Exception as e:
                logger.warning(f"Could not save image {idx}: {e}")
                continue
        
        return image_mapping
    
    def _get_page_hints(self, document) -> Dict[int, int]:
        """Get page numbers from Docling document structure"""
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
        return page_hints
    
    def _estimate_page(self, image_idx: int, total_images: int, total_pages: int) -> int:
        """Estimate page number based on image position"""
        if total_images == 0 or total_pages == 0:
            return 1
        estimated_page = int((image_idx / total_images) * total_pages) + 1
        return max(1, min(estimated_page, total_pages))
    
    def _build_markdown_with_captions(
        self, markdown_content: str, image_mapping: Dict[int, Dict],
        pdf_stem: str, total_pages: int
    ) -> str:
        """
        Build final markdown with proper caption ordering
        Uses CaptionDetector to find and associate captions with images
        """
        lines = markdown_content.split('\n')
        result_lines = []
        
        # Add header
        result_lines.append(f"# Document: {pdf_stem}")
        result_lines.append(f"**Total Pages**: {total_pages}\n")
        
        placeholder_pattern = r'\[IMAGE:\{[^\}]+\}\]'
        image_counter = [1]
        processed_caption_indices = set()
        
        # First pass: Find all captions associated with images
        image_caption_map = {}
        for i, line in enumerate(lines):
            if re.search(placeholder_pattern, line):
                idx = image_counter[0]
                
                # Look ahead for caption in next few lines (FIXED)
                caption_info = None
                for j in range(i + 1, min(i + 6, len(lines))):
                    check_line = lines[j].strip()
                    if not check_line:
                        continue
                    
                    # Try to detect caption
                    result = CaptionDetector.detect_caption(check_line)
                    if result:
                        caption_type, number, description = result
                        # Only accept figure captions for images
                        if caption_type == 'figure':
                            caption_info = {
                                'type': caption_type,
                                'number': number,
                                'description': description,
                                'line_idx': j,
                                'full_text': check_line
                            }
                            processed_caption_indices.add(j)
                            break
                    
                    # Stop if we hit structural elements
                    if check_line.startswith('#') or '[IMAGE:' in check_line:
                        break
                
                if caption_info:
                    image_caption_map[idx] = caption_info
                
                image_counter[0] += 1
        
        # Second pass: Build markdown
        image_counter[0] = 1
        
        for i, line in enumerate(lines):
            # Skip processed captions
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
                    rel_path = os.path.join("extracted_images", pdf_stem, filename)
                    
                    # Add page marker
                    if not result_lines or not result_lines[-1].startswith(f"## Page {page_number}"):
                        result_lines.append(f"\n## Page {page_number}\n")
                    
                    # Add caption as heading if found
                    if idx in image_caption_map:
                        caption = image_caption_map[idx]
                        heading = CaptionDetector.format_as_heading(
                            caption['type'], caption['number'], caption['description']
                        )
                        result_lines.append(f"{heading}\n")
                    
                    # Add image path
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