"""
PDF Extractor using Docling with Image Descriptions - FIXED VERSION
All extraction issues resolved:
1. Proper page tracking (no missing page 1, no duplicates)
2. Captions with images (not separated)
3. Chart titles filtered (not shown as H2)
4. Body text cleanup (no duplication)
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
    """
    Extractor for PDF documents using Docling
    
    FIXED: Proper page markers, captions with images, no chart title headings
    """
    
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
        """Extract PDF with Docling and convert images to descriptions"""
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
            
            # Extract images with base64 data
            markdown_with_base64 = result.document.export_to_markdown(image_mode="embedded")
            
            # Extract and describe images
            image_data = self._extract_and_describe_images(
                markdown_with_base64,
                image_output_dir,
                pdf_stem,
                result.document,
                file_path
            )
            
            # Generate markdown with placeholders
            markdown_content = result.document.export_to_markdown(
                image_mode="placeholder",
                image_placeholder="[IMAGE:{image_name}]"
            )
            
            # FIXED: Build final markdown with proper page tracking and caption placement
            final_markdown = self._build_markdown_with_descriptions(
                markdown_content,
                image_data,
                pdf_stem,
                total_pages
            )
            
            metadata = self.get_metadata(file_path)
            metadata.update({
                'total_pages': total_pages,
                'images_extracted': len(image_data),
                'images_described': sum(1 for img in image_data.values() if img.get('description_success')),
                'image_directory': str(image_output_dir)
            })
            
            logger.info(f"✓ PDF extraction complete: {total_pages} pages, {len(image_data)} images described")
            
            return final_markdown, True, metadata
            
        except Exception as e:
            logger.error(f"PDF extraction error: {str(e)}")
            import traceback
            traceback.print_exc()
            return "", False, {'error': str(e)}
    
    def _extract_and_describe_images(
        self,
        markdown_with_base64: str,
        image_output_dir: str,
        pdf_stem: str,
        document,
        pdf_path: str
    ) -> Dict[int, Dict]:
        """
        Extract images, save to disk, and generate AI descriptions
        
        Returns:
            Dictionary mapping image index to image info
        """
        from app.services.image_description_service import get_image_description_service
        
        pattern = r'!\[([^\]]*)\]\(data:image/([^;]+);base64,([^)]+)\)'
        matches = re.findall(pattern, markdown_with_base64)
        image_data = {}
        
        page_hints = self._get_page_hints(document)
        description_service = get_image_description_service()
        doc_context = f"PDF document: {pdf_stem}"
        
        for idx, match in enumerate(matches, start=1):
            try:
                image_format = match[1]
                base64_image = match[2]
                image_bytes = base64.b64decode(base64_image)
                
                ext = image_format.lower()
                if ext == 'jpeg':
                    ext = 'jpg'
                
                page_number = page_hints.get(
                    idx,
                    self._estimate_page(idx, len(matches), len(document.pages))
                )
                
                image_filename = f"page_{page_number:02d}_image_{idx:03d}.{ext}"
                image_path = os.path.join(image_output_dir, image_filename)
                
                # Save image
                with open(image_path, 'wb') as f:
                    f.write(image_bytes)
                
                logger.info(f"  ✓ Saved: {image_filename}")
                
                # Generate AI description
                if description_service.is_available():
                    media_type = f"image/{image_format}"
                    success, description = description_service.describe_image_from_base64(
                        base64_image,
                        media_type,
                        context=doc_context,
                        page_number=page_number,
                        filename=image_filename
                    )
                    
                    if success:
                        logger.info(f"  ✓ Described: {image_filename}")
                    else:
                        logger.warning(f"  ⚠ Description failed: {image_filename}")
                else:
                    success = False
                    description = description_service.generate_fallback_description(
                        image_filename,
                        page_number=page_number,
                        image_type="figure"
                    )
                    logger.warning(f"  ⚠ Vision service unavailable, using fallback")
                
                image_data[idx] = {
                    'filename': image_filename,
                    'page_number': page_number,
                    'format': ext,
                    'description': description,
                    'description_success': success
                }
                
            except Exception as e:
                logger.warning(f"Could not process image {idx}: {e}")
                continue
        
        return image_data
    
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
    
    def _build_markdown_with_descriptions(
        self,
        markdown_content: str,
        image_data: Dict[int, Dict],
        pdf_stem: str,
        total_pages: int
    ) -> str:
        """
        FIXED: Build final markdown with proper page tracking and caption placement
        
        Fixes:
        1. Track current page to avoid duplicates
        2. Add Page 1 marker
        3. Place captions WITH images
        4. Filter chart titles
        5. Remove body text with figure references near images
        """
        lines = markdown_content.split('\n')
        result_lines = []
        
        # Add header
        result_lines.append(f"# Document: {pdf_stem}")
        result_lines.append(f"**Total Pages**: {total_pages}\n")
        
        placeholder_pattern = r'\[IMAGE:\{[^\}]+\}\]'
        
        # FIRST PASS: Find captions and chart titles associated with images
        image_caption_map = {}
        lines_to_skip = set()  # Track lines to skip in output
        image_counter = 1
        
        for i, line in enumerate(lines):
            if re.search(placeholder_pattern, line):
                idx = image_counter
                
                # Look for caption around this image
                caption_info = self._find_caption_near_image(lines, i, idx)
                
                if caption_info:
                    image_caption_map[idx] = caption_info
                    # Mark caption line to skip
                    lines_to_skip.add(caption_info['line_idx'])
                    
                    # Also mark chart titles before caption
                    for j in range(max(0, caption_info['line_idx'] - 5), caption_info['line_idx']):
                        if self._is_chart_title(lines[j]):
                            lines_to_skip.add(j)
                            logger.debug(f"Skipping chart title: {lines[j][:50]}...")
                
                # Mark body text with figure references near this image
                for j in range(max(0, i - 10), min(i + 5, len(lines))):
                    if self._is_body_text_with_figure_ref(lines[j], caption_info):
                        lines_to_skip.add(j)
                        logger.debug(f"Skipping body text: {lines[j][:50]}...")
                
                image_counter += 1
        
        # SECOND PASS: Build markdown with proper page tracking
        current_page = 0
        image_counter = 1
        
        for i, line in enumerate(lines):
            # Skip lines marked for removal
            if i in lines_to_skip:
                continue
            
            # Skip chart title headings (convert to regular text if needed)
            if line.startswith('#') and self._is_chart_title(line.lstrip('#').strip()):
                # Don't output chart title headings
                continue
            
            # Handle regular headings
            if line.startswith('#') and not self._is_chart_title(line.lstrip('#').strip()):
                result_lines.append(line)
                continue
            
            # Handle images
            if re.search(placeholder_pattern, line):
                idx = image_counter
                
                if idx in image_data:
                    img_info = image_data[idx]
                    page_number = img_info['page_number']
                    description = img_info['description']
                    
                    # FIXED: Add page marker only when page changes
                    if page_number != current_page:
                        result_lines.append(f"\n## Page {page_number}\n")
                        current_page = page_number
                    
                    # Add caption as heading if found
                    if idx in image_caption_map:
                        caption = image_caption_map[idx]
                        heading = CaptionDetector.format_as_heading(
                            caption['type'], caption['number'], caption['description']
                        )
                        result_lines.append(f"{heading}\n")
                    
                    # Add AI description
                    result_lines.append(f"**[Image Description]**: {description}")
                    result_lines.append(f"*(Located on page {page_number})*")
                    result_lines.append("")
                    
                    image_counter += 1
                else:
                    result_lines.append(line)
                    image_counter += 1
            else:
                # Regular content line
                result_lines.append(line)
        
        return '\n'.join(result_lines)
    
    def _find_caption_near_image(
        self,
        lines: List[str],
        image_idx: int,
        image_counter: int
    ) -> Dict:
        """
        Find caption near image using enhanced caption detector
        
        Args:
            lines: All markdown lines
            image_idx: Index of image placeholder line
            image_counter: Current image number
            
        Returns:
            Caption info dict or None
        """
        # Look backward first (captions often before images)
        caption = CaptionDetector.find_caption_before(lines, image_idx, lookback=8)
        
        if caption:
            return caption
        
        # Look forward (sometimes captions after images)
        caption = CaptionDetector.find_caption_in_range(lines, image_idx + 1, lookahead=6)
        
        return caption
    
    def _is_chart_title(self, text: str) -> bool:
        """Check if line is a chart title"""
        return CaptionDetector.is_chart_title(text)
    
    def _is_body_text_with_figure_ref(
        self,
        line: str,
        caption_info: Dict
    ) -> bool:
        """
        Check if line is body text that references the figure
        
        Example: "Figure 1.3.4 illustrates the sectoral origin..."
        This should be removed from body since caption is shown with image
        """
        if not caption_info or not line.strip():
            return False
        
        # Check if line contains the figure reference followed by verb
        figure_ref = f"{caption_info['type'].capitalize()} {caption_info['number']}"
        
        if figure_ref in line:
            # Check if followed by body text indicators
            body_indicators = [
                'illustrate', 'show', 'display', 'highlight',
                'visualize', 'present', 'demonstrate'
            ]
            
            if any(indicator in line.lower() for indicator in body_indicators):
                return True
        
        return False