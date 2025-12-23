"""
PowerPoint (PPTX/PPT) Extractor with Complete Feature Parity
Matches PDF/DOCX extractors with page tracking, images, and captions
"""

import os
from pathlib import Path
from typing import Tuple, Dict, List
import logging
import re
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE
import io
from PIL import Image

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


class PptxExtractor(BaseExtractor):
    """
    Enhanced PowerPoint extractor with full feature parity
    
    Features:
    - Extracts text from all slides
    - Extracts and saves images with slide numbers
    - Extracts tables
    - Extracts speaker notes
    - Slide-based page tracking (slide = page)
    - Caption detection for figures
    - Shape text extraction (text boxes, etc.)
    """
    
    def _get_supported_formats(self) -> List[str]:
        return ['pptx', 'ppt']
    
    def validate_file(self, file_path: str) -> Tuple[bool, str]:
        """Validate PowerPoint file"""
        path = Path(file_path)
        
        if not path.exists():
            return False, "File does not exist"
        
        if path.suffix.lower() not in ['.pptx', '.ppt']:
            return False, "Not a PowerPoint file"
        
        if path.stat().st_size == 0:
            return False, "File is empty"
        
        return True, "Valid PowerPoint file"
    
    def extract(self, file_path: str) -> Tuple[str, bool, Dict]:
        """Extract PowerPoint with full feature support"""
        try:
            is_valid, msg = self.validate_file(file_path)
            if not is_valid:
                return "", False, {'error': msg}
            
            logger.info(f"Extracting PPTX: {file_path}")
            
            # Extract with python-pptx
            return self._extract_with_python_pptx(file_path)
            
        except Exception as e:
            logger.error(f"PPTX extraction error: {str(e)}")
            import traceback
            traceback.print_exc()
            return "", False, {'error': str(e)}
    
    def _extract_with_python_pptx(self, file_path: str) -> Tuple[str, bool, Dict]:
        """
        Extract PowerPoint with comprehensive feature support
        """
        try:
            prs = Presentation(file_path)
            pptx_stem = Path(file_path).stem
            
            # Create image directory
            image_output_dir = os.path.join(
                os.path.dirname(file_path),
                "extracted_images",
                pptx_stem
            )
            os.makedirs(image_output_dir, exist_ok=True)
            
            logger.info(f"Image directory: {image_output_dir}")
            logger.info(f"Total slides: {len(prs.slides)}")
            
            # Extract images from all slides
            image_mapping = self._extract_all_images(
                prs, image_output_dir, pptx_stem
            )
            
            logger.info(f"Extracted {len(image_mapping)} images")
            
            # Build markdown content with slide tracking
            markdown_content = self._build_markdown_from_slides(
                prs, image_mapping, pptx_stem
            )
            
            # Apply caption detection
            final_markdown = self._apply_caption_detection(
                markdown_content, image_mapping, pptx_stem
            )
            
            # Count statistics
            stats = self._gather_statistics(prs, image_mapping)
            
            metadata = self.get_metadata(file_path)
            metadata.update({
                'total_pages': len(prs.slides),  # Slides = Pages
                'total_slides': len(prs.slides),
                'images_extracted': len(image_mapping),
                'image_directory': str(image_output_dir),
                'tables_found': stats['tables'],
                'shapes_with_text': stats['text_shapes'],
                'slides_with_notes': stats['notes_slides'],
                'extraction_method': 'python-pptx'
            })
            
            logger.info("✓ PPTX extraction complete")
            
            return final_markdown, True, metadata
            
        except Exception as e:
            logger.error(f"PPTX extraction failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return "", False, {'error': str(e)}
    
    def _extract_all_images(
        self,
        prs: Presentation,
        output_dir: str,
        pptx_stem: str
    ) -> Dict[int, Dict]:
        """
        Extract all images from presentation with slide tracking
        """
        image_mapping = {}
        image_counter = 0
        
        for slide_num, slide in enumerate(prs.slides, start=1):
            # Extract images from this slide
            slide_images = self._extract_images_from_slide(
                slide, slide_num, output_dir, pptx_stem, image_counter
            )
            
            # Add to mapping
            for img_info in slide_images:
                image_counter += 1
                image_mapping[image_counter] = img_info
        
        return image_mapping
    
    def _extract_images_from_slide(
        self,
        slide,
        slide_num: int,
        output_dir: str,
        pptx_stem: str,
        start_counter: int
    ) -> List[Dict]:
        """
        Extract images from a single slide
        """
        images = []
        img_counter = start_counter
        
        for shape in slide.shapes:
            # Check if shape is a picture
            if shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
                img_counter += 1
                
                try:
                    # Get image data
                    image = shape.image
                    image_bytes = image.blob
                    
                    # Determine format
                    ext = self._get_image_extension(image.content_type)
                    
                    # Generate filename with slide number
                    filename = f"slide_{slide_num:02d}_image_{img_counter:03d}.{ext}"
                    output_path = os.path.join(output_dir, filename)
                    
                    # Save image
                    with open(output_path, 'wb') as f:
                        f.write(image_bytes)
                    
                    images.append({
                        'filename': filename,
                        'slide_number': slide_num,
                        'page_number': slide_num,  # For consistency
                        'format': ext,
                        'content_type': image.content_type,
                        'size_bytes': len(image_bytes)
                    })
                    
                    logger.info(f"  ✓ Saved: {filename} (slide {slide_num})")
                    
                except Exception as e:
                    logger.warning(f"  ⚠ Could not extract image from slide {slide_num}: {e}")
                    continue
            
            # Check for grouped shapes (images might be in groups)
            elif hasattr(shape, 'shapes'):
                # Recursively check grouped shapes
                for sub_shape in shape.shapes:
                    if sub_shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
                        img_counter += 1
                        
                        try:
                            image = sub_shape.image
                            image_bytes = image.blob
                            ext = self._get_image_extension(image.content_type)
                            
                            filename = f"slide_{slide_num:02d}_image_{img_counter:03d}.{ext}"
                            output_path = os.path.join(output_dir, filename)
                            
                            with open(output_path, 'wb') as f:
                                f.write(image_bytes)
                            
                            images.append({
                                'filename': filename,
                                'slide_number': slide_num,
                                'page_number': slide_num,
                                'format': ext,
                                'content_type': image.content_type,
                                'size_bytes': len(image_bytes)
                            })
                            
                            logger.info(f"  ✓ Saved: {filename} (slide {slide_num}, grouped)")
                            
                        except Exception as e:
                            logger.warning(f"  ⚠ Could not extract grouped image: {e}")
        
        return images
    
    def _get_image_extension(self, content_type: str) -> str:
        """Get file extension from MIME type"""
        mime_to_ext = {
            'image/png': 'png',
            'image/jpeg': 'jpg',
            'image/jpg': 'jpg',
            'image/gif': 'gif',
            'image/bmp': 'bmp',
            'image/tiff': 'tiff',
            'image/x-emf': 'emf',
            'image/x-wmf': 'wmf',
        }
        return mime_to_ext.get(content_type, 'png')
    
    def _build_markdown_from_slides(
        self,
        prs: Presentation,
        image_mapping: Dict[int, Dict],
        pptx_stem: str
    ) -> str:
        """
        Build markdown content from presentation slides
        Each slide = one page
        """
        markdown_parts = []
        
        # Add header
        markdown_parts.append(f"# Presentation: {pptx_stem}\n")
        markdown_parts.append(f"**Total Slides**: {len(prs.slides)}\n")
        markdown_parts.append("---\n")
        
        # Track image counter
        image_counter = 0
        
        # Process each slide
        for slide_num, slide in enumerate(prs.slides, start=1):
            # Add slide marker (equivalent to page marker)
            markdown_parts.append(f"\n## Slide {slide_num}\n")
            
            # Extract title
            title = self._get_slide_title(slide)
            if title:
                markdown_parts.append(f"### {title}\n")
            
            # Extract text from all shapes
            slide_text = []
            slide_images = []
            slide_tables = []
            
            for shape in slide.shapes:
                # Handle different shape types
                if shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
                    image_counter += 1
                    if image_counter in image_mapping:
                        slide_images.append(image_counter)
                
                elif shape.shape_type == MSO_SHAPE_TYPE.TABLE:
                    # Extract table
                    table_md = self._extract_table(shape)
                    if table_md:
                        slide_tables.append(table_md)
                
                elif hasattr(shape, "text") and shape.text.strip():
                    # Skip title (already processed)
                    if shape == slide.shapes.title:
                        continue
                    
                    text = shape.text.strip()
                    slide_text.append(text)
                
                elif hasattr(shape, 'shapes'):
                    # Handle grouped shapes
                    for sub_shape in shape.shapes:
                        if sub_shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
                            image_counter += 1
                            if image_counter in image_mapping:
                                slide_images.append(image_counter)
                        elif hasattr(sub_shape, "text") and sub_shape.text.strip():
                            slide_text.append(sub_shape.text.strip())
            
            # Add text content
            if slide_text:
                for text in slide_text:
                    markdown_parts.append(f"{text}\n")
            
            # Add images with placeholders
            if slide_images:
                for img_idx in slide_images:
                    image_info = image_mapping[img_idx]
                    filename = image_info['filename']
                    slide_number = image_info['slide_number']
                    rel_path = os.path.join("extracted_images", pptx_stem, filename)
                    
                    markdown_parts.append(f"\n[IMAGE_PLACEHOLDER_{img_idx}]")
                    markdown_parts.append(f"IMAGE_PATH: {rel_path}")
                    markdown_parts.append(f"PAGE: {slide_number}")  # PAGE tag for consistency
                    markdown_parts.append(f"SLIDE: {slide_number}")  # Additional SLIDE tag
                    markdown_parts.append("")
            
            # Add tables
            if slide_tables:
                for table_md in slide_tables:
                    markdown_parts.append(f"\n{table_md}\n")
            
            # Extract speaker notes
            if slide.has_notes_slide:
                notes_text = slide.notes_slide.notes_text_frame.text.strip()
                if notes_text:
                    markdown_parts.append(f"\n**Speaker Notes**:\n{notes_text}\n")
            
            markdown_parts.append("")  # Blank line between slides
        
        return "\n".join(markdown_parts)
    
    def _get_slide_title(self, slide) -> str:
        """Extract slide title"""
        if slide.shapes.title:
            return slide.shapes.title.text.strip()
        return ""
    
    def _extract_table(self, shape) -> str:
        """
        Extract table from shape and convert to markdown
        """
        try:
            table = shape.table
            rows = []
            
            # Extract all cells
            for row in table.rows:
                row_data = []
                for cell in row.cells:
                    row_data.append(cell.text.strip())
                rows.append(row_data)
            
            if not rows:
                return ""
            
            # Build markdown table
            md_parts = []
            
            # Header row
            md_parts.append("| " + " | ".join(rows[0]) + " |")
            
            # Separator
            md_parts.append("| " + " | ".join(["---"] * len(rows[0])) + " |")
            
            # Data rows
            for row in rows[1:]:
                # Pad row if needed
                while len(row) < len(rows[0]):
                    row.append("")
                md_parts.append("| " + " | ".join(row) + " |")
            
            return "\n".join(md_parts)
            
        except Exception as e:
            logger.warning(f"Could not extract table: {e}")
            return ""
    
    def _gather_statistics(self, prs: Presentation, image_mapping: Dict) -> Dict:
        """Gather presentation statistics"""
        stats = {
            'tables': 0,
            'text_shapes': 0,
            'notes_slides': 0
        }
        
        for slide in prs.slides:
            if slide.has_notes_slide:
                stats['notes_slides'] += 1
            
            for shape in slide.shapes:
                if shape.shape_type == MSO_SHAPE_TYPE.TABLE:
                    stats['tables'] += 1
                elif hasattr(shape, 'text') and shape.text.strip():
                    stats['text_shapes'] += 1
        
        return stats
    
    def _apply_caption_detection(
        self,
        markdown_content: str,
        image_mapping: Dict[int, Dict],
        pptx_stem: str
    ) -> str:
        """
        Apply caption detection to associate figure labels with images
        """
        lines = markdown_content.split('\n')
        result_lines = []
        
        placeholder_pattern = r'^(IMAGE_PATH:\s*(.+)|.*\[IMAGE_PLACEHOLDER_\d+\].*)$'
        image_counter = [0]
        processed_caption_indices = set()
        image_caption_map = {}
        
        # First pass: Find captions
        for i, line in enumerate(lines):
            if re.match(placeholder_pattern, line):
                image_counter[0] += 1
                idx = image_counter[0]
                
                # Look ahead for caption
                caption_info = CaptionDetector.find_caption_in_range(
                    lines, i + 1, lookahead=10
                )
                
                if caption_info:
                    image_caption_map[idx] = caption_info
                    processed_caption_indices.add(caption_info['line_idx'])
                    logger.debug(f"Found caption for image {idx}: {caption_info['full_text']}")
                
                # Look backward for caption
                if idx not in image_caption_map:
                    caption_before = CaptionDetector.find_caption_before(
                        lines, i, lookback=10
                    )
                    if caption_before:
                        image_caption_map[idx] = caption_before
                        processed_caption_indices.add(caption_before['line_idx'])
                        logger.debug(f"Found caption before image {idx}: {caption_before['full_text']}")
        
        # Second pass: Build output with captions
        image_counter[0] = 0
        
        for i, line in enumerate(lines):
            # Skip processed captions
            if i in processed_caption_indices:
                continue
            
            # Keep headings
            if line.startswith('#'):
                result_lines.append(line)
                continue
            
            # Handle image placeholders
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
                
                # Add image path
                if 'IMAGE_PATH:' in line:
                    result_lines.append(line)
                elif '[IMAGE_PLACEHOLDER_' in line:
                    pass  # Skip placeholder, IMAGE_PATH comes next
                
                result_lines.append("")
            else:
                result_lines.append(line)
        
        return '\n'.join(result_lines)