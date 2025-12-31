"""
PowerPoint (PPTX/PPT) Extractor with AI Image Descriptions
Images are described using vision LLM at ingestion time instead of storing paths
"""

import os
from pathlib import Path
from typing import Tuple, Dict, List
import logging
import re
import base64
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE

try:
    from ..base_extractor import BaseExtractor
    from ..processors.caption_detector import CaptionDetector
except ImportError:
    from base_extractor import BaseExtractor
    from caption_detector import CaptionDetector

logger = logging.getLogger(__name__)


class PptxExtractor(BaseExtractor):
    """
    Enhanced PowerPoint extractor with AI image descriptions
    
    Key Change: Images are converted to AI descriptions at ingestion time
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
        """Extract PowerPoint with AI image descriptions"""
        try:
            is_valid, msg = self.validate_file(file_path)
            if not is_valid:
                return "", False, {'error': msg}
            
            logger.info(f"Extracting PPTX: {file_path}")
            
            return self._extract_with_descriptions(file_path)
            
        except Exception as e:
            logger.error(f"PPTX extraction error: {str(e)}")
            import traceback
            traceback.print_exc()
            return "", False, {'error': str(e)}
    
    def _extract_with_descriptions(self, file_path: str) -> Tuple[str, bool, Dict]:
        """Extract PowerPoint with AI image descriptions"""
        try:
            from app.services.image_description_service import get_image_description_service
            
            prs = Presentation(file_path)
            pptx_stem = Path(file_path).stem
            
            image_output_dir = os.path.join(
                os.path.dirname(file_path),
                "extracted_images",
                pptx_stem
            )
            os.makedirs(image_output_dir, exist_ok=True)
            
            logger.info(f"Total slides: {len(prs.slides)}")
            
            # NEW: Extract images with AI descriptions
            image_data = self._extract_images_with_descriptions(
                prs, image_output_dir, pptx_stem
            )
            
            logger.info(f"Extracted {len(image_data)} images")
            
            # Build markdown with descriptions
            markdown_content = self._build_markdown_with_descriptions(
                prs, image_data, pptx_stem
            )
            
            stats = self._gather_statistics(prs, image_data)
            
            metadata = self.get_metadata(file_path)
            metadata.update({
                'total_pages': len(prs.slides),
                'total_slides': len(prs.slides),
                'images_extracted': len(image_data),
                'images_described': sum(1 for img in image_data.values() if img.get('description_success')),
                'image_directory': str(image_output_dir),
                'tables_found': stats['tables'],
                'shapes_with_text': stats['text_shapes'],
                'slides_with_notes': stats['notes_slides'],
                'extraction_method': 'python-pptx'
            })
            
            logger.info("✓ PPTX extraction complete with AI descriptions")
            
            return markdown_content, True, metadata
            
        except Exception as e:
            logger.error(f"PPTX extraction failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return "", False, {'error': str(e)}
    
    def _extract_images_with_descriptions(
        self,
        prs: Presentation,
        output_dir: str,
        pptx_stem: str
    ) -> Dict[int, Dict]:
        """Extract images and generate AI descriptions"""
        from app.services.image_description_service import get_image_description_service
        
        image_data = {}
        image_counter = 0
        
        description_service = get_image_description_service()
        doc_context = f"PowerPoint presentation: {pptx_stem}"
        
        for slide_num, slide in enumerate(prs.slides, start=1):
            for shape in slide.shapes:
                if shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
                    image_counter += 1
                    
                    try:
                        # Get image data
                        image = shape.image
                        image_bytes = image.blob
                        
                        # Determine format
                        ext = self._get_image_extension(image.content_type)
                        
                        filename = f"slide_{slide_num:02d}_image_{image_counter:03d}.{ext}"
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
                                image.content_type,
                                context=doc_context,
                                page_number=slide_num,
                                filename=filename
                            )
                            
                            if success:
                                logger.info(f"  ✓ Described: {filename}")
                            else:
                                logger.warning(f"  ⚠ Description failed: {filename}")
                        else:
                            success = False
                            description = description_service.generate_fallback_description(
                                filename, page_number=slide_num, image_type="slide image"
                            )
                        
                        image_data[image_counter] = {
                            'filename': filename,
                            'slide_number': slide_num,
                            'page_number': slide_num,
                            'format': ext,
                            'description': description,
                            'description_success': success,
                            'content_type': image.content_type,
                            'size_bytes': len(image_bytes)
                        }
                        
                    except Exception as e:
                        logger.warning(f"  ⚠ Could not extract image from slide {slide_num}: {e}")
                
                # Check grouped shapes
                elif hasattr(shape, 'shapes'):
                    for sub_shape in shape.shapes:
                        if sub_shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
                            image_counter += 1
                            try:
                                image = sub_shape.image
                                image_bytes = image.blob
                                ext = self._get_image_extension(image.content_type)
                                
                                filename = f"slide_{slide_num:02d}_image_{image_counter:03d}.{ext}"
                                output_path = os.path.join(output_dir, filename)
                                
                                with open(output_path, 'wb') as f:
                                    f.write(image_bytes)
                                
                                logger.info(f"  ✓ Saved: {filename}")
                                
                                if description_service.is_available():
                                    base64_image = base64.b64encode(image_bytes).decode('utf-8')
                                    success, description = description_service.describe_image_from_base64(
                                        base64_image,
                                        image.content_type,
                                        context=doc_context,
                                        page_number=slide_num,
                                        filename=filename
                                    )
                                    
                                    if success:
                                        logger.info(f"  ✓ Described: {filename}")
                                else:
                                    success = False
                                    description = description_service.generate_fallback_description(
                                        filename, page_number=slide_num
                                    )
                                
                                image_data[image_counter] = {
                                    'filename': filename,
                                    'slide_number': slide_num,
                                    'page_number': slide_num,
                                    'format': ext,
                                    'description': description,
                                    'description_success': success,
                                    'content_type': image.content_type,
                                    'size_bytes': len(image_bytes)
                                }
                                
                            except Exception as e:
                                logger.warning(f"  ⚠ Could not extract grouped image: {e}")
        
        return image_data
    
    def _build_markdown_with_descriptions(
        self,
        prs: Presentation,
        image_data: Dict[int, Dict],
        pptx_stem: str
    ) -> str:
        """
        Build markdown with captions AND AI descriptions
        
        Format:
        ### Figure 1.3.4: Revenue Trends  <-- Caption from slide text
        **[Image Description]**: A bar chart... <-- AI description
        """
        # First pass: Build content with image placeholders
        temp_parts = []
        temp_parts.append(f"# Presentation: {pptx_stem}")
        temp_parts.append(f"**Total Slides**: {len(prs.slides)}\n")
        
        image_counter = 0
        
        for slide_num, slide in enumerate(prs.slides, start=1):
            temp_parts.append(f"## Slide {slide_num}")
            
            # Slide title
            title = self._get_slide_title(slide)
            if title:
                temp_parts.append(f"### {title}\n")
            
            # Extract content
            slide_text = []
            slide_images = []
            slide_tables = []
            
            for shape in slide.shapes:
                if shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
                    image_counter += 1
                    if image_counter in image_data:
                        slide_images.append(image_counter)
                
                elif shape.shape_type == MSO_SHAPE_TYPE.TABLE:
                    table_md = self._extract_table(shape)
                    if table_md:
                        slide_tables.append(table_md)
                
                elif hasattr(shape, "text") and shape.text.strip():
                    if shape == slide.shapes.title:
                        continue
                    text = shape.text.strip()
                    slide_text.append(text)
                
                elif hasattr(shape, 'shapes'):
                    for sub_shape in shape.shapes:
                        if sub_shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
                            image_counter += 1
                            if image_counter in image_data:
                                slide_images.append(image_counter)
                        elif hasattr(sub_shape, "text") and sub_shape.text.strip():
                            slide_text.append(sub_shape.text.strip())
            
            # Add text content
            if slide_text:
                for text in slide_text:
                    temp_parts.append(f"{text}\n")
            
            # Add image placeholders
            if slide_images:
                for img_idx in slide_images:
                    temp_parts.append(f"[IMAGE_PLACEHOLDER_{img_idx}]")
            
            # Add tables
            if slide_tables:
                for table_md in slide_tables:
                    temp_parts.append(f"\n{table_md}\n")
            
            # Speaker notes
            if slide.has_notes_slide:
                notes_text = slide.notes_slide.notes_text_frame.text.strip()
                if notes_text:
                    temp_parts.append(f"\n**Speaker Notes**: {notes_text}\n")
            
            temp_parts.append("")
        
        # Second pass: Detect captions and associate with images
        processed_caption_indices = set()
        image_caption_map = {}
        placeholder_pattern = r'\[IMAGE_PLACEHOLDER_(\d+)\]'
        
        for i, line in enumerate(temp_parts):
            match = re.match(placeholder_pattern, line.strip())
            if match:
                img_idx = int(match.group(1))
                
                # Look ahead for caption (within next 8 lines for slides)
                caption_info = None
                for j in range(i + 1, min(i + 8, len(temp_parts))):
                    check_line = temp_parts[j].strip()
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
                                'line_idx': j,
                                'full_text': check_line
                            }
                            processed_caption_indices.add(j)
                            break
                    
                    # Stop at structural elements
                    if check_line.startswith('#') or '[IMAGE_PLACEHOLDER_' in check_line:
                        break
                
                # Look backward if not found ahead (captions can come before images in slides)
                if not caption_info:
                    for j in range(max(0, i - 8), i):
                        check_line = temp_parts[j].strip()
                        if not check_line:
                            continue
                        
                        result = CaptionDetector.detect_caption(check_line)
                        if result:
                            caption_type, number, description = result
                            if caption_type in ['figure', 'table', 'chart']:
                                # Check if not already processed
                                if j not in processed_caption_indices:
                                    caption_info = {
                                        'type': caption_type,
                                        'number': number,
                                        'description': description,
                                        'line_idx': j,
                                        'full_text': check_line
                                    }
                                    processed_caption_indices.add(j)
                                    break
                        
                        # Stop at structural elements
                        if check_line.startswith('#'):
                            break
                
                if caption_info:
                    image_caption_map[img_idx] = caption_info
        
        # Third pass: Build final output with captions and descriptions
        final_parts = []
        for i, line in enumerate(temp_parts):
            # Skip processed captions
            if i in processed_caption_indices:
                continue
            
            # Check if this is an image placeholder
            match = re.match(placeholder_pattern, line.strip())
            if match:
                img_idx = int(match.group(1))
                if img_idx in image_data:
                    image_info = image_data[img_idx]
                    
                    # Add caption as heading if found (e.g., "### Figure 1.3.4: Revenue Trends")
                    if img_idx in image_caption_map:
                        caption = image_caption_map[img_idx]
                        heading = CaptionDetector.format_as_heading(
                            caption['type'], caption['number'], caption['description']
                        )
                        final_parts.append(f"\n{heading}\n")
                    
                    # Add AI description
                    final_parts.append(f"**[Image Description]**: {image_info['description']}")
                    final_parts.append(f"*(Located on slide {image_info['slide_number']})*")
                    final_parts.append("")
            else:
                final_parts.append(line)
        
        return "\n".join(final_parts)
    
    def _get_slide_title(self, slide) -> str:
        """Extract slide title"""
        if slide.shapes.title:
            return slide.shapes.title.text.strip()
        return ""
    
    def _extract_table(self, shape) -> str:
        """Extract table as markdown"""
        try:
            table = shape.table
            rows = []
            
            for row in table.rows:
                row_data = []
                for cell in row.cells:
                    row_data.append(cell.text.strip())
                rows.append(row_data)
            
            if not rows:
                return ""
            
            md_parts = []
            md_parts.append("| " + " | ".join(rows[0]) + " |")
            md_parts.append("| " + " | ".join(["---"] * len(rows[0])) + " |")
            
            for row in rows[1:]:
                while len(row) < len(rows[0]):
                    row.append("")
                md_parts.append("| " + " | ".join(row) + " |")
            
            return "\n".join(md_parts)
            
        except Exception as e:
            logger.warning(f"Could not extract table: {e}")
            return ""
    
    def _get_image_extension(self, content_type: str) -> str:
        """Get file extension from content type"""
        extensions = {
            'image/jpeg': 'jpg',
            'image/png': 'png',
            'image/gif': 'gif',
            'image/bmp': 'bmp',
            'image/tiff': 'tiff',
            'image/webp': 'webp'
        }
        ext = extensions.get(content_type, 'png')
        return ext
    
    def _gather_statistics(self, prs: Presentation, image_data: Dict) -> Dict:
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