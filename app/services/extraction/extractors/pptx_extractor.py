"""
Enhanced PowerPoint (PPTX/PPT) Extractor - UPDATED
Matches PDF extractor pattern with image caching, duplicate detection, and consistent flow
"""

import os
from pathlib import Path
from typing import Tuple, Dict, List, Optional
import logging
import re
import base64
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE

try:
    from ..base_extractor import BaseExtractor
except ImportError:
    from base_extractor import BaseExtractor

logger = logging.getLogger(__name__)


class PptxExtractor(BaseExtractor):
    """
    Enhanced PowerPoint extractor matching PDF extractor pattern
    
    Features:
    - Image caching (avoid redundant API calls)
    - Duplicate detection (skip same image multiple times)
    - Proper page/slide numbering
    - Consistent markdown structure
    - Smart table and text extraction
    """
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        
        from app.services.image_cache_manager import get_image_cache
        from app.services.image_description_service import get_image_description_service
        
        self.image_cache = get_image_cache()
        self.image_service = get_image_description_service()
        
        # Tracking counters
        self._duplicates_skipped = 0
        
        logger.info("PPTX Extractor initialized with caching and duplicate detection")
    
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
        """Extract PowerPoint with proper flow matching PDF extractor"""
        try:
            is_valid, msg = self.validate_file(file_path)
            if not is_valid:
                return "", False, {'error': msg}
            
            pptx_stem = Path(file_path).stem
            image_output_dir = self._create_image_directory(file_path, pptx_stem)
            
            logger.info(f"📊 Extracting: {file_path}")
            
            # Load presentation
            prs = Presentation(file_path)
            total_slides = len(prs.slides)
            
            logger.info(f"📑 Presentation has {total_slides} slides")
            
            # Reset counters
            self._duplicates_skipped = 0
            
            # Process images with caching and duplicate detection
            image_data = self._process_images_with_caching(
                prs,
                image_output_dir,
                pptx_stem
            )
            
            # Build markdown slide-by-slide
            final_markdown = self._build_markdown_from_slides(
                prs,
                total_slides,
                image_data,
                pptx_stem
            )
            
            # Gather statistics
            cache_stats = self.image_cache.get_stats()
            stats = self._gather_statistics(prs, image_data)
            
            metadata = self.get_metadata(file_path)
            metadata.update({
                'extractor': 'PptxExtractor_Enhanced',
                'total_pages': total_slides,
                'total_slides': total_slides,
                'images_extracted': len(image_data),
                'images_described': sum(1 for img in image_data.values() if img.get('success')),
                'images_cached': cache_stats['hits'],
                'images_skipped_duplicates': self._duplicates_skipped,
                'cache_hit_rate': cache_stats['hit_rate_percent'],
                'image_directory': str(image_output_dir),
                'tables_found': stats['tables'],
                'shapes_with_text': stats['text_shapes'],
                'slides_with_notes': stats['notes_slides']
            })
            
            logger.info(
                f"✅ Complete: {total_slides} slides, {len(image_data)} images "
                f"({metadata['images_cached']} cached, {self._duplicates_skipped} dupes skipped)"
            )
            
            return final_markdown, True, metadata
            
        except Exception as e:
            logger.error(f"❌ PPTX extraction error: {str(e)}")
            import traceback
            traceback.print_exc()
            return "", False, {'error': str(e)}
    
    def _create_image_directory(self, file_path: str, pptx_stem: str) -> str:
        """Create directory for extracted images"""
        image_dir = os.path.join(
            os.path.dirname(file_path),
            "extracted_images",
            pptx_stem
        )
        os.makedirs(image_dir, exist_ok=True)
        return image_dir
    
    def _process_images_with_caching(
        self,
        prs: Presentation,
        image_output_dir: str,
        pptx_stem: str
    ) -> Dict[int, Dict]:
        """
        Process images with caching and duplicate detection
        
        Key features matching PDF extractor:
        - Perceptual hashing for duplicate detection
        - Image cache to avoid redundant API calls
        - Slide number tracking
        """
        image_data = {}
        seen_hashes = set()
        image_counter = 0
        
        doc_context = f"PowerPoint presentation: {pptx_stem}"
        
        for slide_num, slide in enumerate(prs.slides, start=1):
            # Extract images from slide shapes
            slide_images = self._extract_images_from_shapes(
                slide.shapes, slide_num, image_counter, seen_hashes,
                image_output_dir, doc_context, image_data
            )
            image_counter += slide_images
        
        logger.info(
            f"Processed {len(image_data)} images "
            f"(skipped {self._duplicates_skipped} duplicates)"
        )
        
        return image_data
    
    def _extract_images_from_shapes(
        self,
        shapes,
        slide_num: int,
        start_counter: int,
        seen_hashes: set,
        image_output_dir: str,
        doc_context: str,
        image_data: Dict
    ) -> int:
        """Extract images from shapes recursively"""
        images_found = 0
        
        for shape in shapes:
            # Picture shape
            if shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
                images_found += 1
                image_idx = start_counter + images_found
                
                try:
                    # Get image data
                    image = shape.image
                    image_bytes = image.blob
                    
                    # Check for duplicates using perceptual hash
                    img_hash = self.image_cache.compute_hash(image_bytes)
                    if img_hash in seen_hashes:
                        logger.debug(f"  ⏭ Skipping duplicate image on slide {slide_num}")
                        self._duplicates_skipped += 1
                        continue
                    seen_hashes.add(img_hash)
                    
                    # Get cached description or generate new one
                    cached_desc = self.image_cache.get(image_bytes)
                    
                    if cached_desc:
                        description = cached_desc
                        from_cache = True
                        success = True
                    else:
                        # Generate AI description
                        if self.image_service.is_available():
                            base64_image = base64.b64encode(image_bytes).decode('utf-8')
                            success, description = self.image_service.describe_image_from_base64(
                                base64_image,
                                image.content_type,
                                context=doc_context,
                                page_number=slide_num
                            )
                            from_cache = False
                            
                            if success:
                                self.image_cache.put(image_bytes, description)
                        else:
                            success = False
                            description = self.image_service.generate_fallback_description(
                                f"slide_{slide_num}_image",
                                page_number=slide_num,
                                image_type="slide image"
                            )
                            from_cache = False
                    
                    # Determine format
                    ext = self._get_image_extension(image.content_type)
                    
                    # Save image with consistent naming
                    filename = f"slide_{slide_num:03d}_image_{image_idx:03d}.{ext}"
                    output_path = os.path.join(image_output_dir, filename)
                    
                    with open(output_path, 'wb') as f:
                        f.write(image_bytes)
                    
                    # Store image data
                    image_data[image_idx] = {
                        'extract_idx': image_idx,
                        'filename': filename,
                        'slide_number': slide_num,
                        'page_number': slide_num,
                        'format': ext,
                        'description': description,
                        'success': success,
                        'from_cache': from_cache,
                        'content_type': image.content_type,
                        'size_bytes': len(image_bytes)
                    }
                    
                    cache_str = "💾" if from_cache else "✓"
                    logger.info(f"  {cache_str} Image {image_idx} -> {filename} (slide {slide_num})")
                    
                except Exception as e:
                    logger.warning(f"  ⚠ Error processing image on slide {slide_num}: {e}")
                    continue
            
            # Grouped shapes - recurse
            elif hasattr(shape, 'shapes'):
                nested_images = self._extract_images_from_shapes(
                    shape.shapes, slide_num, start_counter + images_found,
                    seen_hashes, image_output_dir, doc_context, image_data
                )
                images_found += nested_images
        
        return images_found
    
    def _build_markdown_from_slides(
        self,
        prs: Presentation,
        total_slides: int,
        image_data: Dict[int, Dict],
        pptx_stem: str
    ) -> str:
        """
        Build markdown by processing each slide separately
        
        Matches PDF extractor structure with proper page markers
        """
        result_lines = []
        
        # Header
        result_lines.append(f"# Document: {pptx_stem}")
        result_lines.append(f"**Total Pages**: {total_slides}")
        result_lines.append(f"**Format**: PowerPoint")
        result_lines.append("")
        
        # Build mapping: slide_number -> list of images on that slide
        images_by_slide = {}
        for img_idx, img_info in sorted(image_data.items()):
            slide_num = img_info['slide_number']
            if slide_num not in images_by_slide:
                images_by_slide[slide_num] = []
            images_by_slide[slide_num].append(img_info)
        
        # Process each slide
        for slide_num, slide in enumerate(prs.slides, start=1):
            # Add slide/page marker
            result_lines.append(f"## Page {slide_num}\n")
            
            # Slide title
            title = self._get_slide_title(slide)
            if title:
                result_lines.append(f"### {title}\n")
            
            # Get images for this slide
            slide_images = images_by_slide.get(slide_num, [])
            image_counter_on_slide = 0
            
            # Extract content from shapes
            slide_content = self._extract_slide_content(
                slide, slide_images, image_counter_on_slide
            )
            
            result_lines.extend(slide_content)
            result_lines.append("")  # Blank line after slide
        
        return '\n'.join(result_lines)
    
    def _extract_slide_content(
        self,
        slide,
        slide_images: List[Dict],
        image_counter: int
    ) -> List[str]:
        """Extract content from a single slide"""
        content_lines = []
        image_idx = 0
        
        # Process shapes in order
        for shape in slide.shapes:
            # Text shapes
            if hasattr(shape, 'text') and shape.text.strip():
                if shape != slide.shapes.title:  # Skip title (already added)
                    content_lines.append(shape.text.strip())
                    content_lines.append("")
            
            # Picture shapes - add description
            elif shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
                if image_idx < len(slide_images):
                    img_info = slide_images[image_idx]
                    
                    content_lines.append(f"**[Image Description]**: {img_info['description']}")
                    cache_str = "cached" if img_info.get('from_cache') else "described"
                    content_lines.append(f"*(Slide {img_info['slide_number']}, {cache_str})*")
                    content_lines.append("")
                    
                    image_idx += 1
            
            # Tables
            elif shape.shape_type == MSO_SHAPE_TYPE.TABLE:
                table_md = self._extract_table(shape)
                if table_md:
                    content_lines.append(table_md)
                    content_lines.append("")
            
            # Grouped shapes
            elif hasattr(shape, 'shapes'):
                grouped_content = self._extract_grouped_content(
                    shape.shapes, slide_images, image_idx
                )
                content_lines.extend(grouped_content['lines'])
                image_idx = grouped_content['image_idx']
        
        # Speaker notes
        if slide.has_notes_slide:
            notes_text = slide.notes_slide.notes_text_frame.text.strip()
            if notes_text:
                content_lines.append(f"**Speaker Notes**: {notes_text}")
                content_lines.append("")
        
        return content_lines
    
    def _extract_grouped_content(
        self,
        shapes,
        slide_images: List[Dict],
        start_image_idx: int
    ) -> Dict:
        """Extract content from grouped shapes"""
        lines = []
        image_idx = start_image_idx
        
        for shape in shapes:
            if hasattr(shape, 'text') and shape.text.strip():
                lines.append(shape.text.strip())
                lines.append("")
            
            elif shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
                if image_idx < len(slide_images):
                    img_info = slide_images[image_idx]
                    lines.append(f"**[Image Description]**: {img_info['description']}")
                    lines.append(f"*(Slide {img_info['slide_number']})*")
                    lines.append("")
                    image_idx += 1
        
        return {'lines': lines, 'image_idx': image_idx}
    
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
        return extensions.get(content_type, 'png')
    
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