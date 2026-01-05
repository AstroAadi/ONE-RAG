"""
Enhanced PDF Extractor - PROPER SOLUTION
Uses Docling's internal page structure to extract content page-by-page
"""

import os
from pathlib import Path
from typing import Tuple, Dict, List, Optional
import base64
import re
import logging

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat

from ..base_extractor import BaseExtractor

logger = logging.getLogger(__name__)


class PDFExtractor(BaseExtractor):
    """
    PDF Extractor with proper page-by-page extraction
    
    SOLUTION: Use Docling's page objects to get content per page
    """
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        
        from app.services.header_footer_detector import get_header_footer_detector
        from app.services.image_cache_manager import get_image_cache
        from app.services.image_description_service import get_image_description_service
        from app.services.extraction.processors.caption_detector import CaptionDetector
        
        self.header_footer_detector = get_header_footer_detector()
        self.image_cache = get_image_cache()
        self.image_service = get_image_description_service()
        self.caption_detector = CaptionDetector
        
        logger.info("PDF Extractor initialized with page-aware extraction")
    
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
        """Extract PDF with proper page structure"""
        try:
            is_valid, msg = self.validate_file(file_path)
            if not is_valid:
                return "", False, {'error': msg}
            
            pdf_stem = Path(file_path).stem
            image_output_dir = self._create_image_directory(file_path, pdf_stem)
            
            logger.info(f"📄 Extracting: {file_path}")
            
            # Detect header/footer zones
            zones = self.header_footer_detector.detect(file_path, sample_pages=5)
            if zones:
                logger.info(f"🎯 {zones}")
            
            # Extract with Docling
            result, total_pages = self._extract_with_docling(file_path)
            
            logger.info(f"📑 Document has {total_pages} pages")
            
            # Build content index: map each content item to its page
            content_to_page_map = self._build_content_page_map(result.document)
            
            # Process images with page information
            image_data = self._process_images_with_pages(
                result.document,
                image_output_dir,
                pdf_stem,
                file_path,
                zones,
                content_to_page_map
            )
            
            # Build markdown page-by-page using Docling's structure
            final_markdown = self._build_markdown_from_pages(
                result.document,
                total_pages,
                image_data,
                pdf_stem,
                zones,
                content_to_page_map
            )
            
            # Metadata
            cache_stats = self.image_cache.get_stats()
            metadata = {
                'extractor': 'PDFExtractor_PageAware',
                'total_pages': total_pages,
                'images_extracted': len(image_data),
                'images_described': sum(1 for img in image_data.values() if img.get('success')),
                'images_cached': cache_stats['hits'],
                'images_skipped_duplicates': getattr(self, '_duplicates_skipped', 0),
                'images_skipped_headers': getattr(self, '_headers_skipped', 0),
                'header_footer_detected': zones is not None,
                'cache_hit_rate': cache_stats['hit_rate_percent'],
                'image_directory': str(image_output_dir)
            }
            
            logger.info(
                f"✅ Complete: {total_pages} pages, {len(image_data)} images "
                f"({metadata['images_cached']} cached)"
            )
            
            return final_markdown, True, metadata
            
        except Exception as e:
            logger.error(f"❌ PDF extraction error: {str(e)}")
            import traceback
            traceback.print_exc()
            return "", False, {'error': str(e)}
    
    def _create_image_directory(self, file_path: str, pdf_stem: str) -> str:
        """Create directory for extracted images"""
        image_dir = os.path.join(
            os.path.dirname(file_path),
            "extracted_images",
            pdf_stem
        )
        os.makedirs(image_dir, exist_ok=True)
        return image_dir
    
    def _extract_with_docling(self, file_path: str) -> Tuple[any, int]:
        """Extract content using Docling"""
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = self.config.get('do_ocr', True)
        pipeline_options.do_table_structure = self.config.get('do_table_structure', True)
        pipeline_options.images_scale = self.config.get('images_scale', 2.0)
        pipeline_options.generate_picture_images = True
        
        try:
            pipeline_options.generate_page_images = False
            pipeline_options.artifacts_path = None
        except AttributeError:
            pass
        
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
        
        logger.info("🔄 Starting Docling conversion...")
        result = converter.convert(file_path)
        total_pages = len(result.document.pages)
        logger.info(f"✅ Docling extracted {total_pages} pages")
        
        return result, total_pages
    
    def _build_content_page_map(self, document) -> Dict:
        """
        Build a map of content items to their source pages
        
        This is the KEY to proper page-by-page extraction
        
        Returns:
            {
                'texts': {text_index: page_number},
                'pictures': {picture_index: page_number},
                'tables': {table_index: page_number}
            }
        """
        content_map = {
            'texts': {},
            'pictures': {},
            'tables': {}
        }
        
        try:
            # Map pictures to pages
            if hasattr(document, 'pictures') and document.pictures:
                for idx, picture in enumerate(document.pictures):
                    if hasattr(picture, 'prov') and picture.prov:
                        for prov in picture.prov:
                            if hasattr(prov, 'page_no'):
                                # Docling uses 0-indexed pages
                                content_map['pictures'][idx] = prov.page_no + 1
                                break
            
            # Map tables to pages
            if hasattr(document, 'tables') and document.tables:
                for idx, table in enumerate(document.tables):
                    if hasattr(table, 'prov') and table.prov:
                        for prov in table.prov:
                            if hasattr(prov, 'page_no'):
                                content_map['tables'][idx] = prov.page_no + 1
                                break
            
            # Map text items to pages (if available)
            if hasattr(document, 'texts') and document.texts:
                for idx, text in enumerate(document.texts):
                    if hasattr(text, 'prov') and text.prov:
                        for prov in text.prov:
                            if hasattr(prov, 'page_no'):
                                content_map['texts'][idx] = prov.page_no + 1
                                break
            
            logger.debug(
                f"Mapped content: {len(content_map['pictures'])} images, "
                f"{len(content_map['tables'])} tables, "
                f"{len(content_map['texts'])} text items"
            )
            
        except Exception as e:
            logger.warning(f"Error building content map: {e}")
        
        return content_map
    
    def _process_images_with_pages(
        self,
        document,
        image_output_dir: str,
        pdf_stem: str,
        pdf_path: str,
        zones,
        content_map: Dict
    ) -> Dict[int, Dict]:
        """Process images with accurate page numbers from content map"""
        try:
            markdown_with_base64 = document.export_to_markdown(image_mode="embedded")
        except Exception as e:
            logger.warning(f"Failed to export with embedded images: {e}")
            return {}
        
        pattern = r'!\[([^\]]*)\]\(data:image/([^;]+);base64,([^)]+)\)'
        matches = re.findall(pattern, markdown_with_base64)
        
        if not matches:
            return {}
        
        # Track statistics
        self._duplicates_skipped = 0
        self._headers_skipped = 0
        seen_hashes = set()
        
        # Process images
        image_data = {}
        doc_context = f"PDF document: {pdf_stem}"
        
        for extract_idx, match in enumerate(matches, start=1):
            try:
                image_format = match[1]
                base64_image = match[2]
                image_bytes = base64.b64decode(base64_image)
                
                # Get page number from content map (0-indexed in enumerate, so extract_idx-1)
                page_number = content_map['pictures'].get(extract_idx - 1, 1)
                
                # Check duplicate
                img_hash = self.image_cache.compute_hash(image_bytes)
                if img_hash in seen_hashes:
                    self._duplicates_skipped += 1
                    continue
                seen_hashes.add(img_hash)
                
                # Check header/footer
                if zones and self.header_footer_detector.is_image_in_zone(
                    pdf_path, page_number, image_bytes, zones
                ):
                    self._headers_skipped += 1
                    continue
                
                # Get description (with caching)
                cached_desc = self.image_cache.get(image_bytes)
                
                if cached_desc:
                    description = cached_desc
                    from_cache = True
                    success = True
                else:
                    success, description = self.image_service.describe_image_from_base64(
                        base64_image,
                        self._get_media_type(image_format),
                        context=doc_context,
                        page_number=page_number
                    )
                    from_cache = False
                    
                    if success:
                        self.image_cache.put(image_bytes, description)
                
                # Save image
                ext = image_format.lower()
                if ext == 'jpeg':
                    ext = 'jpg'
                
                filename = f"page_{page_number:03d}_image_{extract_idx:03d}.{ext}"
                image_path = os.path.join(image_output_dir, filename)
                
                with open(image_path, 'wb') as f:
                    f.write(image_bytes)
                
                image_data[extract_idx] = {
                    'extract_idx': extract_idx,
                    'filename': filename,
                    'page_number': page_number,
                    'format': image_format,
                    'description': description,
                    'success': success,
                    'from_cache': from_cache
                }
                
                cache_str = "💾" if from_cache else "✓"
                logger.info(f"  {cache_str} {filename} (page {page_number})")
                
            except Exception as e:
                logger.warning(f"  ⚠ Error processing image {extract_idx}: {e}")
                continue
        
        return image_data
    
    def _build_markdown_from_pages(
    self,
    document,  # This is the DoclingDocument
    total_pages: int,
    image_data: Dict[int, Dict],
    pdf_stem: str,
    zones,
    content_map: Dict
) -> str:
        """
        Build markdown by processing each page separately
        
        FIXED: Iterate over dict keys; use document.export_to_markdown with page_no filter
        """
        result_lines = []
        
        # Header
        result_lines.append(f"# Document: {pdf_stem}")
        result_lines.append(f"**Total Pages**: {total_pages}")
        if zones:
            result_lines.append("**Processing**: Header/footer filtered")
        result_lines.append("")
        
        # Get sorted page numbers (dict keys are ints like 1, 2, ...)
        page_numbers = sorted(document.pages.keys())
        
        # Process each page individually
        for page_number in page_numbers:
            # Add page marker
            result_lines.append(f"## Page {page_number}\n")
            
            try:
                # Export this specific page to markdown using the document-level method
                page_markdown = document.export_to_markdown(
                    image_mode="placeholder",
                    image_placeholder="[IMAGE_PLACEHOLDER_{image_name}]",
                    page_no=page_number,  # Filter to this page only
                    # Add other params as needed, e.g., delim="\n\n", indent=4
                )
                
                if not page_markdown.strip():
                    result_lines.append("*(No content on this page)*\n")
                    result_lines.append("")  # Blank line after page
                    continue
                
                # Process page content and insert image descriptions
                page_lines = page_markdown.split('\n')
                
                for line in page_lines:
                    # Check for image placeholders (your existing logic)
                    if '[IMAGE_PLACEHOLDER_' in line:
                        # Find matching images for this page
                        for img_idx, img_info in image_data.items():
                            if img_info['page_number'] == page_number:
                                result_lines.append(f"**[Image Description]**: {img_info['description']}")
                                cache_str = "cached" if img_info.get('from_cache') else "described"
                                result_lines.append(f"*(Page {page_number}, {cache_str})*")
                                result_lines.append("")
                                break  # Assuming one placeholder per image; adjust if multiple
                    else:
                        # Regular content
                        result_lines.append(line)
                
                result_lines.append("")  # Blank line after page content
                
            except Exception as e:
                logger.warning(f"Error processing page {page_number}: {e}")
                result_lines.append(f"*(Error extracting content from page {page_number})*")
                result_lines.append("")
        
        return '\n'.join(result_lines)
    
    def _get_media_type(self, extension: str) -> str:
        """Map extension to MIME type"""
        return {
            'jpg': 'image/jpeg',
            'jpeg': 'image/jpeg',
            'png': 'image/png',
            'gif': 'image/gif',
            'webp': 'image/webp',
            'bmp': 'image/bmp'
        }.get(extension.lower(), 'image/jpeg')
    
    def get_metadata(self, file_path: str) -> Dict:
        """Get file metadata"""
        path = Path(file_path)
        return {
            'filename': path.name,
            'size_bytes': path.stat().st_size,
            'extension': path.suffix.lower()
        }