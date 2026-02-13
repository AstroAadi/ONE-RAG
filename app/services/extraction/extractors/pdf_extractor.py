"""
Enhanced PDF Extractor with Docling + Unstructured Fallback
=============================================================

COMPLETE IMPLEMENTATION with Direct Markdown Table Extraction

Strategy:
1. Get page count from Docling and Unstructured (BEFORE extraction)
2. Compare with PyMuPDF page count (ground truth)
3. If Docling matches → Use Docling (fast, better layout)
4. If Unstructured matches → Use Unstructured (robust fallback)
5. Both methods: Extract images, describe with Vision LLM, preserve layout
6. Tables: Extract directly as Markdown (no HTML conversion!)

Author: Complete Dual-Extractor Solution with Direct Markdown Tables
Date: 2026-01-21
"""

import os
from pathlib import Path
from typing import Tuple, Dict, List, Optional
import base64
import re
import logging
import shutil

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
from app.core.config import settings 
from ..base_extractor import BaseExtractor

logger = logging.getLogger(__name__)

# Import PyMuPDF for page count validation
try:
    import fitz
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    logger.warning("⚠️  PyMuPDF not available - page count validation disabled")

# Import Unstructured for fallback
try:
    from unstructured.partition.pdf import partition_pdf
    UNSTRUCTURED_AVAILABLE = True
    logger.info("✓ Unstructured library available")
except ImportError:
    UNSTRUCTURED_AVAILABLE = False
    logger.warning("⚠️  Unstructured not available - fallback disabled")
    logger.warning("   Install with: pip install 'unstructured[all-docs]'")


class PDFExtractor(BaseExtractor):
    """
    PDF Extractor with Docling + Unstructured dual-extraction
    
    Features:
    - Page count validation BEFORE extraction
    - Automatic fallback to Unstructured when Docling fails
    - Full Vision LLM image description pipeline
    - Direct Markdown table extraction (no HTML conversion!)
    - Layout-aware extraction
    - Header/footer filtering
    - Image caching
    """
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        
        from app.services.header_footer_detector import get_header_footer_detector
        from app.services.image_cache_manager import get_image_cache
        from app.services.image_description_service import get_image_description_service
        # from app.services.extraction.processors.caption_detector import CaptionDetector
        
        self.header_footer_detector = get_header_footer_detector()
        self.image_cache = get_image_cache()
        self.image_service = get_image_description_service()
        # self.caption_detector = CaptionDetector
        
        # Statistics tracking
        self.extraction_stats = {
            'total_extractions': 0,
            'docling_used': 0,
            'unstructured_used': 0,
            'validation_failures': 0
        }
        
        logger.info("📄 PDF Extractor initialized with Docling + Unstructured fallback")
    
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
        """
        Extract PDF with dual-extractor validation
        
        Process:
        1. Get page counts from both extractors (BEFORE full extraction)
        2. Compare with PyMuPDF count (ground truth)
        3. Use appropriate extractor
        4. Maintain full image description pipeline
        """
        self.extraction_stats['total_extractions'] += 1
        
        try:
            is_valid, msg = self.validate_file(file_path)
            if not is_valid:
                return "", False, {'error': msg}
            
            pdf_stem = Path(file_path).stem
            image_output_dir = self._create_image_directory(file_path, pdf_stem)
            
            logger.info(f"📄 Extracting: {file_path}")
            
            # Step 1: Get actual page count (ground truth)
            actual_page_count = self._get_pymupdf_page_count(file_path)
            
            if actual_page_count == 0:
                logger.error("❌ Could not determine PDF page count")
                return "", False, {'error': 'Could not read PDF'}
            
            logger.info(f"📑 PDF has {actual_page_count} pages (PyMuPDF - ground truth)")
            
            # Step 2: Quick page count checks
            docling_page_count = self._get_docling_page_count(file_path)
            logger.info(f"📊 Docling detected {docling_page_count} pages")
            
            unstructured_page_count = 0
            if UNSTRUCTURED_AVAILABLE:
                unstructured_page_count = self._get_unstructured_page_count(file_path)
                logger.info(f"📊 Unstructured detected {unstructured_page_count} pages")
            
            # Detect header/footer zones
            zones = self.header_footer_detector.detect(file_path, sample_pages=5)
            if zones:
                logger.info(f"🎯 {zones}")
            
            # Step 3: Decide which extractor to use
            if docling_page_count == actual_page_count:
                # ✅ Docling page count matches - use Docling
                logger.info(
                    f"✅ Validation passed: Docling={docling_page_count} == Actual={actual_page_count}\n"
                    f"→ Using Docling extraction (full pipeline)"
                )
                
                final_markdown, metadata = self._extract_with_docling_full(
                    file_path,
                    pdf_stem,
                    image_output_dir,
                    actual_page_count,
                    zones
                )
                
                metadata['extraction_method'] = 'docling'
                metadata['validation_passed'] = True
                metadata['page_count_match'] = True
                
                self.extraction_stats['docling_used'] += 1
                
                return final_markdown, True, metadata
            
            elif UNSTRUCTURED_AVAILABLE and unstructured_page_count == actual_page_count:
                # ✅ Unstructured page count matches - use Unstructured
                logger.warning(
                    f"⚠️  Docling validation failed: Docling={docling_page_count} != Actual={actual_page_count}\n"
                    f"✅ Unstructured validation passed: Unstructured={unstructured_page_count} == Actual={actual_page_count}\n"
                    f"→ Using Unstructured extraction (fallback)"
                )
                
                final_markdown, metadata = self._extract_with_unstructured_full(
                    file_path,
                    pdf_stem,
                    image_output_dir,
                    actual_page_count,
                    zones
                )
                
                metadata['extraction_method'] = 'unstructured_fallback'
                metadata['validation_passed'] = True
                metadata['page_count_match'] = True
                
                self.extraction_stats['unstructured_used'] += 1
                
                return final_markdown, True, metadata
            
            else:
                # ❌ Both failed validation
                logger.error(
                    f"❌ Both extractors failed validation:\n"
                    f"   - Actual (PyMuPDF): {actual_page_count} pages\n"
                    f"   - Docling: {docling_page_count} pages\n"
                    f"   - Unstructured: {unstructured_page_count} pages"
                )
                
                self.extraction_stats['validation_failures'] += 1
                
                return "", False, {
                    'error': 'Both Docling and Unstructured failed page count validation',
                    'actual_pages': actual_page_count,
                    'docling_pages': docling_page_count,
                    'unstructured_pages': unstructured_page_count
                }
            
        except Exception as e:
            logger.error(f"❌ PDF extraction error: {str(e)}")
            import traceback
            traceback.print_exc()
            return "", False, {'error': str(e)}
    
    # ============================================================================
    # PAGE COUNT METHODS
    # ============================================================================
    
    def _get_pymupdf_page_count(self, file_path: str) -> int:
        """Get actual page count using PyMuPDF (ground truth) - FAST"""
        if not PYMUPDF_AVAILABLE:
            logger.warning("PyMuPDF not available")
            return 0
        
        try:
            doc = fitz.open(file_path)
            page_count = len(doc)
            doc.close()
            logger.debug(f"✓ PyMuPDF page count: {page_count}")
            return page_count
        except Exception as e:
            logger.error(f"Error getting PyMuPDF page count: {e}")
            return 0
    
    def _get_docling_page_count(self, file_path: str) -> int:
        """Get Docling page count (quick check - minimal processing)"""
        try:
            # Use minimal options for speed
            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_ocr = False  # Faster - don't need OCR for page count
            pipeline_options.do_table_structure = False  # Faster
            pipeline_options.generate_picture_images = False  # Faster
            
            # OCR disabled for page count - no need to configure GPU
            
            converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
                }
            )
            
            result = converter.convert(file_path)
            page_count = len(result.document.pages)
            logger.debug(f"✓ Docling page count: {page_count}")
            return page_count
            
        except Exception as e:
            logger.warning(f"Docling page count failed: {e}")
            return 0
    
    def _get_unstructured_page_count(self, file_path: str) -> int:
        """Get Unstructured page count (quick check - fast strategy)"""
        if not UNSTRUCTURED_AVAILABLE:
            return 0
        
        try:
            # Use fast strategy for quick page count
            elements = partition_pdf(
                filename=file_path,
                extract_images_in_pdf=False,  # Faster
                infer_table_structure=False,  # Faster
                strategy="fast",  # Fastest strategy
            )
            
            # Get unique page numbers
            page_numbers = {
                elem.metadata.page_number 
                for elem in elements 
                if elem.metadata.page_number
            }
            
            page_count = max(page_numbers) if page_numbers else 0
            logger.debug(f"✓ Unstructured page count: {page_count}")
            return page_count
            
        except Exception as e:
            logger.warning(f"Unstructured page count failed: {e}")
            return 0
    
    # ============================================================================
    # DOCLING EXTRACTION (FULL PIPELINE)
    # ============================================================================
    
    def _extract_with_docling_full(
        self,
        file_path: str,
        pdf_stem: str,
        image_output_dir: str,
        total_pages: int,
        zones: any
    ) -> Tuple[str, Dict]:
        """
        Full Docling extraction with complete pipeline
        
        Maintains your existing functionality:
        - Layout-aware extraction
        - Image extraction and description
        - Vision LLM integration
        - Header/footer filtering
        """
        logger.info("🔄 Full Docling extraction...")
        
        # Extract with Docling (full options)
        result, extracted_pages = self._extract_with_docling(file_path)
        
        logger.info(f"📑 Docling extracted {extracted_pages} pages")
        
        # Build content page map
        content_map = self._build_content_page_map(result.document)
        
        logger.info(f"Content map: {len(content_map['pictures'])} pictures mapped")
        
        # Process images WITH Vision LLM
        image_data = self._process_images_with_pages(
            result.document,
            image_output_dir,
            pdf_stem,
            file_path,
            zones,
            content_map
        )
        
        # Build markdown
        final_markdown = self._build_markdown_from_pages(
            result.document,
            total_pages,
            image_data,
            pdf_stem,
            zones,
            content_map
        )
        
        # Metadata
        cache_stats = self.image_cache.get_stats()
        metadata = {
            'extractor': 'Docling',
            'total_pages': total_pages,
            'images_extracted': len(image_data),
            'images_described': sum(1 for img in image_data.values() if img.get('success')),
            'images_cached': cache_stats['hits'],
            'images_skipped_duplicates': getattr(self, '_duplicates_skipped', 0),
            'images_skipped_headers': getattr(self, '_headers_skipped', 0),
            'header_footer_detected': zones is not None,
            'cache_hit_rate': cache_stats['hit_rate_percent'],
            'image_directory': str(image_output_dir),
        }
        
        logger.info(
            f"✅ Docling extraction complete: {total_pages} pages, "
            f"{len(image_data)} images ({metadata['images_cached']} cached)"
        )
        
        return final_markdown, metadata
    
    # ============================================================================
    # UNSTRUCTURED EXTRACTION (FALLBACK PIPELINE WITH DIRECT MARKDOWN TABLES)
    # ============================================================================
    
    def _extract_with_unstructured_full(
        self,
        file_path: str,
        pdf_stem: str,
        image_output_dir: str,
        total_pages: int,
        zones: any
    ) -> Tuple[str, Dict]:
        """
        Full Unstructured extraction with DIRECT Markdown tables
        
        Key Feature:
        - Tables extracted directly as Markdown (no HTML conversion!)
        - Uses elem.metadata.text_as_markdown for clean, proper table format
        """
        logger.info("🔄 Full Unstructured extraction (fallback)...")
        
        # Create temp directory for extracted images
        temp_image_dir = Path(image_output_dir) / "unstructured_temp"
        temp_image_dir.mkdir(exist_ok=True)
        
        try:
            # Extract with Unstructured (hi_res strategy)
            elements = partition_pdf(
                filename=file_path,
                extract_images_in_pdf=True,  # Extract images
                infer_table_structure=True,  # Infer tables
                strategy="hi_res",  # High resolution
                extract_image_block_output_dir=str(temp_image_dir)
            )
            
            logger.info(f"Unstructured extracted {len(elements)} elements")
            
            # Build markdown with image descriptions
            result_lines = []
            
            # Header
            result_lines.append(f"# Document: {pdf_stem}")
            result_lines.append(f"**Total Pages**: {total_pages}")
            result_lines.append("**Extraction Method**: Unstructured (fallback)")
            if zones:
                result_lines.append("**Processing**: Header/footer filtering attempted")
            result_lines.append("")
            
            # Track images and statistics
            image_data = {}
            table_counter = 0
            image_counter = 0
            current_page = 0
            
            # Reset counters
            self._duplicates_skipped = 0
            self._headers_skipped = 0
            seen_hashes = set()
            
            # Process elements by page
            for elem in elements:
                # Track current page
                page_num = elem.metadata.page_number if elem.metadata.page_number else 1
                
                if page_num != current_page:
                    current_page = page_num
                    result_lines.append(f"\n## Page {page_num}\n")
                
                # Handle different element types
                if elem.category == "Image":
                    # Image element - describe it
                    image_counter += 1
                    
                    if elem.metadata.image_path and Path(elem.metadata.image_path).exists():
                        image_path = elem.metadata.image_path
                        
                        # Read image
                        with open(image_path, 'rb') as f:
                            image_bytes = f.read()
                        
                        # Check duplicate
                        img_hash = self.image_cache.compute_hash(image_bytes)
                        if img_hash in seen_hashes:
                            logger.debug(f"  ⏭ Skipping duplicate image {image_counter}")
                            self._duplicates_skipped += 1
                            continue
                        seen_hashes.add(img_hash)
                        
                        # Check cache
                        cached_desc = self.image_cache.get(image_bytes)
                        
                        if cached_desc:
                            description = cached_desc
                            from_cache = True
                            success = True
                        else:
                            # Describe with Vision LLM
                            success, description = self._describe_image_file(
                                image_path,
                                pdf_stem,
                                page_num
                            )
                            from_cache = False
                            
                            if success:
                                self.image_cache.put(image_bytes, description)
                        
                        # Save to final location
                        final_filename = f"page_{page_num:03d}_image_{image_counter:03d}.png"
                        final_path = Path(image_output_dir) / final_filename
                        
                        shutil.copy(image_path, final_path)
                        
                        # Add to markdown
                        result_lines.append(f"**[Image Description]**: {description}")
                        cache_str = "cached" if from_cache else "described"
                        result_lines.append(f"*(Page {page_num}, {cache_str})*")
                        result_lines.append("")
                        
                        # Track
                        image_data[image_counter] = {
                            'filename': final_filename,
                            'page_number': page_num,
                            'description': description,
                            'success': success,
                            'from_cache': from_cache
                        }
                        
                        cache_icon = "💾" if from_cache else "✓"
                        logger.info(f"  {cache_icon} Image {image_counter} -> {final_filename} (page {page_num})")
                    
                elif elem.category == "Table":
                    # ============================================================
                    # TABLE HANDLING - DIRECT MARKDOWN EXTRACTION (NO CONVERSION!)
                    # ============================================================
                    table_counter += 1
                    
                    # Try to get table as Markdown DIRECTLY!
                    if hasattr(elem.metadata, 'text_as_markdown') and elem.metadata.text_as_markdown:
                        # SUCCESS - Direct Markdown table! ✅
                        result_lines.append(f"**Table {table_counter}** (Page {page_num}):")
                        result_lines.append("")
                        result_lines.append(elem.metadata.text_as_markdown)
                        result_lines.append("")
                        logger.info(f"  ✓ Table {table_counter} -> Markdown (page {page_num})")
                    
                    # Fallback: Try HTML (if markdown not available)
                    elif hasattr(elem.metadata, 'text_as_html') and elem.metadata.text_as_html:
                        result_lines.append(f"**Table {table_counter}** (Page {page_num}):")
                        result_lines.append("")
                        result_lines.append(elem.metadata.text_as_html)
                        result_lines.append("")
                        logger.warning(f"  ⚠️  Table {table_counter} -> HTML fallback (page {page_num})")
                    
                    # Last resort: Plain text
                    else:
                        result_lines.append(f"**Table {table_counter}** (Page {page_num}):")
                        result_lines.append("")
                        result_lines.append(str(elem))
                        result_lines.append("")
                        logger.warning(f"  ⚠️  Table {table_counter} -> Plain text fallback (page {page_num})")
                
                else:
                    # Text, Title, etc.
                    result_lines.append(str(elem))
                    result_lines.append("")
            
            # Cleanup temp directory
            try:
                shutil.rmtree(temp_image_dir)
            except Exception as e:
                logger.warning(f"Could not cleanup temp dir: {e}")
            
            # Combine markdown
            final_markdown = '\n'.join(result_lines)
            
            # Metadata
            cache_stats = self.image_cache.get_stats()
            metadata = {
                'extractor': 'Unstructured',
                'total_pages': total_pages,
                'images_extracted': len(image_data),
                'images_described': sum(1 for img in image_data.values() if img.get('success')),
                'images_cached': cache_stats['hits'],
                'images_skipped_duplicates': self._duplicates_skipped,
                'images_skipped_headers': self._headers_skipped,
                'tables_extracted': table_counter,
                'header_footer_detected': zones is not None,
                'cache_hit_rate': cache_stats['hit_rate_percent'],
                'image_directory': str(image_output_dir),
            }
            
            logger.info(
                f"✅ Unstructured extraction complete: {total_pages} pages, "
                f"{len(image_data)} images, {table_counter} tables"
            )
            
            return final_markdown, metadata
            
        except Exception as e:
            logger.error(f"Unstructured extraction failed: {e}")
            raise
    
    def _describe_image_file(
        self,
        image_path: str,
        doc_context: str,
        page_number: int
    ) -> Tuple[bool, str]:
        """
        Describe image file using Vision LLM
        
        Converts image to base64 and calls ImageDescriptionService
        """
        try:
            # Read image and convert to base64
            with open(image_path, 'rb') as f:
                image_bytes = f.read()
            
            base64_image = base64.b64encode(image_bytes).decode('utf-8')
            
            # Determine media type from extension
            ext = Path(image_path).suffix.lower()
            media_type_map = {
                '.png': 'image/png',
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.gif': 'image/gif',
                '.webp': 'image/webp',
                '.bmp': 'image/bmp'
            }
            media_type = media_type_map.get(ext, 'image/png')
            
            # Describe with Vision LLM (your existing service!)
            success, description = self.image_service.describe_image_from_base64(
                base64_image,
                media_type,
                context=f"PDF document: {doc_context}",
                page_number=page_number
            )
            
            return success, description
            
        except Exception as e:
            logger.error(f"Image description failed: {e}")
            return False, "[Image description unavailable]"
    
    # ============================================================================
    # HELPER METHODS (FROM ORIGINAL pdf_extractor.py)
    # ============================================================================
    
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
        """Extract content using Docling with GPU-enabled OCR"""
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = self.config.get('do_ocr', True)
        pipeline_options.do_table_structure = self.config.get('do_table_structure', True)
        pipeline_options.images_scale = self.config.get('images_scale', 2.0)
        pipeline_options.generate_picture_images = True
        
        # ⚡ ENABLE GPU FOR OCR
        # OCR configuration removed - use Docling defaults
        # Custom ocr_options can cause compatibility issues with different Docling versions
        
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
        
        logger.debug("🔄 Starting Docling conversion (GPU-enabled)...")
        result = converter.convert(file_path)
        total_pages = len(result.document.pages)
        logger.debug(f"✅ Docling converted {total_pages} pages")
        
        return result, total_pages
    
    def _build_content_page_map(self, document) -> Dict:
        """
        Build a map of content items to their source pages
        
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
                                content_map['pictures'][idx] = prov.page_no
                                break
            
            # Map tables to pages
            if hasattr(document, 'tables') and document.tables:
                for idx, table in enumerate(document.tables):
                    if hasattr(table, 'prov') and table.prov:
                        for prov in table.prov:
                            if hasattr(prov, 'page_no'):
                                content_map['tables'][idx] = prov.page_no
                                break
            
            # Map text items to pages
            if hasattr(document, 'texts') and document.texts:
                for idx, text in enumerate(document.texts):
                    if hasattr(text, 'prov') and text.prov:
                        for prov in text.prov:
                            if hasattr(prov, 'page_no'):
                                content_map['texts'][idx] = prov.page_no
                                break
            
            logger.debug(
                f"Mapped content: {len(content_map['pictures'])} pictures, "
                f"{len(content_map['tables'])} tables, "
                f"{len(content_map['texts'])} texts"
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
        """
        Process images with accurate page numbers from content map
        
        Extracts images, describes with Vision LLM, handles caching
        """
        try:
            markdown_with_base64 = document.export_to_markdown(image_mode="embedded")
        except Exception as e:
            logger.warning(f"Failed to export with embedded images: {e}")
            return {}
        
        pattern = r'!\[([^\]]*)\]\(data:image/([^;]+);base64,([^)]+)\)'
        matches = re.findall(pattern, markdown_with_base64)
        
        if not matches:
            logger.info("No images found in markdown")
            return {}
        
        logger.info(f"Found {len(matches)} images in markdown")
        
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
                
                # Get page number from content map
                picture_idx = extract_idx - 1
                page_number = content_map['pictures'].get(picture_idx, 1)
                
                logger.debug(f"Image {extract_idx} -> page {page_number}")
                
                # Check duplicate
                img_hash = self.image_cache.compute_hash(image_bytes)
                if img_hash in seen_hashes:
                    logger.debug(f"  ⏭ Skipping duplicate image {extract_idx}")
                    self._duplicates_skipped += 1
                    continue
                seen_hashes.add(img_hash)
                
                # Check header/footer
                if zones and self.header_footer_detector.is_image_in_zone(
                    pdf_path, page_number, image_bytes, zones
                ):
                    logger.debug(f"  ⏭ Skipping header/footer image {extract_idx}")
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
                
                # Store image data
                image_data[extract_idx] = {
                    'extract_idx': extract_idx,
                    'picture_idx': picture_idx,
                    'filename': filename,
                    'page_number': page_number,
                    'format': image_format,
                    'description': description,
                    'success': success,
                    'from_cache': from_cache
                }
                
                cache_str = "💾" if from_cache else "✓"
                logger.info(f"  {cache_str} Image {extract_idx} -> {filename} (page {page_number})")
                
            except Exception as e:
                logger.warning(f"  ⚠ Error processing image {extract_idx}: {e}")
                continue
        
        logger.info(
            f"Processed {len(image_data)} images "
            f"(skipped {self._duplicates_skipped} dupes, {self._headers_skipped} headers)"
        )
        
        return image_data
    
    def _build_markdown_from_pages(
        self,
        document,
        total_pages: int,
        image_data: Dict[int, Dict],
        pdf_stem: str,
        zones,
        content_map: Dict
    ) -> str:
        """
        Build markdown by processing each page separately
        
        Properly matches image placeholders to descriptions
        """
        result_lines = []
        
        # Header
        result_lines.append(f"# Document: {pdf_stem}")
        result_lines.append(f"**Total Pages**: {total_pages}")
        if zones:
            result_lines.append("**Processing**: Header/footer filtered")
        result_lines.append("")
        
        # Build images by page mapping
        images_by_page = {}
        for img_idx, img_info in sorted(image_data.items()):
            page_num = img_info['page_number']
            if page_num not in images_by_page:
                images_by_page[page_num] = []
            images_by_page[page_num].append(img_info)
        
        # Process each page
        for page_number in sorted(document.pages.keys()):
            result_lines.append(f"## Page {page_number}\n")
            
            try:
                # Export this specific page to markdown
                page_markdown = document.export_to_markdown(
                    page_no=page_number,
                    image_mode="placeholder",
                    image_placeholder="[IMAGE_PLACEHOLDER_{image_name}]"
                )
                
                if not page_markdown.strip():
                    result_lines.append("*(No content on this page)*\n")
                    continue
                
                # Get images for this page
                page_images = images_by_page.get(page_number, [])
                image_counter_on_page = 0
                
                # Process page content and insert image descriptions
                page_lines = page_markdown.split('\n')
                
                for line in page_lines:
                    if '[IMAGE_PLACEHOLDER_' in line:
                        # Match placeholder to the next image on this page
                        if image_counter_on_page < len(page_images):
                            img_info = page_images[image_counter_on_page]
                            
                            # Add image description
                            result_lines.append(f"**[Image Description]**: {img_info['description']}")
                            cache_str = "cached" if img_info.get('from_cache') else "described"
                            result_lines.append(f"*(Page {page_number}, {cache_str})*")
                            result_lines.append("")
                            
                            image_counter_on_page += 1
                        else:
                            # Placeholder exists but image was filtered
                            result_lines.append("*(Image filtered - duplicate or header/footer)*")
                            result_lines.append("")
                    else:
                        # Regular content
                        result_lines.append(line)
                
                result_lines.append("")  # Blank line after page
                
            except Exception as e:
                logger.warning(f"Error processing page {page_number}: {e}")
                result_lines.append(f"*(Error extracting content from page {page_number})*\n")
        
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
    
    def get_extraction_stats(self) -> Dict:
        """Get extraction statistics"""
        total = self.extraction_stats['total_extractions']
        
        if total == 0:
            return self.extraction_stats
        
        return {
            **self.extraction_stats,
            'docling_usage_rate': round(
                self.extraction_stats['docling_used'] / total * 100, 1
            ),
            'unstructured_usage_rate': round(
                self.extraction_stats['unstructured_used'] / total * 100, 1
            ),
            'validation_failure_rate': round(
                self.extraction_stats['validation_failures'] / total * 100, 1
            )
        }