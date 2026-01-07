"""
Enhanced DOCX/DOC Extractor - UPDATED
Matches PDF extractor pattern with image caching, duplicate detection, and consistent flow
Uses Docling when available, falls back to python-docx
"""

import os
from pathlib import Path
from typing import Tuple, Dict, List, Optional
import logging
import re
import base64
from docx import Document

try:
    from ..base_extractor import BaseExtractor
except ImportError:
    from base_extractor import BaseExtractor

logger = logging.getLogger(__name__)


class DocxExtractor(BaseExtractor):
    """
    Enhanced Word document extractor matching PDF extractor pattern
    
    Features:
    - Image caching (avoid redundant API calls)
    - Duplicate detection (skip same image multiple times)
    - Proper page numbering
    - Consistent markdown structure
    - Smart table and text extraction
    - Docling support with fallback to python-docx
    """
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        
        from app.services.image_cache_manager import get_image_cache
        from app.services.image_description_service import get_image_description_service
        
        self.image_cache = get_image_cache()
        self.image_service = get_image_description_service()
        
        # Tracking counters
        self._duplicates_skipped = 0
        
        logger.info("DOCX Extractor initialized with caching and duplicate detection")
    
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
        """Extract DOCX with proper flow matching PDF extractor"""
        try:
            is_valid, msg = self.validate_file(file_path)
            if not is_valid:
                return "", False, {'error': msg}
            
            logger.info(f"📄 Extracting: {file_path}")
            
            # Try Docling first (better structure preservation)
            docling_result = self._try_docling_extraction(file_path)
            if docling_result[1]:  # success
                logger.info("✅ Docling extraction successful")
                return docling_result
            
            # Fallback to python-docx
            logger.info("Using python-docx with caching")
            return self._extract_with_python_docx(file_path)
            
        except Exception as e:
            logger.error(f"❌ DOCX extraction error: {str(e)}")
            import traceback
            traceback.print_exc()
            return "", False, {'error': str(e)}
    
    def _try_docling_extraction(self, file_path: str) -> Tuple[str, bool, Dict]:
        """Try Docling extraction with full PDF-like features"""
        try:
            from docling.document_converter import DocumentConverter
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            
            doc_stem = Path(file_path).stem
            image_output_dir = self._create_image_directory(file_path, doc_stem)
            
            logger.info("🔄 Starting Docling conversion...")
            
            # Convert with Docling
            converter = DocumentConverter()
            result = converter.convert(file_path)
            
            total_pages = len(result.document.pages) if hasattr(result.document, 'pages') else 1
            logger.info(f"📑 Document has {total_pages} pages")
            
            # Reset counters
            self._duplicates_skipped = 0
            
            # Build content-to-page map (similar to PDF)
            content_map = self._build_content_page_map_docling(result.document)
            
            # Process images with caching and duplicate detection
            image_data = self._process_images_with_caching_docling(
                result.document,
                image_output_dir,
                doc_stem,
                file_path,
                content_map
            )
            
            # Build markdown page-by-page
            final_markdown = self._build_markdown_from_pages_docling(
                result.document,
                total_pages,
                image_data,
                doc_stem,
                content_map
            )
            
            # Gather statistics
            cache_stats = self.image_cache.get_stats()
            
            metadata = self.get_metadata(file_path)
            metadata.update({
                'extractor': 'DocxExtractor_Docling_Enhanced',
                'total_pages': total_pages,
                'images_extracted': len(image_data),
                'images_described': sum(1 for img in image_data.values() if img.get('success')),
                'images_cached': cache_stats['hits'],
                'images_skipped_duplicates': self._duplicates_skipped,
                'cache_hit_rate': cache_stats['hit_rate_percent'],
                'image_directory': str(image_output_dir),
                'extraction_method': 'docling'
            })
            
            logger.info(
                f"✅ Complete: {total_pages} pages, {len(image_data)} images "
                f"({metadata['images_cached']} cached, {self._duplicates_skipped} dupes skipped)"
            )
            
            return final_markdown, True, metadata
            
        except ImportError:
            logger.debug("Docling not available")
            return "", False, {'error': 'Docling not installed'}
        except Exception as e:
            logger.warning(f"Docling extraction failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return "", False, {'error': str(e)}
    
    def _build_content_page_map_docling(self, document) -> Dict:
        """Build map of content items to their source pages (Docling)"""
        content_map = {
            'texts': {},
            'pictures': {},
            'tables': {}
        }
        
        try:
            if hasattr(document, 'texts'):
                for idx, text in enumerate(document.texts):
                    if hasattr(text, 'prov'):
                        for prov in text.prov:
                            if hasattr(prov, 'page_no'):
                                content_map['texts'][idx] = prov.page_no + 1
                                break
            
            if hasattr(document, 'pictures'):
                for idx, picture in enumerate(document.pictures):
                    if hasattr(picture, 'prov'):
                        for prov in picture.prov:
                            if hasattr(prov, 'page_no'):
                                content_map['pictures'][idx] = prov.page_no + 1
                                break
            
            if hasattr(document, 'tables'):
                for idx, table in enumerate(document.tables):
                    if hasattr(table, 'prov'):
                        for prov in table.prov:
                            if hasattr(prov, 'page_no'):
                                content_map['tables'][idx] = prov.page_no + 1
                                break
        
        except Exception as e:
            logger.debug(f"Error building content map: {e}")
        
        logger.debug(f"Content map: {len(content_map['pictures'])} pictures mapped")
        return content_map
    
    def _process_images_with_caching_docling(
        self,
        document,
        image_output_dir: str,
        doc_stem: str,
        file_path: str,
        content_map: Dict
    ) -> Dict[int, Dict]:
        """Process images with caching and duplicate detection (Docling path)"""
        image_data = {}
        seen_hashes = set()
        
        # Export with embedded images to extract them
        markdown_with_base64 = document.export_to_markdown(image_mode="embedded")
        
        pattern = r'!\[([^\]]*)\]\(data:image/([^;]+);base64,([^)]+)\)'
        matches = re.findall(pattern, markdown_with_base64)
        
        doc_context = f"Word document: {doc_stem}"
        total_pages = len(document.pages) if hasattr(document, 'pages') else 1
        
        for extract_idx, match in enumerate(matches, start=1):
            try:
                image_format = match[1]
                base64_image = match[2]
                image_bytes = base64.b64decode(base64_image)
                
                # Get page number from content map
                picture_idx = extract_idx - 1
                page_number = content_map['pictures'].get(picture_idx, 1)
                
                logger.debug(f"Image {extract_idx} (picture_idx={picture_idx}) -> page {page_number}")
                
                # Check for duplicates
                img_hash = self.image_cache.compute_hash(image_bytes)
                if img_hash in seen_hashes:
                    logger.debug(f"  ⏭ Skipping duplicate image {extract_idx}")
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
            f"(skipped {self._duplicates_skipped} duplicates)"
        )
        
        return image_data
    
    def _build_markdown_from_pages_docling(
        self,
        document,
        total_pages: int,
        image_data: Dict[int, Dict],
        doc_stem: str,
        content_map: Dict
    ) -> str:
        """Build markdown by processing each page separately (Docling path)"""
        result_lines = []
        
        # Header
        result_lines.append(f"# Document: {doc_stem}")
        result_lines.append(f"**Total Pages**: {total_pages}")
        result_lines.append(f"**Format**: Word Document")
        result_lines.append("")
        
        # Build mapping: page_number -> list of images on that page
        images_by_page = {}
        for img_idx, img_info in sorted(image_data.items()):
            page_num = img_info['page_number']
            if page_num not in images_by_page:
                images_by_page[page_num] = []
            images_by_page[page_num].append(img_info)
        
        logger.debug(f"Images distribution: {[(p, len(imgs)) for p, imgs in images_by_page.items()]}")
        
        # Process each page individually
        for page_number in sorted(document.pages.keys()):
            # Add page marker
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
                    # Check for image placeholders
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
                            # Placeholder exists but image was filtered out (duplicate)
                            logger.debug(f"Placeholder on page {page_number} but no image data (likely filtered)")
                            result_lines.append("*(Image filtered - duplicate)*")
                            result_lines.append("")
                    else:
                        # Regular content
                        result_lines.append(line)
                
                result_lines.append("")  # Blank line after page content
                
            except Exception as e:
                logger.warning(f"Error processing page {page_number}: {e}")
                result_lines.append(f"*(Error extracting content from page {page_number})*\n")
        
        return '\n'.join(result_lines)
    
    def _extract_with_python_docx(self, file_path: str) -> Tuple[str, bool, Dict]:
        """Fallback extraction using python-docx with caching"""
        try:
            doc = Document(file_path)
            doc_stem = Path(file_path).stem
            image_output_dir = self._create_image_directory(file_path, doc_stem)
            
            # Reset counters
            self._duplicates_skipped = 0
            
            # Process images with caching
            image_data = self._process_images_python_docx(
                doc,
                image_output_dir,
                doc_stem
            )
            
            # Build markdown
            final_markdown = self._build_markdown_python_docx(
                doc,
                image_data,
                doc_stem
            )
            
            cache_stats = self.image_cache.get_stats()
            
            metadata = self.get_metadata(file_path)
            metadata.update({
                'extractor': 'DocxExtractor_PythonDocx_Enhanced',
                'total_pages': len(doc.paragraphs) // 20 + 1,  # Rough estimate
                'images_extracted': len(image_data),
                'images_described': sum(1 for img in image_data.values() if img.get('success')),
                'images_cached': cache_stats['hits'],
                'images_skipped_duplicates': self._duplicates_skipped,
                'cache_hit_rate': cache_stats['hit_rate_percent'],
                'image_directory': str(image_output_dir),
                'extraction_method': 'python-docx'
            })
            
            logger.info(f"✅ Complete with python-docx: {len(image_data)} images")
            
            return final_markdown, True, metadata
            
        except Exception as e:
            logger.error(f"python-docx extraction failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return "", False, {'error': str(e)}
    
    def _process_images_python_docx(
        self,
        doc: Document,
        image_output_dir: str,
        doc_stem: str
    ) -> Dict[int, Dict]:
        """Process images with caching (python-docx path)"""
        image_data = {}
        seen_hashes = set()
        image_counter = 0
        
        doc_context = f"Word document: {doc_stem}"
        
        # Extract from inline shapes
        for rel in doc.part.rels.values():
            if "image" in rel.target_ref:
                image_counter += 1
                
                try:
                    image_bytes = rel.target_part.blob
                    
                    # Check for duplicates
                    img_hash = self.image_cache.compute_hash(image_bytes)
                    if img_hash in seen_hashes:
                        logger.debug(f"  ⏭ Skipping duplicate image {image_counter}")
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
                        base64_image = base64.b64encode(image_bytes).decode('utf-8')
                        content_type = rel.target_part.content_type
                        
                        success, description = self.image_service.describe_image_from_base64(
                            base64_image,
                            content_type,
                            context=doc_context,
                            page_number=None  # No page info in python-docx
                        )
                        from_cache = False
                        
                        if success:
                            self.image_cache.put(image_bytes, description)
                    
                    # Determine format
                    ext = self._get_extension_from_content_type(rel.target_part.content_type)
                    
                    # Save image
                    filename = f"image_{image_counter:03d}.{ext}"
                    output_path = os.path.join(image_output_dir, filename)
                    
                    with open(output_path, 'wb') as f:
                        f.write(image_bytes)
                    
                    image_data[image_counter] = {
                        'extract_idx': image_counter,
                        'filename': filename,
                        'page_number': None,
                        'format': ext,
                        'description': description,
                        'success': success,
                        'from_cache': from_cache
                    }
                    
                    cache_str = "💾" if from_cache else "✓"
                    logger.info(f"  {cache_str} Image {image_counter} -> {filename}")
                    
                except Exception as e:
                    logger.warning(f"  ⚠ Error processing image {image_counter}: {e}")
                    continue
        
        logger.info(
            f"Processed {len(image_data)} images "
            f"(skipped {self._duplicates_skipped} duplicates)"
        )
        
        return image_data
    
    def _build_markdown_python_docx(
        self,
        doc: Document,
        image_data: Dict[int, Dict],
        doc_stem: str
    ) -> str:
        """Build markdown from python-docx extraction"""
        result_lines = []
        
        # Header
        result_lines.append(f"# Document: {doc_stem}")
        result_lines.append(f"**Format**: Word Document")
        result_lines.append("")
        
        image_counter = 1
        
        # Process paragraphs
        for para in doc.paragraphs:
            text = para.text.strip()
            
            if text:
                # Check if it's a heading
                if para.style.name.startswith('Heading'):
                    level = para.style.name.replace('Heading ', '')
                    try:
                        level_num = int(level)
                        result_lines.append(f"{'#' * level_num} {text}\n")
                    except:
                        result_lines.append(f"{text}\n")
                else:
                    result_lines.append(f"{text}\n")
            
            # Check for images in paragraph
            if hasattr(para, '_element') and hasattr(para._element, 'xpath'):
                # Check for images
                images_in_para = para._element.xpath('.//pic:pic', namespaces={
                    'pic': 'http://schemas.openxmlformats.org/drawingml/2006/picture'
                })
                
                if images_in_para and image_counter in image_data:
                    img_info = image_data[image_counter]
                    result_lines.append(f"**[Image Description]**: {img_info['description']}")
                    cache_str = "cached" if img_info.get('from_cache') else "described"
                    result_lines.append(f"*({cache_str})*")
                    result_lines.append("")
                    image_counter += 1
        
        # Process tables
        for table in doc.tables:
            table_md = self._extract_table(table)
            if table_md:
                result_lines.append(table_md)
                result_lines.append("")
        
        return '\n'.join(result_lines)
    
    def _create_image_directory(self, file_path: str, doc_stem: str) -> str:
        """Create directory for extracted images"""
        image_dir = os.path.join(
            os.path.dirname(file_path),
            "extracted_images",
            doc_stem
        )
        os.makedirs(image_dir, exist_ok=True)
        return image_dir
    
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
    
    def _get_extension_from_content_type(self, content_type: str) -> str:
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