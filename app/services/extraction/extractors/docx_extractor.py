"""
DOCX Extractor - DEBUG & FIX VERSION
Added extensive logging to identify extraction failure
"""

import os
from pathlib import Path
from typing import Tuple, Dict, List, Optional
import logging
import re
import base64
from docx import Document
from docx.opc.constants import RELATIONSHIP_TYPE as RT

try:
    from ..base_extractor import BaseExtractor
except ImportError:
    from base_extractor import BaseExtractor

logger = logging.getLogger(__name__)


class DocxExtractor(BaseExtractor):
    """
    DOCX Extractor with debugging and multiple fallback strategies
    """
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        
        from app.services.image_cache_manager import get_image_cache
        from app.services.image_description_service import get_image_description_service
        
        self.image_cache = get_image_cache()
        self.image_service = get_image_description_service()
        
        self._duplicates_skipped = 0
        
        logger.info("DOCX Extractor initialized (DEBUG MODE)")
    
    def _get_supported_formats(self) -> List[str]:
        return ['docx']
    
    def validate_file(self, file_path: str) -> Tuple[bool, str]:
        """Validate DOCX file"""
        path = Path(file_path)
        
        if not path.exists():
            return False, f"File does not exist: {file_path}"
        if path.suffix.lower() != '.docx':
            return False, f"Not a Word document: {path.suffix}"
        if path.stat().st_size == 0:
            return False, "File is empty (0 bytes)"
        
        # Additional check: try to open as ZIP (DOCX is ZIP format)
        try:
            import zipfile
            if not zipfile.is_zipfile(file_path):
                return False, "File is not a valid DOCX (not a ZIP archive)"
        except Exception as e:
            logger.warning(f"Could not validate ZIP format: {e}")
        
        return True, "Valid Word document"
    
    def extract(self, file_path: str) -> Tuple[str, bool, Dict]:
        """
        Extract DOCX with extensive debugging
        """
        try:
            logger.info(f"=" * 80)
            logger.info(f"DOCX EXTRACTION DEBUG - START")
            logger.info(f"=" * 80)
            logger.info(f"File path: {file_path}")
            
            # Validate
            is_valid, msg = self.validate_file(file_path)
            if not is_valid:
                logger.error(f"❌ Validation failed: {msg}")
                return "", False, {'error': msg}
            
            logger.info(f"✅ File validation passed")
            
            file_size = Path(file_path).stat().st_size
            logger.info(f"📊 File size: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")
            
            # Strategy 1: Try Docling
            logger.info("\n" + "="*60)
            logger.info("STRATEGY 1: Trying Docling extraction...")
            logger.info("="*60)
            
            docling_result = self._try_docling_extraction_debug(file_path)
            if docling_result[1]:  # success
                logger.info("✅✅✅ Docling extraction SUCCESSFUL")
                return docling_result
            else:
                logger.warning(f"⚠️ Docling failed: {docling_result[2].get('error', 'Unknown')}")
            
            # Strategy 2: Try python-docx
            logger.info("\n" + "="*60)
            logger.info("STRATEGY 2: Trying python-docx extraction...")
            logger.info("="*60)
            
            result = self._extract_with_python_docx_debug(file_path)
            
            if result[1]:  # success
                logger.info("✅✅✅ python-docx extraction SUCCESSFUL")
            else:
                logger.error(f"❌❌❌ python-docx extraction FAILED: {result[2].get('error', 'Unknown')}")
            
            logger.info(f"=" * 80)
            logger.info(f"DOCX EXTRACTION DEBUG - END")
            logger.info(f"=" * 80)
            
            return result
            
        except Exception as e:
            logger.error(f"❌❌❌ CRITICAL ERROR in extract(): {str(e)}")
            import traceback
            traceback.print_exc()
            return "", False, {'error': str(e)}
    
    def _try_docling_extraction_debug(self, file_path: str) -> Tuple[str, bool, Dict]:
        """Try Docling with detailed logging"""
        try:
            logger.info("Attempting to import Docling...")
            from docling.document_converter import DocumentConverter
            logger.info("✅ Docling imported successfully")
            
            doc_stem = Path(file_path).stem
            logger.info(f"Document stem: {doc_stem}")
            
            # Get user_id
            user_id = Path(file_path).parent.name
            if user_id == "uploads":
                user_id = "default"
            logger.info(f"User ID: {user_id}")
            
            # Create image directory
            image_output_dir = self._create_image_directory_user_isolated(
                file_path, doc_stem, user_id
            )
            logger.info(f"Image directory: {image_output_dir}")
            
            # Convert
            logger.info("Starting Docling conversion...")
            converter = DocumentConverter()
            result = converter.convert(file_path)
            logger.info("✅ Docling conversion completed")
            
            # Check result
            if not hasattr(result, 'document'):
                logger.error("❌ Result has no 'document' attribute")
                return "", False, {'error': 'No document in result'}
            
            logger.info(f"Document object type: {type(result.document)}")
            
            # Check pages
            if hasattr(result.document, 'pages'):
                total_pages = len(result.document.pages)
                logger.info(f"✅ Document has {total_pages} pages")
            else:
                total_pages = 1
                logger.warning("⚠️ No 'pages' attribute, assuming 1 page")
            
            # Export to markdown to check content
            logger.info("Exporting to markdown...")
            try:
                markdown_text = result.document.export_to_markdown()
                text_length = len(markdown_text)
                logger.info(f"✅ Exported markdown: {text_length:,} characters")
                
                if text_length == 0:
                    logger.error("❌ Exported markdown is EMPTY!")
                    return "", False, {'error': 'Empty markdown from Docling'}
                
                # Show preview
                preview = markdown_text[:500].replace('\n', ' ')
                logger.info(f"📄 Content preview: {preview}...")
                
                # Continue with full extraction
                return self._complete_docling_extraction(
                    result, file_path, doc_stem, user_id, image_output_dir
                )
                
            except Exception as e:
                logger.error(f"❌ Error exporting markdown: {e}")
                import traceback
                traceback.print_exc()
                return "", False, {'error': f'Markdown export failed: {str(e)}'}
            
        except ImportError as e:
            logger.info(f"⚠️ Docling not available: {e}")
            return "", False, {'error': 'Docling not installed'}
        except Exception as e:
            logger.error(f"❌ Docling error: {str(e)}")
            import traceback
            traceback.print_exc()
            return "", False, {'error': str(e)}
    
    def _complete_docling_extraction(
        self,
        result,
        file_path: str,
        doc_stem: str,
        user_id: str,
        image_output_dir: str
    ) -> Tuple[str, bool, Dict]:
        """Complete the Docling extraction process"""
        try:
            total_pages = len(result.document.pages) if hasattr(result.document, 'pages') else 1
            
            # Reset counters
            self._duplicates_skipped = 0
            
            # Build content map
            content_map = self._build_content_page_map_docling(result.document)
            logger.info(f"Content map: {len(content_map['pictures'])} pictures, {len(content_map['tables'])} tables")
            
            # Process images
            logger.info("Processing images...")
            image_data = self._process_images_docling(
                result.document,
                image_output_dir,
                doc_stem,
                doc_stem,  # file_id
                content_map
            )
            logger.info(f"✅ Processed {len(image_data)} images")
            
            # Build markdown
            logger.info("Building final markdown...")
            final_markdown = self._build_markdown_docling(
                result.document,
                total_pages,
                image_data,
                doc_stem,
                content_map
            )
            
            logger.info(f"✅ Final markdown: {len(final_markdown):,} characters")
            
            # Metadata
            cache_stats = self.image_cache.get_stats()
            
            metadata = self.get_metadata(file_path)
            metadata.update({
                'extractor': 'DocxExtractor_Docling',
                'total_pages': total_pages,
                'images_extracted': len(image_data),
                'images_described': sum(1 for img in image_data.values() if img.get('success')),
                'images_cached': cache_stats['hits'],
                'images_skipped_duplicates': self._duplicates_skipped,
                'cache_hit_rate': cache_stats['hit_rate_percent'],
                'image_directory': str(image_output_dir),
                'extraction_method': 'docling'
            })
            
            return final_markdown, True, metadata
            
        except Exception as e:
            logger.error(f"❌ Error completing Docling extraction: {e}")
            import traceback
            traceback.print_exc()
            return "", False, {'error': str(e)}
    
    def _extract_with_python_docx_debug(self, file_path: str) -> Tuple[str, bool, Dict]:
        """
        Extract with python-docx and extensive debugging
        """
        try:
            logger.info("Loading document with python-docx...")
            doc = Document(file_path)
            logger.info("✅ Document loaded")
            
            # Count elements
            num_paragraphs = len(doc.paragraphs)
            num_tables = len(doc.tables)
            
            logger.info(f"📊 Document structure:")
            logger.info(f"  - Paragraphs: {num_paragraphs}")
            logger.info(f"  - Tables: {num_tables}")
            
            if num_paragraphs == 0 and num_tables == 0:
                logger.error("❌ Document has NO paragraphs and NO tables!")
                return "", False, {'error': 'Empty document - no content found'}
            
            # Check paragraph content
            non_empty_paragraphs = 0
            total_text_length = 0
            
            for i, para in enumerate(doc.paragraphs[:10]):  # Check first 10
                text = para.text.strip()
                if text:
                    non_empty_paragraphs += 1
                    total_text_length += len(text)
                    if i < 3:  # Show first 3
                        preview = text[:100].replace('\n', ' ')
                        logger.info(f"  Para {i+1}: {preview}...")
            
            logger.info(f"  - Non-empty paragraphs (first 10): {non_empty_paragraphs}/10")
            logger.info(f"  - Total text length (first 10): {total_text_length:,} chars")
            
            if non_empty_paragraphs == 0:
                logger.warning("⚠️ First 10 paragraphs are ALL EMPTY!")
                # Check if there's ANY text in the document
                all_text = '\n'.join([p.text for p in doc.paragraphs if p.text.strip()])
                if not all_text:
                    logger.error("❌ NO text found in ENTIRE document!")
                    return "", False, {'error': 'No text content in document'}
                else:
                    logger.info(f"✅ Found text in later paragraphs: {len(all_text):,} chars")
            
            # Continue with extraction
            doc_stem = Path(file_path).stem
            user_id = Path(file_path).parent.name
            if user_id == "uploads":
                user_id = "default"
            
            image_output_dir = self._create_image_directory_user_isolated(
                file_path, doc_stem, user_id
            )
            
            logger.info("Processing images...")
            self._duplicates_skipped = 0
            image_data = self._process_images_python_docx(
                doc,
                image_output_dir,
                doc_stem
            )
            logger.info(f"✅ Processed {len(image_data)} images")
            
            logger.info("Building markdown...")
            final_markdown = self._build_markdown_python_docx(
                doc,
                image_data,
                doc_stem
            )
            
            logger.info(f"✅ Final markdown: {len(final_markdown):,} characters")
            
            if len(final_markdown) < 100:
                logger.error(f"❌ Final markdown too short: {len(final_markdown)} chars")
                logger.error(f"Content: {final_markdown}")
                return "", False, {'error': 'Generated markdown is too short'}
            
            # Metadata
            cache_stats = self.image_cache.get_stats()
            estimated_pages = max(1, num_paragraphs // 20)
            
            metadata = self.get_metadata(file_path)
            metadata.update({
                'extractor': 'DocxExtractor_PythonDocx',
                'total_pages': estimated_pages,
                'paragraphs': num_paragraphs,
                'tables': num_tables,
                'images_extracted': len(image_data),
                'images_described': sum(1 for img in image_data.values() if img.get('success')),
                'images_cached': cache_stats['hits'],
                'images_skipped_duplicates': self._duplicates_skipped,
                'cache_hit_rate': cache_stats['hit_rate_percent'],
                'image_directory': str(image_output_dir),
                'extraction_method': 'python-docx'
            })
            
            logger.info("✅ python-docx extraction successful")
            return final_markdown, True, metadata
            
        except Exception as e:
            logger.error(f"❌ python-docx extraction error: {str(e)}")
            import traceback
            traceback.print_exc()
            return "", False, {'error': str(e)}
    
    def _create_image_directory_user_isolated(
        self, 
        file_path: str, 
        doc_stem: str,
        user_id: str
    ) -> str:
        """Create user-isolated image directory"""
        base_dir = Path(file_path).parent
        image_dir = base_dir / "extracted_images" / doc_stem
        image_dir.mkdir(parents=True, exist_ok=True)
        return str(image_dir)
    
    def _process_images_python_docx(
        self,
        doc: Document,
        image_output_dir: str,
        doc_stem: str
    ) -> Dict[int, Dict]:
        """Process images from python-docx"""
        image_data = {}
        seen_hashes = set()
        image_counter = 0
        
        doc_context = f"Word document: {doc_stem}"
        
        total_paragraphs = len(doc.paragraphs)
        paragraphs_per_page = 20
        current_paragraph = 0
        
        # Extract from relationships
        for rel_id, rel in doc.part.rels.items():
            if "image" in rel.target_ref:
                image_counter += 1
                
                try:
                    image_bytes = rel.target_part.blob
                    
                    # Check duplicates
                    img_hash = self.image_cache.compute_hash(image_bytes)
                    if img_hash in seen_hashes:
                        self._duplicates_skipped += 1
                        continue
                    seen_hashes.add(img_hash)
                    
                    # Estimate page
                    estimated_page = max(1, (current_paragraph // paragraphs_per_page) + 1)
                    current_paragraph += 1
                    
                    # Get description
                    cached_desc = self.image_cache.get(image_bytes)
                    
                    if cached_desc:
                        description = cached_desc
                        from_cache = True
                        success = True
                    else:
                        base64_image = base64.b64encode(image_bytes).decode('utf-8')
                        content_type = rel.target_part.content_type
                        
                        success, description = self.image_service.describe_image_from_base64(
                            base64_image,
                            content_type,
                            context=doc_context,
                            page_number=estimated_page
                        )
                        from_cache = False
                        
                        if success:
                            self.image_cache.put(image_bytes, description)
                    
                    ext = self._get_extension_from_content_type(rel.target_part.content_type)
                    filename = f"page_{estimated_page:03d}_image_{image_counter:03d}.{ext}"
                    output_path = os.path.join(image_output_dir, filename)
                    
                    with open(output_path, 'wb') as f:
                        f.write(image_bytes)
                    
                    image_data[image_counter] = {
                        'extract_idx': image_counter,
                        'filename': filename,
                        'page_number': estimated_page,
                        'format': ext,
                        'description': description,
                        'success': success,
                        'from_cache': from_cache
                    }
                    
                except Exception as e:
                    logger.warning(f"Error processing image {image_counter}: {e}")
                    continue
        
        return image_data
    
    def _build_markdown_python_docx(
        self,
        doc: Document,
        image_data: Dict[int, Dict],
        doc_stem: str
    ) -> str:
        """Build markdown from python-docx"""
        result_lines = []
        
        # Header
        result_lines.append(f"# Document: {doc_stem}")
        result_lines.append(f"**Format**: Word Document")
        result_lines.append(f"**Paragraphs**: {len(doc.paragraphs)}")
        result_lines.append(f"**Tables**: {len(doc.tables)}")
        result_lines.append("")
        
        used_images = set()
        image_counter = 1
        
        paragraphs_per_page = 20
        current_para = 0
        current_page = 1
        
        result_lines.append(f"## Page {current_page}\n")
        
        # Process paragraphs
        for para in doc.paragraphs:
            text = para.text.strip()
            
            # Page break
            if current_para > 0 and current_para % paragraphs_per_page == 0:
                current_page += 1
                result_lines.append(f"\n## Page {current_page}\n")
            
            current_para += 1
            
            # Add text
            if text:
                if para.style.name.startswith('Heading'):
                    level = para.style.name.replace('Heading ', '')
                    try:
                        level_num = int(level)
                        result_lines.append(f"{'#' * level_num} {text}\n")
                    except:
                        result_lines.append(f"{text}\n")
                else:
                    result_lines.append(f"{text}\n")
            
            # Check for images
            if hasattr(para, '_element'):
                images_in_para = para._element.xpath(
                    './/pic:pic',
                    namespaces={'pic': 'http://schemas.openxmlformats.org/drawingml/2006/picture'}
                )
                
                if images_in_para and image_counter in image_data and image_counter not in used_images:
                    img_info = image_data[image_counter]
                    result_lines.append(f"\n**[Image Description]**: {img_info['description']}")
                    cache_str = "cached" if img_info.get('from_cache') else "described"
                    result_lines.append(f"*(Page ~{img_info['page_number']}, {cache_str})*\n")
                    used_images.add(image_counter)
                    image_counter += 1
        
        # Process tables
        for table_idx, table in enumerate(doc.tables, start=1):
            table_md = self._extract_table(table)
            if table_md:
                result_lines.append(f"\n**Table {table_idx}**:\n")
                result_lines.append(table_md)
                result_lines.append("")
        
        return '\n'.join(result_lines)
    
    def _build_content_page_map_docling(self, document) -> Dict:
        """Build content map for Docling"""
        content_map = {'texts': {}, 'pictures': {}, 'tables': {}}
        
        try:
            if hasattr(document, 'pictures'):
                for idx, picture in enumerate(document.pictures):
                    if hasattr(picture, 'prov'):
                        for prov in picture.prov:
                            if hasattr(prov, 'page_no'):
                                content_map['pictures'][idx] = prov.page_no + 1
                                break
        except Exception as e:
            logger.debug(f"Error building content map: {e}")
        
        return content_map
    
    def _process_images_docling(
        self,
        document,
        image_output_dir: str,
        doc_stem: str,
        file_id: str,
        content_map: Dict
    ) -> Dict[int, Dict]:
        """Process images from Docling"""
        image_data = {}
        seen_hashes = set()
        
        markdown_with_base64 = document.export_to_markdown(image_mode="embedded")
        
        pattern = r'!\[([^\]]*)\]\(data:image/([^;]+);base64,([^)]+)\)'
        matches = re.findall(pattern, markdown_with_base64)
        
        doc_context = f"Word document: {doc_stem}"
        
        for extract_idx, match in enumerate(matches, start=1):
            try:
                image_format = match[1]
                base64_image = match[2]
                image_bytes = base64.b64decode(base64_image)
                
                picture_idx = extract_idx - 1
                page_number = content_map['pictures'].get(picture_idx, 1)
                
                img_hash = self.image_cache.compute_hash(image_bytes)
                if img_hash in seen_hashes:
                    self._duplicates_skipped += 1
                    continue
                seen_hashes.add(img_hash)
                
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
                
                ext = image_format.lower()
                if ext == 'jpeg':
                    ext = 'jpg'
                
                filename = f"page_{page_number:03d}_image_{extract_idx:03d}.{ext}"
                image_path = os.path.join(image_output_dir, filename)
                
                with open(image_path, 'wb') as f:
                    f.write(image_bytes)
                
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
                
            except Exception as e:
                logger.warning(f"Error processing image {extract_idx}: {e}")
                continue
        
        return image_data
    
    def _build_markdown_docling(
        self,
        document,
        total_pages: int,
        image_data: Dict[int, Dict],
        doc_stem: str,
        content_map: Dict
    ) -> str:
        """Build markdown from Docling"""
        result_lines = []
        
        result_lines.append(f"# Document: {doc_stem}")
        result_lines.append(f"**Total Pages**: {total_pages if total_pages > 0 else 'N/A (DOCX)'}")
        result_lines.append(f"**Format**: Word Document")
        result_lines.append("")
        
        images_by_page = {}
        for img_idx, img_info in sorted(image_data.items()):
            page_num = img_info['page_number']
            if page_num not in images_by_page:
                images_by_page[page_num] = []
            images_by_page[page_num].append(img_info)
        
        # CRITICAL FIX: Handle DOCX with 0 pages (Docling doesn't paginate DOCX)
        if total_pages == 0 or not hasattr(document, 'pages') or len(document.pages) == 0:
            logger.info("DOCX has no pages - exporting entire document as single unit")
            
            try:
                # Export entire document at once
                full_markdown = document.export_to_markdown(
                    image_mode="placeholder",
                    image_placeholder="[IMAGE_PLACEHOLDER_{image_name}]"
                )
                
                if not full_markdown.strip():
                    result_lines.append("*(No content extracted)*")
                    return '\n'.join(result_lines)
                
                # Process all images in order
                all_images = sorted(image_data.values(), key=lambda x: x['extract_idx'])
                image_counter = 0
                
                for line in full_markdown.split('\n'):
                    if '[IMAGE_PLACEHOLDER_' in line and image_counter < len(all_images):
                        img_info = all_images[image_counter]
                        result_lines.append(f"\n**[Image Description]**: {img_info['description']}")
                        cache_str = "cached" if img_info.get('from_cache') else "described"
                        result_lines.append(f"*(Image {image_counter + 1}, {cache_str})*\n")
                        image_counter += 1
                    else:
                        result_lines.append(line)
                
                logger.info(f"Exported full DOCX document: {len(full_markdown)} chars")
                return '\n'.join(result_lines)
                
            except Exception as e:
                logger.error(f"Error exporting full DOCX: {e}")
                result_lines.append(f"*(Error extracting content: {str(e)})*")
                return '\n'.join(result_lines)
        
        # Original logic for paginated documents (PDFs)
        # FIX: document.pages is a list, not a dict - iterate by index
        for page_idx in range(len(document.pages)):
            page_number = page_idx + 1  # Convert to 1-based page number
            result_lines.append(f"## Page {page_number}\n")
            
            try:
                # Use 0-based index for Docling API
                page_markdown = document.export_to_markdown(
                    page_no=page_idx,  # Use 0-based index
                    image_mode="placeholder",
                    image_placeholder="[IMAGE_PLACEHOLDER_{image_name}]"
                )
                
                if not page_markdown.strip():
                    result_lines.append("*(No content on this page)*\n")
                    continue
                
                page_images = images_by_page.get(page_number, [])
                image_counter = 0
                
                for line in page_markdown.split('\n'):
                    if '[IMAGE_PLACEHOLDER_' in line:
                        if image_counter < len(page_images):
                            img_info = page_images[image_counter]
                            result_lines.append(f"**[Image Description]**: {img_info['description']}")
                            cache_str = "cached" if img_info.get('from_cache') else "described"
                            result_lines.append(f"*(Page {page_number}, {cache_str})*")
                            result_lines.append("")
                            image_counter += 1
                    else:
                        result_lines.append(line)
                
                result_lines.append("")
                
            except Exception as e:
                logger.warning(f"Error processing page {page_number}: {e}")
                result_lines.append(f"*(Error extracting content from page {page_number})*\n")
        
        return '\n'.join(result_lines)
    
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
        """Get extension from content type"""
        extensions = {
            'image/jpeg': 'jpg',
            'image/png': 'png',
            'image/gif': 'gif',
            'image/bmp': 'bmp',
            'image/tiff': 'tiff',
            'image/webp': 'webp'
        }
        return extensions.get(content_type, 'png')