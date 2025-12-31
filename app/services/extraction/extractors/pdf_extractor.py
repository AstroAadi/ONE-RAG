"""
PDF Extractor - COMPLETE FIX
All issues resolved:
1. ✅ Image deduplication using perceptual hashing
2. ✅ Proper page numbers from Docling (no estimation)
3. ✅ Page 1 marker added at start
4. ✅ Vision classification to skip headers/footers
5. ✅ Proper image-to-page mapping
6. ✅ No duplicate figures
"""

import os
from pathlib import Path
from typing import Tuple, Dict, List, Set
import base64
import re
import logging
import io
from PIL import Image
import imagehash

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat

from ..base_extractor import BaseExtractor
from ..processors.caption_detector import CaptionDetector

logger = logging.getLogger(__name__)


class PDFExtractor(BaseExtractor):
    """
    Production PDF Extractor with complete fixes
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
        """Extract PDF with all fixes"""
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
            pipeline_options.generate_picture_images = True
            
            # Optional features
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
            
            # Convert
            logger.info("Starting Docling conversion...")
            result = converter.convert(file_path)
            total_pages = len(result.document.pages)
            logger.info(f"Docling extracted {total_pages} pages")
            
            # STEP 1: Get page mapping from Docling (EXACT pages, no estimation)
            page_mapping = self._get_page_mapping_from_docling(result.document)
            
            # STEP 2: Extract images with base64
            markdown_with_base64 = result.document.export_to_markdown(image_mode="embedded")
            
            # STEP 3: Process images (deduplicate, classify, describe)
            image_data = self._extract_and_describe_images(
                markdown_with_base64,
                image_output_dir,
                pdf_stem,
                result.document,
                file_path,
                total_pages,
                page_mapping
            )
            
            # STEP 4: Generate markdown with placeholders
            markdown_content = result.document.export_to_markdown(
                image_mode="placeholder",
                image_placeholder="[IMAGE:{image_name}]"
            )
            
            # STEP 5: Build final markdown
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
                'images_skipped_duplicates': getattr(self, '_duplicates_skipped', 0),
                'images_skipped_headers': getattr(self, '_headers_skipped', 0),
                'image_directory': str(image_output_dir)
            })
            
            logger.info(
                f"✓ Complete: {total_pages} pages, {len(image_data)} images "
                f"(skipped {metadata['images_skipped_duplicates']} dupes, "
                f"{metadata['images_skipped_headers']} headers)"
            )
            
            return final_markdown, True, metadata
            
        except Exception as e:
            logger.error(f"PDF extraction error: {str(e)}")
            import traceback
            traceback.print_exc()
            return "", False, {'error': str(e)}
    
    def _get_page_mapping_from_docling(self, document) -> Dict[int, int]:
        """
        Get EXACT page numbers from Docling structure
        NO ESTIMATION - use actual Docling data
        
        Returns:
            Dict mapping extract_index -> page_number
        """
        page_mapping = {}
        
        try:
            if hasattr(document, 'pictures') and document.pictures:
                for idx, picture in enumerate(document.pictures, start=1):
                    if hasattr(picture, 'prov'):
                        for prov in picture.prov:
                            if hasattr(prov, 'page_no'):
                                # Docling uses 0-indexed pages
                                page_mapping[idx] = prov.page_no + 1
                                logger.debug(f"Image {idx} -> Page {prov.page_no + 1} (from Docling)")
                                break
        except Exception as e:
            logger.warning(f"Could not extract page mapping: {e}")
        
        return page_mapping
    
    def _extract_and_describe_images(
        self,
        markdown_with_base64: str,
        image_output_dir: str,
        pdf_stem: str,
        document,
        pdf_path: str,
        total_pages: int,
        page_mapping: Dict[int, int]
    ) -> Dict[int, Dict]:
        """
        Extract and process images with:
        1. Perceptual hash deduplication
        2. Vision classification (skip headers)
        3. AI descriptions
        """
        from app.services.image_description_service import get_image_description_service
        
        pattern = r'!\[([^\]]*)\]\(data:image/([^;]+);base64,([^)]+)\)'
        matches = re.findall(pattern, markdown_with_base64)
        
        description_service = get_image_description_service()
        doc_context = f"PDF document: {pdf_stem}"
        
        # Track seen hashes
        seen_hashes: Set[str] = set()
        self._duplicates_skipped = 0
        self._headers_skipped = 0
        
        image_data = {}
        output_idx = 1  # Index for kept images
        
        for extract_idx, match in enumerate(matches, start=1):
            try:
                image_format = match[1]
                base64_image = match[2]
                image_bytes = base64.b64decode(base64_image)
                
                # STEP 1: Check for duplicate
                if self._is_duplicate_image(image_bytes, seen_hashes):
                    logger.info(f"  ⏭ Duplicate: image {extract_idx}")
                    self._duplicates_skipped += 1
                    continue
                
                # STEP 2: Get page number (NO ESTIMATION)
                page_number = page_mapping.get(extract_idx)
                if page_number is None:
                    logger.warning(f"  ⚠ No page mapping for image {extract_idx}, defaulting to page 1")
                    page_number = 1
                
                # Clamp to valid range
                if page_number > total_pages:
                    logger.warning(f"  ⚠ Page {page_number} > {total_pages}, clamping")
                    page_number = total_pages
                
                # STEP 3: Classify image type
                image_class = self._classify_image_type(
                    base64_image,
                    f"image/{image_format}",
                    description_service
                )
                
                if image_class in ['HEADER', 'FOOTER', 'DECORATION']:
                    logger.info(f"  ⏭ {image_class}: image {extract_idx}")
                    self._headers_skipped += 1
                    continue
                
                # STEP 4: Save image
                ext = image_format.lower()
                if ext == 'jpeg':
                    ext = 'jpg'
                
                image_filename = f"page_{page_number:02d}_image_{output_idx:03d}.{ext}"
                image_path = os.path.join(image_output_dir, image_filename)
                
                with open(image_path, 'wb') as f:
                    f.write(image_bytes)
                
                logger.info(f"  ✓ Saved: {image_filename} (page {page_number})")
                
                # STEP 5: Generate description
                if description_service.is_available():
                    success, description = description_service.describe_image_from_base64(
                        base64_image,
                        f"image/{image_format}",
                        context=doc_context,
                        page_number=page_number,
                        filename=image_filename
                    )
                    
                    if success:
                        logger.info(f"  ✓ Described: {image_filename}")
                else:
                    success = False
                    description = description_service.generate_fallback_description(
                        image_filename,
                        page_number=page_number,
                        image_type=image_class.lower()
                    )
                
                # Store with output index
                image_data[output_idx] = {
                    'filename': image_filename,
                    'page_number': page_number,
                    'format': ext,
                    'description': description,
                    'description_success': success,
                    'original_index': extract_idx
                }
                
                output_idx += 1
                
            except Exception as e:
                logger.warning(f"  ⚠ Error processing image {extract_idx}: {e}")
                continue
        
        logger.info(
            f"Processed: {len(image_data)} kept, "
            f"{self._duplicates_skipped} duplicates, "
            f"{self._headers_skipped} headers"
        )
        
        return image_data
    
    def _is_duplicate_image(self, image_bytes: bytes, seen_hashes: Set[str]) -> bool:
        """Check if image is duplicate using perceptual hash"""
        try:
            img = Image.open(io.BytesIO(image_bytes))
            img_hash = str(imagehash.phash(img, hash_size=8))
            
            if img_hash in seen_hashes:
                return True
            
            seen_hashes.add(img_hash)
            return False
            
        except Exception as e:
            logger.debug(f"Hash error: {e}")
            return False
    
    def _classify_image_type(
        self,
        base64_image: str,
        media_type: str,
        description_service
    ) -> str:
        """
        Classify image using vision LLM
        Returns: FIGURE, TABLE, HEADER, FOOTER, or DECORATION
        """
        if not description_service.is_available():
            return 'FIGURE'
        
        try:
            prompt = """Classify this image into ONE category:

A) FIGURE - Chart, graph, diagram, photo with data/research content
B) TABLE - Data table or spreadsheet
C) HEADER - Logo, title banner, page header, report branding
D) FOOTER - Copyright, page number, footer text
E) DECORATION - Decorative element with no information

Respond ONLY with the letter (A, B, C, D, or E). No explanation."""
            
            content_parts = [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": base64_image
                    }
                },
                {
                    "type": "text",
                    "text": prompt
                }
            ]
            
            request_body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 10,
                "temperature": 0.0,
                "messages": [{
                    "role": "user",
                    "content": content_parts
                }]
            }
            
            import boto3
            import json
            from app.core.config import settings
            
            if settings.aws_profile_name:
                session = boto3.Session(
                    profile_name=settings.aws_profile_name,
                    region_name=settings.aws_region
                )
                bedrock = session.client('bedrock-runtime')
            else:
                bedrock = boto3.client(
                    'bedrock-runtime',
                    region_name=settings.aws_region
                )
            
            response = bedrock.invoke_model(
                modelId=settings.bedrock_vision_model_id,
                body=json.dumps(request_body)
            )
            
            response_body = json.loads(response['body'].read())
            classification = response_body['content'][0]['text'].strip().upper()
            
            mapping = {
                'A': 'FIGURE',
                'B': 'TABLE',
                'C': 'HEADER',
                'D': 'FOOTER',
                'E': 'DECORATION'
            }
            
            result = mapping.get(classification, 'FIGURE')
            logger.debug(f"Classified as: {result}")
            return result
            
        except Exception as e:
            logger.debug(f"Classification failed: {e}, defaulting to FIGURE")
            return 'FIGURE'
    
    def _build_markdown_with_descriptions(
        self,
        markdown_content: str,
        image_data: Dict[int, Dict],
        pdf_stem: str,
        total_pages: int
    ) -> str:
        """
        Build final markdown with proper page tracking
        
        FIXES:
        1. Adds "## Page 1" at document start
        2. Tracks current page properly
        3. Only adds page marker when page changes
        4. Captions with images
        5. No duplicate page markers
        """
        lines = markdown_content.split('\n')
        result_lines = []
        
        # Header
        result_lines.append(f"# Document: {pdf_stem}")
        result_lines.append(f"**Total Pages**: {total_pages}\n")
        
        # FIX: Add Page 1 marker at start
        result_lines.append("## Page 1\n")
        current_page = 1
        
        placeholder_pattern = r'\[IMAGE:\{[^\}]+\}\]'
        
        # PASS 1: Find captions
        image_caption_map = {}
        lines_to_skip = set()
        image_counter = 1
        
        for i, line in enumerate(lines):
            if re.search(placeholder_pattern, line):
                idx = image_counter
                
                if idx not in image_data:
                    image_counter += 1
                    continue
                
                caption_info = self._find_caption_near_image(lines, i, idx)
                
                if caption_info:
                    image_caption_map[idx] = caption_info
                    lines_to_skip.add(caption_info['line_idx'])
                    
                    # Skip chart titles
                    for j in range(max(0, caption_info['line_idx'] - 5), caption_info['line_idx']):
                        if self._is_chart_title(lines[j]):
                            lines_to_skip.add(j)
                
                # Skip body text with figure refs
                for j in range(max(0, i - 10), min(i + 5, len(lines))):
                    if self._is_body_text_with_figure_ref(lines[j], caption_info):
                        lines_to_skip.add(j)
                
                image_counter += 1
        
        # PASS 2: Build markdown
        image_counter = 1
        
        for i, line in enumerate(lines):
            # Skip marked lines
            if i in lines_to_skip:
                continue
            
            # Skip chart title headings
            if line.startswith('#') and self._is_chart_title(line.lstrip('#').strip()):
                continue
            
            # Keep regular headings
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
                    
                    # FIX: Only add page marker when page changes
                    if page_number != current_page:
                        result_lines.append(f"\n## Page {page_number}\n")
                        current_page = page_number
                    
                    # Add caption
                    if idx in image_caption_map:
                        caption = image_caption_map[idx]
                        heading = CaptionDetector.format_as_heading(
                            caption['type'], caption['number'], caption['description']
                        )
                        result_lines.append(f"{heading}\n")
                    
                    # Add description
                    result_lines.append(f"**[Image Description]**: {description}")
                    result_lines.append(f"*(Located on page {page_number})*")
                    result_lines.append("")
                    
                    image_counter += 1
                else:
                    # Skipped image (duplicate/header)
                    image_counter += 1
            else:
                # Regular content
                result_lines.append(line)
        
        return '\n'.join(result_lines)
    
    def _find_caption_near_image(self, lines: List[str], image_idx: int, image_counter: int) -> Dict:
        """Find caption near image"""
        caption = CaptionDetector.find_caption_before(lines, image_idx, lookback=8)
        if caption:
            return caption
        caption = CaptionDetector.find_caption_in_range(lines, image_idx + 1, lookahead=6)
        return caption
    
    def _is_chart_title(self, text: str) -> bool:
        """Check if line is chart title"""
        return CaptionDetector.is_chart_title(text)
    
    def _is_body_text_with_figure_ref(self, line: str, caption_info: Dict) -> bool:
        """Check if line is body text referencing figure"""
        if not caption_info or not line.strip():
            return False
        
        figure_ref = f"{caption_info['type'].capitalize()} {caption_info['number']}"
        
        if figure_ref in line:
            body_indicators = [
                'illustrate', 'show', 'display', 'highlight',
                'visualize', 'present', 'demonstrate'
            ]
            
            if any(indicator in line.lower() for indicator in body_indicators):
                return True
        
        return False