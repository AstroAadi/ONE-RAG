"""
Docling-Based Document Extractor with Page Tracking
Clean, simple implementation - no unused parameters
Images named with page numbers: page_05_image_001.png
"""

import os
from typing import Tuple, Dict, List
from pathlib import Path
import base64
import logging
import re

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat

logger = logging.getLogger(__name__)


class DoclingDocumentExtractor:
    """
    Clean Docling-based extractor with proper page tracking
    Images named as: page_XX_image_YYY.png
    """
    
    def __init__(self):
        """Initialize extractor"""
        logger.info("Docling extractor initialized")
    
    @staticmethod
    def extract_text_from_document(
        file_path: str,
        file_extension: str,
        **kwargs  # Ignore any extra parameters for compatibility
    ) -> Tuple[str, bool]:
        """
        Main extraction entry point
        
        Args:
            file_path: Path to document
            file_extension: File extension (pdf, docx, doc)
            
        Returns:
            Tuple of (extracted_markdown_text, success_boolean)
        """
        
        extractor = DoclingDocumentExtractor()
        
        ext = file_extension.lower().lstrip('.')
        if ext == 'pdf':
            return extractor.extract_pdf_with_docling(file_path)
        elif ext in ['docx', 'doc']:
            return extractor.extract_docx_with_docling(file_path)
        else:
            logger.error(f"Unsupported file type: {ext}")
            return "", False
    
    def extract_pdf_with_docling(self, pdf_path: str) -> Tuple[str, bool]:
        """Extract PDF with Docling - images named with page numbers"""
        try:
            pdf_stem = Path(pdf_path).stem
            image_output_dir = os.path.join(
                os.path.dirname(pdf_path),
                "extracted_images",
                pdf_stem
            )
            os.makedirs(image_output_dir, exist_ok=True)
            
            logger.info(f"Extracting: {pdf_path}")
            logger.info(f"Images: {image_output_dir}")
            
            # Configure Docling
            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_ocr = True
            pipeline_options.do_table_structure = True
            pipeline_options.images_scale = 2.0
            pipeline_options.generate_picture_images = True
            
            converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
                }
            )
            
            logger.info("Converting PDF...")
            result = converter.convert(pdf_path)
            
            total_pages = len(result.document.pages)
            logger.info(f"Converted {total_pages} pages")
            
            # Extract images
            logger.info("Extracting images...")
            markdown_with_base64 = result.document.export_to_markdown(image_mode="embedded")
            
            image_mapping = self._extract_and_save_images(
                markdown_with_base64,
                image_output_dir,
                pdf_stem,
                result.document
            )
            
            # Generate markdown
            logger.info("Generating markdown...")
            markdown_content = result.document.export_to_markdown(
                image_mode="placeholder",
                image_placeholder="[IMAGE:{image_name}]"
            )
            
            # Build final markdown with page info
            final_markdown = self._build_markdown_with_pages(
                markdown_content,
                image_mapping,
                pdf_stem,
                total_pages
            )
            
            logger.info(f"✓ Complete: {len(image_mapping)} images extracted")
            
            return final_markdown, True
            
        except Exception as e:
            logger.error(f"Extraction error: {str(e)}")
            import traceback
            traceback.print_exc()
            return "", False
    
    def extract_docx_with_docling(self, docx_path: str) -> Tuple[str, bool]:
        """Extract DOCX with Docling"""
        try:
            logger.info(f"Extracting DOCX: {docx_path}")
            
            converter = DocumentConverter()
            result = converter.convert(docx_path)
            markdown_content = result.document.export_to_markdown()
            
            logger.info("✓ DOCX extraction complete")
            return markdown_content, True
            
        except Exception as e:
            logger.error(f"DOCX extraction error: {str(e)}")
            
            # Fallback to python-docx
            try:
                from docx import Document
                
                doc = Document(docx_path)
                markdown_parts = []
                
                for para in doc.paragraphs:
                    if para.text.strip():
                        text = para.text.strip()
                        style = para.style.name.lower()
                        
                        if 'heading' in style:
                            level = 2
                            try:
                                level = int(style.split()[-1]) + 1
                            except:
                                pass
                            markdown_parts.append(f"\n{'#' * level} {text}\n")
                        else:
                            markdown_parts.append(text)
                
                # Handle tables
                for table in doc.tables:
                    rows = [[cell.text.strip() for cell in row.cells] for row in table.rows]
                    
                    if rows:
                        max_cols = max(len(r) for r in rows)
                        rows = [r + [''] * (max_cols - len(r)) for r in rows]
                        
                        md_table = ["\n"]
                        md_table.append("| " + " | ".join(rows[0]) + " |")
                        md_table.append("| " + " | ".join(["---"] * max_cols) + " |")
                        for row in rows[1:]:
                            md_table.append("| " + " | ".join(row) + " |")
                        md_table.append("")
                        
                        markdown_parts.append("\n".join(md_table))
                
                return "\n\n".join(markdown_parts), True
                
            except Exception as fallback_error:
                logger.error(f"Fallback extraction failed: {fallback_error}")
                return "", False
    
    def _extract_and_save_images(
        self,
        markdown_with_base64: str,
        image_output_dir: str,
        pdf_stem: str,
        document
    ) -> Dict[int, Dict]:
        """Extract and save images with page numbers"""
        pattern = r'!\[([^\]]*)\]\(data:image/([^;]+);base64,([^)]+)\)'
        matches = re.findall(pattern, markdown_with_base64)
        image_mapping = {}
        
        # Get page hints from Docling
        page_hints = self._get_page_hints(document)
        
        for idx, match in enumerate(matches, start=1):
            try:
                image_format = match[1]
                base64_data = match[2]
                
                # Decode image
                image_data = base64.b64decode(base64_data)
                
                ext = image_format.lower()
                if ext == 'jpeg':
                    ext = 'jpg'
                
                # Determine page number
                page_number = page_hints.get(idx, self._estimate_page(idx, len(matches), len(document.pages)))
                
                # Save with page number in filename
                image_filename = f"page_{page_number:02d}_image_{idx:03d}.{ext}"
                image_path = os.path.join(image_output_dir, image_filename)
                
                with open(image_path, 'wb') as f:
                    f.write(image_data)
                
                image_mapping[idx] = {
                    'filename': image_filename,
                    'page_number': page_number,
                    'format': ext
                }
                
                logger.info(f"  ✓ Saved: {image_filename} (page {page_number})")
                
            except Exception as e:
                logger.warning(f"  ⚠ Could not save image {idx}: {e}")
                continue
        
        logger.info(f"Total images: {len(image_mapping)}")
        return image_mapping
    
    def _get_page_hints(self, document) -> Dict[int, int]:
        """Try to get page numbers from Docling document structure"""
        page_hints = {}
        
        try:
            if hasattr(document, 'pictures') and document.pictures:
                for idx, picture in enumerate(document.pictures, start=1):
                    if hasattr(picture, 'prov'):
                        for prov in picture.prov:
                            if hasattr(prov, 'page_no'):
                                page_hints[idx] = prov.page_no + 1
                                break
                    
                    if idx not in page_hints and hasattr(picture, 'data'):
                        if hasattr(picture.data, 'page'):
                            page_hints[idx] = picture.data.page
        except Exception as e:
            logger.debug(f"Could not extract page hints: {e}")
        
        return page_hints
    
    def _estimate_page(self, image_idx: int, total_images: int, total_pages: int) -> int:
        """Estimate page number based on image position"""
        if total_images == 0 or total_pages == 0:
            return 1
        
        estimated_page = int((image_idx / total_images) * total_pages) + 1
        return max(1, min(estimated_page, total_pages))
    
    def _build_markdown_with_pages(
        self,
        markdown_content: str,
        image_mapping: Dict[int, Dict],
        pdf_stem: str,
        total_pages: int
    ) -> str:
        """Build final markdown with page numbers for each image"""
        lines = markdown_content.split('\n')
        result_lines = []
        
        # Add header
        result_lines.append(f"# Document: {pdf_stem}")
        result_lines.append(f"**Total Pages**: {total_pages}\n")
        
        placeholder_pattern = r'\[IMAGE:\{[^\}]+\}\]'
        image_counter = [1]
        
        for line in lines:
            # Track headings
            if line.startswith('#'):
                result_lines.append(line)
                continue
            
            # Check for image placeholder
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
                    
                    # Add image path and page tag
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


# Backwards compatibility
PageByPagePDFExtractor = DoclingDocumentExtractor


def extract_text_from_document(file_path: str, file_extension: str, **kwargs) -> Tuple[str, bool]:
    """Module-level function for backwards compatibility"""
    return DoclingDocumentExtractor.extract_text_from_document(file_path, file_extension, **kwargs)