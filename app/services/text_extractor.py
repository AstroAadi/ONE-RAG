"""
Page-by-Page PDF Extractor
Converts tables, charts, graphs to images with LLM descriptions
Preserves exact layout and reading order
FIXED: Robust error handling for problematic PDFs
"""

import os
from typing import Tuple, List, Dict, Optional
from pathlib import Path
import base64
import boto3
import fitz  # PyMuPDF
from PIL import Image
import io
import json
import logging

logger = logging.getLogger(__name__)


class PageByPagePDFExtractor:
    """
    Extract PDF page-by-page, converting all non-text elements to images
    """
    
    def __init__(self, aws_region: str = "us-east-1", aws_profile: Optional[str] = None):
        """Initialize with Bedrock for image descriptions"""
        self.aws_region = aws_region
        self.aws_profile = aws_profile
        self.bedrock_runtime = None
        
        # Initialize Bedrock
        try:
            if aws_profile:
                session = boto3.Session(profile_name=aws_profile, region_name=aws_region)
                self.bedrock_runtime = session.client('bedrock-runtime')
            else:
                self.bedrock_runtime = boto3.client('bedrock-runtime', region_name=aws_region)
        except Exception as e:
            print(f"Warning: Bedrock not available: {e}")
            self.bedrock_runtime = None
    
    @staticmethod
    def extract_text_from_document(
        file_path: str,
        file_extension: str,
        aws_region: str = "us-east-1",
        aws_profile: Optional[str] = None
    ) -> Tuple[str, bool]:
        """Main extraction entry point"""
        
        extractor = PageByPagePDFExtractor(aws_region, aws_profile)
        
        ext = file_extension.lower().lstrip('.')
        if ext == 'pdf':
            return extractor.extract_pdf_page_by_page(file_path)
        elif ext in ['docx', 'doc']:
            # DOCX handled separately (simpler)
            from docx import Document
            return extractor.extract_docx_simple(file_path)
        else:
            return "", False
    
    def extract_pdf_page_by_page(self, pdf_path: str) -> Tuple[str, bool]:
        """
        Extract PDF page by page
        Convert tables/images/charts to image files with descriptions
        """
        try:
            # Create image output directory
            image_output_dir = os.path.join(
                os.path.dirname(pdf_path),
                "extracted_images",
                Path(pdf_path).stem
            )
            os.makedirs(image_output_dir, exist_ok=True)
            
            # Open PDF
            doc = fitz.open(pdf_path)
            
            markdown_pages = []
            total_images_created = 0
            
            print(f"Processing {len(doc)} pages...")
            
            for page_num in range(len(doc)):
                try:
                    page = doc[page_num]
                    
                    print(f"\nProcessing page {page_num + 1}/{len(doc)}...")
                    
                    # Process page
                    page_markdown, images_count = self._process_page(
                        page,
                        page_num,
                        image_output_dir,
                        pdf_path
                    )
                    
                    markdown_pages.append(page_markdown)
                    total_images_created += images_count
                    
                    print(f"  ✓ Page {page_num + 1}: {images_count} images created")
                    
                except Exception as page_error:
                    print(f"  ⚠ Error processing page {page_num + 1}: {page_error}")
                    # Add a placeholder for failed page
                    markdown_pages.append(f"\n## Page {page_num + 1}\n\n[Error processing this page]\n")
                    continue
            
            doc.close()
            
            # Combine all pages
            final_markdown = "\n\n---PAGE BREAK---\n\n".join(markdown_pages)
            
            print(f"\n✓ Extraction complete: {total_images_created} total images")
            
            return final_markdown, True
            
        except Exception as e:
            print(f"Error extracting PDF: {str(e)}")
            import traceback
            traceback.print_exc()
            return "", False
    
    def _process_page(
        self,
        page,
        page_num: int,
        image_output_dir: str,
        pdf_path: str
    ) -> Tuple[str, int]:
        """
        Process a single page
        Returns: (markdown_content, images_created_count)
        """
        
        page_markdown = []
        images_created = 0
        
        # Add page header
        page_markdown.append(f"\n## Page {page_num + 1}\n")
        
        # Extract text blocks with position info
        try:
            blocks = page.get_text("dict")["blocks"]
        except Exception as e:
            logger.warning(f"Could not extract text blocks from page {page_num + 1}: {e}")
            blocks = []
        
        # Extract images - with error handling
        try:
            image_list = page.get_images(full=True)  # Use full=True for complete image info
        except Exception as e:
            logger.warning(f"Could not extract image list from page {page_num + 1}: {e}")
            image_list = []
        
        # Detect tables using heuristics
        tables = self._detect_tables_on_page(page, blocks)
        
        # Process all elements in reading order (top to bottom)
        all_elements = []
        
        # Add text blocks
        for block_idx, block in enumerate(blocks):
            if block.get("type") == 0:  # Text block
                bbox = block["bbox"]
                text = self._extract_text_from_block(block)
                if text.strip():
                    all_elements.append({
                        'type': 'text',
                        'bbox': bbox,
                        'y_position': bbox[1],  # Top Y coordinate for sorting
                        'content': text,
                        'block_idx': block_idx
                    })
        
        # Add tables
        for table_idx, table_bbox in enumerate(tables):
            all_elements.append({
                'type': 'table',
                'bbox': table_bbox,
                'y_position': table_bbox[1],
                'table_idx': table_idx
            })
        
        # Add images - with robust error handling
        for img_idx, img_info in enumerate(image_list):
            try:
                # Try to get bbox - this can fail for some PDFs
                img_bbox = None
                try:
                    img_bbox = page.get_image_bbox(img_info)
                except (ValueError, AttributeError) as e:
                    # If get_image_bbox fails, try alternative method
                    logger.debug(f"get_image_bbox failed for image {img_idx}: {e}")
                    # Use a default position at the end of page
                    img_bbox = fitz.Rect(0, 9999, 100, 10000)  # Will be sorted to end
                
                if img_bbox:
                    all_elements.append({
                        'type': 'image',
                        'bbox': img_bbox,
                        'y_position': img_bbox[1] if img_bbox else 9999,
                        'img_idx': img_idx,
                        'img_info': img_info
                    })
            except Exception as e:
                logger.warning(f"Could not process image {img_idx} metadata: {e}")
                continue
        
        # Sort by Y position (reading order: top to bottom)
        all_elements.sort(key=lambda x: x['y_position'])
        
        # Process elements in order
        for element in all_elements:
            if element['type'] == 'text':
                # Add text content
                text = element['content']
                
                # Detect if it's a heading (larger font, bold, etc.)
                if self._is_heading(text):
                    # Count heading level (simple heuristic)
                    if len(text) < 50 and text.isupper():
                        page_markdown.append(f"### {text}\n")
                    elif len(text) < 80:
                        page_markdown.append(f"#### {text}\n")
                    else:
                        page_markdown.append(f"{text}\n")
                else:
                    page_markdown.append(f"{text}\n")
            
            elif element['type'] == 'table':
                # Convert table to image
                table_bbox = element['bbox']
                
                try:
                    # Capture table as image
                    image_path = os.path.join(
                        image_output_dir,
                        f"page{page_num + 1}_table{element['table_idx'] + 1}.png"
                    )
                    
                    self._capture_region_as_image(
                        page,
                        table_bbox,
                        image_path
                    )
                    
                    # Generate description
                    description = self._generate_description(
                        image_path,
                        "table",
                        page_num + 1
                    )
                    
                    # Add to markdown
                    rel_path = os.path.relpath(image_path, start=os.path.dirname(pdf_path))
                    
                    page_markdown.append(f"\n---\n")
                    page_markdown.append(f"**[TABLE]** Table {element['table_idx'] + 1}\n\n")
                    page_markdown.append(f"IMAGE_PATH: {rel_path}\n")
                    page_markdown.append(f"PAGE: {page_num + 1}\n")
                    page_markdown.append(f"\nDESCRIPTION:\n{description}\n")
                    page_markdown.append(f"---\n\n")
                    
                    images_created += 1
                except Exception as table_error:
                    logger.warning(f"Could not process table {element['table_idx'] + 1}: {table_error}")
                    page_markdown.append(f"\n[Table {element['table_idx'] + 1} - extraction failed]\n\n")
            
            elif element['type'] == 'image':
                # Extract and save image
                try:
                    img_info = element['img_info']
                    xref = img_info[0]
                    
                    image_path = os.path.join(
                        image_output_dir,
                        f"page{page_num + 1}_image{element['img_idx'] + 1}.png"
                    )
                    
                    # Extract image from PDF - with error handling
                    try:
                        base_image = page.parent.extract_image(xref)
                        image_bytes = base_image["image"]
                        
                        # Save image
                        with open(image_path, "wb") as img_file:
                            img_file.write(image_bytes)
                        
                        # Generate description
                        description = self._generate_description(
                            image_path,
                            "image",
                            page_num + 1
                        )
                        
                        # Add to markdown
                        rel_path = os.path.relpath(image_path, start=os.path.dirname(pdf_path))
                        
                        page_markdown.append(f"\n---\n")
                        page_markdown.append(f"**[IMAGE]** Image {element['img_idx'] + 1}\n\n")
                        page_markdown.append(f"IMAGE_PATH: {rel_path}\n")
                        page_markdown.append(f"PAGE: {page_num + 1}\n")
                        page_markdown.append(f"\nDESCRIPTION:\n{description}\n")
                        page_markdown.append(f"---\n\n")
                        
                        images_created += 1
                    except Exception as extract_error:
                        logger.warning(f"Could not extract image {element['img_idx'] + 1}: {extract_error}")
                        page_markdown.append(f"\n[Image {element['img_idx'] + 1} - extraction failed]\n\n")
                        
                except Exception as img_error:
                    logger.warning(f"Error processing image element: {img_error}")
                    continue
        
        return "\n".join(page_markdown), images_created
    
    def _detect_tables_on_page(self, page, blocks) -> List[tuple]:
        """
        Detect tables using layout heuristics
        Returns list of bounding boxes
        """
        tables = []
        
        try:
            # Look for patterns: multiple aligned text blocks in grid
            # Group blocks by Y coordinate (rows)
            rows = {}
            for block in blocks:
                if block.get("type") == 0:  # Text
                    bbox = block["bbox"]
                    y = round(bbox[1], 0)  # Round Y to group rows
                    
                    if y not in rows:
                        rows[y] = []
                    rows[y].append(bbox)
            
            # Find potential table regions (multiple aligned rows)
            table_rows = []
            sorted_ys = sorted(rows.keys())
            
            for i, y in enumerate(sorted_ys):
                row_boxes = rows[y]
                
                # If row has 2+ elements, could be table
                if len(row_boxes) >= 2:
                    # Check alignment with previous rows
                    is_aligned = False
                    if i > 0:
                        prev_y = sorted_ys[i-1]
                        prev_boxes = rows[prev_y]
                        
                        # Check if X positions are similar
                        if len(prev_boxes) >= 2:
                            x_positions = sorted([b[0] for b in row_boxes])
                            prev_x_positions = sorted([b[0] for b in prev_boxes])
                            
                            # Similar X positions = aligned columns
                            if self._are_positions_aligned(x_positions, prev_x_positions):
                                is_aligned = True
                    
                    if is_aligned or len(row_boxes) >= 3:
                        table_rows.append((y, row_boxes))
            
            # Group consecutive table rows into table regions
            if table_rows:
                current_table_rows = [table_rows[0]]
                
                for i in range(1, len(table_rows)):
                    y, boxes = table_rows[i]
                    prev_y, _ = table_rows[i-1]
                    
                    # If rows are close, add to current table
                    if y - prev_y < 50:  # Within 50 points
                        current_table_rows.append((y, boxes))
                    else:
                        # End current table, start new
                        if len(current_table_rows) >= 3:  # FIXED: Require minimum 3 rows to reduce false positives
                            bbox = self._get_table_bbox(current_table_rows)
                            tables.append(bbox)
                        current_table_rows = [(y, boxes)]
                
                # Add last table
                if len(current_table_rows) >= 3:  # FIXED: Require minimum 3 rows
                    bbox = self._get_table_bbox(current_table_rows)
                    tables.append(bbox)
        
        except Exception as e:
            logger.warning(f"Table detection failed: {e}")
        
        return tables
    
    def _are_positions_aligned(self, positions1: List[float], positions2: List[float], tolerance: float = 15) -> bool:
        """
        Check if two lists of positions are aligned
        FIXED: Increased tolerance and allow partial alignment
        """
        if abs(len(positions1) - len(positions2)) > 1:  # Allow 1 column difference
            return False
        
        # Check alignment for common positions
        min_len = min(len(positions1), len(positions2))
        aligned_count = 0
        
        for i in range(min_len):
            if abs(positions1[i] - positions2[i]) < tolerance:
                aligned_count += 1
        
        # Require 70% alignment instead of 100%
        return aligned_count >= min_len * 0.7
    
    def _get_table_bbox(self, table_rows: List[tuple]) -> tuple:
        """Calculate bounding box for table region"""
        all_boxes = []
        for y, boxes in table_rows:
            all_boxes.extend(boxes)
        
        x0 = min(b[0] for b in all_boxes)
        y0 = min(b[1] for b in all_boxes)
        x1 = max(b[2] for b in all_boxes)
        y1 = max(b[3] for b in all_boxes)
        
        # Add small padding
        padding = 5
        return (x0 - padding, y0 - padding, x1 + padding, y1 + padding)
    
    def _extract_text_from_block(self, block: dict) -> str:
        """Extract text from a text block"""
        text_parts = []
        
        for line in block.get("lines", []):
            line_text = ""
            for span in line.get("spans", []):
                line_text += span.get("text", "")
            if line_text.strip():
                text_parts.append(line_text.strip())
        
        return " ".join(text_parts)
    
    def _is_heading(self, text: str) -> bool:
        """Heuristic to detect if text is a heading"""
        # Short text, all caps, or ends with colon
        return (
            len(text) < 100 and
            (text.isupper() or text.endswith(':') or text.endswith('-:'))
        )
    
    def _capture_region_as_image(
        self,
        page,
        bbox: tuple,
        output_path: str,
        zoom: float = 2.0
    ):
        """Capture a region of the page as an image"""
        
        # Create a rect from bbox
        rect = fitz.Rect(bbox)
        
        # Render with higher resolution
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, clip=rect)
        
        # Save as PNG
        pix.save(output_path)
    
    def _generate_description(
        self,
        image_path: str,
        element_type: str,
        page_num: int
    ) -> str:
        """Generate description using Bedrock"""
        
        if not self.bedrock_runtime:
            return f"{element_type.title()} from page {page_num} (AI description unavailable)"
        
        try:
            # Read image
            with open(image_path, 'rb') as f:
                image_data = base64.standard_b64encode(f.read()).decode('utf-8')
            
            # Customize prompt by type
            if element_type == 'table':
                prompt = """Describe this table in detail:
1. What data does it show? What is the table about?
2. How many rows and columns?
3. Column headers (if visible)

Be specific and detailed."""
            
            elif element_type == 'chart':
                prompt = """Describe this chart/graph:
1. Type (bar, line, pie, etc.)
2. Title and labels
3. Data being shown"""
            
            else:  # image
                prompt = """Describe this image:
1. What does it show?
2. Any text or labels visible?"""
            
            # Call Bedrock
            request_body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 300,
                "messages": [{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": image_data
                            }
                        },
                        {"type": "text", "text": prompt}
                    ]
                }]
            }
            
            response = self.bedrock_runtime.invoke_model(
                modelId="us.anthropic.claude-3-5-sonnet-20241022-v2:0",
                body=json.dumps(request_body)
            )
            
            response_body = json.loads(response['body'].read())
            
            if 'content' in response_body and response_body['content']:
                return response_body['content'][0]['text']
            
            return f"{element_type.title()} from page {page_num}"
            
        except Exception as e:
            print(f"  ⚠ Description generation failed: {e}")
            return f"{element_type.title()} from page {page_num}"
    
    def extract_docx_simple(self, file_path: str) -> Tuple[str, bool]:
        """Simple DOCX extraction"""
        try:
            from docx import Document
            
            doc = Document(file_path)
            markdown_parts = []
            
            for element in doc.element.body:
                if element.tag.endswith('p'):
                    para_obj = next((p for p in doc.paragraphs if p._element == element), None)
                    if para_obj and para_obj.text.strip():
                        text = para_obj.text.strip()
                        style = para_obj.style.name.lower()
                        
                        if 'heading' in style:
                            level = 1
                            try:
                                level = int(style.split()[-1])
                            except:
                                pass
                            markdown_parts.append(f"\n{'#' * (level + 1)} {text}\n")
                        else:
                            markdown_parts.append(text)
                
                elif element.tag.endswith('tbl'):
                    table_obj = next((t for t in doc.tables if t._element == element), None)
                    if table_obj:
                        rows = []
                        for row in table_obj.rows:
                            cells = [cell.text.strip() for cell in row.cells]
                            rows.append(cells)
                        
                        if rows:
                            # Convert to markdown table
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
            
        except Exception as e:
            print(f"DOCX extraction error: {e}")
            return "", False