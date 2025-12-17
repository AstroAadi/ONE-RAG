"""
Streamlined Document Text Extractor
- NO JSON file generation (just markdown)
- Treats tables/charts as images with LLM descriptions
- Stores image paths with descriptions in markdown
- Fast and efficient
"""

import os
from typing import Tuple, List, Dict, Optional
from docx import Document
import nltk
from pathlib import Path
import base64
import boto3
import concurrent.futures

# Download required NLTK data
def _download_nltk_data():
    resources = [
        ('tokenizers/punkt_tab', 'punkt_tab'),
        ('taggers/averaged_perceptron_tagger_eng', 'averaged_perceptron_tagger_eng'),
    ]
    for path, name in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name, quiet=True)

_download_nltk_data()

# Import unstructured
try:
    from unstructured.partition.pdf import partition_pdf
    import pandas as pd
except ImportError:
    raise ImportError("Install: pip install \"unstructured[all-docs]\" pandas pillow")


class DocumentTextExtractor:
    """
    Streamlined extractor - outputs only markdown with image references
    """
    
    def __init__(self, aws_region: str = "us-east-1", aws_profile: Optional[str] = None):
        """Initialize with AWS Bedrock for image/table analysis"""
        self.aws_region = aws_region
        self.aws_profile = aws_profile
        self.bedrock_runtime = None
        
        # Initialize Bedrock client
        try:
            if aws_profile:
                session = boto3.Session(profile_name=aws_profile, region_name=aws_region)
                self.bedrock_runtime = session.client('bedrock-runtime')
            else:
                self.bedrock_runtime = boto3.client('bedrock-runtime', region_name=aws_region)
        except Exception as e:
            print(f"Warning: Bedrock not available for image analysis: {e}")
            self.bedrock_runtime = None
        
    @staticmethod
    def extract_text_from_document(
        file_path: str, 
        file_extension: str,
        aws_region: str = "us-east-1",
        aws_profile: Optional[str] = None
    ) -> Tuple[str, bool]:
        """
        Extract text and return markdown only (no JSON)
        
        Returns:
            Tuple of (markdown_text, success)
        """
        extractor = DocumentTextExtractor(aws_region, aws_profile)
        
        ext = file_extension.lower().lstrip('.')
        if ext == 'pdf':
            return extractor.extract_text_from_pdf(file_path)
        elif ext in ['docx', 'doc']:
            return extractor.extract_text_from_docx(file_path)
        else:
            return "", False

    def extract_text_from_pdf(self, file_path: str) -> Tuple[str, bool]:
        """PDF extraction with fallback strategies"""
        result = self._extract_pdf_fast(file_path)  # Start with fast
        if not result[1]:
            result = self._extract_pdf_hi_res(file_path)
        if not result[1]:
            result = self._extract_pdf_basic(file_path)
        return result

    def _extract_pdf_fast(self, file_path: str) -> Tuple[str, bool]:
        """Fast strategy - good for most PDFs"""
        try:
            image_output_dir = os.path.join(
                os.path.dirname(file_path), 
                "extracted_images",
                Path(file_path).stem
            )
            os.makedirs(image_output_dir, exist_ok=True)

            elements = partition_pdf(
                filename=file_path,
                strategy="fast",
                infer_table_structure=True,
                languages=["eng"],
                extract_images_in_pdf=True,
                extract_image_block_types=["Image", "Table", "Figure"],
                extract_image_block_output_dir=image_output_dir,
            )

            return self._process_elements(elements, file_path, image_output_dir)

        except Exception as e:
            print(f"Fast extraction failed: {str(e)}")
            return "", False

    def _extract_pdf_hi_res(self, file_path: str) -> Tuple[str, bool]:
        """Hi-res strategy - for complex layouts"""
        try:
            image_output_dir = os.path.join(
                os.path.dirname(file_path), 
                "extracted_images",
                Path(file_path).stem
            )
            os.makedirs(image_output_dir, exist_ok=True)

            elements = partition_pdf(
                filename=file_path,
                strategy="hi_res",
                infer_table_structure=True,
                languages=["eng"],
                extract_images_in_pdf=True,
                extract_image_block_types=["Image", "Table", "Figure"],
                extract_image_block_output_dir=image_output_dir,
            )

            return self._process_elements(elements, file_path, image_output_dir)

        except Exception as e:
            print(f"Hi-res extraction failed: {str(e)}")
            return "", False

    def _extract_pdf_basic(self, file_path: str) -> Tuple[str, bool]:
        """Basic strategy - most compatible"""
        try:
            elements = partition_pdf(filename=file_path, strategy="auto")
            return self._process_elements(elements, file_path)
        except Exception as e:
            print(f"Basic extraction failed: {str(e)}")
            return "", False

    def _process_elements(
        self, 
        elements, 
        file_path: str,
        image_dir: Optional[str] = None
    ) -> Tuple[str, bool]:
        """
        Process elements into markdown with image references
        """
        try:
            markdown_parts = []
            current_section = None
            
            # Collect all images/tables for batch processing
            visual_elements = {}
            
            for idx, element in enumerate(elements):
                element_type = self._get_element_type(element)
                page_num = getattr(element.metadata, 'page_number', 0) if hasattr(element, 'metadata') else 0
                
                # Handle different element types
                if element_type == 'heading':
                    content = self._get_text_content(element)
                    level = self._detect_heading_level(element)
                    current_section = content
                    markdown_parts.append(f"\n{'#' * level} {content}\n")
                
                elif element_type == 'text':
                    content = self._get_text_content(element)
                    if content.strip():
                        markdown_parts.append(content)
                
                elif element_type == 'list':
                    content = self._get_text_content(element)
                    if content.strip():
                        # Format as list if not already
                        if not content.startswith(('- ', '* ', '1.')):
                            content = f"- {content}"
                        markdown_parts.append(content)
                
                elif element_type in ['table', 'image', 'figure', 'chart']:
                    # Store for batch processing
                    visual_elements[idx] = {
                        'element': element,
                        'type': element_type,
                        'page': page_num,
                        'section': current_section
                    }
            
            # Batch process visual elements
            if visual_elements and self.bedrock_runtime:
                print(f"Analyzing {len(visual_elements)} visual elements...")
                visual_descriptions = self._batch_analyze_visuals(
                    visual_elements, 
                    image_dir, 
                    file_path
                )
                
                # Insert visual descriptions in order
                for idx in sorted(visual_elements.keys()):
                    if idx in visual_descriptions:
                        markdown_parts.insert(
                            self._find_insertion_point(markdown_parts, idx),
                            visual_descriptions[idx]
                        )
            
            markdown_content = "\n\n".join(markdown_parts)
            
            if not markdown_content.strip():
                return "", False
                
            return markdown_content, True

        except Exception as e:
            print(f"Error processing elements: {str(e)}")
            return "", False

    def _batch_analyze_visuals(
        self,
        visual_elements: Dict,
        image_dir: Optional[str],
        file_path: str
    ) -> Dict[int, str]:
        """
        Batch analyze all visual elements (images, tables, charts)
        Returns markdown snippets with image references
        """
        results = {}
        
        # Process in parallel (max 3 concurrent)
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            future_to_idx = {}
            
            for idx, visual_data in visual_elements.items():
                element = visual_data['element']
                visual_type = visual_data['type']
                page = visual_data['page']
                section = visual_data['section']
                
                # Get image path
                image_path = None
                if hasattr(element, 'metadata'):
                    image_path = getattr(element.metadata, 'image_path', None)
                
                # Get initial caption/text
                caption = ""
                if hasattr(element, 'metadata'):
                    caption = getattr(element.metadata, 'caption', '')
                if not caption:
                    caption = self._get_text_content(element)
                
                # Submit for analysis
                if image_path and os.path.exists(image_path):
                    future = executor.submit(
                        self._analyze_and_format_visual,
                        image_path,
                        visual_type,
                        caption,
                        page,
                        section,
                        file_path
                    )
                    future_to_idx[future] = idx
                else:
                    # No image file - just use text content
                    results[idx] = f"\n**[{visual_type.upper()}]** {caption}\n"
            
            # Collect results
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    result = future.result(timeout=15)
                    if result:
                        results[idx] = result
                        print(f"✓ Analyzed visual {idx}")
                except Exception as e:
                    print(f"✗ Visual {idx} analysis failed: {e}")
                    # Fallback
                    visual_data = visual_elements[idx]
                    results[idx] = f"\n**[{visual_data['type'].upper()}]** (Page {visual_data['page']})\n"
        
        return results

    def _analyze_and_format_visual(
        self,
        image_path: str,
        visual_type: str,
        caption: str,
        page: int,
        section: Optional[str],
        file_path: str
    ) -> str:
        """
        Analyze visual with LLM and format as markdown with image reference
        
        Returns markdown like:
        ---
        **[TABLE/CHART/IMAGE]** Caption Here
        
        IMAGE_PATH: /path/to/image.png
        PAGE: 5
        SECTION: Financial Results
        
        DESCRIPTION: [LLM generated description]
        The table shows quarterly revenue data with 4 columns and 5 rows...
        ---
        """
        # Get LLM description
        description = self._analyze_visual_with_bedrock(image_path, visual_type)
        
        if not description:
            description = caption or f"Visual content from page {page}"
        
        # Build markdown with metadata
        relative_path = os.path.relpath(image_path, start=os.path.dirname(file_path))
        
        markdown = f"\n---\n"
        markdown += f"**[{visual_type.upper()}]** {caption or f'{visual_type.title()} from page {page}'}\n\n"
        markdown += f"IMAGE_PATH: {relative_path}\n"
        markdown += f"PAGE: {page}\n"
        if section:
            markdown += f"SECTION: {section}\n"
        markdown += f"\nDESCRIPTION:\n{description}\n"
        markdown += f"---\n"
        
        return markdown

    def _analyze_visual_with_bedrock(
        self, 
        image_path: str, 
        visual_type: str
    ) -> Optional[str]:
        """
        Use Bedrock to analyze image/table/chart
        """
        if not self.bedrock_runtime:
            return None
        
        try:
            # Read and encode image
            with open(image_path, 'rb') as f:
                image_data = base64.standard_b64encode(f.read()).decode('utf-8')
            
            # Determine media type
            ext = Path(image_path).suffix.lower()
            media_types = {
                '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
                '.png': 'image/png', '.gif': 'image/gif', '.webp': 'image/webp'
            }
            media_type = media_types.get(ext, 'image/jpeg')
            
            # Customize prompt based on visual type
            if visual_type == 'table':
                prompt = """Describe this table in detail:
1. What data does it show?
2. Column headers and row structure
3. Key values and patterns
4. Any notable trends or outliers
Be specific about numbers and labels."""
            elif visual_type in ['chart', 'figure']:
                prompt = """Describe this chart/graph:
1. Type of chart (bar, line, pie, etc.)
2. What data is being visualized?
3. Axis labels and scale
4. Key trends and insights
5. Any important data points"""
            else:  # image
                prompt = """Describe this image from a document:
1. What is shown in the image?
2. Any text, labels, or captions visible
3. Relevant details for understanding the document
4. Key visual elements"""
            
            # Call Bedrock
            request_body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 400,
                "messages": [{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": image_data
                            }
                        },
                        {"type": "text", "text": prompt}
                    ]
                }]
            }
            
            response = self.bedrock_runtime.invoke_model(
                modelId="us.anthropic.claude-3-5-sonnet-20241022-v2:0",
                body=__import__('json').dumps(request_body)
            )
            
            response_body = __import__('json').loads(response['body'].read())
            
            if 'content' in response_body and response_body['content']:
                return response_body['content'][0]['text']
            
            return None
            
        except Exception as e:
            print(f"Visual analysis error: {str(e)}")
            return None

    def _get_element_type(self, element) -> str:
        """Determine element type from category"""
        if hasattr(element, 'category'):
            category = element.category.lower()
            if 'title' in category or category == 'heading':
                return 'heading'
            elif 'table' in category:
                return 'table'
            elif 'image' in category:
                return 'image'
            elif 'figure' in category or 'figcaption' in category:
                return 'figure'
            elif 'list' in category:
                return 'list'
        return 'text'

    def _detect_heading_level(self, element) -> int:
        """Detect heading level (1-6)"""
        level = 1
        if hasattr(element, 'metadata'):
            if hasattr(element.metadata, 'emphasized_text_tag'):
                tag = element.metadata.emphasized_text_tag
                if tag and 'h' in str(tag).lower():
                    try:
                        level = int(str(tag).lower().replace('h', ''))
                    except:
                        pass
        return min(level, 6)

    def _get_text_content(self, element) -> str:
        """Extract text content"""
        if hasattr(element, 'text'):
            return element.text
        elif hasattr(element, 'to_markdown'):
            return element.to_markdown()
        return str(element)

    def _find_insertion_point(self, markdown_parts: List[str], idx: int) -> int:
        """Find appropriate insertion point for visual element"""
        # Insert after corresponding text position
        # This is a simple heuristic - you might want to improve this
        return min(idx, len(markdown_parts))

    # =========================================================================
    # DOCX EXTRACTION
    # =========================================================================

    def extract_text_from_docx(self, file_path: str) -> Tuple[str, bool]:
        """DOCX extraction - outputs markdown only"""
        try:
            doc = Document(file_path)
            markdown_parts = []
            current_section = None
            
            for element in doc.element.body:
                # Handle Paragraphs
                if element.tag.endswith('p'):
                    para_obj = next((p for p in doc.paragraphs if p._element == element), None)
                    
                    if para_obj and para_obj.text.strip():
                        style = para_obj.style.name.lower()
                        text = para_obj.text.strip()
                        
                        if 'heading' in style or 'title' in style:
                            level = 1
                            if 'heading' in style:
                                try:
                                    level = int(style.split()[-1])
                                except:
                                    level = 1
                            current_section = text
                            markdown_parts.append(f"\n{'#' * level} {text}\n")
                        
                        elif 'list' in style:
                            markdown_parts.append(f"- {text}")
                        
                        else:
                            if para_obj.runs and para_obj.runs[0].bold and len(text) < 100:
                                markdown_parts.append(f"**{text}**")
                            else:
                                markdown_parts.append(text)
                
                # Handle Tables
                elif element.tag.endswith('tbl'):
                    table_obj = next((t for t in doc.tables if t._element == element), None)
                    
                    if table_obj:
                        rows = []
                        for row in table_obj.rows:
                            cell_data = [cell.text.strip().replace('\n', ' ') for cell in row.cells]
                            rows.append(cell_data)
                        
                        if rows:
                            markdown_table = self._convert_table_to_markdown(rows)
                            markdown_parts.append(markdown_table)
            
            markdown_content = "\n\n".join(markdown_parts)
            return markdown_content, True
            
        except Exception as e:
            print(f"Error extracting DOCX: {e}")
            return "", False

    @staticmethod
    def _convert_table_to_markdown(rows: List[List[str]]) -> str:
        """Convert table to markdown"""
        if not rows:
            return ""
        
        rows = [r for r in rows if any(str(c).strip() for c in r)]
        if not rows:
            return ""

        max_cols = max(len(r) for r in rows)
        rows = [r + [""] * (max_cols - len(r)) for r in rows]
        
        header = rows[0]
        md = ["\n"]
        md.append("| " + " | ".join(str(c).replace("|", "\\|") for c in header) + " |")
        md.append("| " + " | ".join(["---"] * max_cols) + " |")
        
        for row in rows[1:]:
            md.append("| " + " | ".join(str(c).replace("|", "\\|") for c in row) + " |")
        
        md.append("")
        return "\n".join(md)