import fitz  # PyMuPDF
import pdfplumber
from docx import Document
from typing import Tuple, List, Dict, Any
import re
import unicodedata

class DocumentTextExtractor:
    """
    Hybrid extractor: Uses PyMuPDF (fitz) for structure/headings and 
    pdfplumber for robust table extraction.
    """
    
    @staticmethod
    def extract_text_from_document(file_path: str, file_extension: str) -> Tuple[str, bool]:
        """Dispatch based on file type"""
        ext = file_extension.lower().strip('.')
        if ext == 'pdf':
            return DocumentTextExtractor.extract_text_from_pdf(file_path)
        elif ext in ['docx', 'doc']:
            return DocumentTextExtractor.extract_text_from_docx(file_path)
        else:
            return "", False

    # =========================================================================
    # PDF EXTRACTION LOGIC (Hybrid: fitz + pdfplumber)
    # =========================================================================

    @staticmethod
    def extract_text_from_pdf(file_path: str) -> Tuple[str, bool]:
        try:
            # 1. Analyze font sizes to detect what a "Heading" looks like
            doc_fitz = fitz.open(file_path)
            body_font_size = DocumentTextExtractor._analyze_font_size_distribution(doc_fitz)
            
            # 2. Open with pdfplumber for tables
            pdf_plumber = pdfplumber.open(file_path)
            
            full_content = []

            for page_index, page_fitz in enumerate(doc_fitz):
                # Page Header
                full_content.append(f"\n\n")
                
                # A. Extract Tables & their vertical positions (Y-coordinates) using pdfplumber
                # We use find_tables() to get both the data AND the bounding box (bbox)
                page_plumber = pdf_plumber.pages[page_index]
                found_tables = page_plumber.find_tables()
                
                page_elements = []
                
                if found_tables:
                    for table_obj in found_tables:
                        # Extract data from this specific table object
                        table_data = table_obj.extract()
                        if table_data:
                            md_table = DocumentTextExtractor._convert_table_to_markdown(table_data)
                            # bbox is (x0, top, x1, bottom)
                            page_elements.append({
                                'y0': table_obj.bbox[1],
                                'y1': table_obj.bbox[3],
                                'type': 'table',
                                'content': md_table
                            })

                # B. Extract Text Blocks using fitz
                # We ignore text that overlaps with identified table regions
                text_blocks = page_fitz.get_text("dict")["blocks"]
                
                for block in text_blocks:
                    if block['type'] != 0: # 0 is text, 1 is image
                        continue
                        
                    block_y0 = block['bbox'][1]
                    block_y1 = block['bbox'][3]
                    
                    # Check if this text block is inside a table we already extracted
                    is_inside_table = False
                    for tbl in page_elements:
                        if tbl['type'] == 'table':
                            # Check for overlap (if block is mostly inside the table vertical region)
                            # We use a slight buffer to avoid skipping headers just above tables
                            if block_y0 >= tbl['y0'] and block_y1 <= tbl['y1']:
                                is_inside_table = True
                                break
                    
                    if is_inside_table:
                        continue

                    # Process text block lines
                    block_text_lines = []
                    for line in block["lines"]:
                        for span in line["spans"]:
                            text = span["text"].strip()
                            if not text: continue
                            
                            size = span["size"]
                            # Detect formatting based on font size vs body size
                            formatted_text = DocumentTextExtractor._format_span(text, size, span["font"], body_font_size)
                            block_text_lines.append(formatted_text)

                    if block_text_lines:
                        # Join lines in the block
                        text_content = " ".join(block_text_lines).replace("  ", " ")
                        
                        # Add to elements list
                        page_elements.append({
                            'y0': block_y0,
                            'type': 'text',
                            'content': text_content
                        })

                # C. Sort all elements (tables and text) by vertical position (Y0)
                page_elements.sort(key=lambda x: x['y0'])
                
                # D. Append to final output
                for elem in page_elements:
                    full_content.append(elem['content'])
                    full_content.append("\n") # Spacing between blocks

            doc_fitz.close()
            pdf_plumber.close()
            
            return "\n".join(full_content), True

        except Exception as e:
            print(f"Error in hybrid extraction: {str(e)}")
            return "", False

    @staticmethod
    def _analyze_font_size_distribution(doc) -> float:
        """
        Scans the PDF to find the most common font size (Body Text).
        """
        font_sizes = {}
        for page in doc:
            blocks = page.get_text("dict")["blocks"]
            for b in blocks:
                if b['type'] == 0:
                    for l in b["lines"]:
                        for s in l["spans"]:
                            size = round(s["size"], 1)
                            font_sizes[size] = font_sizes.get(size, 0) + len(s["text"])
        
        if not font_sizes:
            return 11.0 # Default fallback
            
        # Return the font size with the highest character count
        return max(font_sizes, key=font_sizes.get)

    @staticmethod
    def _format_span(text: str, size: float, font_name: str, body_size: float) -> str:
        """Apply markdown based on font characteristics"""
        
        # Clean special chars
        text = DocumentTextExtractor._clean_text(text)
        
        # Heading Logic (Size based)
        if size > body_size + 6:
            return f"\n# {text}\n"
        elif size > body_size + 3:
            return f"\n## {text}\n"
        elif size > body_size + 1.5:
             # Check if it looks like a list item
             if text.strip().startswith(('•', '-', '1.')):
                 return text
             return f"\n### {text}\n"
        
        # Bold Logic (Font name based)
        if "bold" in font_name.lower() or "black" in font_name.lower():
             return f"**{text}**"
             
        return text

    @staticmethod
    def _clean_text(text: str) -> str:
        # Normalize unicode
        text = unicodedata.normalize('NFKD', text)
        return text

    # =========================================================================
    # DOCX EXTRACTION LOGIC
    # =========================================================================

    @staticmethod
    def extract_text_from_docx(file_path: str) -> Tuple[str, bool]:
        try:
            doc = Document(file_path)
            content = []
            
            for element in doc.element.body:
                # 1. Handle Paragraphs
                if element.tag.endswith('p'):
                    # Find the paragraph object
                    para_obj = None
                    for p in doc.paragraphs:
                        if p._element == element:
                            para_obj = p
                            break
                    
                    if para_obj and para_obj.text.strip():
                        # Determine style
                        style = para_obj.style.name.lower()
                        text = para_obj.text.strip()
                        
                        if 'title' in style:
                            content.append(f"# {text}")
                        elif 'heading 1' in style:
                            content.append(f"# {text}")
                        elif 'heading 2' in style:
                            content.append(f"## {text}")
                        elif 'heading 3' in style:
                            content.append(f"### {text}")
                        elif 'list' in style:
                            content.append(f"- {text}")
                        else:
                            # Check for bold runs acting as inline headers
                            if para_obj.runs and para_obj.runs[0].bold and len(para_obj.runs[0].text) < 50:
                                content.append(f"**{text}**")
                            else:
                                content.append(text)

                # 2. Handle Tables
                elif element.tag.endswith('tbl'):
                    # Find table object
                    for table in doc.tables:
                        if table._element == element:
                            rows = []
                            for row in table.rows:
                                cell_data = [cell.text.strip().replace('\n', ' ') for cell in row.cells]
                                rows.append(cell_data)
                            
                            if rows:
                                content.append(DocumentTextExtractor._convert_table_to_markdown(rows))
                            break
                            
            return "\n\n".join(content), True

        except Exception as e:
            print(f"Error extracting DOCX: {e}")
            return "", False

    # =========================================================================
    # UTILS
    # =========================================================================

    @staticmethod
    def _convert_table_to_markdown(rows: List[List[str]]) -> str:
        """Converts a list of lists into a Markdown table"""
        if not rows: return ""
        
        # Filter empty rows
        rows = [r for r in rows if any(str(c).strip() for c in r)]
        if not rows: return ""

        # Normalize column count
        max_cols = max(len(r) for r in rows)
        rows = [r + [""] * (max_cols - len(r)) for r in rows]
        
        # Header
        header = rows[0]
        md = []
        md.append("| " + " | ".join(str(c).replace("|", "/") for c in header) + " |")
        md.append("| " + " | ".join(["---"] * max_cols) + " |")
        
        # Body
        for row in rows[1:]:
            md.append("| " + " | ".join(str(c).replace("|", "/") for c in row) + " |")
            
        return "\n" + "\n".join(md) + "\n"