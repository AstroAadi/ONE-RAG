"""
XLS (Legacy Excel) File Extractor
Processes older Excel format (.xls) files for RAG pipeline

Note: This extends the Excel extractor to support the older .xls format
"""

import os
from pathlib import Path
from typing import Tuple, Dict, List
import logging

try:
    from ..base_extractor import BaseExtractor
except ImportError:
    from base_extractor import BaseExtractor

logger = logging.getLogger(__name__)


class XLSExtractor(BaseExtractor):
    """
    Extractor for legacy XLS (Excel 97-2003) files
    
    Features:
    - Reads .xls binary format
    - Extracts all sheets
    - Converts to markdown tables
    - Handles formulas and formatting
    
    Requires: xlrd library for reading .xls files
    """
    
    def _get_supported_formats(self) -> List[str]:
        return ['xls', 'csv', 'xlsx', 'xlsm']
    
    def validate_file(self, file_path: str) -> Tuple[bool, str]:
        """Validate XLS file"""
        path = Path(file_path)
        
        if not path.exists():
            return False, "File does not exist"
        
        ext = path.suffix.lower().lstrip('.')
        if ext not in self.supported_formats:
            return False, "Not an XLS file"
        
        # Check file signature
        try:
            with open(file_path, 'rb') as f:
                header = f.read(8)
                # XLS files start with specific BIFF signature
                if not (header[:4] == b'\xd0\xcf\x11\xe0' or header[:2] == b'\x09\x08'):
                    return False, "Not a valid XLS file"
        except Exception as e:
            return False, f"Could not read file: {e}"
        
        return True, "Valid XLS file"
    
    def extract(self, file_path: str) -> Tuple[str, bool, Dict]:
        """Extract XLS file"""
        try:
            is_valid, msg = self.validate_file(file_path)
            if not is_valid:
                return "", False, {'error': msg}
            
            logger.info(f"Extracting XLS: {file_path}")
            
            xls_stem = Path(file_path).stem
            
            # Try different extraction methods
            extraction_result = None
            method_used = None
            
            # Method 1: xlrd
            try:
                extraction_result = self._extract_with_xlrd(file_path)
                method_used = 'xlrd'
            except ImportError:
                logger.debug("xlrd not available")
            except Exception as e:
                logger.warning(f"xlrd extraction failed: {e}")
            
            # Method 2: pandas with xlrd engine
            if not extraction_result:
                try:
                    extraction_result = self._extract_with_pandas(file_path)
                    method_used = 'pandas'
                except ImportError:
                    logger.debug("pandas not available")
                except Exception as e:
                    logger.warning(f"pandas extraction failed: {e}")
            
            if not extraction_result:
                return "", False, {
                    'error': 'No XLS parser available. Install: xlrd or pandas with xlrd'
                }
            
            sheets = extraction_result['sheets']
            
            if not sheets:
                return "", False, {'error': 'No sheets found in XLS file'}
            
            # Build markdown
            markdown_content = self._build_markdown(xls_stem, sheets)
            
            metadata = self.get_metadata(file_path)
            metadata.update({
                'total_pages': len(sheets),
                'sheet_count': len(sheets),
                'sheet_names': [s['name'] for s in sheets],
                'total_rows': sum(s.get('rows', 0) for s in sheets),
                'total_cols': sum(s.get('cols', 0) for s in sheets),
                'extraction_method': f'xls_{method_used}'
            })
            
            logger.info(f"✓ XLS extraction complete: {len(sheets)} sheets")
            
            return markdown_content, True, metadata
            
        except Exception as e:
            logger.error(f"XLS extraction error: {str(e)}")
            import traceback
            traceback.print_exc()
            return "", False, {'error': str(e)}
    
    def _extract_with_xlrd(self, file_path: str) -> Dict:
        """Extract using xlrd library"""
        import xlrd
        
        workbook = xlrd.open_workbook(file_path)
        sheets = []
        
        for sheet_name in workbook.sheet_names():
            sheet = workbook.sheet_by_name(sheet_name)
            
            # Extract data
            data = []
            for row_idx in range(sheet.nrows):
                row = []
                for col_idx in range(sheet.ncols):
                    cell = sheet.cell(row_idx, col_idx)
                    
                    # Convert cell value to string
                    if cell.ctype == xlrd.XL_CELL_DATE:
                        # Date cell
                        date_tuple = xlrd.xldate_as_tuple(cell.value, workbook.datemode)
                        value = f"{date_tuple[0]}-{date_tuple[1]:02d}-{date_tuple[2]:02d}"
                    elif cell.ctype == xlrd.XL_CELL_BOOLEAN:
                        value = "TRUE" if cell.value else "FALSE"
                    elif cell.ctype == xlrd.XL_CELL_ERROR:
                        value = "#ERROR"
                    elif cell.ctype == xlrd.XL_CELL_EMPTY:
                        value = ""
                    else:
                        value = str(cell.value)
                    
                    row.append(value)
                
                data.append(row)
            
            sheets.append({
                'name': sheet_name,
                'data': data,
                'rows': sheet.nrows,
                'cols': sheet.ncols
            })
        
        return {'sheets': sheets}
    
    def _extract_with_pandas(self, file_path: str) -> Dict:
        """Extract using pandas"""
        import pandas as pd
        
        # Read all sheets
        excel_file = pd.ExcelFile(file_path, engine='xlrd')
        sheets = []
        
        for sheet_name in excel_file.sheet_names:
            df = pd.read_excel(excel_file, sheet_name=sheet_name)
            
            # Convert to list of lists
            data = [df.columns.tolist()] + df.fillna('').values.tolist()
            
            # Convert all values to strings
            data = [[str(cell) for cell in row] for row in data]
            
            sheets.append({
                'name': sheet_name,
                'data': data,
                'rows': len(data),
                'cols': len(data[0]) if data else 0
            })
        
        return {'sheets': sheets}
    
    def _build_markdown(self, xls_stem: str, sheets: List[Dict]) -> str:
        """Build markdown from XLS sheets"""
        lines = []
        
        # Header
        lines.append(f"# Spreadsheet: {xls_stem}")
        lines.append(f"**Format**: Excel 97-2003 (XLS)")
        lines.append(f"**Total Sheets**: {len(sheets)}")
        lines.append("")
        lines.append("---\n")
        
        # Process each sheet
        for sheet_idx, sheet in enumerate(sheets, 1):
            lines.append(f"## Sheet {sheet_idx}: {sheet['name']}\n")
            lines.append(f"**Dimensions**: {sheet['rows']} rows × {sheet['cols']} columns\n")
            
            # Convert to markdown table
            data = sheet['data']
            
            if not data:
                lines.append("*(Empty sheet)*\n")
                continue
            
            # Limit preview to first 100 rows
            preview_data = data[:100]
            
            # Create markdown table
            if len(preview_data) > 0:
                # Header row
                header = preview_data[0]
                lines.append("| " + " | ".join(str(cell) for cell in header) + " |")
                lines.append("| " + " | ".join(["---"] * len(header)) + " |")
                
                # Data rows
                for row in preview_data[1:]:
                    # Pad row if needed
                    while len(row) < len(header):
                        row.append("")
                    lines.append("| " + " | ".join(str(cell) for cell in row[:len(header)]) + " |")
            
            if len(data) > 100:
                lines.append(f"\n*(... {len(data) - 100} more rows not shown)*")
            
            lines.append("")
        
        return '\n'.join(lines)
