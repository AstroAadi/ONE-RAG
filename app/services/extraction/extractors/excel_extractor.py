"""
Excel File Extractor - XLSX/XLS/CSV
Matches PDF extractor pattern with image caching and consistent flow
Handles spreadsheets with multiple sheets, images, charts, and data
"""

import os
from pathlib import Path
from typing import Tuple, Dict, List, Optional
import logging
import re
import base64
import pandas as pd
from openpyxl import load_workbook
from openpyxl.drawing.image import Image as OpenpyxlImage

try:
    from ..base_extractor import BaseExtractor
except ImportError:
    from base_extractor import BaseExtractor

logger = logging.getLogger(__name__)


class ExcelExtractor(BaseExtractor):
    """
    Enhanced Excel extractor matching PDF extractor pattern
    
    Features:
    - Image caching (avoid redundant API calls)
    - Duplicate detection (skip same image multiple times)
    - Multi-sheet support (each sheet = page)
    - Chart extraction and description
    - Consistent markdown structure
    - Smart table formatting
    
    Supported formats:
    - XLSX (Excel 2007+)
    - XLS (Excel 97-2003) via pandas
    - CSV (Comma-separated values)
    - TSV (Tab-separated values)
    """
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        
        from app.services.image_cache_manager import get_image_cache
        from app.services.image_description_service import get_image_description_service
        
        self.image_cache = get_image_cache()
        self.image_service = get_image_description_service()
        
        # Tracking counters
        self._duplicates_skipped = 0
        self._charts_found = 0
        
        # Configuration
        self.max_rows_per_table = config.get('max_rows_per_table', 100) if config else 100
        self.max_preview_rows = config.get('max_preview_rows', 20) if config else 20
        self.include_formulas = config.get('include_formulas', True) if config else True
        
        logger.info("Excel Extractor initialized with caching and duplicate detection")
    
    def _get_supported_formats(self) -> List[str]:
        return ['xlsx', 'xls', 'xlsm', 'csv', 'tsv']
    
    def validate_file(self, file_path: str) -> Tuple[bool, str]:
        """Validate Excel file"""
        path = Path(file_path)
        
        if not path.exists():
            return False, "File does not exist"
        
        ext = path.suffix.lower().lstrip('.')
        if ext not in self.supported_formats:
            return False, f"Not a supported Excel format. Supported: {', '.join(self.supported_formats)}"
        
        if path.stat().st_size == 0:
            return False, "File is empty"
        
        return True, "Valid Excel file"
    
    def extract(self, file_path: str) -> Tuple[str, bool, Dict]:
        """Extract Excel file with proper flow matching PDF extractor"""
        try:
            is_valid, msg = self.validate_file(file_path)
            if not is_valid:
                return "", False, {'error': msg}
            
            excel_stem = Path(file_path).stem
            file_ext = Path(file_path).suffix.lower().lstrip('.')
            
            logger.info(f"📊 Extracting: {file_path}")
            
            # Route to appropriate extraction method
            if file_ext in ['csv', 'tsv']:
                return self._extract_csv(file_path, excel_stem, file_ext)
            elif file_ext in ['xlsx', 'xlsm']:
                return self._extract_xlsx(file_path, excel_stem)
            elif file_ext == 'xls':
                return self._extract_xls(file_path, excel_stem)
            else:
                return "", False, {'error': f'Unsupported format: {file_ext}'}
            
        except Exception as e:
            logger.error(f"❌ Excel extraction error: {str(e)}")
            import traceback
            traceback.print_exc()
            return "", False, {'error': str(e)}
    
    def _extract_csv(
        self,
        file_path: str,
        excel_stem: str,
        file_ext: str
    ) -> Tuple[str, bool, Dict]:
        """Extract CSV/TSV files"""
        try:
            # Determine delimiter
            delimiter = '\t' if file_ext == 'tsv' else ','
            
            # Read CSV
            df = pd.read_csv(file_path, delimiter=delimiter, encoding='utf-8', on_bad_lines='skip')
            
            logger.info(f"📑 CSV has {len(df)} rows, {len(df.columns)} columns")
            
            # Build markdown
            markdown_content = self._build_csv_markdown(df, excel_stem, file_ext)
            
            metadata = self.get_metadata(file_path)
            metadata.update({
                'extractor': 'ExcelExtractor_CSV',
                'total_pages': 1,
                'total_sheets': 1,
                'row_count': len(df),
                'column_count': len(df.columns),
                'images_extracted': 0,
                'extraction_method': 'pandas_csv'
            })
            
            logger.info(f"✅ Complete: {len(df)} rows, {len(df.columns)} columns")
            
            return markdown_content, True, metadata
            
        except Exception as e:
            logger.error(f"CSV extraction failed: {str(e)}")
            return "", False, {'error': str(e)}
    
    def _extract_xls(
        self,
        file_path: str,
        excel_stem: str
    ) -> Tuple[str, bool, Dict]:
        """Extract XLS files using pandas (no image support)"""
        try:
            # Read all sheets
            excel_file = pd.ExcelFile(file_path)
            sheet_names = excel_file.sheet_names
            
            logger.info(f"📑 XLS has {len(sheet_names)} sheets")
            
            # Process each sheet
            sheets_data = {}
            for sheet_name in sheet_names:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                sheets_data[sheet_name] = df
                logger.info(f"  Sheet '{sheet_name}': {len(df)} rows, {len(df.columns)} columns")
            
            # Build markdown
            markdown_content = self._build_xls_markdown(sheets_data, excel_stem)
            
            total_rows = sum(len(df) for df in sheets_data.values())
            total_cols = max(len(df.columns) for df in sheets_data.values()) if sheets_data else 0
            
            metadata = self.get_metadata(file_path)
            metadata.update({
                'extractor': 'ExcelExtractor_XLS',
                'total_pages': len(sheet_names),
                'total_sheets': len(sheet_names),
                'row_count': total_rows,
                'column_count': total_cols,
                'images_extracted': 0,
                'extraction_method': 'pandas_xls',
                'note': 'XLS format - images not supported'
            })
            
            logger.info(f"✅ Complete: {len(sheet_names)} sheets")
            
            return markdown_content, True, metadata
            
        except Exception as e:
            logger.error(f"XLS extraction failed: {str(e)}")
            return "", False, {'error': str(e)}
    
    def _extract_xlsx(
        self,
        file_path: str,
        excel_stem: str
    ) -> Tuple[str, bool, Dict]:
        """Extract XLSX files with full support (images, charts, data)"""
        try:
            # Create image directory
            image_output_dir = self._create_image_directory(file_path, excel_stem)
            
            # Load workbook with data_only=True for values, plus a second pass
            # for formulas (include_formulas=True). Previously formulas were
            # completely discarded despite the config flag existing.
            wb = load_workbook(file_path, data_only=True)
            wb_formulas = None
            if self.include_formulas:
                try:
                    wb_formulas = load_workbook(file_path, data_only=False)
                except Exception as e:
                    logger.warning(f"Could not load formula workbook: {e}")
            sheet_names = wb.sheetnames
            
            logger.info(f"📑 XLSX has {len(sheet_names)} sheets")
            
            # Reset counters
            self._duplicates_skipped = 0
            self._charts_found = 0
            
            # Process all sheets
            sheets_data = []
            for sheet_idx, sheet_name in enumerate(sheet_names, start=1):
                sheet_data = self._process_sheet(
                    wb,
                    sheet_name,
                    sheet_idx,
                    file_path,
                    image_output_dir,
                    excel_stem,
                    wb_formulas=wb_formulas
                )
                sheets_data.append(sheet_data)
            
            wb.close()
            if wb_formulas:
                wb_formulas.close()
            
            # Build markdown sheet-by-sheet
            final_markdown = self._build_xlsx_markdown(
                sheets_data,
                excel_stem,
                len(sheet_names)
            )
            
            # Gather statistics
            cache_stats = self.image_cache.get_stats()
            total_images = sum(sheet['image_count'] for sheet in sheets_data)
            total_charts = sum(sheet['chart_count'] for sheet in sheets_data)
            total_rows = sum(sheet['row_count'] for sheet in sheets_data)
            
            metadata = self.get_metadata(file_path)
            metadata.update({
                'extractor': 'ExcelExtractor_XLSX_Enhanced',
                'total_pages': len(sheet_names),
                'total_sheets': len(sheet_names),
                'row_count': total_rows,
                'images_extracted': total_images,
                'charts_found': total_charts,
                'images_described': sum(
                    len([img for img in sheet['images'] if img.get('success')])
                    for sheet in sheets_data
                ),
                'images_cached': cache_stats['hits'],
                'images_skipped_duplicates': self._duplicates_skipped,
                'cache_hit_rate': cache_stats['hit_rate_percent'],
                'image_directory': str(image_output_dir),
                'extraction_method': 'openpyxl+pandas'
            })
            
            logger.info(
                f"✅ Complete: {len(sheet_names)} sheets, {total_images} images, "
                f"{total_charts} charts ({metadata['images_cached']} cached)"
            )
            
            return final_markdown, True, metadata
            
        except Exception as e:
            logger.error(f"XLSX extraction failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return "", False, {'error': str(e)}
    
    def _process_sheet(
        self,
        wb,
        sheet_name: str,
        sheet_idx: int,
        file_path: str,
        image_output_dir: str,
        excel_stem: str,
        wb_formulas=None
    ) -> Dict:
        """Process a single sheet (data, images, charts, formulas, statistics)"""
        try:
            sheet = wb[sheet_name]
            
            # Extract data using pandas for better handling
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            
            # FIX: Use max_rows_per_table (default 100) not max_preview_rows (20).
            # Previously only 20 rows were written to markdown, so the vector DB
            # had no data to answer queries against. Now we embed the full table
            # up to max_rows_per_table so aggregation/retrieval actually works.
            if len(df) > self.max_rows_per_table:
                logger.warning(
                    f"Sheet '{sheet_name}' has {len(df)} rows, "
                    f"limiting to first {self.max_rows_per_table} for embedding"
                )
                df_preview = df.head(self.max_rows_per_table)
                truncated = True
            else:
                df_preview = df
                truncated = False
            
            # FIX: Compute per-column statistics for numeric columns.
            # These aggregates (sum, mean, min, max, count) are embedded directly
            # in the markdown so retrieval can answer "what is the total X?" queries.
            column_stats = self._compute_column_statistics(df)
            
            # FIX: Extract formulas from the formula workbook.
            # Previously include_formulas config was ignored — data_only=True
            # silently discards all formula text.
            formula_map = {}
            if self.include_formulas and wb_formulas and sheet_name in wb_formulas.sheetnames:
                formula_map = self._extract_formulas(wb_formulas[sheet_name])
            
            # Process images in sheet
            images_data = self._process_sheet_images(
                sheet,
                sheet_idx,
                sheet_name,
                image_output_dir,
                excel_stem
            )
            
            # Detect charts
            charts_count = len(sheet._charts) if hasattr(sheet, '_charts') else 0
            if charts_count > 0:
                logger.info(f"  Found {charts_count} charts in sheet '{sheet_name}'")
                self._charts_found += charts_count
            
            return {
                'sheet_name': sheet_name,
                'sheet_idx': sheet_idx,
                'dataframe': df_preview,
                'row_count': len(df),
                'column_count': len(df.columns),
                'column_stats': column_stats,
                'formula_map': formula_map,
                'images': images_data,
                'image_count': len(images_data),
                'chart_count': charts_count,
                'truncated': truncated,
                'has_data': not df.empty
            }
            
        except Exception as e:
            logger.error(f"Error processing sheet '{sheet_name}': {e}")
            return {
                'sheet_name': sheet_name,
                'sheet_idx': sheet_idx,
                'dataframe': pd.DataFrame(),
                'row_count': 0,
                'column_count': 0,
                'column_stats': {},
                'formula_map': {},
                'images': [],
                'image_count': 0,
                'chart_count': 0,
                'truncated': False,
                'has_data': False,
                'error': str(e)
            }

    def _compute_column_statistics(self, df: pd.DataFrame) -> Dict:
        """
        FIX: Compute per-column numeric statistics so the vector DB can answer
        aggregation queries like 'total revenue', 'average score', 'min/max price'.
        Without this, retrieval returns raw rows with no computed values.
        """
        stats = {}
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                series = df[col].dropna()
                if len(series) == 0:
                    continue
                stats[str(col)] = {
                    'count': int(series.count()),
                    'sum': float(round(series.sum(), 4)),
                    'mean': float(round(series.mean(), 4)),
                    'median': float(round(series.median(), 4)),
                    'min': float(series.min()),
                    'max': float(series.max()),
                    'std': float(round(series.std(), 4)) if len(series) > 1 else 0.0,
                }
            elif pd.api.types.is_string_dtype(df[col]) or df[col].dtype == object:
                # For text columns store unique value count and top values
                series = df[col].dropna().astype(str)
                value_counts = series.value_counts()
                stats[str(col)] = {
                    'count': int(series.count()),
                    'unique_values': int(series.nunique()),
                    'top_values': value_counts.head(10).index.tolist(),
                }
        return stats

    def _extract_formulas(self, sheet) -> Dict:
        """
        FIX: Extract formula strings from the formula-mode workbook.
        data_only=True (used for values) silently drops all formula text,
        so include_formulas config was previously a no-op.
        Returns {cell_address: formula_string} for cells that contain formulas.
        """
        formula_map = {}
        for row in sheet.iter_rows():
            for cell in row:
                val = cell.value
                if isinstance(val, str) and val.startswith('='):
                    formula_map[cell.coordinate] = val
        return formula_map

    def _process_sheet_images(
        self,
        sheet,
        sheet_idx: int,
        sheet_name: str,
        image_output_dir: str,
        excel_stem: str
    ) -> List[Dict]:
        """Process images in a sheet with caching and duplicate detection"""
        images_data = []
        seen_hashes = set()
        
        doc_context = f"Excel workbook: {excel_stem}, Sheet: {sheet_name}"
        
        # Get images from sheet
        if not hasattr(sheet, '_images') or not sheet._images:
            return images_data
        
        for img_idx, img in enumerate(sheet._images, start=1):
            try:
                # Get image data
                image_bytes = img._data()
                
                # Check for duplicates
                img_hash = self.image_cache.compute_hash(image_bytes)
                if img_hash in seen_hashes:
                    logger.debug(f"  ⏭ Skipping duplicate image in sheet '{sheet_name}'")
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
                    if self.image_service.is_available():
                        base64_image = base64.b64encode(image_bytes).decode('utf-8')
                        # Determine media type from image format
                        media_type = self._get_media_type_from_bytes(image_bytes)
                        
                        success, description = self.image_service.describe_image_from_base64(
                            base64_image,
                            media_type,
                            context=doc_context,
                            page_number=sheet_idx
                        )
                        from_cache = False
                        
                        if success:
                            self.image_cache.put(image_bytes, description)
                    else:
                        success = False
                        description = self.image_service.generate_fallback_description(
                            f"sheet_{sheet_idx}_image",
                            page_number=sheet_idx,
                            image_type="spreadsheet image"
                        )
                        from_cache = False
                
                # Determine format
                ext = self._detect_image_format(image_bytes)
                
                # Save image
                filename = f"sheet_{sheet_idx:02d}_image_{img_idx:03d}.{ext}"
                output_path = os.path.join(image_output_dir, filename)
                
                with open(output_path, 'wb') as f:
                    f.write(image_bytes)
                
                images_data.append({
                    'img_idx': img_idx,
                    'filename': filename,
                    'sheet_idx': sheet_idx,
                    'sheet_name': sheet_name,
                    'format': ext,
                    'description': description,
                    'success': success,
                    'from_cache': from_cache,
                    'size_bytes': len(image_bytes)
                })
                
                cache_str = "💾" if from_cache else "✓"
                logger.info(f"  {cache_str} Image {img_idx} -> {filename} (sheet {sheet_idx})")
                
            except Exception as e:
                logger.warning(f"  ⚠ Error processing image {img_idx} in sheet '{sheet_name}': {e}")
                continue
        
        return images_data
    
    def _build_csv_markdown(
        self,
        df: pd.DataFrame,
        excel_stem: str,
        file_ext: str
    ) -> str:
        """Build markdown for CSV/TSV files — includes statistics for retrieval"""
        result_lines = []
        
        # Header
        result_lines.append(f"# Document: {excel_stem}")
        result_lines.append(f"**Format**: {file_ext.upper()}")
        result_lines.append(f"**Total Pages**: 1")
        result_lines.append(f"**Rows**: {len(df)}")
        result_lines.append(f"**Columns**: {len(df.columns)}")
        result_lines.append("")
        
        # Page marker
        result_lines.append("## Page 1\n")
        
        # FIX: Embed all rows up to max_rows_per_table, not just max_preview_rows
        if len(df) > self.max_rows_per_table:
            result_lines.append(f"**Note**: Showing first {self.max_rows_per_table} of {len(df)} rows\n")
            df_to_show = df.head(self.max_rows_per_table)
        else:
            df_to_show = df
        
        # Convert to markdown table
        table_md = self._dataframe_to_markdown(df_to_show)
        result_lines.append(table_md)
        result_lines.append("")
        
        # Summary statistics
        result_lines.append("### Data Summary\n")
        result_lines.append(f"- **Total Rows**: {len(df)}")
        result_lines.append(f"- **Total Columns**: {len(df.columns)}")
        result_lines.append(f"- **Columns**: {', '.join(df.columns.tolist())}")
        result_lines.append("")
        
        # FIX: Embed per-column numeric statistics for aggregation queries
        col_stats = self._compute_column_statistics(df)
        if col_stats:
            result_lines.append("### Column Statistics\n")
            for col_name, stats in col_stats.items():
                if 'sum' in stats:
                    result_lines.append(
                        f"- **{col_name}**: count={stats['count']}, "
                        f"sum={stats['sum']}, mean={stats['mean']}, "
                        f"median={stats['median']}, min={stats['min']}, "
                        f"max={stats['max']}"
                    )
                else:
                    top = ', '.join(str(v) for v in stats.get('top_values', [])[:5])
                    result_lines.append(
                        f"- **{col_name}**: {stats['count']} values, "
                        f"{stats['unique_values']} unique — top: {top}"
                    )
            result_lines.append("")
        
        return '\n'.join(result_lines)
    
    def _build_xls_markdown(
        self,
        sheets_data: Dict[str, pd.DataFrame],
        excel_stem: str
    ) -> str:
        """Build markdown for XLS files — includes statistics for retrieval"""
        result_lines = []
        
        # Header
        result_lines.append(f"# Document: {excel_stem}")
        result_lines.append(f"**Format**: Excel 97-2003 (XLS)")
        result_lines.append(f"**Total Pages**: {len(sheets_data)}")
        result_lines.append("")
        
        # Process each sheet as a page
        for page_num, (sheet_name, df) in enumerate(sheets_data.items(), start=1):
            result_lines.append(f"## Page {page_num}: {sheet_name}\n")
            
            if df.empty:
                result_lines.append("*(Empty sheet)*\n")
                continue
            
            # FIX: Embed all rows up to max_rows_per_table, not max_preview_rows
            if len(df) > self.max_rows_per_table:
                result_lines.append(f"**Note**: Showing first {self.max_rows_per_table} of {len(df)} rows\n")
                df_to_show = df.head(self.max_rows_per_table)
            else:
                df_to_show = df
            
            table_md = self._dataframe_to_markdown(df_to_show)
            result_lines.append(table_md)
            result_lines.append("")
            
            # Sheet statistics
            result_lines.append(f"**Sheet Statistics**: {len(df)} rows × {len(df.columns)} columns")
            result_lines.append("")
            
            # FIX: Embed per-column numeric statistics for aggregation queries
            col_stats = self._compute_column_statistics(df)
            if col_stats:
                result_lines.append("### Column Statistics\n")
                for col_name, stats in col_stats.items():
                    if 'sum' in stats:
                        result_lines.append(
                            f"- **{col_name}**: count={stats['count']}, "
                            f"sum={stats['sum']}, mean={stats['mean']}, "
                            f"median={stats['median']}, min={stats['min']}, "
                            f"max={stats['max']}"
                        )
                    else:
                        top = ', '.join(str(v) for v in stats.get('top_values', [])[:5])
                        result_lines.append(
                            f"- **{col_name}**: {stats['count']} values, "
                            f"{stats['unique_values']} unique — top: {top}"
                        )
                result_lines.append("")
        
        return '\n'.join(result_lines)
    
    def _build_xlsx_markdown(
        self,
        sheets_data: List[Dict],
        excel_stem: str,
        total_sheets: int
    ) -> str:
        """Build markdown for XLSX files — includes statistics and formulas for retrieval"""
        result_lines = []
        
        # Header
        result_lines.append(f"# Document: {excel_stem}")
        result_lines.append(f"**Format**: Excel 2007+ (XLSX)")
        result_lines.append(f"**Total Pages**: {total_sheets}")
        result_lines.append("")
        
        # Process each sheet as a page
        for sheet_data in sheets_data:
            page_num = sheet_data['sheet_idx']
            sheet_name = sheet_data['sheet_name']
            
            result_lines.append(f"## Page {page_num}: {sheet_name}\n")
            
            # Check for errors
            if 'error' in sheet_data:
                result_lines.append(f"*(Error: {sheet_data['error']})*\n")
                continue
            
            # Add images first
            for img_info in sheet_data['images']:
                result_lines.append(f"**[Image Description]**: {img_info['description']}")
                cache_str = "cached" if img_info.get('from_cache') else "described"
                result_lines.append(f"*(Sheet {page_num}, {cache_str})*")
                result_lines.append("")
            
            # Add chart note
            if sheet_data['chart_count'] > 0:
                result_lines.append(f"**Charts**: {sheet_data['chart_count']} chart(s) found in this sheet")
                result_lines.append("")
            
            # Add data table
            if not sheet_data['has_data']:
                result_lines.append("*(No data in this sheet)*\n")
                continue
            
            df = sheet_data['dataframe']
            
            if sheet_data['truncated']:
                result_lines.append(
                    f"**Note**: Showing first {self.max_rows_per_table} of "
                    f"{sheet_data['row_count']} rows\n"
                )
            
            table_md = self._dataframe_to_markdown(df)
            result_lines.append(table_md)
            result_lines.append("")
            
            # Sheet statistics
            result_lines.append(
                f"**Sheet Statistics**: {sheet_data['row_count']} rows × "
                f"{sheet_data['column_count']} columns"
            )
            result_lines.append("")
            
            # FIX: Embed computed column statistics so the vector DB can answer
            # numeric queries (sum, average, min, max) without seeing all rows.
            col_stats = sheet_data.get('column_stats', {})
            if col_stats:
                result_lines.append("### Column Statistics\n")
                for col_name, stats in col_stats.items():
                    if 'sum' in stats:
                        result_lines.append(
                            f"- **{col_name}**: count={stats['count']}, "
                            f"sum={stats['sum']}, mean={stats['mean']}, "
                            f"median={stats['median']}, min={stats['min']}, "
                            f"max={stats['max']}, std={stats['std']}"
                        )
                    else:
                        top = ', '.join(str(v) for v in stats.get('top_values', [])[:5])
                        result_lines.append(
                            f"- **{col_name}**: {stats['count']} values, "
                            f"{stats['unique_values']} unique — top: {top}"
                        )
                result_lines.append("")
            
            # FIX: Embed formulas so the vector DB understands computed columns.
            formula_map = sheet_data.get('formula_map', {})
            if formula_map:
                result_lines.append("### Formulas\n")
                for cell_addr, formula in list(formula_map.items())[:50]:  # cap at 50
                    result_lines.append(f"- `{cell_addr}`: `{formula}`")
                result_lines.append("")
        
        return '\n'.join(result_lines)
    
    def _dataframe_to_markdown(self, df: pd.DataFrame) -> str:
        """
        Convert pandas DataFrame to markdown table.
        FIX: Removed the raw integer index column — it added noise to embeddings
        and confused retrieval (e.g. queries for row 0 vs actual data values).
        """
        if df.empty:
            return "*(No data)*"
        
        # Replace NaN with empty string
        df = df.fillna('')
        
        lines = []
        
        # FIX: Header uses only actual column names — no index prefix column
        headers = [str(h) for h in df.columns.tolist()]
        lines.append("| " + " | ".join(headers) + " |")
        
        # Separator
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        
        # Data rows — values only, no index
        for _, row in df.iterrows():
            row_data = [str(val) for val in row.values]
            lines.append("| " + " | ".join(row_data) + " |")
        
        return '\n'.join(lines)
    
    def _create_image_directory(self, file_path: str, excel_stem: str) -> str:
        """Create directory for extracted images"""
        image_dir = os.path.join(
            os.path.dirname(file_path),
            "extracted_images",
            excel_stem
        )
        os.makedirs(image_dir, exist_ok=True)
        return image_dir
    
    def _detect_image_format(self, image_bytes: bytes) -> str:
        """Detect image format from bytes"""
        # Check magic numbers
        if image_bytes[:2] == b'\xff\xd8':
            return 'jpg'
        elif image_bytes[:8] == b'\x89PNG\r\n\x1a\n':
            return 'png'
        elif image_bytes[:3] == b'GIF':
            return 'gif'
        elif image_bytes[:2] == b'BM':
            return 'bmp'
        else:
            return 'png'  # Default
    
    def _get_media_type_from_bytes(self, image_bytes: bytes) -> str:
        """Get MIME type from image bytes"""
        ext = self._detect_image_format(image_bytes)
        return {
            'jpg': 'image/jpeg',
            'jpeg': 'image/jpeg',
            'png': 'image/png',
            'gif': 'image/gif',
            'bmp': 'image/bmp'
        }.get(ext, 'image/png')