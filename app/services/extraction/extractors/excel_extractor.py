"""
Excel File Extractor  –  XLSX / XLS / XLSM
===========================================
Handles Excel workbooks with two distinct pipelines:

  A) Semantic pipeline  (text-heavy workbooks)
     extract() → returns markdown → caller saves it → /chunks/create →
     /chunks/finalize  (any chunking strategy)

  B) Row-wise pipeline  (tabular / financial workbooks)
     extract_and_store() → embeds every data row → upserts each row as
     a separate Qdrant point → no markdown produced for tabular sheets.
     Text-heavy sheets inside the same workbook still produce a markdown
     fragment that is saved to disk and used by the semantic pipeline.

CSV and TSV files are handled by csv_extractor.py, NOT here.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from openpyxl import load_workbook

try:
    from ..base_extractor import BaseExtractor
except ImportError:
    from base_extractor import BaseExtractor

logger = logging.getLogger(__name__)


# ── helpers ───────────────────────────────────────────────────────────────────

def _string_to_uuid(text: str) -> str:
    """Deterministic UUID v5 from any string."""
    namespace = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")
    return str(uuid.uuid5(namespace, text))


def _row_to_text(sheet_name: str, row_index: int, row: pd.Series) -> str:
    """
    Convert one DataFrame row to a searchable plain-text sentence.

    Format:
        Sheet: <name> | Row <N>: col1: val1 | col2: val2 | …

    Empty / NaN cells are skipped so they don't pollute the embedding.
    """
    parts = []
    for col, val in row.items():
        if pd.notna(val) and str(val).strip():
            parts.append(f"{col}: {val}")
    body = " | ".join(parts) if parts else "(empty row)"
    return f"Sheet: {sheet_name} | Row {row_index}: {body}"


# ── extractor ─────────────────────────────────────────────────────────────────

class ExcelExtractor(BaseExtractor):
    """
    Excel extractor supporting both semantic and row-wise pipelines.

    Supported formats: xlsx, xls, xlsm
    (csv / tsv → handled by CSVExtractor)

    Key methods
    -----------
    extract()            Standard BaseExtractor contract.  Returns markdown.
                         Used by the semantic pipeline.
    extract_and_store()  Row-wise pipeline entry point.  Embeds every data
                         row of tabular sheets and upserts to Qdrant.
                         Returns (success, summary, fallback_markdown) where
                         fallback_markdown covers any text-heavy sheets found
                         inside the same workbook.
    """

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)

        from app.services.image_cache_manager import get_image_cache
        from app.services.image_description_service import get_image_description_service

        self.image_cache   = get_image_cache()
        self.image_service = get_image_description_service()

        self._duplicates_skipped = 0
        self._charts_found       = 0

        # max_rows_per_table is ONLY used in the markdown (semantic) path.
        # The row-wise path always processes ALL rows — no cap.
        cfg = config or {}
        self.max_rows_per_table = cfg.get("max_rows_per_table", 100)
        self.max_preview_rows   = cfg.get("max_preview_rows",   20)
        self.include_formulas   = cfg.get("include_formulas",   True)

        logger.info("ExcelExtractor initialised (csv/tsv → CSVExtractor)")

    # ── format support ────────────────────────────────────────────────────────

    def _get_supported_formats(self) -> List[str]:
        # CSV and TSV intentionally removed — owned by CSVExtractor.
        return ["xlsx", "xls", "xlsm"]

    # ── validation ────────────────────────────────────────────────────────────

    def validate_file(self, file_path: str) -> Tuple[bool, str]:
        path = Path(file_path)
        if not path.exists():
            return False, "File does not exist"
        ext = path.suffix.lower().lstrip(".")
        if ext not in self.supported_formats:
            return False, (
                f"Not a supported Excel format. Supported: "
                f"{', '.join(self.supported_formats)}"
            )
        if path.stat().st_size == 0:
            return False, "File is empty"
        return True, "Valid Excel file"

    # =========================================================================
    # PUBLIC API — extract()  (semantic / markdown pipeline)
    # =========================================================================

    def extract(self, file_path: str) -> Tuple[str, bool, Dict]:
        """
        Extract Excel workbook to markdown.

        Routing:
            .xlsx / .xlsm  →  _extract_xlsx()
            .xls           →  _extract_xls()
        """
        try:
            is_valid, msg = self.validate_file(file_path)
            if not is_valid:
                return "", False, {"error": msg}

            excel_stem = Path(file_path).stem
            file_ext   = Path(file_path).suffix.lower().lstrip(".")

            logger.info("📊 Extracting (semantic): %s", file_path)

            if file_ext in ("xlsx", "xlsm"):
                return self._extract_xlsx(file_path, excel_stem)
            elif file_ext == "xls":
                return self._extract_xls(file_path, excel_stem)
            else:
                return "", False, {"error": f"Unsupported format: {file_ext}"}

        except Exception as exc:
            logger.error("❌ Excel extraction error: %s", exc, exc_info=True)
            return "", False, {"error": str(exc)}

    # =========================================================================
    # PUBLIC API — extract_and_store()  (row-wise pipeline)
    # =========================================================================

    def extract_and_store(
        self,
        file_path: str,
        file_id: str,
        filename: str,
        knowledge_base_id: str,
        knowledge_base_name: str,
        user_id: str,
        vector_store,               # VectorStoreService instance
        batch_size: int = 50,
    ) -> Tuple[bool, Dict, str]:
        """
        Row-wise extraction + Qdrant storage for tabular Excel sheets.

        Each sheet is individually classified by ExcelSheetClassifier:
          • TABULAR   → every data row becomes one Qdrant point.
          • TEXT_HEAVY → sheet is rendered to markdown for the fallback
                         semantic pipeline.

        Parameters
        ----------
        file_path          : absolute path to the Excel file.
        file_id            : unique identifier for this upload.
        filename           : human-readable filename (stored in payload).
        knowledge_base_id  : KB this file belongs to.
        knowledge_base_name: KB display name.
        user_id            : owning user.
        vector_store       : initialised VectorStoreService.
        batch_size         : rows per embedding/upsert batch.

        Returns
        -------
        (success, summary_dict, fallback_markdown_str)

        summary_dict keys:
            total_rows_stored  int   – rows successfully indexed in Qdrant.
            total_rows_failed  int   – rows that failed.
            tabular_sheets     list  – sheet names processed row-wise.
            semantic_sheets    list  – sheet names rendered to markdown.
            sheet_detail       dict  – per-sheet breakdown.

        fallback_markdown_str:
            Markdown content for all text-heavy sheets (may be empty string
            if every sheet was tabular).  Caller is responsible for saving
            this to disk and setting file_metadata.markdown_path.
        """
        from app.services.extraction.extractors.excel_classifier import ExcelSheetClassifier

        try:
            is_valid, msg = self.validate_file(file_path)
            if not is_valid:
                return False, {"error": msg}, ""

            classifier   = ExcelSheetClassifier()
            xl           = pd.ExcelFile(file_path)
            excel_stem   = Path(file_path).stem
            file_ext     = Path(file_path).suffix.lower().lstrip(".")

            total_rows_stored  = 0
            total_rows_failed  = 0
            tabular_sheets:  List[str] = []
            semantic_sheets: List[str] = []
            sheet_detail:    Dict      = {}
            fallback_md_parts: List[str] = []

            logger.info(
                "📊 extract_and_store: %d sheets in '%s'",
                len(xl.sheet_names), filename,
            )

            for sheet_idx, sheet_name in enumerate(xl.sheet_names, start=1):
                try:
                    # Always read the FULL DataFrame — no row cap here.
                    df = pd.read_excel(file_path, sheet_name=sheet_name)
                except Exception as exc:
                    logger.error("Cannot read sheet '%s': %s", sheet_name, exc)
                    sheet_detail[sheet_name] = {"error": str(exc)}
                    continue

                decision = classifier.classify_sheet(df)

                # ── TABULAR sheet → row-wise Qdrant storage ───────────────
                if decision == "tabular":
                    tabular_sheets.append(sheet_name)
                    stored, failed = self._store_sheet_rows(
                        df         = df,
                        sheet_name = sheet_name,
                        sheet_idx  = sheet_idx,
                        file_id    = file_id,
                        filename   = filename,
                        knowledge_base_id   = knowledge_base_id,
                        knowledge_base_name = knowledge_base_name,
                        user_id    = user_id,
                        vector_store= vector_store,
                        batch_size = batch_size,
                    )
                    total_rows_stored += stored
                    total_rows_failed += failed
                    sheet_detail[sheet_name] = {
                        "pipeline": "row_wise",
                        "rows_total" : len(df),
                        "rows_stored": stored,
                        "rows_failed": failed,
                    }
                    logger.info(
                        "  ✅ Sheet '%s' (tabular): %d/%d rows stored",
                        sheet_name, stored, len(df),
                    )

                # ── TEXT_HEAVY sheet → render markdown ────────────────────
                else:
                    semantic_sheets.append(sheet_name)
                    sheet_md = self._sheet_to_markdown(
                        df         = df,
                        sheet_name = sheet_name,
                        sheet_idx  = sheet_idx,
                        excel_stem = excel_stem,
                    )
                    fallback_md_parts.append(sheet_md)
                    sheet_detail[sheet_name] = {
                        "pipeline": "semantic",
                        "rows": len(df),
                    }
                    logger.info(
                        "  📝 Sheet '%s' (text-heavy): rendered to markdown",
                        sheet_name,
                    )

            fallback_markdown = "\n\n".join(fallback_md_parts)

            summary = {
                "total_rows_stored": total_rows_stored,
                "total_rows_failed": total_rows_failed,
                "tabular_sheets"   : tabular_sheets,
                "semantic_sheets"  : semantic_sheets,
                "sheet_detail"     : sheet_detail,
                "extraction_method": "excel_hybrid_row_wise",
            }

            success = total_rows_stored > 0 or bool(fallback_md_parts)
            logger.info(
                "🎉 extract_and_store complete: %d rows in Qdrant, "
                "%d semantic sheets → markdown",
                total_rows_stored, len(semantic_sheets),
            )
            return success, summary, fallback_markdown

        except Exception as exc:
            logger.error("❌ extract_and_store error: %s", exc, exc_info=True)
            return False, {"error": str(exc)}, ""

    # =========================================================================
    # PRIVATE — row-wise storage helpers
    # =========================================================================

    def _store_sheet_rows(
        self,
        df: pd.DataFrame,
        sheet_name: str,
        sheet_idx: int,
        file_id: str,
        filename: str,
        knowledge_base_id: str,
        knowledge_base_name: str,
        user_id: str,
        vector_store,
        batch_size: int,
    ) -> Tuple[int, int]:
        """
        Embed and upsert every non-empty row of *df* as a Qdrant point.

        Returns (rows_stored, rows_failed).

        Bug fix applied here
        --------------------
        The old markdown path capped rows at max_rows_per_table (100).
        This method intentionally iterates ALL rows of df — the caller
        must pass the complete (uncapped) DataFrame.
        """
        # Drop rows where every cell is NaN (truly empty rows)
        df = df.dropna(how="all").reset_index(drop=True)

        if df.empty:
            logger.warning("Sheet '%s' has no non-empty rows after cleaning.", sheet_name)
            return 0, 0

        total_stored = 0
        total_failed = 0

        for batch_start in range(0, len(df), batch_size):
            batch = df.iloc[batch_start : batch_start + batch_size]
            chunks: List[Dict] = []

            for local_idx, (_, row) in enumerate(batch.iterrows()):
                global_row_idx = batch_start + local_idx + 1   # 1-based
                row_text  = _row_to_text(sheet_name, global_row_idx, row)
                chunk_id  = _string_to_uuid(
                    f"{file_id}::{sheet_name}::row::{global_row_idx}"
                )
                chunks.append({
                    "text":        row_text,
                    "chunk_id":    chunk_id,
                    "chunk_index": global_row_idx - 1,   # 0-based for storage
                    "metadata": {
                        "chunk_type":  "excel_row",
                        "sheet_name":  sheet_name,
                        "row_index":   global_row_idx,
                        "page_number": sheet_idx,
                        "source_file": filename,
                    },
                })

            try:
                stored = vector_store.add_document_chunks(
                    chunks             = chunks,
                    file_id            = file_id,
                    filename           = filename,
                    knowledge_base_id  = knowledge_base_id,
                    knowledge_base_name= knowledge_base_name,
                    user_id            = user_id,
                )
                total_stored += stored
                logger.debug(
                    "  Sheet '%s' batch %d–%d: stored %d rows",
                    sheet_name,
                    batch_start + 1,
                    batch_start + len(batch),
                    stored,
                )
            except Exception as exc:
                logger.error(
                    "  Batch %d–%d of sheet '%s' failed: %s",
                    batch_start + 1,
                    batch_start + len(batch),
                    sheet_name,
                    exc,
                )
                total_failed += len(batch)

        return total_stored, total_failed

    # =========================================================================
    # PRIVATE — markdown rendering (semantic / text-heavy path)
    # =========================================================================

    def _sheet_to_markdown(
        self,
        df: pd.DataFrame,
        sheet_name: str,
        sheet_idx: int,
        excel_stem: str,
    ) -> str:
        """
        Render a single text-heavy sheet to markdown.

        This reuses the same table-capped rendering as before so the
        semantic pipeline receives reasonably sized markdown chunks.
        The row cap (max_preview_rows) only applies here, never in the
        row-wise path.
        """
        lines: List[str] = [f"## Sheet {sheet_idx}: {sheet_name}\n"]

        if df.empty:
            lines.append("*(Empty sheet)*\n")
            return "\n".join(lines)

        if len(df) > self.max_rows_per_table:
            lines.append(
                f"**Note**: Showing first {self.max_preview_rows} of {len(df)} rows\n"
            )
            df_show = df.head(self.max_preview_rows)
        else:
            df_show = df

        lines.append(self._dataframe_to_markdown(df_show))
        lines.append("")
        lines.append(
            f"**Sheet Statistics**: {len(df)} rows × {len(df.columns)} columns"
        )
        lines.append("")
        return "\n".join(lines)

    # ── existing markdown helpers (unchanged) ─────────────────────────────────

    def _extract_xls(self, file_path: str, excel_stem: str) -> Tuple[str, bool, Dict]:
        """Extract XLS files using pandas (no image support)."""
        try:
            excel_file = pd.ExcelFile(file_path)
            sheet_names = excel_file.sheet_names
            logger.info("📑 XLS has %d sheets", len(sheet_names))

            sheets_data: Dict[str, pd.DataFrame] = {}
            for sheet_name in sheet_names:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                sheets_data[sheet_name] = df
                logger.info("  Sheet '%s': %d rows, %d cols", sheet_name, len(df), len(df.columns))

            markdown_content = self._build_xls_markdown(sheets_data, excel_stem)

            total_rows = sum(len(df) for df in sheets_data.values())
            total_cols = max((len(df.columns) for df in sheets_data.values()), default=0)

            metadata = self.get_metadata(file_path)
            metadata.update({
                "extractor":         "ExcelExtractor_XLS",
                "total_pages":       len(sheet_names),
                "total_sheets":      len(sheet_names),
                "row_count":         total_rows,
                "column_count":      total_cols,
                "images_extracted":  0,
                "extraction_method": "pandas_xls",
                "note":              "XLS format — images not supported",
            })
            logger.info("✅ XLS complete: %d sheets", len(sheet_names))
            return markdown_content, True, metadata

        except Exception as exc:
            logger.error("XLS extraction failed: %s", exc)
            return "", False, {"error": str(exc)}

    def _extract_xlsx(self, file_path: str, excel_stem: str) -> Tuple[str, bool, Dict]:
        """Extract XLSX/XLSM with images, charts, and data."""
        try:
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
            logger.info("📑 XLSX has %d sheets", len(sheet_names))

            self._duplicates_skipped = 0
            self._charts_found       = 0

            sheets_data: List[Dict] = []
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
                "extractor":              "ExcelExtractor_XLSX",
                "total_pages":            len(sheet_names),
                "total_sheets":           len(sheet_names),
                "row_count":              total_rows,
                "images_extracted":       total_images,
                "charts_found":           total_charts,
                "images_described":       sum(
                    len([img for img in s["images"] if img.get("success")])
                    for s in sheets_data
                ),
                "images_cached":          cache_stats["hits"],
                "images_skipped_duplicates": self._duplicates_skipped,
                "cache_hit_rate":         cache_stats["hit_rate_percent"],
                "image_directory":        str(image_output_dir),
                "extraction_method":      "openpyxl+pandas",
            })
            logger.info(
                "✅ XLSX complete: %d sheets, %d images, %d charts",
                len(sheet_names), total_images, total_charts,
            )
            return final_markdown, True, metadata

        except Exception as exc:
            logger.error("XLSX extraction failed: %s", exc, exc_info=True)
            return "", False, {"error": str(exc)}

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
            charts_count = len(sheet._charts) if hasattr(sheet, "_charts") else 0
            if charts_count:
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

        except Exception as exc:
            logger.error("Error processing sheet '%s': %s", sheet_name, exc)
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
        self, sheet, sheet_idx: int, sheet_name: str,
        image_output_dir: str, excel_stem: str,
    ) -> List[Dict]:
        """Extract and describe images from a single XLSX sheet."""
        images_data: List[Dict] = []
        seen_hashes: set = set()
        doc_context = f"Excel workbook: {excel_stem}, Sheet: {sheet_name}"

        if not hasattr(sheet, "_images") or not sheet._images:
            return images_data

        for img_idx, img in enumerate(sheet._images, start=1):
            try:
                image_bytes = img._data()
                img_hash    = self.image_cache.compute_hash(image_bytes)

                if img_hash in seen_hashes:
                    logger.debug("Skipping duplicate image in sheet '%s'", sheet_name)
                    self._duplicates_skipped += 1
                    continue
                seen_hashes.add(img_hash)

                cached_desc = self.image_cache.get(image_bytes)
                if cached_desc:
                    description = cached_desc
                    from_cache  = True
                    success     = True
                else:
                    if self.image_service.is_available():
                        b64        = base64.b64encode(image_bytes).decode("utf-8")
                        media_type = self._get_media_type_from_bytes(image_bytes)
                        success, description = self.image_service.describe_image_from_base64(
                            b64, media_type, context=doc_context, page_number=sheet_idx
                        )
                        from_cache = False
                        if success:
                            self.image_cache.put(image_bytes, description)
                    else:
                        success     = False
                        description = self.image_service.generate_fallback_description(
                            f"sheet_{sheet_idx}_image",
                            page_number=sheet_idx,
                            image_type="spreadsheet image",
                        )
                        from_cache = False

                ext      = self._detect_image_format(image_bytes)
                img_file = f"sheet_{sheet_idx:02d}_image_{img_idx:03d}.{ext}"
                out_path = os.path.join(image_output_dir, img_file)
                with open(out_path, "wb") as fh:
                    fh.write(image_bytes)

                images_data.append({
                    "img_idx":    img_idx,
                    "filename":   img_file,
                    "sheet_idx":  sheet_idx,
                    "sheet_name": sheet_name,
                    "format":     ext,
                    "description":description,
                    "success":    success,
                    "from_cache": from_cache,
                    "size_bytes": len(image_bytes),
                })
                logger.info("  %s Image %d → %s (sheet %d)",
                            "💾" if from_cache else "✓", img_idx, img_file, sheet_idx)

            except Exception as exc:
                logger.warning("Error processing image %d in sheet '%s': %s",
                               img_idx, sheet_name, exc)
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
        self, sheets_data: Dict[str, pd.DataFrame], excel_stem: str
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
            lines.append(f"## Page {page_num}: {sheet_name}\n")
            if df.empty:
                lines.append("*(Empty sheet)*\n")
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
        self, sheets_data: List[Dict], excel_stem: str, total_sheets: int
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

            for img_info in sd["images"]:
                lines.append(f"**[Image Description]**: {img_info['description']}")
                lines.append(
                    f"*(Sheet {page_num}, "
                    f"{'cached' if img_info.get('from_cache') else 'described'})*"
                )
                lines.append("")

            if sd["chart_count"] > 0:
                lines.append(f"**Charts**: {sd['chart_count']} chart(s) found in this sheet")
                lines.append("")

            if not sd["has_data"]:
                lines.append("*(No data in this sheet)*\n")
                continue
            
            df = sheet_data['dataframe']
            
            if sheet_data['truncated']:
                result_lines.append(
                    f"**Note**: Showing first {self.max_rows_per_table} of "
                    f"{sheet_data['row_count']} rows\n"
                )
            lines.append(self._dataframe_to_markdown(sd["dataframe"]))
            lines.append("")
            lines.append(
                f"**Sheet Statistics**: {sd['row_count']} rows × {sd['column_count']} columns"
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
        image_dir = os.path.join(
            os.path.dirname(file_path), "extracted_images", excel_stem
        )
        os.makedirs(image_dir, exist_ok=True)
        return image_dir

    def _detect_image_format(self, image_bytes: bytes) -> str:
        if image_bytes[:2] == b"\xff\xd8":
            return "jpg"
        if image_bytes[:8] == b"\x89PNG\r\n\x1a\n":
            return "png"
        if image_bytes[:3] == b"GIF":
            return "gif"
        if image_bytes[:2] == b"BM":
            return "bmp"
        return "png"

    def _get_media_type_from_bytes(self, image_bytes: bytes) -> str:
        return {
            "jpg":  "image/jpeg",
            "jpeg": "image/jpeg",
            "png":  "image/png",
            "gif":  "image/gif",
            "bmp":  "image/bmp",
        }.get(self._detect_image_format(image_bytes), "image/png")
