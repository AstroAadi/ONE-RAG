"""
CSV / TSV File Extractor
Processes .csv and .tsv files for the RAG pipeline.

"""

import csv
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple

try:
    from ..base_extractor import BaseExtractor
except ImportError:
    from base_extractor import BaseExtractor

logger = logging.getLogger(__name__)


def _string_to_uuid(text: str) -> str:
    """Convert any string to a deterministic UUID v5."""
    namespace = uuid.UUID('6ba7b810-9dad-11d1-80b4-00c04fd430c8')
    return str(uuid.uuid5(namespace, text))


class CSVExtractor(BaseExtractor):
    """
    Extractor for CSV and TSV files.

    Each data row is:
      1. Converted to a plain-text string  →  "col1: val1 | col2: val2 | …"
      2. Embedded (dense + sparse).
      3. Stored as a separate Qdrant point.

    The standard extract() method returns the row texts joined by newlines so
    the file can still be processed by the normal chunking / storage pipeline
    if needed. For the row-level Qdrant storage workflow, call
    extract_and_store() instead.
    """

    # Maximum number of rows to process (safety guard for huge files).
    MAX_ROWS = 100_000

    def _get_supported_formats(self) -> List[str]:
        return ['csv', 'tsv']

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_file(self, file_path: str) -> Tuple[bool, str]:
        path = Path(file_path)

        if not path.exists():
            return False, "File does not exist"

        ext = path.suffix.lower().lstrip('.')
        if ext not in self.supported_formats:
            return False, (
                f"Not a supported delimited file. "
                f"Supported: {', '.join(self.supported_formats)}"
            )

        if path.stat().st_size == 0:
            return False, "File is empty"

        return True, "Valid CSV/TSV file"

    # ------------------------------------------------------------------
    # Low-level helpers
    # ------------------------------------------------------------------

    def _detect_delimiter(self, file_path: str, file_ext: str) -> str:
        """Return the delimiter character for this file."""
        if file_ext == 'tsv':
            return '\t'
        # For .csv files sniff the first 4 KB to handle edge cases.
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as fh:
                sample = fh.read(4096)
            dialect = csv.Sniffer().sniff(sample, delimiters=',\t;|')
            return dialect.delimiter
        except csv.Error:
            return ','  # Safe default

    def _read_with_encoding(self, file_path: str, delimiter: str) -> Tuple[List[str], List[Dict]]:
        """
        Read the CSV/TSV file and return (headers, rows).

        Each item in *rows* is an ordered dict mapping header → cell value.
        """
        encodings = ['utf-8', 'utf-8-sig', 'utf-16', 'latin-1', 'cp1252']

        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding, newline='') as fh:
                    reader = csv.DictReader(fh, delimiter=delimiter)
                    headers = reader.fieldnames or []
                    rows = []
                    for i, row in enumerate(reader):
                        if i >= self.MAX_ROWS:
                            logger.warning(
                                f"File exceeds {self.MAX_ROWS:,} rows — truncating."
                            )
                            break
                        rows.append(dict(row))
                logger.info(f"Read {len(rows)} rows with {encoding} encoding")
                return list(headers), rows
            except (UnicodeDecodeError, UnicodeError):
                continue
            except Exception as exc:
                logger.warning(f"Error reading with {encoding}: {exc}")
                continue

        raise RuntimeError(f"Could not read {file_path} with any supported encoding")

    # ------------------------------------------------------------------
    # Row → plain text
    # ------------------------------------------------------------------

    @staticmethod
    def row_to_text(headers: List[str], row: Dict, row_index: int) -> str:
        """
        Convert one CSV row to a plain-text representation.

        Format:  "Row <N>: col1: val1 | col2: val2 | …"

        Empty / whitespace-only cells are omitted to keep the string clean.
        """
        parts = []
        for header in headers:
            value = str(row.get(header, '')).strip()
            if value:
                parts.append(f"{header}: {value}")

        body = " | ".join(parts) if parts else "(empty row)"
        return f"Row {row_index}: {body}"

    # ------------------------------------------------------------------
    # Standard extract() — satisfies BaseExtractor contract
    # ------------------------------------------------------------------

    def extract(self, file_path: str) -> Tuple[str, bool, Dict]:
        """
        Extract CSV/TSV file.

        Returns a plain-text representation (one line per data row) so the
        file can be handled by the normal pipeline. For row-level Qdrant
        storage use extract_and_store() instead.
        """
        try:
            is_valid, msg = self.validate_file(file_path)
            if not is_valid:
                return "", False, {'error': msg}

            logger.info(f"Extracting CSV/TSV: {file_path}")

            file_ext = Path(file_path).suffix.lower().lstrip('.')
            delimiter = self._detect_delimiter(file_path, file_ext)
            headers, rows = self._read_with_encoding(file_path, delimiter)

            if not rows:
                return "", False, {'error': 'File contains no data rows'}

            # Build plain-text content (one row per line, 1-based index)
            text_lines = [
                self.row_to_text(headers, row, idx + 1)
                for idx, row in enumerate(rows)
            ]
            plain_text = "\n".join(text_lines)

            metadata = self.get_metadata(file_path)
            metadata.update({
                'total_pages': 1,
                'row_count': len(rows),
                'column_count': len(headers),
                'columns': headers,
                'extraction_method': 'csv_row_by_row',
                'delimiter': repr(delimiter),
            })

            logger.info(
                f"✓ CSV extraction complete: {len(rows)} rows, "
                f"{len(headers)} columns"
            )
            return plain_text, True, metadata

        except Exception as exc:
            logger.error(f"CSV extraction error: {exc}")
            return "", False, {'error': str(exc)}

    # ------------------------------------------------------------------
    # Row-level Qdrant storage
    # ------------------------------------------------------------------

    def extract_and_store(
        self,
        file_path: str,
        file_id: str,
        filename: str,
        knowledge_base_id: str,
        knowledge_base_name: str,
        user_id: str,
        vector_store,          # VectorStoreService instance
        batch_size: int = 50,
    ) -> Tuple[bool, Dict]:
        """
        Extract the CSV/TSV file and store each row as a separate Qdrant point.

        Args:
            file_path:            Path to the CSV/TSV file.
            file_id:              Unique identifier for this file.
            filename:             Human-readable file name.
            knowledge_base_id:    Knowledge-base this file belongs to.
            knowledge_base_name:  Display name of the knowledge base.
            user_id:              Owning user.
            vector_store:         Initialised VectorStoreService.
            batch_size:           Number of rows to embed + upsert per batch.

        Returns:
            (success, summary_dict)
        """
        try:
            is_valid, msg = self.validate_file(file_path)
            if not is_valid:
                return False, {'error': msg}

            file_ext = Path(file_path).suffix.lower().lstrip('.')
            delimiter = self._detect_delimiter(file_path, file_ext)
            headers, rows = self._read_with_encoding(file_path, delimiter)

            if not rows:
                return False, {'error': 'File contains no data rows'}

            logger.info(
                f"Storing {len(rows)} CSV rows for file '{filename}' "
                f"(KB: {knowledge_base_name})"
            )

            total_stored = 0
            total_failed = 0

            # Process in batches so we don't overwhelm the embedding API
            for batch_start in range(0, len(rows), batch_size):
                batch_rows = rows[batch_start: batch_start + batch_size]

                # Build chunks in the format expected by VectorStoreService
                chunks = []
                for local_idx, row in enumerate(batch_rows):
                    global_idx = batch_start + local_idx + 1   # 1-based
                    row_text = self.row_to_text(headers, row, global_idx)

                    # Unique, deterministic chunk ID
                    chunk_id = _string_to_uuid(
                        f"{file_id}::row::{global_idx}"
                    )

                    chunks.append({
                        'text':        row_text,
                        'chunk_id':    chunk_id,
                        'chunk_index': global_idx - 1,   # 0-based for storage
                        'metadata': {
                            'chunk_type':  'csv_row',
                            'row_index':   global_idx,
                            'page_number': 1,
                            'columns':     ', '.join(headers),
                            'source_file': filename,
                            'file_ext':    file_ext,
                        },
                    })

                try:
                    stored = vector_store.add_document_chunks(
                        chunks=chunks,
                        file_id=file_id,
                        filename=filename,
                        knowledge_base_id=knowledge_base_id,
                        knowledge_base_name=knowledge_base_name,
                        user_id=user_id,
                    )
                    total_stored += stored
                    logger.info(
                        f"  Batch {batch_start + 1}–{batch_start + len(batch_rows)}: "
                        f"stored {stored} rows"
                    )
                except Exception as batch_exc:
                    logger.error(
                        f"  Batch {batch_start + 1}–{batch_start + len(batch_rows)} "
                        f"failed: {batch_exc}"
                    )
                    total_failed += len(batch_rows)

            summary = {
                'total_rows':    len(rows),
                'total_stored':  total_stored,
                'total_failed':  total_failed,
                'row_count':     len(rows),
                'column_count':  len(headers),
                'columns':       headers,
                'file_ext':      file_ext,
                'extraction_method': 'csv_row_by_row_qdrant',
            }

            success = total_stored > 0
            if success:
                logger.info(
                    f"✓ CSV storage complete for '{filename}': "
                    f"{total_stored}/{len(rows)} rows stored"
                )
            else:
                logger.error(f"✗ No rows stored for '{filename}'")

            return success, summary

        except Exception as exc:
            logger.error(f"CSV extract_and_store error: {exc}")
            return False, {'error': str(exc)}

    # ------------------------------------------------------------------
    # Streaming helper (optional — for very large files)
    # ------------------------------------------------------------------

    def iter_row_texts(
        self, file_path: str
    ) -> Generator[Tuple[int, str], None, None]:
        """
        Lazily yield (row_index, row_text) pairs without loading the whole
        file into memory. Useful for very large CSV files.

        Args:
            file_path: Path to the CSV/TSV file.

        Yields:
            (1-based row index, plain-text representation of that row)
        """
        file_ext = Path(file_path).suffix.lower().lstrip('.')
        delimiter = self._detect_delimiter(file_path, file_ext)

        encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding, newline='') as fh:
                    reader = csv.DictReader(fh, delimiter=delimiter)
                    headers = list(reader.fieldnames or [])
                    for row_idx, row in enumerate(reader, start=1):
                        if row_idx > self.MAX_ROWS:
                            logger.warning(
                                f"Streaming stopped at {self.MAX_ROWS:,} rows."
                            )
                            return
                        yield row_idx, self.row_to_text(headers, dict(row), row_idx)
                return  # Success — stop trying encodings
            except (UnicodeDecodeError, UnicodeError):
                continue
            except Exception as exc:
                logger.error(f"Streaming error with {encoding}: {exc}")
                continue
