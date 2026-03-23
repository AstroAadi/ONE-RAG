# """
# ZIP Extractor
# Mirrors DocxExtractor architecture exactly:
#   validate → extract (Strategy 1: Docling per-file) → (Strategy 2: native per-file fallback)
#   → image description pipeline → markdown assembly

# Folder traversal:
#   The extractor builds a nested folder tree from the ZIP's flat namelist and
#   processes entries recursively, folder by folder.  Each folder gets its own
#   Markdown heading so the output reflects the real directory structure:

#     # Archive: my_archive
#     ## 📁 / (root)
#     ### 📄 readme.txt
#     ## 📁 docs/
#     ### 📄 docs/report.pdf
#     ## 📁 docs/images/
#     ### 🖼️ docs/images/chart.png

# Supported entry types inside the ZIP:
#   - Registry-routed files: any extension registered in ExtractorFactory
#     (for example .xml, .json, .html, .pdf, .docx, .xlsx, .pptx, .msg, .epub, .md, .txt)
#   - Native text fallback : .yaml, .yml, .log and other plain-text formats without a registered extractor
#   - Images               : .png, .jpg, .jpeg, .gif, .webp, .bmp  (described via Vision LLM)
#   - Nested ZIPs : .zip  (extracted recursively up to MAX_NESTED_ZIP_DEPTH=5 levels)
#   - Skipped     : binaries, executables, hidden files, password-protected ZIPs (logged)
# """

# import os
# import io
# import re
# import zipfile
# import logging
# import traceback
# import collections
# import tempfile
# from pathlib import Path
# from typing import Tuple, Dict, List, Optional

# try:
#     from ..base_extractor import BaseExtractor
# except ImportError:
#     from base_extractor import BaseExtractor

# logger = logging.getLogger(__name__)

# # ── Supported entry categories ─────────────────────────────────────────────────
# TEXT_EXTENSIONS  = {'.txt', '.md', '.csv', '.json', '.xml', '.html', '.htm',
#                     '.yaml', '.yml', '.log', '.eml', '.properties', '.vtt'}
# IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp', '.tiff', '.tif'}
# DOC_EXTENSIONS   = {'.pdf', '.docx', '.xlsx', '.xlsm', '.pptx', '.msg', '.epub'}
# ZIP_EXTENSIONS   = {'.zip'}                # nested ZIPs — extracted recursively

# # Max raw size we attempt to read per text entry (avoid huge CSVs/logs)
# MAX_TEXT_ENTRY_BYTES  = 512 * 1024   # 512 KB
# # Max nesting depth for ZIP-inside-ZIP (prevents infinite recursion / zip bombs)
# MAX_NESTED_ZIP_DEPTH  = 5


# class ZIPExtractor(BaseExtractor):
#     """
#     ZIP Extractor with Docling + native fallback strategies.

#     Mirrors DocxExtractor structure:
#       Strategy 1 — Docling  (tries to convert each extractable entry)
#       Strategy 2 — Native   (direct text read / image description via Vision LLM)

#     Returns the same (markdown, success, metadata) tuple as all other extractors.
#     """

#     # ── Construction ──────────────────────────────────────────────────────────

#     def __init__(self, config: Optional[Dict] = None):
#         super().__init__(config)

#         from app.services.image_cache_manager import get_image_cache
#         from app.services.image_description_service import get_image_description_service

#         self.image_cache    = get_image_cache()
#         self.image_service  = get_image_description_service()

#         self._duplicates_skipped = 0

#         logger.info("ZIP Extractor initialized")

#     # ── BaseExtractor interface ────────────────────────────────────────────────

#     def _get_supported_formats(self) -> List[str]:
#         return ['zip']

#     # ── Validation ────────────────────────────────────────────────────────────

#     def validate_file(self, file_path: str) -> Tuple[bool, str]:
#         """
#         Validate ZIP file — mirrors DocxExtractor.validate_file().

#         Checks:
#           1. File exists
#           2. Extension is .zip
#           3. File is not empty
#           4. zipfile.is_zipfile() confirms valid ZIP magic bytes
#           5. Archive is not password-protected (best-effort check)
#         """
#         path = Path(file_path)

#         if not path.exists():
#             return False, f"File does not exist: {file_path}"
#         if path.suffix.lower() != '.zip':
#             return False, f"Not a ZIP file: {path.suffix}"
#         if path.stat().st_size == 0:
#             return False, "File is empty (0 bytes)"

#         # Check ZIP magic bytes
#         if not zipfile.is_zipfile(file_path):
#             return False, "File is not a valid ZIP archive (bad magic bytes)"

#         # Best-effort: detect encryption by checking first entry flags
#         try:
#             with zipfile.ZipFile(file_path, 'r') as zf:
#                 if not zf.namelist():
#                     return False, "ZIP archive is empty (no entries)"
#                 first = zf.infolist()[0]
#                 if first.flag_bits & 0x1:   # bit 0 = encryption flag
#                     return False, "ZIP archive is password-protected — decryption not supported"
#         except zipfile.BadZipFile as e:
#             return False, f"Corrupt ZIP archive: {e}"
#         except Exception as e:
#             logger.warning(f"Could not inspect ZIP entries: {e}")

#         return True, "Valid ZIP archive"

#     # ── Main extract ──────────────────────────────────────────────────────────

#     def extract(self, file_path: str) -> Tuple[str, bool, Dict]:
#         """
#         Extract ZIP with two-strategy fallback — mirrors DocxExtractor.extract().

#         Strategy 1: Docling (attempts to convert each extractable entry)
#         Strategy 2: Native  (direct text read + Vision LLM for images)
#         """
#         try:
#             logger.info("=" * 80)
#             logger.info("ZIP EXTRACTION - START")
#             logger.info("=" * 80)
#             logger.info(f"File path: {file_path}")

#             # Validate
#             is_valid, msg = self.validate_file(file_path)
#             if not is_valid:
#                 logger.error(f"❌ Validation failed: {msg}")
#                 return "", False, {'error': msg}

#             logger.info("✅ File validation passed")

#             file_size = Path(file_path).stat().st_size
#             logger.info(f"📊 File size: {file_size:,} bytes ({file_size / 1024 / 1024:.2f} MB)")

#             # List entries
#             with zipfile.ZipFile(file_path, 'r') as zf:
#                 all_entries   = zf.namelist()
#                 total_entries = len(all_entries)
#             logger.info(f"📦 Total ZIP entries: {total_entries}")

#             zip_stem = Path(file_path).stem

#             # ── Strategy 1: Docling ───────────────────────────────────────
#             logger.info("\n" + "=" * 60)
#             logger.info("STRATEGY 1: Trying Docling extraction...")
#             logger.info("=" * 60)

#             docling_result = self._try_docling_extraction(file_path, zip_stem)
#             if docling_result[1]:
#                 logger.info("✅✅✅ Docling extraction SUCCESSFUL")
#                 return docling_result
#             else:
#                 logger.warning(f"⚠️ Docling failed: {docling_result[2].get('error', 'Unknown')}")

#             # ── Strategy 2: Native ────────────────────────────────────────
#             logger.info("\n" + "=" * 60)
#             logger.info("STRATEGY 2: Trying native extraction...")
#             logger.info("=" * 60)

#             result = self._extract_native(file_path, zip_stem)

#             if result[1]:
#                 logger.info("✅✅✅ Native extraction SUCCESSFUL")
#             else:
#                 logger.error(f"❌❌❌ Native extraction FAILED: {result[2].get('error', 'Unknown')}")

#             logger.info("=" * 80)
#             logger.info("ZIP EXTRACTION - END")
#             logger.info("=" * 80)

#             return result

#         except Exception as e:
#             logger.error(f"❌❌❌ CRITICAL ERROR in extract(): {str(e)}")
#             traceback.print_exc()
#             return "", False, {'error': str(e)}

#     # ── Per-file chunked indexing ─────────────────────────────────────────────

#     def extract_chunks(self, file_path: str) -> Tuple[List[Dict], bool, Dict]:
#         """
#         Extract ZIP and return one chunk dict per file entry, ready for indexing.

#         Each chunk has:
#           {
#             'content'           : str,   # Markdown content for this file only
#             'source_path'       : str,   # e.g. "docs/report.pdf"
#             'file_name'         : str,   # e.g. "report.pdf"
#             'file_type'         : str,   # e.g. ".pdf"
#             'chunk_index'       : int,   # 0-based position within the archive
#             'archive_name'      : str,   # ZIP stem, e.g. "my_archive"
#             'extraction_method' : str,   # "docling" | "native_text" | "vision_llm" | ...
#           }

#         Returns (chunks, success, metadata) — mirrors the extract() contract but
#         replaces the monolithic markdown string with a list of per-file chunks.
#         """
#         is_valid, msg = self.validate_file(file_path)
#         if not is_valid:
#             return [], False, {'error': msg}

#         archive_name = Path(file_path).stem

#         try:
#             with tempfile.TemporaryDirectory(prefix=f"zip_extract_{archive_name}_") as temp_dir:
#                 extract_root = self._extract_archive_to_temp_tree(file_path, temp_dir)
#                 chunks = self._build_file_chunks_from_directory(
#                     root_dir=extract_root,
#                     archive_name=archive_name,
#                 )
#                 metadata = {
#                     'archive_name': archive_name,
#                     'extraction_method': 'temp_dir_per_file',
#                     'file_chunks': chunks,
#                     'files_extracted': len(chunks),
#                 }
#                 logger.info(
#                     f"extract_chunks: ZIP temp extraction complete for '{archive_name}' "
#                     f"-> {len(chunks)} file chunk(s)"
#                 )
#                 return chunks, True, metadata
#         except Exception as e:
#             logger.error(f"extract_chunks failed for ZIP '{file_path}': {e}", exc_info=True)
#             return [], False, {'error': str(e)}

#     # ── Strategy 1: Docling ───────────────────────────────────────────────────

#     def _extract_archive_to_temp_tree(self, file_path: str, temp_dir: str) -> Path:
#         """Extract the full ZIP tree into a temporary directory, preserving folders."""
#         archive_root = Path(temp_dir) / Path(file_path).stem
#         archive_root.mkdir(parents=True, exist_ok=True)

#         with zipfile.ZipFile(file_path, 'r') as zf:
#             for info in zf.infolist():
#                 raw_name = info.filename
#                 if not raw_name:
#                     continue

#                 normalized = Path(raw_name)
#                 if normalized.is_absolute() or ".." in normalized.parts:
#                     logger.warning(f"Skipping unsafe ZIP entry path: {raw_name}")
#                     continue

#                 target_path = archive_root / normalized
#                 target_path.parent.mkdir(parents=True, exist_ok=True)

#                 if info.is_dir():
#                     target_path.mkdir(parents=True, exist_ok=True)
#                     continue

#                 with zf.open(info, 'r') as src, open(target_path, 'wb') as dst:
#                     dst.write(src.read())

#         return archive_root

#     def _build_file_chunks_from_directory(
#         self,
#         root_dir: Path,
#         archive_name: str,
#     ) -> List[Dict]:
#         """Process each extracted file independently and return one chunk record per file."""
#         chunks: List[Dict] = []

#         for file_path in sorted(p for p in root_dir.rglob('*') if p.is_file()):
#             source_path = file_path.relative_to(root_dir).as_posix()

#             if self._should_skip_entry(source_path):
#                 continue

#             suffix = file_path.suffix.lower()
#             method = "unknown"

#             entry_type = self._classify_entry_suffix(suffix)

#             if entry_type == "nested_zip":
#                 content, ok, _ = self.extract(str(file_path))
#                 if not ok or not content.strip():
#                     continue
#                 method = "nested_zip"

#             elif entry_type == "delegated":
#                 content = self._delegate_to_sub_extractor(
#                     str(file_path),
#                     suffix,
#                     source_path=source_path,
#                 ) or ""
#                 if not content.strip():
#                     continue
#                 method = "registered_extractor"

#             elif entry_type == "native_text":
#                 content, ok = self._read_text_file(file_path)
#                 if not ok or not content.strip():
#                     continue
#                 method = "native_text"
#                 content = self._format_text_file_chunk(source_path, suffix, content)

#             elif entry_type == "image":
#                 content, ok = self._describe_image_file(file_path, source_path, archive_name)
#                 if not content.strip():
#                     continue
#                 method = "vision_llm" if ok else "vision_fallback"

#             else:
#                 logger.debug(f"Skipping unsupported ZIP entry for chunking: {source_path}")
#                 continue

#             chunks.append({
#                 'content': content,
#                 'source_path': source_path,
#                 'file_name': file_path.name,
#                 'file_type': suffix,
#                 'chunk_index': len(chunks),
#                 'archive_name': archive_name,
#                 'extraction_method': method,
#             })

#         return chunks

#     def _get_extractor_factory(self):
#         """Load the shared extractor registry lazily to avoid import cycles."""
#         try:
#             from app.services.extraction.extractor_factory import (
#                 ExtractorFactory,
#                 auto_register_extractors,
#             )
#         except ImportError:
#             from ..extractor_factory import ExtractorFactory, auto_register_extractors

#         if not ExtractorFactory.get_supported_formats() and not ExtractorFactory.get_unavailable_formats():
#             auto_register_extractors()

#         return ExtractorFactory

#     def _has_registered_extractor(self, suffix: str) -> bool:
#         """Return True when the shared extractor registry knows this suffix."""
#         clean_suffix = suffix.lower().lstrip('.')
#         if not clean_suffix:
#             return False

#         try:
#             return self._get_extractor_factory().is_supported(clean_suffix)
#         except Exception as e:
#             logger.debug("Could not inspect extractor registry for %s: %s", suffix, e)
#             return False

#     def _classify_entry_suffix(self, suffix: str) -> str:
#         """
#         Classify a ZIP entry so routing stays extension-driven.

#         Priority matters:
#         - nested ZIPs recurse
#         - images use the ZIP-specific image pipeline
#         - registered extractors handle everything else they support
#         - plain text falls back to native reading only when no extractor exists
#         """
#         clean_suffix = suffix.lower()

#         if clean_suffix in ZIP_EXTENSIONS:
#             return "nested_zip"
#         if clean_suffix in IMAGE_EXTENSIONS:
#             return "image"
#         if self._has_registered_extractor(clean_suffix) or clean_suffix in DOC_EXTENSIONS:
#             return "delegated"
#         if clean_suffix in TEXT_EXTENSIONS:
#             return "native_text"
#         return "unsupported"

#     def _get_entry_emoji(self, suffix: str) -> str:
#         """Choose a heading emoji based on the routed entry type."""
#         entry_type = self._classify_entry_suffix(suffix)
#         if entry_type == "nested_zip":
#             return '📦'
#         if entry_type == "image":
#             return '🖼️'
#         if entry_type == "delegated" and suffix in TEXT_EXTENSIONS:
#             return '📄'
#         if entry_type == "delegated":
#             return '📑'
#         return '📄'

#     def _read_text_file(self, file_path: Path) -> Tuple[str, bool]:
#         """Read a text file from disk with the same truncation policy as ZIP entries."""
#         try:
#             raw = file_path.read_bytes()
#             if len(raw) > MAX_TEXT_ENTRY_BYTES:
#                 logger.warning(
#                     f"  ⚠️ {file_path.name} is {len(raw):,} bytes — truncating to {MAX_TEXT_ENTRY_BYTES:,}"
#                 )
#                 raw = raw[:MAX_TEXT_ENTRY_BYTES]

#             try:
#                 return raw.decode('utf-8'), True
#             except UnicodeDecodeError:
#                 return raw.decode('latin-1', errors='replace'), True
#         except Exception as e:
#             logger.warning(f"Could not read extracted text file {file_path}: {e}")
#             return "", False

#     def _format_text_file_chunk(self, source_path: str, suffix: str, content: str) -> str:
#         """Wrap plain text entries with a small header so chunk content stays self-describing."""
#         lang = suffix.lstrip('.')
#         return f"### `{source_path}`\n\n```{lang}\n{content}\n```"

#     def _build_image_description_body(
#         self,
#         description: str,
#         source_path: Optional[str] = None,
#         status_note: Optional[str] = None,
#     ) -> str:
#         """Build a consistent markdown body for image descriptions."""
#         safe_description = (description or "[Image description unavailable]").strip()
#         lines = [
#             "Description:",
#             safe_description,
#         ]

#         notes: List[str] = []
#         if source_path:
#             notes.append(f"Source: `{source_path}`")
#         if status_note:
#             notes.append(status_note)

#         if notes:
#             lines.extend(["", f"*{' | '.join(notes)}*"])

#         return "\n".join(lines)

#     def _build_image_markdown_block(
#         self,
#         image_name: str,
#         description: str,
#         source_path: Optional[str] = None,
#         status_note: Optional[str] = None,
#     ) -> str:
#         """Build the full markdown block for a ZIP image."""
#         body = self._build_image_description_body(
#             description,
#             source_path=source_path,
#             status_note=status_note,
#         )
#         return f"### 🖼️ {image_name}\n\n{body}"

#     def _describe_image_file(
#         self,
#         file_path: Path,
#         source_path: str,
#         archive_name: str,
#     ) -> Tuple[str, bool]:
#         """Describe an extracted image file and return markdown content for chunking."""
#         try:
#             image_bytes = file_path.read_bytes()
#             if not image_bytes:
#                 return "", False

#             success = True
#             status_note = None
#             cached_desc = self.image_cache.get(image_bytes)
#             if cached_desc:
#                 description = cached_desc
#                 status_note = "cached"
#             else:
#                 media_type = self._get_media_type(file_path.suffix.lower().lstrip('.'))
#                 success, description = self.image_service.describe_image_from_bytes(
#                     image_bytes,
#                     media_type,
#                     context=f"ZIP archive: {archive_name}",
#                     page_number=0,
#                     filename=Path(source_path).name,
#                 )
#                 status_note = "generated" if success else "fallback"
#                 if success:
#                     self.image_cache.put(image_bytes, description)

#             if not success:
#                 logger.warning("Image description fallback used for %s", source_path)

#             markdown = self._build_image_markdown_block(
#                 Path(source_path).name,
#                 description,
#                 source_path=source_path,
#                 status_note=status_note,
#             )
#             return markdown, success
#         except Exception as e:
#             logger.warning(f"Could not describe extracted image {file_path}: {e}")
#             fallback = self.image_service.generate_fallback_description(
#                 Path(source_path).name,
#                 page_number=0,
#                 image_type="image",
#             )
#             markdown = self._build_image_markdown_block(
#                 Path(source_path).name,
#                 fallback,
#                 source_path=source_path,
#                 status_note="fallback",
#             )
#             return markdown, False

#     def _try_docling_extraction(
#         self, file_path: str, zip_stem: str
#     ) -> Tuple[str, bool, Dict]:
#         """
#         Try Docling on each extractable entry — mirrors
#         DocxExtractor._try_docling_extraction_debug().

#         Docling still handles PDF and DOCX entries directly.
#         Other registered formats are delegated through the shared extractor
#         registry; images keep the ZIP-specific image pipeline.
#         """
#         try:
#             logger.info("Attempting to import Docling...")
#             from docling.document_converter import DocumentConverter
#             logger.info("✅ Docling imported successfully")

#             user_id = Path(file_path).parent.name
#             if user_id == "uploads":
#                 user_id = "default"

#             image_output_dir = self._create_image_directory_user_isolated(
#                 file_path, zip_stem, user_id
#             )
#             logger.info(f"Image directory: {image_output_dir}")

#             converter = DocumentConverter()

#             result_lines: List[str] = []
#             result_lines.append(f"# Archive: {zip_stem}")
#             result_lines.append(f"**Format**: ZIP Archive")
#             result_lines.append(f"**Extraction Method**: Docling")
#             result_lines.append("")

#             self._duplicates_skipped = 0
#             images_extracted  = 0
#             images_described  = 0
#             entries_processed = 0
#             entries_skipped   = 0
#             doc_entries: List[Dict] = []   # collect for metadata
#             file_chunks: List[Dict] = []   # one chunk per file, for per-file indexing

#             def _make_chunk_docling(entry_name: str, content: str, method: str) -> Dict:
#                 return {
#                     'content'           : content,
#                     'source_path'       : entry_name,
#                     'file_name'         : Path(entry_name).name,
#                     'file_type'         : Path(entry_name).suffix.lower(),
#                     'chunk_index'       : len(file_chunks),
#                     'archive_name'      : zip_stem,
#                     'extraction_method' : method,
#                 }

#             with zipfile.ZipFile(file_path, 'r') as zf:
#                 all_names = zf.namelist()
#                 folder_tree = self._build_folder_tree(all_names)

#                 logger.info(f"📁 Folder tree built: {len(folder_tree)} top-level nodes")

#                 def _walk_docling(tree: Dict, zf, depth: int = 0) -> None:
#                     nonlocal images_extracted, images_described, entries_processed, entries_skipped

#                     for node_name in sorted(tree.keys()):
#                         node = tree[node_name]

#                         if node['type'] == 'folder':
#                             folder_path = node['path']
#                             heading_level = '#' * min(depth + 2, 6)
#                             result_lines.append(
#                                 f"\n{heading_level} 📁 {'/ (root)' if not folder_path else folder_path + '/'}\n"
#                             )
#                             logger.info(f"  {'  ' * depth}📁 Entering folder: {folder_path or '(root)'}")
#                             _walk_docling(node['children'], zf, depth + 1)

#                         else:
#                             # It's a file node
#                             entry_name = node['path']

#                             if self._should_skip_entry(entry_name):
#                                 entries_skipped += 1
#                                 continue

#                             suffix = Path(entry_name).suffix.lower()
#                             entry_type = self._classify_entry_suffix(suffix)
#                             file_heading = '#' * min(depth + 3, 6)
#                             emoji = self._get_entry_emoji(suffix)
#                             result_lines.append(f"\n{file_heading} {emoji} `{Path(entry_name).name}`\n")

#                             if entry_type == "nested_zip":
#                                 # ── Nested ZIP ──────────────────────────────
#                                 if depth >= MAX_NESTED_ZIP_DEPTH:
#                                     logger.warning(f"  {'  ' * depth}⚠️ Max nesting depth reached — skipping {entry_name}")
#                                     result_lines.append(
#                                         f"*(Nested ZIP skipped — max depth {MAX_NESTED_ZIP_DEPTH} reached)*"
#                                     )
#                                     entries_skipped += 1
#                                 else:
#                                     logger.info(f"  {'  ' * depth}📦 Nested ZIP found: {entry_name}")
#                                     nested_md = self._extract_nested_zip(
#                                         zf, entry_name, image_output_dir, zip_stem, depth
#                                     )
#                                     result_lines.append(nested_md)
#                                     entries_processed += 1
#                                     doc_entries.append({
#                                         'name': entry_name, 'suffix': suffix, 'method': 'nested_zip'
#                                     })

#                             elif entry_type == "delegated":
#                                 tmp_path = self._write_temp_entry(zf, entry_name, image_output_dir)
#                                 if not tmp_path:
#                                     result_lines.append("*(Could not extract entry to temp file)*")
#                                 else:
#                                     try:
#                                         # Docling handles PDF/DOCX natively.
#                                         # Everything else goes through the shared extractor registry.
#                                         if suffix in {'.pdf', '.docx'}:
#                                             logger.info(f"  {'  ' * depth}Docling converting: {entry_name}")
#                                             doc_result = converter.convert(tmp_path)
#                                             md_text = doc_result.document.export_to_markdown()
#                                             method = 'docling'
#                                         else:
#                                             logger.info(f"  {'  ' * depth}Registered extractor for: {entry_name}")
#                                             md_text = self._delegate_to_sub_extractor(
#                                                 tmp_path,
#                                                 suffix,
#                                                 source_path=entry_name,
#                                             ) or ''
#                                             method = 'registered_extractor'

#                                         if md_text.strip():
#                                             result_lines.append(md_text)
#                                             file_chunks.append(
#                                                 _make_chunk_docling(entry_name, md_text, method)
#                                             )
#                                             logger.info(f"  {'  ' * depth}✅ {entry_name}: {len(md_text):,} chars")
#                                         else:
#                                             logger.warning(f"  ⚠️ {entry_name}: no content extracted")
#                                             result_lines.append("*(No content extracted)*")

#                                         entries_processed += 1
#                                         doc_entries.append({'name': entry_name, 'suffix': suffix, 'method': method})
#                                     except Exception as e:
#                                         logger.warning(f"  ⚠️ Extraction failed for {entry_name}: {e}")
#                                         result_lines.append(f"*(Extraction failed: {e})*")
#                                     finally:
#                                         try:
#                                             os.unlink(tmp_path)
#                                         except Exception:
#                                             pass

#                             elif entry_type == "native_text":
#                                 content, ok = self._read_text_entry(zf, entry_name)
#                                 if ok and content:
#                                     block = f"```{suffix.lstrip('.')}\n{content}\n```"
#                                     result_lines.append(block)
#                                     file_chunks.append(
#                                         _make_chunk_docling(entry_name, content, 'native_text')
#                                     )
#                                     entries_processed += 1
#                                     doc_entries.append({'name': entry_name, 'suffix': suffix, 'method': 'native_text'})
#                                 else:
#                                     result_lines.append("*(Could not read text content)*")

#                             elif entry_type == "image":
#                                 img_md, ok = self._describe_image_entry(
#                                     zf, entry_name, image_output_dir, zip_stem
#                                 )
#                                 result_lines.append(img_md)
#                                 if img_md.strip():
#                                     file_chunks.append(
#                                         _make_chunk_docling(
#                                             entry_name,
#                                             img_md,
#                                             'vision_llm' if ok else 'vision_fallback',
#                                         )
#                                     )
#                                     doc_entries.append({
#                                         'name': entry_name,
#                                         'suffix': suffix,
#                                         'method': 'vision_llm' if ok else 'vision_fallback',
#                                     })
#                                 if ok:
#                                     images_extracted += 1
#                                     images_described += 1
#                                     entries_processed += 1

#                             else:
#                                 result_lines.append(f"*(Unsupported type — skipped: `{suffix or 'no extension'}`)*")
#                                 entries_skipped += 1

#                             result_lines.append("")

#                 _walk_docling(folder_tree, zf, depth=0)

#             final_markdown = '\n'.join(result_lines)

#             if len(final_markdown.strip()) < 100:
#                 return "", False, {'error': 'Docling pass produced insufficient content'}

#             cache_stats = self.image_cache.get_stats()
#             metadata = self.get_metadata(file_path)
#             metadata.update({
#                 'extractor':               'ZipExtractor_Docling',
#                 'total_entries':           len(zf.namelist()) if False else entries_processed + entries_skipped,
#                 'entries_processed':       entries_processed,
#                 'entries_skipped':         entries_skipped,
#                 'images_extracted':        images_extracted,
#                 'images_described':        images_described,
#                 'images_skipped_duplicates': self._duplicates_skipped,
#                 'images_cached':           cache_stats['hits'],
#                 'cache_hit_rate':          cache_stats['hit_rate_percent'],
#                 'image_directory':         str(image_output_dir),
#                 'extraction_method':       'docling',
#                 'doc_entries':             doc_entries,
#                 'file_chunks':             file_chunks,
#             })

#             return final_markdown, True, metadata

#         except ImportError as e:
#             logger.info(f"⚠️ Docling not available: {e}")
#             return "", False, {'error': 'Docling not installed'}
#         except Exception as e:
#             logger.error(f"❌ Docling strategy error: {str(e)}")
#             traceback.print_exc()
#             return "", False, {'error': str(e)}

#     # ── Strategy 2: Native ────────────────────────────────────────────────────

#     def _extract_native(
#         self, file_path: str, zip_stem: str
#     ) -> Tuple[str, bool, Dict]:
#         """
#         Native extraction — mirrors DocxExtractor._extract_with_python_docx_debug().

#         - Registry-routed entries → call the registered extractor for that suffix
#         - Native text entries     → read and embed as fenced code blocks
#         - Image entries           → describe via Vision LLM (with dedup + caching)
#         - Nested ZIPs             → recurse up to MAX_NESTED_ZIP_DEPTH
#         - Everything else         → log and skip
#         """
#         try:
#             user_id = Path(file_path).parent.name
#             if user_id == "uploads":
#                 user_id = "default"

#             image_output_dir = self._create_image_directory_user_isolated(
#                 file_path, zip_stem, user_id
#             )

#             with zipfile.ZipFile(file_path, 'r') as zf:
#                 all_entries   = [n for n in zf.namelist() if not n.endswith('/')]
#                 total_entries = len(all_entries)

#             logger.info(f"📊 ZIP structure:")
#             logger.info(f"  - Total file entries : {total_entries}")

#             entry_types = {
#                 entry_name: self._classify_entry_suffix(Path(entry_name).suffix.lower())
#                 for entry_name in all_entries
#             }
#             native_text_entries = [n for n, entry_type in entry_types.items() if entry_type == "native_text"]
#             image_entries = [n for n, entry_type in entry_types.items() if entry_type == "image"]
#             delegated_entries = [n for n, entry_type in entry_types.items() if entry_type == "delegated"]
#             zip_entries = [n for n, entry_type in entry_types.items() if entry_type == "nested_zip"]
#             other_entries = [n for n, entry_type in entry_types.items() if entry_type == "unsupported"]

#             logger.info(f"  - Native text  : {len(native_text_entries)}")
#             logger.info(f"  - Image files  : {len(image_entries)}")
#             logger.info(f"  - Delegated    : {len(delegated_entries)}")
#             logger.info(f"  - Nested ZIPs  : {len(zip_entries)}")
#             logger.info(f"  - Other/skipped: {len(other_entries)}")

#             if total_entries == 0:
#                 return "", False, {'error': 'ZIP archive has no file entries'}

#             self._duplicates_skipped = 0
#             result_lines: List[str] = []
#             image_data:   Dict[int, Dict] = {}
#             doc_entry_meta: List[Dict] = []
#             file_chunks: List[Dict] = []   # one chunk per file, for per-file indexing

#             def _make_chunk_native(entry_name: str, content: str, method: str) -> Dict:
#                 return {
#                     'content'           : content,
#                     'source_path'       : entry_name,
#                     'file_name'         : Path(entry_name).name,
#                     'file_type'         : Path(entry_name).suffix.lower(),
#                     'chunk_index'       : len(file_chunks),
#                     'archive_name'      : zip_stem,
#                     'extraction_method' : method,
#                 }

#             # ── Document header ───────────────────────────────────────────
#             result_lines.append(f"# Archive: {zip_stem}")
#             result_lines.append(f"**Format**: ZIP Archive")
#             result_lines.append(f"**Total Entries**: {total_entries}")
#             result_lines.append(f"**Native Text Files**: {len(native_text_entries)}")
#             result_lines.append(f"**Image Files**: {len(image_entries)}")
#             result_lines.append(f"**Extractor-routed Files**: {len(delegated_entries)}")
#             if zip_entries:
#                 result_lines.append(f"**Nested ZIP Archives**: {len(zip_entries)}")
#             result_lines.append("")

#             with zipfile.ZipFile(file_path, 'r') as zf:
#                 folder_tree = self._build_folder_tree(zf.namelist())
#                 logger.info(f"📁 Folder tree built: {len(folder_tree)} top-level nodes")

#                 def _walk_native(tree: Dict, zf, depth: int = 0) -> None:
#                     for node_name in sorted(tree.keys()):
#                         node = tree[node_name]

#                         if node['type'] == 'folder':
#                             folder_path = node['path']
#                             heading_level = '#' * min(depth + 2, 6)
#                             result_lines.append(
#                                 f"\n{heading_level} 📁 {'/ (root)' if not folder_path else folder_path + '/'}\n"
#                             )
#                             logger.info(f"  {'  ' * depth}📁 Entering folder: {folder_path or '(root)'}")
#                             _walk_native(node['children'], zf, depth + 1)

#                         else:
#                             entry_name = node['path']

#                             if self._should_skip_entry(entry_name):
#                                 continue

#                             suffix = Path(entry_name).suffix.lower()
#                             entry_type = self._classify_entry_suffix(suffix)
#                             file_heading = '#' * min(depth + 3, 6)

#                             if entry_type == "nested_zip":
#                                 # ── Nested ZIP ──────────────────────────────
#                                 result_lines.append(f"\n{file_heading} 📦 `{Path(entry_name).name}`\n")
#                                 if depth >= MAX_NESTED_ZIP_DEPTH:
#                                     logger.warning(f"  {'  ' * depth}⚠️ Max nesting depth reached — skipping {entry_name}")
#                                     result_lines.append(
#                                         f"*(Nested ZIP skipped — max depth {MAX_NESTED_ZIP_DEPTH} reached)*"
#                                     )
#                                 else:
#                                     logger.info(f"  {'  ' * depth}📦 Nested ZIP found: {entry_name}")
#                                     nested_md = self._extract_nested_zip(
#                                         zf, entry_name, image_output_dir, zip_stem, depth
#                                     )
#                                     result_lines.append(nested_md)
#                                     img_idx = len(image_data) + 1
#                                     doc_entry_meta.append({
#                                         'name': entry_name, 'type': 'nested_zip', 'method': 'recursive'
#                                     })

#                             elif entry_type == "native_text":
#                                 result_lines.append(f"\n{file_heading} 📄 `{Path(entry_name).name}`\n")
#                                 logger.info(f"  {'  ' * depth}📄 Reading text: {entry_name}")
#                                 content, ok = self._read_text_entry(zf, entry_name)
#                                 if ok and content:
#                                     result_lines.append(f"```{suffix.lstrip('.')}\n{content}\n```")
#                                     file_chunks.append(
#                                         _make_chunk_native(entry_name, content, 'native_text')
#                                     )
#                                     logger.info(f"  {'  ' * depth}  ✅ {len(content):,} chars")
#                                     doc_entry_meta.append({'name': entry_name, 'type': 'text',
#                                                            'chars': len(content), 'method': 'native'})
#                                 else:
#                                     result_lines.append("*(Could not read or empty file)*")
#                                     logger.warning(f"  {'  ' * depth}  ⚠️ Could not read {entry_name}")

#                             elif entry_type == "image":
#                                 result_lines.append(f"\n{file_heading} 🖼️ `{Path(entry_name).name}`\n")
#                                 logger.info(f"  {'  ' * depth}🖼️  Describing image: {entry_name}")
#                                 img_md, ok = self._describe_image_entry(
#                                     zf, entry_name, image_output_dir, zip_stem
#                                 )
#                                 result_lines.append(img_md)
#                                 if img_md.strip():
#                                     file_chunks.append(
#                                         _make_chunk_native(
#                                             entry_name,
#                                             img_md,
#                                             'vision_llm' if ok else 'vision_fallback',
#                                         )
#                                     )
#                                     doc_entry_meta.append({
#                                         'name': entry_name,
#                                         'type': 'image',
#                                         'method': 'vision_llm' if ok else 'vision_fallback',
#                                     })
#                                 if ok:
#                                     img_idx = len(image_data) + 1
#                                     image_data[img_idx] = {'name': entry_name, 'success': True}

#                             elif entry_type == "delegated":
#                                 result_lines.append(
#                                     f"\n{file_heading} {self._get_entry_emoji(suffix)} `{Path(entry_name).name}`\n"
#                                 )
#                                 logger.info(f"  {'  ' * depth}📑 Routing through extractor: {entry_name}")
#                                 tmp_path = self._write_temp_entry(zf, entry_name, image_output_dir)
#                                 if not tmp_path:
#                                     result_lines.append("*(Could not extract to temp file)*")
#                                 else:
#                                     try:
#                                         sub_md = self._delegate_to_sub_extractor(
#                                             tmp_path,
#                                             suffix,
#                                             source_path=entry_name,
#                                         )
#                                         if sub_md:
#                                             result_lines.append(sub_md)
#                                             file_chunks.append(
#                                                 _make_chunk_native(entry_name, sub_md, 'registered_extractor')
#                                             )
#                                             doc_entry_meta.append({'name': entry_name, 'type': 'document',
#                                                                     'suffix': suffix, 'method': 'registered_extractor'})
#                                             logger.info(f"  {'  ' * depth}  ✅ Sub-extractor: {len(sub_md):,} chars")
#                                         else:
#                                             result_lines.append("*(Sub-extractor returned no content)*")
#                                     except Exception as e:
#                                         logger.warning(f"  {'  ' * depth}  ⚠️ Sub-extractor failed for {entry_name}: {e}")
#                                         result_lines.append(f"*(Sub-extractor failed: {e})*")
#                                     finally:
#                                         try:
#                                             os.unlink(tmp_path)
#                                         except Exception:
#                                             pass

#                             else:
#                                 result_lines.append(f"\n{file_heading} ⏭️ `{Path(entry_name).name}`\n")
#                                 result_lines.append(
#                                     f"*(Unsupported type — skipped: `{suffix or 'no extension'}`)*"
#                                 )

#                             result_lines.append("")

#                 _walk_native(folder_tree, zf, depth=0)

#             final_markdown = '\n'.join(result_lines)

#             logger.info(f"✅ Final markdown: {len(final_markdown):,} characters")

#             if len(final_markdown.strip()) < 100:
#                 logger.error(f"❌ Final markdown too short: {len(final_markdown)} chars")
#                 return "", False, {'error': 'Generated markdown is too short — ZIP may contain no readable content'}

#             # Metadata
#             cache_stats = self.image_cache.get_stats()
#             metadata = self.get_metadata(file_path)
#             metadata.update({
#                 'extractor':               'ZipExtractor_Native',
#                 'total_entries':           total_entries,
#                 'text_entries':            len(native_text_entries),
#                 'native_text_entries':     len(native_text_entries),
#                 'image_entries':           len(image_entries),
#                 'doc_entries':             len(delegated_entries),
#                 'delegated_entries':       len(delegated_entries),
#                 'other_entries':           len(other_entries),
#                 'images_extracted':        len(image_data),
#                 'images_described':        sum(1 for v in image_data.values() if v.get('success')),
#                 'images_skipped_duplicates': self._duplicates_skipped,
#                 'images_cached':           cache_stats['hits'],
#                 'cache_hit_rate':          cache_stats['hit_rate_percent'],
#                 'image_directory':         str(image_output_dir),
#                 'extraction_method':       'native',
#                 'folder_aware':            True,
#                 'entry_details':           doc_entry_meta,
#                 'file_chunks':             file_chunks,
#             })

#             logger.info("✅ Native extraction successful")
#             return final_markdown, True, metadata

#         except Exception as e:
#             logger.error(f"❌ Native extraction error: {str(e)}")
#             traceback.print_exc()
#             return "", False, {'error': str(e)}

#     # ── Folder tree builder ───────────────────────────────────────────────────

#     def _build_folder_tree(self, namelist: List[str]) -> Dict:
#         """
#         Build a nested folder tree from the ZIP's flat namelist.

#         ZIP files store all entries as flat paths, e.g.:
#             docs/report.pdf
#             docs/images/chart.png
#             readme.txt
#             src/utils/helper.py

#         This method converts that into a nested dict tree:
#             {
#               '(root)': {
#                 'type': 'folder', 'path': '',
#                 'children': {
#                   'readme.txt': {'type': 'file', 'path': 'readme.txt'},
#                   ...
#                 }
#               },
#               '(root)': {
#                 'type': 'folder', 'path': 'docs',
#                 'children': {
#                   'report.pdf': {'type': 'file', 'path': 'docs/report.pdf'},
#                   'images': {
#                     'type': 'folder', 'path': 'docs/images',
#                     'children': {
#                       'chart.png': {'type': 'file', 'path': 'docs/images/chart.png'}
#                     }
#                   }
#                 }
#               },
#               ...
#             }

#         Folder nodes: {'type': 'folder', 'path': str, 'children': dict}
#         File nodes  : {'type': 'file',   'path': str}
#         """
#         root: Dict = {}

#         for entry in namelist:
#             # Normalise — strip trailing slash (directory marker) and split
#             parts = [p for p in entry.replace('\\', '/').split('/') if p]

#             if not parts:
#                 continue

#             node = root
#             accumulated_path = ''

#             for i, part in enumerate(parts):
#                 accumulated_path = '/'.join(parts[:i + 1])
#                 is_last = (i == len(parts) - 1)
#                 is_dir_entry = entry.endswith('/') and is_last

#                 if is_last and not is_dir_entry:
#                     # This part is a file
#                     if part not in node:
#                         node[part] = {'type': 'file', 'path': accumulated_path}
#                 else:
#                     # This part is a folder component
#                     if part not in node:
#                         node[part] = {
#                             'type': 'folder',
#                             'path': accumulated_path,
#                             'children': {},
#                         }
#                     elif node[part]['type'] == 'file':
#                         # Edge case: a name appears as both file and directory prefix
#                         node[part] = {
#                             'type': 'folder',
#                             'path': accumulated_path,
#                             'children': {},
#                         }
#                     node = node[part]['children']

#         # Hoist root-level files under a virtual '(root)' folder node so the
#         # walk function has a uniform structure to iterate
#         root_files   = {k: v for k, v in root.items() if v['type'] == 'file'}
#         root_folders = {k: v for k, v in root.items() if v['type'] == 'folder'}

#         if root_files:
#             root_folders['(root)'] = {
#                 'type': 'folder',
#                 'path': '',
#                 'children': root_files,
#             }

#         return root_folders

#     # ── Nested ZIP extraction ─────────────────────────────────────────────────

#     def _extract_nested_zip(
#         self,
#         parent_zf: zipfile.ZipFile,
#         entry_name: str,
#         image_output_dir: str,
#         parent_stem: str,
#         current_depth: int,
#     ) -> str:
#         """
#         Recursively extract a nested ZIP entry found inside a parent ZipFile.

#         Reads the nested ZIP bytes into memory (BytesIO), opens it as a new
#         ZipFile, builds its folder tree, then walks it exactly like the outer
#         archive — producing indented Markdown content that is returned as a
#         string to be spliced into the parent's result_lines.

#         Guards:
#           - MAX_NESTED_ZIP_DEPTH  : stops recursion if too deep
#           - zipfile.BadZipFile    : gracefully handles corrupt inner ZIPs
#           - Password protection   : detects and skips encrypted inner ZIPs
#         """
#         nested_stem = Path(entry_name).stem
#         indent_note = f"*Contents of nested archive `{entry_name}`:*"

#         try:
#             # ── Read nested ZIP bytes from parent ─────────────────────────
#             raw_bytes = parent_zf.read(entry_name)
#             if not raw_bytes:
#                 return "*(Nested ZIP is empty — skipped)*"

#             nested_buf = io.BytesIO(raw_bytes)

#             if not zipfile.is_zipfile(nested_buf):
#                 return f"*(Not a valid ZIP: `{entry_name}`)*"

#             nested_buf.seek(0)

#             try:
#                 nested_zf = zipfile.ZipFile(nested_buf, 'r')
#             except zipfile.BadZipFile as e:
#                 return f"*(Corrupt nested ZIP `{entry_name}`: {e})*"

#             with nested_zf:
#                 # Encryption check
#                 inner_entries = nested_zf.infolist()
#                 if inner_entries and inner_entries[0].flag_bits & 0x1:
#                     return f"*(Nested ZIP `{entry_name}` is password-protected — skipped)*"

#                 inner_names = nested_zf.namelist()
#                 if not inner_names:
#                     return f"*(Nested ZIP `{entry_name}` is empty)*"

#                 logger.info(
#                     f"  {'  ' * current_depth}  📦 Extracting nested ZIP: {entry_name} "
#                     f"({len(inner_names)} entries, depth={current_depth + 1})"
#                 )

#                 # ── Build folder tree for nested archive ──────────────────
#                 nested_tree = self._build_folder_tree(inner_names)
#                 nested_lines: List[str] = [indent_note, ""]

#                 # ── Walk nested tree ──────────────────────────────────────
#                 def _walk_nested(tree: Dict, zf: zipfile.ZipFile, depth: int) -> None:
#                     for node_name in sorted(tree.keys()):
#                         node = tree[node_name]

#                         if node['type'] == 'folder':
#                             folder_path = node['path']
#                             h = '#' * min(depth + 3, 6)
#                             label = '/ (root)' if not folder_path else folder_path + '/'
#                             nested_lines.append(f"\n{h} 📁 {label}\n")
#                             logger.info(f"  {'  ' * depth}  📁 Nested folder: {folder_path or '(root)'}")
#                             _walk_nested(node['children'], zf, depth + 1)

#                         else:
#                             inner_entry = node['path']

#                             if self._should_skip_entry(inner_entry):
#                                 continue

#                             suffix = Path(inner_entry).suffix.lower()
#                             entry_type = self._classify_entry_suffix(suffix)
#                             fh = '#' * min(depth + 4, 6)

#                             if entry_type == "nested_zip":
#                                 # Further nesting
#                                 nested_lines.append(f"\n{fh} 📦 `{Path(inner_entry).name}`\n")
#                                 if depth + 1 >= MAX_NESTED_ZIP_DEPTH:
#                                     nested_lines.append(
#                                         f"*(Max nesting depth {MAX_NESTED_ZIP_DEPTH} reached — skipped)*"
#                                     )
#                                 else:
#                                     deeper_md = self._extract_nested_zip(
#                                         zf, inner_entry, image_output_dir,
#                                         f"{parent_stem}/{nested_stem}", depth + 1
#                                     )
#                                     nested_lines.append(deeper_md)

#                             elif entry_type == "delegated":
#                                 nested_lines.append(
#                                     f"\n{fh} {self._get_entry_emoji(suffix)} `{Path(inner_entry).name}`\n"
#                                 )
#                                 tmp_path = self._write_temp_entry(zf, inner_entry, image_output_dir)
#                                 if not tmp_path:
#                                     nested_lines.append("*(Could not extract to temp file)*")
#                                 else:
#                                     try:
#                                         sub_md = self._delegate_to_sub_extractor(
#                                             tmp_path,
#                                             suffix,
#                                             source_path=inner_entry,
#                                         )
#                                         if sub_md:
#                                             nested_lines.append(sub_md)
#                                             logger.info(f"  {'  ' * depth}  ✅ doc {inner_entry}: {len(sub_md):,} chars")
#                                         else:
#                                             nested_lines.append("*(Sub-extractor returned no content)*")
#                                     except Exception as e:
#                                         nested_lines.append(f"*(Sub-extractor failed: {e})*")
#                                     finally:
#                                         try:
#                                             os.unlink(tmp_path)
#                                         except Exception:
#                                             pass

#                             elif entry_type == "native_text":
#                                 nested_lines.append(f"\n{fh} 📄 `{Path(inner_entry).name}`\n")
#                                 content, ok = self._read_text_entry(zf, inner_entry)
#                                 if ok and content:
#                                     nested_lines.append(f"```{suffix.lstrip('.')}\n{content}\n```")
#                                     logger.info(f"  {'  ' * depth}  ✅ text {inner_entry}: {len(content):,} chars")
#                                 else:
#                                     nested_lines.append("*(Could not read or empty file)*")

#                             elif entry_type == "image":
#                                 nested_lines.append(f"\n{fh} 🖼️ `{Path(inner_entry).name}`\n")
#                                 img_md, ok = self._describe_image_entry(
#                                     zf, inner_entry, image_output_dir,
#                                     f"{parent_stem}/{nested_stem}"
#                                 )
#                                 nested_lines.append(img_md)
#                                 logger.info(f"  {'  ' * depth}  {'✅' if ok else '⚠️'} image {inner_entry}")

#                             else:
#                                 nested_lines.append(f"\n{fh} ⏭️ `{Path(inner_entry).name}`\n")
#                                 nested_lines.append(
#                                     f"*(Unsupported type — skipped: `{suffix or 'no extension'}`)*"
#                                 )

#                             nested_lines.append("")

#                 _walk_nested(nested_tree, nested_zf, depth=current_depth + 1)

#             return '\n'.join(nested_lines)

#         except Exception as e:
#             logger.error(f"❌ Failed to extract nested ZIP {entry_name}: {e}")
#             traceback.print_exc()
#             return f"*(Error extracting nested ZIP `{entry_name}`: {e})*"

#     # ── Image description ─────────────────────────────────────────────────────

#     def _describe_image_entry(
#         self,
#         zf: zipfile.ZipFile,
#         entry_name: str,
#         image_output_dir: str,
#         zip_stem: str,
#     ) -> Tuple[str, bool]:
#         """
#         Read an image entry from the open ZipFile, deduplicate,
#         describe via Vision LLM, save to disk.
#         Mirrors DocxExtractor._process_images_python_docx() image loop.

#         Returns (markdown_string, success_bool).
#         """
#         try:
#             image_bytes = zf.read(entry_name)

#             if not image_bytes:
#                 return self._build_image_description_body(
#                     "[Empty image file]",
#                     source_path=entry_name,
#                     status_note="fallback",
#                 ), False

#             # Deduplication
#             img_hash = self.image_cache.compute_hash(image_bytes)
#             if img_hash in getattr(self, '_seen_hashes', set()):
#                 self._duplicates_skipped += 1
#                 logger.debug(f"  ⏭ Duplicate image skipped: {entry_name}")
#                 return self._build_image_description_body(
#                     "[Duplicate image skipped]",
#                     source_path=entry_name,
#                     status_note="duplicate",
#                 ), False

#             if not hasattr(self, '_seen_hashes'):
#                 self._seen_hashes: set = set()
#             self._seen_hashes.add(img_hash)

#             # Check cache
#             cached_desc = self.image_cache.get(image_bytes)
#             if cached_desc:
#                 description = cached_desc
#                 from_cache  = True
#                 success     = True
#             else:
#                 ext        = Path(entry_name).suffix.lower().lstrip('.')
#                 media_type = self._get_media_type(ext)
#                 success, description = self.image_service.describe_image_from_bytes(
#                     image_bytes,
#                     media_type,
#                     context=f"ZIP archive: {zip_stem}",
#                     page_number=0,
#                     filename=Path(entry_name).name,
#                 )
#                 from_cache = False

#                 if success:
#                     self.image_cache.put(image_bytes, description)

#             # Save image to output directory
#             safe_name  = re.sub(r'[^\w.\-]', '_', Path(entry_name).name)
#             image_path = os.path.join(image_output_dir, safe_name)
#             with open(image_path, 'wb') as f:
#                 f.write(image_bytes)

#             cache_str = "cached" if from_cache else "described"
#             if not success:
#                 logger.warning("Image description fallback used for %s", entry_name)
#                 cache_str = "fallback"

#             md = self._build_image_description_body(
#                 description,
#                 source_path=entry_name,
#                 status_note=cache_str,
#             )
#             return md, success

#         except Exception as e:
#             logger.warning(f"  ⚠️ Error describing image {entry_name}: {e}")
#             fallback = self.image_service.generate_fallback_description(
#                 Path(entry_name).name,
#                 page_number=0,
#                 image_type="image",
#             )
#             return self._build_image_description_body(
#                 fallback,
#                 source_path=entry_name,
#                 status_note="fallback",
#             ), False

#     # ── Text entry reader ─────────────────────────────────────────────────────

#     def _read_text_entry(
#         self, zf: zipfile.ZipFile, entry_name: str
#     ) -> Tuple[str, bool]:
#         """
#         Read a text entry from the open ZipFile.
#         Respects MAX_TEXT_ENTRY_BYTES; tries UTF-8 then latin-1.
#         """
#         try:
#             raw = zf.read(entry_name)
#             if len(raw) > MAX_TEXT_ENTRY_BYTES:
#                 logger.warning(
#                     f"  ⚠️ {entry_name} is {len(raw):,} bytes — truncating to {MAX_TEXT_ENTRY_BYTES:,}"
#                 )
#                 raw = raw[:MAX_TEXT_ENTRY_BYTES]

#             try:
#                 return raw.decode('utf-8'), True
#             except UnicodeDecodeError:
#                 return raw.decode('latin-1', errors='replace'), True

#         except Exception as e:
#             logger.warning(f"  ⚠️ Could not read {entry_name}: {e}")
#             return "", False

#     # ── Temp file helper ──────────────────────────────────────────────────────

#     def _write_temp_entry(
#         self,
#         zf: zipfile.ZipFile,
#         entry_name: str,
#         output_dir: str,
#     ) -> Optional[str]:
#         """
#         Extract a single ZIP entry to a temp file in output_dir.
#         Returns the temp file path, or None on failure.
#         """
#         try:
#             data     = zf.read(entry_name)
#             suffix   = Path(entry_name).suffix
#             safe_name = re.sub(r'[^\w.\-]', '_', Path(entry_name).name)
#             tmp_path  = os.path.join(output_dir, f"__tmp_{safe_name}")

#             with open(tmp_path, 'wb') as f:
#                 f.write(data)
#             return tmp_path
#         except Exception as e:
#             logger.warning(f"  ⚠️ Could not write temp file for {entry_name}: {e}")
#             return None

#     # ── Sub-extractor delegation ──────────────────────────────────────────────

#     def _delegate_to_sub_extractor(
#         self,
#         tmp_path: str,
#         suffix: str,
#         source_path: Optional[str] = None,
#     ) -> Optional[str]:
#         """
#         Delegate to the shared extractor registry for files discovered inside ZIPs.

#         Falls back to:
#         - native text read for text-like formats when a registered extractor fails
#         - Docling for document formats where registry extraction is unavailable
#         """
#         clean_suffix = suffix.lower()
#         display_path = source_path or Path(tmp_path).name

#         try:
#             extractor = self._get_extractor_factory().get_extractor(
#                 clean_suffix,
#                 config=dict(self.config),
#             )
#             if extractor:
#                 md, success, metadata = extractor.extract(tmp_path)
#                 if success and md and md.strip():
#                     return md

#                 error_message = None
#                 if isinstance(metadata, dict):
#                     error_message = metadata.get('error')
#                 logger.warning(
#                     "  Registered extractor returned no content for %s (%s): %s",
#                     display_path,
#                     clean_suffix,
#                     error_message or "empty result",
#                 )
#             else:
#                 logger.info(
#                     "  No registered extractor available for %s (%s)",
#                     display_path,
#                     clean_suffix,
#                 )
#         except Exception as e:
#             logger.warning(f"  Registered extractor error for {display_path} ({clean_suffix}): {e}")

#         if clean_suffix in TEXT_EXTENSIONS:
#             content, ok = self._read_text_file(Path(tmp_path))
#             if ok and content.strip():
#                 return self._format_text_file_chunk(display_path, clean_suffix, content)

#         if clean_suffix in DOC_EXTENSIONS:
#             return self._delegate_to_docling(tmp_path, clean_suffix)

#         return None

#     def _delegate_to_docling(self, tmp_path: str, suffix: str) -> Optional[str]:
#         """
#         Fallback: convert a temp file through Docling.
#         Used when a dedicated extractor is not available.
#         """
#         try:
#             from docling.document_converter import DocumentConverter
#             converter = DocumentConverter()
#             doc_result = converter.convert(tmp_path)
#             md_text = doc_result.document.export_to_markdown()
#             if md_text.strip():
#                 logger.info(f"  ✅ Docling fallback OK for {suffix}: {len(md_text):,} chars")
#                 return md_text
#             logger.warning(f"  ⚠️ Docling returned empty markdown for {suffix}")
#         except ImportError:
#             logger.info("  Docling not available — no fallback for sub-extractor")
#         except Exception as e:
#             logger.warning(f"  Docling fallback failed for {suffix}: {e}")
#         return None

#     # ── Image directory helper ────────────────────────────────────────────────

#     def _create_image_directory_user_isolated(
#         self,
#         file_path: str,
#         zip_stem: str,
#         user_id: str,
#     ) -> str:
#         """
#         Create user-isolated image output directory.
#         Mirrors DocxExtractor._create_image_directory_user_isolated().
#         """
#         base_dir  = Path(file_path).parent
#         image_dir = base_dir / "extracted_images" / zip_stem
#         image_dir.mkdir(parents=True, exist_ok=True)
#         return str(image_dir)

#     # ── Skip filter ───────────────────────────────────────────────────────────

#     @staticmethod
#     def _should_skip_entry(entry_name: str) -> bool:
#         """
#         Return True for entries that should be silently skipped:
#           - macOS __MACOSX metadata entries
#           - Hidden files (dot-files)
#           - Windows thumbs / desktop.ini artefacts
#           - Zero-length name
#         """
#         name = Path(entry_name).name
#         if not name:
#             return True
#         if '__MACOSX' in entry_name or entry_name.startswith('.'):
#             return True
#         if name.startswith('.'):
#             return True
#         if name.lower() in {'thumbs.db', 'desktop.ini', '.ds_store'}:
#             return True
#         return False

#     # ── MIME helpers ──────────────────────────────────────────────────────────

#     def _get_media_type(self, extension: str) -> str:
#         """Map file extension to MIME type — mirrors DocxExtractor._get_media_type()."""
#         return {
#             'jpg'  : 'image/jpeg',
#             'jpeg' : 'image/jpeg',
#             'png'  : 'image/png',
#             'gif'  : 'image/gif',
#             'webp' : 'image/webp',
#             'bmp'  : 'image/bmp',
#         }.get(extension.lower(), 'image/jpeg')

#     # ── BaseExtractor.get_metadata ────────────────────────────────────────────

#     def get_metadata(self, file_path: str) -> Dict:
#         """Basic file metadata — mirrors DocxExtractor pattern."""
#         path = Path(file_path)
#         return {
#             'filename'      : path.name,
#             'size_bytes'    : path.stat().st_size,
#             'extension'     : path.suffix.lower(),
#         }

#     # ── Extraction stats ──────────────────────────────────────────────────────

#     def get_extraction_stats(self) -> Dict:
#         """Return extraction statistics."""
#         return {
#             'duplicates_skipped': self._duplicates_skipped,
#         }
"""
ZIP Extractor
Mirrors DocxExtractor architecture exactly:
  validate → extract (Strategy 1: Docling per-file) → (Strategy 2: native per-file fallback)
  → image description pipeline → markdown assembly

Folder traversal:
  The extractor builds a nested folder tree from the ZIP's flat namelist and
  processes entries recursively, folder by folder.  Each folder gets its own
  Markdown heading so the output reflects the real directory structure:

    # Archive: my_archive
    ## 📁 / (root)
    ### 📄 readme.txt
    ## 📁 docs/
    ### 📄 docs/report.pdf
    ## 📁 docs/images/
    ### 🖼️ docs/images/chart.png

Supported entry types inside the ZIP:
  - Text-based  : .txt, .md, .csv, .json, .xml, .html, .htm, .yaml, .yml, .log
  - Documents   : .pdf, .docx  (delegated to their own extractors when available)
  - Images      : .png, .jpg, .jpeg, .gif, .webp, .bmp  (described via Vision LLM)
  - Nested ZIPs : .zip  (extracted recursively up to MAX_NESTED_ZIP_DEPTH=5 levels)
  - Skipped     : binaries, executables, hidden files, password-protected ZIPs (logged)
"""

import os
import io
import re
import base64
import zipfile
import logging
import traceback
import collections
import tempfile
from pathlib import Path
from typing import Tuple, Dict, List, Optional

try:
    from ..base_extractor import BaseExtractor
except ImportError:
    from base_extractor import BaseExtractor

logger = logging.getLogger(__name__)

# ── Supported entry categories ─────────────────────────────────────────────────
TEXT_EXTENSIONS  = {'.txt', '.md', '.csv', '.json', '.xml', '.html', '.htm',
                    '.yaml', '.yml', '.log', '.eml', '.properties', '.vtt'}
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp', '.tiff', '.tif'}
DOC_EXTENSIONS   = {'.pdf', '.docx', '.xlsx', '.xlsm', '.pptx', '.msg', '.epub'}
ZIP_EXTENSIONS   = {'.zip'}                # nested ZIPs — extracted recursively

# Max raw size we attempt to read per text entry (avoid huge CSVs/logs)
MAX_TEXT_ENTRY_BYTES  = 512 * 1024   # 512 KB
# Max nesting depth for ZIP-inside-ZIP (prevents infinite recursion / zip bombs)
MAX_NESTED_ZIP_DEPTH  = 5


class ZIPExtractor(BaseExtractor):
    """
    ZIP Extractor with Docling + native fallback strategies.

    Mirrors DocxExtractor structure:
      Strategy 1 — Docling  (tries to convert each extractable entry)
      Strategy 2 — Native   (direct text read / image description via Vision LLM)

    Returns the same (markdown, success, metadata) tuple as all other extractors.
    """

    # ── Construction ──────────────────────────────────────────────────────────

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)

        from app.services.image_cache_manager import get_image_cache
        from app.services.image_description_service import get_image_description_service

        self.image_cache    = get_image_cache()
        self.image_service  = get_image_description_service()

        self._duplicates_skipped = 0

        logger.info("ZIP Extractor initialized")

    # ── BaseExtractor interface ────────────────────────────────────────────────

    def _get_supported_formats(self) -> List[str]:
        return ['zip']

    # ── Validation ────────────────────────────────────────────────────────────

    def validate_file(self, file_path: str) -> Tuple[bool, str]:
        """
        Validate ZIP file — mirrors DocxExtractor.validate_file().

        Checks:
          1. File exists
          2. Extension is .zip
          3. File is not empty
          4. zipfile.is_zipfile() confirms valid ZIP magic bytes
          5. Archive is not password-protected (best-effort check)
        """
        path = Path(file_path)

        if not path.exists():
            return False, f"File does not exist: {file_path}"
        if path.suffix.lower() != '.zip':
            return False, f"Not a ZIP file: {path.suffix}"
        if path.stat().st_size == 0:
            return False, "File is empty (0 bytes)"

        # Check ZIP magic bytes
        if not zipfile.is_zipfile(file_path):
            return False, "File is not a valid ZIP archive (bad magic bytes)"

        # Best-effort: detect encryption by checking first entry flags
        try:
            with zipfile.ZipFile(file_path, 'r') as zf:
                if not zf.namelist():
                    return False, "ZIP archive is empty (no entries)"
                first = zf.infolist()[0]
                if first.flag_bits & 0x1:   # bit 0 = encryption flag
                    return False, "ZIP archive is password-protected — decryption not supported"
        except zipfile.BadZipFile as e:
            return False, f"Corrupt ZIP archive: {e}"
        except Exception as e:
            logger.warning(f"Could not inspect ZIP entries: {e}")

        return True, "Valid ZIP archive"

    # ── Main extract ──────────────────────────────────────────────────────────

    def extract(self, file_path: str) -> Tuple[str, bool, Dict]:
        """
        Extract ZIP with two-strategy fallback — mirrors DocxExtractor.extract().

        Strategy 1: Docling (attempts to convert each extractable entry)
        Strategy 2: Native  (direct text read + Vision LLM for images)
        """
        try:
            logger.info("=" * 80)
            logger.info("ZIP EXTRACTION - START")
            logger.info("=" * 80)
            logger.info(f"File path: {file_path}")

            # Validate
            is_valid, msg = self.validate_file(file_path)
            if not is_valid:
                logger.error(f"❌ Validation failed: {msg}")
                return "", False, {'error': msg}

            logger.info("✅ File validation passed")

            file_size = Path(file_path).stat().st_size
            logger.info(f"📊 File size: {file_size:,} bytes ({file_size / 1024 / 1024:.2f} MB)")

            # List entries
            with zipfile.ZipFile(file_path, 'r') as zf:
                all_entries   = zf.namelist()
                total_entries = len(all_entries)
            logger.info(f"📦 Total ZIP entries: {total_entries}")

            zip_stem = Path(file_path).stem

            # ── Strategy 1: Docling ───────────────────────────────────────
            logger.info("\n" + "=" * 60)
            logger.info("STRATEGY 1: Trying Docling extraction...")
            logger.info("=" * 60)

            docling_result = self._try_docling_extraction(file_path, zip_stem)
            if docling_result[1]:
                logger.info("✅✅✅ Docling extraction SUCCESSFUL")
                return docling_result
            else:
                logger.warning(f"⚠️ Docling failed: {docling_result[2].get('error', 'Unknown')}")

            # ── Strategy 2: Native ────────────────────────────────────────
            logger.info("\n" + "=" * 60)
            logger.info("STRATEGY 2: Trying native extraction...")
            logger.info("=" * 60)

            result = self._extract_native(file_path, zip_stem)

            if result[1]:
                logger.info("✅✅✅ Native extraction SUCCESSFUL")
            else:
                logger.error(f"❌❌❌ Native extraction FAILED: {result[2].get('error', 'Unknown')}")

            logger.info("=" * 80)
            logger.info("ZIP EXTRACTION - END")
            logger.info("=" * 80)

            return result

        except Exception as e:
            logger.error(f"❌❌❌ CRITICAL ERROR in extract(): {str(e)}")
            traceback.print_exc()
            return "", False, {'error': str(e)}

    # ── Per-file chunked indexing ─────────────────────────────────────────────

    def extract_chunks(self, file_path: str) -> Tuple[List[Dict], bool, Dict]:
        """
        Extract ZIP and return one chunk dict per file entry, ready for indexing.

        Each chunk has:
          {
            'content'           : str,   # Markdown content for this file only
            'source_path'       : str,   # e.g. "docs/report.pdf"
            'file_name'         : str,   # e.g. "report.pdf"
            'file_type'         : str,   # e.g. ".pdf"
            'chunk_index'       : int,   # 0-based position within the archive
            'archive_name'      : str,   # ZIP stem, e.g. "my_archive"
            'extraction_method' : str,   # "docling" | "native_text" | "vision_llm" | ...
          }

        Returns (chunks, success, metadata) — mirrors the extract() contract but
        replaces the monolithic markdown string with a list of per-file chunks.
        """
        is_valid, msg = self.validate_file(file_path)
        if not is_valid:
            return [], False, {'error': msg}

        archive_name = Path(file_path).stem

        try:
            with tempfile.TemporaryDirectory(prefix=f"zip_extract_{archive_name}_") as temp_dir:
                extract_root = self._extract_archive_to_temp_tree(file_path, temp_dir)
                chunks = self._build_file_chunks_from_directory(
                    root_dir=extract_root,
                    archive_name=archive_name,
                )
                metadata = {
                    'archive_name': archive_name,
                    'extraction_method': 'temp_dir_per_file',
                    'file_chunks': chunks,
                    'files_extracted': len(chunks),
                }
                logger.info(
                    f"extract_chunks: ZIP temp extraction complete for '{archive_name}' "
                    f"-> {len(chunks)} file chunk(s)"
                )
                return chunks, True, metadata
        except Exception as e:
            logger.error(f"extract_chunks failed for ZIP '{file_path}': {e}", exc_info=True)
            return [], False, {'error': str(e)}

    # ── Strategy 1: Docling ───────────────────────────────────────────────────

    def _extract_archive_to_temp_tree(self, file_path: str, temp_dir: str) -> Path:
        """Extract the full ZIP tree into a temporary directory, preserving folders."""
        archive_root = Path(temp_dir) / Path(file_path).stem
        archive_root.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(file_path, 'r') as zf:
            for info in zf.infolist():
                raw_name = info.filename
                if not raw_name:
                    continue

                normalized = Path(raw_name)
                if normalized.is_absolute() or ".." in normalized.parts:
                    logger.warning(f"Skipping unsafe ZIP entry path: {raw_name}")
                    continue

                target_path = archive_root / normalized
                target_path.parent.mkdir(parents=True, exist_ok=True)

                if info.is_dir():
                    target_path.mkdir(parents=True, exist_ok=True)
                    continue

                with zf.open(info, 'r') as src, open(target_path, 'wb') as dst:
                    dst.write(src.read())

        return archive_root

    def _build_file_chunks_from_directory(
        self,
        root_dir: Path,
        archive_name: str,
    ) -> List[Dict]:
        """Process each extracted file independently and return one chunk record per file."""
        chunks: List[Dict] = []

        for file_path in sorted(p for p in root_dir.rglob('*') if p.is_file()):
            source_path = file_path.relative_to(root_dir).as_posix()

            if self._should_skip_entry(source_path):
                continue

            suffix = file_path.suffix.lower()
            method = "unknown"

            if suffix in TEXT_EXTENSIONS:
                content, ok = self._read_text_file(file_path)
                if not ok or not content.strip():
                    continue
                method = "native_text"
                content = self._format_text_file_chunk(source_path, suffix, content)

            elif suffix in IMAGE_EXTENSIONS:
                content, ok = self._describe_image_file(file_path, source_path, archive_name)
                if not ok or not content.strip():
                    continue
                method = "vision_llm"

            elif suffix in DOC_EXTENSIONS:
                content = self._delegate_to_sub_extractor(str(file_path), suffix) or ""
                if not content.strip():
                    continue
                method = "sub_extractor"

            else:
                logger.debug(f"Skipping unsupported ZIP entry for chunking: {source_path}")
                continue

            chunks.append({
                'content': content,
                'source_path': source_path,
                'file_name': file_path.name,
                'file_type': suffix,
                'chunk_index': len(chunks),
                'archive_name': archive_name,
                'extraction_method': method,
            })

        return chunks

    def _read_text_file(self, file_path: Path) -> Tuple[str, bool]:
        """Read a text file from disk with the same truncation policy as ZIP entries."""
        try:
            raw = file_path.read_bytes()
            if len(raw) > MAX_TEXT_ENTRY_BYTES:
                logger.warning(
                    f"  ⚠️ {file_path.name} is {len(raw):,} bytes — truncating to {MAX_TEXT_ENTRY_BYTES:,}"
                )
                raw = raw[:MAX_TEXT_ENTRY_BYTES]

            try:
                return raw.decode('utf-8'), True
            except UnicodeDecodeError:
                return raw.decode('latin-1', errors='replace'), True
        except Exception as e:
            logger.warning(f"Could not read extracted text file {file_path}: {e}")
            return "", False

    def _format_text_file_chunk(self, source_path: str, suffix: str, content: str) -> str:
        """Wrap plain text entries with a small header so chunk content stays self-describing."""
        lang = suffix.lstrip('.')
        return f"### `{source_path}`\n\n```{lang}\n{content}\n```"

    def _describe_image_file(
        self,
        file_path: Path,
        source_path: str,
        archive_name: str,
    ) -> Tuple[str, bool]:
        """Describe an extracted image file and return markdown content for chunking."""
        try:
            image_bytes = file_path.read_bytes()
            if not image_bytes:
                return "", False

            cached_desc = self.image_cache.get(image_bytes)
            if cached_desc:
                description = cached_desc
            else:
                media_type = self._get_media_type(file_path.suffix.lower().lstrip('.'))
                base64_img = base64.b64encode(image_bytes).decode('utf-8')
                success, description = self.image_service.describe_image_from_base64(
                    base64_img,
                    media_type,
                    context=f"ZIP archive: {archive_name}",
                    page_number=0
                )
                if not success:
                    return "", False
                self.image_cache.put(image_bytes, description)

            markdown = (
                f"### `{source_path}`\n\n"
                f"**[Image Description]**: {description}\n"
                f"*(File: `{source_path}`)*"
            )
            return markdown, True
        except Exception as e:
            logger.warning(f"Could not describe extracted image {file_path}: {e}")
            return "", False

    def _try_docling_extraction(
        self, file_path: str, zip_stem: str
    ) -> Tuple[str, bool, Dict]:
        """
        Try Docling on each extractable entry — mirrors
        DocxExtractor._try_docling_extraction_debug().

        Docling works best with PDF and DOCX entries.
        Text / image entries fall through to the native handler even here.
        """
        try:
            logger.info("Attempting to import Docling...")
            from docling.document_converter import DocumentConverter
            logger.info("✅ Docling imported successfully")

            user_id = Path(file_path).parent.name
            if user_id == "uploads":
                user_id = "default"

            image_output_dir = self._create_image_directory_user_isolated(
                file_path, zip_stem, user_id
            )
            logger.info(f"Image directory: {image_output_dir}")

            converter = DocumentConverter()

            result_lines: List[str] = []
            result_lines.append(f"# Archive: {zip_stem}")
            result_lines.append(f"**Format**: ZIP Archive")
            result_lines.append(f"**Extraction Method**: Docling")
            result_lines.append("")

            self._duplicates_skipped = 0
            images_extracted  = 0
            images_described  = 0
            entries_processed = 0
            entries_skipped   = 0
            doc_entries: List[Dict] = []   # collect for metadata
            file_chunks: List[Dict] = []   # one chunk per file, for per-file indexing

            def _make_chunk_docling(entry_name: str, content: str, method: str) -> Dict:
                return {
                    'content'           : content,
                    'source_path'       : entry_name,
                    'file_name'         : Path(entry_name).name,
                    'file_type'         : Path(entry_name).suffix.lower(),
                    'chunk_index'       : len(file_chunks),
                    'archive_name'      : zip_stem,
                    'extraction_method' : method,
                }

            with zipfile.ZipFile(file_path, 'r') as zf:
                all_names = zf.namelist()
                folder_tree = self._build_folder_tree(all_names)

                logger.info(f"📁 Folder tree built: {len(folder_tree)} top-level nodes")

                def _walk_docling(tree: Dict, zf, depth: int = 0) -> None:
                    nonlocal images_extracted, images_described, entries_processed, entries_skipped

                    for node_name in sorted(tree.keys()):
                        node = tree[node_name]

                        if node['type'] == 'folder':
                            folder_path = node['path']
                            heading_level = '#' * min(depth + 2, 6)
                            result_lines.append(
                                f"\n{heading_level} 📁 {'/ (root)' if not folder_path else folder_path + '/'}\n"
                            )
                            logger.info(f"  {'  ' * depth}📁 Entering folder: {folder_path or '(root)'}")
                            _walk_docling(node['children'], zf, depth + 1)

                        else:
                            # It's a file node
                            entry_name = node['path']

                            if self._should_skip_entry(entry_name):
                                entries_skipped += 1
                                continue

                            suffix = Path(entry_name).suffix.lower()
                            file_heading = '#' * min(depth + 3, 6)
                            # Emit heading with type-specific emoji
                            if suffix in ZIP_EXTENSIONS:
                                emoji = '📦'
                            elif suffix in IMAGE_EXTENSIONS:
                                emoji = '🖼️'
                            elif suffix in DOC_EXTENSIONS:
                                emoji = '📑'
                            else:
                                emoji = '📄'
                            result_lines.append(f"\n{file_heading} {emoji} `{Path(entry_name).name}`\n")

                            if suffix in ZIP_EXTENSIONS:
                                # ── Nested ZIP ──────────────────────────────
                                if depth >= MAX_NESTED_ZIP_DEPTH:
                                    logger.warning(f"  {'  ' * depth}⚠️ Max nesting depth reached — skipping {entry_name}")
                                    result_lines.append(
                                        f"*(Nested ZIP skipped — max depth {MAX_NESTED_ZIP_DEPTH} reached)*"
                                    )
                                    entries_skipped += 1
                                else:
                                    logger.info(f"  {'  ' * depth}📦 Nested ZIP found: {entry_name}")
                                    nested_md = self._extract_nested_zip(
                                        zf, entry_name, image_output_dir, zip_stem, depth
                                    )
                                    result_lines.append(nested_md)
                                    entries_processed += 1
                                    doc_entries.append({
                                        'name': entry_name, 'suffix': suffix, 'method': 'nested_zip'
                                    })

                            elif suffix in DOC_EXTENSIONS:
                                tmp_path = self._write_temp_entry(zf, entry_name, image_output_dir)
                                if not tmp_path:
                                    result_lines.append("*(Could not extract entry to temp file)*")
                                else:
                                    try:
                                        # Docling handles PDF/DOCX natively.
                                        # All other formats go through dedicated sub-extractors.
                                        if suffix in {'.pdf', '.docx'}:
                                            logger.info(f"  {'  ' * depth}Docling converting: {entry_name}")
                                            doc_result = converter.convert(tmp_path)
                                            md_text = doc_result.document.export_to_markdown()
                                            method = 'docling'
                                        else:
                                            logger.info(f"  {'  ' * depth}Sub-extractor for: {entry_name}")
                                            md_text = self._delegate_to_sub_extractor(tmp_path, suffix) or ''
                                            method = 'sub_extractor'

                                        if md_text.strip():
                                            result_lines.append(md_text)
                                            file_chunks.append(
                                                _make_chunk_docling(entry_name, md_text, method)
                                            )
                                            logger.info(f"  {'  ' * depth}✅ {entry_name}: {len(md_text):,} chars")
                                        else:
                                            logger.warning(f"  ⚠️ {entry_name}: no content extracted")
                                            result_lines.append("*(No content extracted)*")

                                        entries_processed += 1
                                        doc_entries.append({'name': entry_name, 'suffix': suffix, 'method': method})
                                    except Exception as e:
                                        logger.warning(f"  ⚠️ Extraction failed for {entry_name}: {e}")
                                        result_lines.append(f"*(Extraction failed: {e})*")
                                    finally:
                                        try:
                                            os.unlink(tmp_path)
                                        except Exception:
                                            pass

                            elif suffix in TEXT_EXTENSIONS:
                                content, ok = self._read_text_entry(zf, entry_name)
                                if ok and content:
                                    block = f"```{suffix.lstrip('.')}\n{content}\n```"
                                    result_lines.append(block)
                                    file_chunks.append(
                                        _make_chunk_docling(entry_name, content, 'native_text')
                                    )
                                    entries_processed += 1
                                    doc_entries.append({'name': entry_name, 'suffix': suffix, 'method': 'native_text'})
                                else:
                                    result_lines.append("*(Could not read text content)*")

                            elif suffix in IMAGE_EXTENSIONS:
                                img_md, ok = self._describe_image_entry(
                                    zf, entry_name, image_output_dir, zip_stem
                                )
                                result_lines.append(img_md)
                                if ok:
                                    images_extracted += 1
                                    images_described += 1
                                    entries_processed += 1
                                    file_chunks.append(
                                        _make_chunk_docling(entry_name, img_md, 'vision_llm')
                                    )
                                    doc_entries.append({'name': entry_name, 'suffix': suffix, 'method': 'vision_llm'})

                            else:
                                result_lines.append(f"*(Unsupported type — skipped: `{suffix or 'no extension'}`)*")
                                entries_skipped += 1

                            result_lines.append("")

                _walk_docling(folder_tree, zf, depth=0)

            final_markdown = '\n'.join(result_lines)

            if len(final_markdown.strip()) < 100:
                return "", False, {'error': 'Docling pass produced insufficient content'}

            cache_stats = self.image_cache.get_stats()
            metadata = self.get_metadata(file_path)
            metadata.update({
                'extractor':               'ZipExtractor_Docling',
                'total_entries':           len(zf.namelist()) if False else entries_processed + entries_skipped,
                'entries_processed':       entries_processed,
                'entries_skipped':         entries_skipped,
                'images_extracted':        images_extracted,
                'images_described':        images_described,
                'images_skipped_duplicates': self._duplicates_skipped,
                'images_cached':           cache_stats['hits'],
                'cache_hit_rate':          cache_stats['hit_rate_percent'],
                'image_directory':         str(image_output_dir),
                'extraction_method':       'docling',
                'doc_entries':             doc_entries,
                'file_chunks':             file_chunks,
            })

            return final_markdown, True, metadata

        except ImportError as e:
            logger.info(f"⚠️ Docling not available: {e}")
            return "", False, {'error': 'Docling not installed'}
        except Exception as e:
            logger.error(f"❌ Docling strategy error: {str(e)}")
            traceback.print_exc()
            return "", False, {'error': str(e)}

    # ── Strategy 2: Native ────────────────────────────────────────────────────

    def _extract_native(
        self, file_path: str, zip_stem: str
    ) -> Tuple[str, bool, Dict]:
        """
        Native extraction — mirrors DocxExtractor._extract_with_python_docx_debug().

        - Text entries  → read and embed as fenced code blocks
        - Image entries → describe via Vision LLM (with dedup + caching)
        - Doc entries   → attempt sub-extractor delegation (PDF/DOCX extractors)
        - Everything else → log and skip
        """
        try:
            user_id = Path(file_path).parent.name
            if user_id == "uploads":
                user_id = "default"

            image_output_dir = self._create_image_directory_user_isolated(
                file_path, zip_stem, user_id
            )

            with zipfile.ZipFile(file_path, 'r') as zf:
                all_entries   = [n for n in zf.namelist() if not n.endswith('/')]
                total_entries = len(all_entries)

            logger.info(f"📊 ZIP structure:")
            logger.info(f"  - Total file entries : {total_entries}")

            text_entries  = [n for n in all_entries if Path(n).suffix.lower() in TEXT_EXTENSIONS]
            image_entries = [n for n in all_entries if Path(n).suffix.lower() in IMAGE_EXTENSIONS]
            doc_entries_l = [n for n in all_entries if Path(n).suffix.lower() in DOC_EXTENSIONS]
            zip_entries   = [n for n in all_entries if Path(n).suffix.lower() in ZIP_EXTENSIONS]
            other_entries = [n for n in all_entries
                             if Path(n).suffix.lower() not in
                             TEXT_EXTENSIONS | IMAGE_EXTENSIONS | DOC_EXTENSIONS | ZIP_EXTENSIONS]

            logger.info(f"  - Text files   : {len(text_entries)}")
            logger.info(f"  - Image files  : {len(image_entries)}")
            logger.info(f"  - Doc files    : {len(doc_entries_l)}")
            logger.info(f"  - Nested ZIPs  : {len(zip_entries)}")
            logger.info(f"  - Other/skipped: {len(other_entries)}")

            if total_entries == 0:
                return "", False, {'error': 'ZIP archive has no file entries'}

            self._duplicates_skipped = 0
            result_lines: List[str] = []
            image_data:   Dict[int, Dict] = {}
            doc_entry_meta: List[Dict] = []
            file_chunks: List[Dict] = []   # one chunk per file, for per-file indexing

            def _make_chunk_native(entry_name: str, content: str, method: str) -> Dict:
                return {
                    'content'           : content,
                    'source_path'       : entry_name,
                    'file_name'         : Path(entry_name).name,
                    'file_type'         : Path(entry_name).suffix.lower(),
                    'chunk_index'       : len(file_chunks),
                    'archive_name'      : zip_stem,
                    'extraction_method' : method,
                }

            # ── Document header ───────────────────────────────────────────
            result_lines.append(f"# Archive: {zip_stem}")
            result_lines.append(f"**Format**: ZIP Archive")
            result_lines.append(f"**Total Entries**: {total_entries}")
            result_lines.append(f"**Text Files**: {len(text_entries)}")
            result_lines.append(f"**Image Files**: {len(image_entries)}")
            result_lines.append(f"**Document Files**: {len(doc_entries_l)}")
            if zip_entries:
                result_lines.append(f"**Nested ZIP Archives**: {len(zip_entries)}")
            result_lines.append("")

            with zipfile.ZipFile(file_path, 'r') as zf:
                folder_tree = self._build_folder_tree(zf.namelist())
                logger.info(f"📁 Folder tree built: {len(folder_tree)} top-level nodes")

                def _walk_native(tree: Dict, zf, depth: int = 0) -> None:
                    for node_name in sorted(tree.keys()):
                        node = tree[node_name]

                        if node['type'] == 'folder':
                            folder_path = node['path']
                            heading_level = '#' * min(depth + 2, 6)
                            result_lines.append(
                                f"\n{heading_level} 📁 {'/ (root)' if not folder_path else folder_path + '/'}\n"
                            )
                            logger.info(f"  {'  ' * depth}📁 Entering folder: {folder_path or '(root)'}")
                            _walk_native(node['children'], zf, depth + 1)

                        else:
                            entry_name = node['path']

                            if self._should_skip_entry(entry_name):
                                continue

                            suffix = Path(entry_name).suffix.lower()
                            file_heading = '#' * min(depth + 3, 6)

                            if suffix in ZIP_EXTENSIONS:
                                # ── Nested ZIP ──────────────────────────────
                                result_lines.append(f"\n{file_heading} 📦 `{Path(entry_name).name}`\n")
                                if depth >= MAX_NESTED_ZIP_DEPTH:
                                    logger.warning(f"  {'  ' * depth}⚠️ Max nesting depth reached — skipping {entry_name}")
                                    result_lines.append(
                                        f"*(Nested ZIP skipped — max depth {MAX_NESTED_ZIP_DEPTH} reached)*"
                                    )
                                else:
                                    logger.info(f"  {'  ' * depth}📦 Nested ZIP found: {entry_name}")
                                    nested_md = self._extract_nested_zip(
                                        zf, entry_name, image_output_dir, zip_stem, depth
                                    )
                                    result_lines.append(nested_md)
                                    img_idx = len(image_data) + 1
                                    doc_entry_meta.append({
                                        'name': entry_name, 'type': 'nested_zip', 'method': 'recursive'
                                    })

                            elif suffix in TEXT_EXTENSIONS:
                                result_lines.append(f"\n{file_heading} 📄 `{Path(entry_name).name}`\n")
                                logger.info(f"  {'  ' * depth}📄 Reading text: {entry_name}")
                                content, ok = self._read_text_entry(zf, entry_name)
                                if ok and content:
                                    result_lines.append(f"```{suffix.lstrip('.')}\n{content}\n```")
                                    file_chunks.append(
                                        _make_chunk_native(entry_name, content, 'native_text')
                                    )
                                    logger.info(f"  {'  ' * depth}  ✅ {len(content):,} chars")
                                    doc_entry_meta.append({'name': entry_name, 'type': 'text',
                                                           'chars': len(content), 'method': 'native'})
                                else:
                                    result_lines.append("*(Could not read or empty file)*")
                                    logger.warning(f"  {'  ' * depth}  ⚠️ Could not read {entry_name}")

                            elif suffix in IMAGE_EXTENSIONS:
                                result_lines.append(f"\n{file_heading} 🖼️ `{Path(entry_name).name}`\n")
                                logger.info(f"  {'  ' * depth}🖼️  Describing image: {entry_name}")
                                img_md, ok = self._describe_image_entry(
                                    zf, entry_name, image_output_dir, zip_stem
                                )
                                result_lines.append(img_md)
                                if ok:
                                    img_idx = len(image_data) + 1
                                    image_data[img_idx] = {'name': entry_name, 'success': True}
                                    file_chunks.append(
                                        _make_chunk_native(entry_name, img_md, 'vision_llm')
                                    )
                                    doc_entry_meta.append({'name': entry_name, 'type': 'image',
                                                           'method': 'vision_llm'})

                            elif suffix in DOC_EXTENSIONS:
                                result_lines.append(f"\n{file_heading} 📑 `{Path(entry_name).name}`\n")
                                logger.info(f"  {'  ' * depth}📑 Extracting document: {entry_name}")
                                tmp_path = self._write_temp_entry(zf, entry_name, image_output_dir)
                                if not tmp_path:
                                    result_lines.append("*(Could not extract to temp file)*")
                                else:
                                    try:
                                        sub_md = self._delegate_to_sub_extractor(tmp_path, suffix)
                                        if sub_md:
                                            result_lines.append(sub_md)
                                            file_chunks.append(
                                                _make_chunk_native(entry_name, sub_md, 'sub_extractor')
                                            )
                                            doc_entry_meta.append({'name': entry_name, 'type': 'document',
                                                                    'suffix': suffix, 'method': 'sub_extractor'})
                                            logger.info(f"  {'  ' * depth}  ✅ Sub-extractor: {len(sub_md):,} chars")
                                        else:
                                            result_lines.append("*(Sub-extractor returned no content)*")
                                    except Exception as e:
                                        logger.warning(f"  {'  ' * depth}  ⚠️ Sub-extractor failed for {entry_name}: {e}")
                                        result_lines.append(f"*(Sub-extractor failed: {e})*")
                                    finally:
                                        try:
                                            os.unlink(tmp_path)
                                        except Exception:
                                            pass

                            else:
                                result_lines.append(f"\n{file_heading} ⏭️ `{Path(entry_name).name}`\n")
                                result_lines.append(
                                    f"*(Unsupported type — skipped: `{suffix or 'no extension'}`)*"
                                )

                            result_lines.append("")

                _walk_native(folder_tree, zf, depth=0)

            final_markdown = '\n'.join(result_lines)

            logger.info(f"✅ Final markdown: {len(final_markdown):,} characters")

            if len(final_markdown.strip()) < 100:
                logger.error(f"❌ Final markdown too short: {len(final_markdown)} chars")
                return "", False, {'error': 'Generated markdown is too short — ZIP may contain no readable content'}

            # Metadata
            cache_stats = self.image_cache.get_stats()
            metadata = self.get_metadata(file_path)
            metadata.update({
                'extractor':               'ZipExtractor_Native',
                'total_entries':           total_entries,
                'text_entries':            len(text_entries),
                'image_entries':           len(image_entries),
                'doc_entries':             len(doc_entries_l),
                'other_entries':           len(other_entries),
                'images_extracted':        len(image_data),
                'images_described':        sum(1 for v in image_data.values() if v.get('success')),
                'images_skipped_duplicates': self._duplicates_skipped,
                'images_cached':           cache_stats['hits'],
                'cache_hit_rate':          cache_stats['hit_rate_percent'],
                'image_directory':         str(image_output_dir),
                'extraction_method':       'native',
                'folder_aware':            True,
                'entry_details':           doc_entry_meta,
                'file_chunks':             file_chunks,
            })

            logger.info("✅ Native extraction successful")
            return final_markdown, True, metadata

        except Exception as e:
            logger.error(f"❌ Native extraction error: {str(e)}")
            traceback.print_exc()
            return "", False, {'error': str(e)}

    # ── Folder tree builder ───────────────────────────────────────────────────

    def _build_folder_tree(self, namelist: List[str]) -> Dict:
        """
        Build a nested folder tree from the ZIP's flat namelist.

        ZIP files store all entries as flat paths, e.g.:
            docs/report.pdf
            docs/images/chart.png
            readme.txt
            src/utils/helper.py

        This method converts that into a nested dict tree:
            {
              '(root)': {
                'type': 'folder', 'path': '',
                'children': {
                  'readme.txt': {'type': 'file', 'path': 'readme.txt'},
                  ...
                }
              },
              '(root)': {
                'type': 'folder', 'path': 'docs',
                'children': {
                  'report.pdf': {'type': 'file', 'path': 'docs/report.pdf'},
                  'images': {
                    'type': 'folder', 'path': 'docs/images',
                    'children': {
                      'chart.png': {'type': 'file', 'path': 'docs/images/chart.png'}
                    }
                  }
                }
              },
              ...
            }

        Folder nodes: {'type': 'folder', 'path': str, 'children': dict}
        File nodes  : {'type': 'file',   'path': str}
        """
        root: Dict = {}

        for entry in namelist:
            # Normalise — strip trailing slash (directory marker) and split
            parts = [p for p in entry.replace('\\', '/').split('/') if p]

            if not parts:
                continue

            node = root
            accumulated_path = ''

            for i, part in enumerate(parts):
                accumulated_path = '/'.join(parts[:i + 1])
                is_last = (i == len(parts) - 1)
                is_dir_entry = entry.endswith('/') and is_last

                if is_last and not is_dir_entry:
                    # This part is a file
                    if part not in node:
                        node[part] = {'type': 'file', 'path': accumulated_path}
                else:
                    # This part is a folder component
                    if part not in node:
                        node[part] = {
                            'type': 'folder',
                            'path': accumulated_path,
                            'children': {},
                        }
                    elif node[part]['type'] == 'file':
                        # Edge case: a name appears as both file and directory prefix
                        node[part] = {
                            'type': 'folder',
                            'path': accumulated_path,
                            'children': {},
                        }
                    node = node[part]['children']

        # Hoist root-level files under a virtual '(root)' folder node so the
        # walk function has a uniform structure to iterate
        root_files   = {k: v for k, v in root.items() if v['type'] == 'file'}
        root_folders = {k: v for k, v in root.items() if v['type'] == 'folder'}

        if root_files:
            root_folders['(root)'] = {
                'type': 'folder',
                'path': '',
                'children': root_files,
            }

        return root_folders

    # ── Nested ZIP extraction ─────────────────────────────────────────────────

    def _extract_nested_zip(
        self,
        parent_zf: zipfile.ZipFile,
        entry_name: str,
        image_output_dir: str,
        parent_stem: str,
        current_depth: int,
    ) -> str:
        """
        Recursively extract a nested ZIP entry found inside a parent ZipFile.

        Reads the nested ZIP bytes into memory (BytesIO), opens it as a new
        ZipFile, builds its folder tree, then walks it exactly like the outer
        archive — producing indented Markdown content that is returned as a
        string to be spliced into the parent's result_lines.

        Guards:
          - MAX_NESTED_ZIP_DEPTH  : stops recursion if too deep
          - zipfile.BadZipFile    : gracefully handles corrupt inner ZIPs
          - Password protection   : detects and skips encrypted inner ZIPs
        """
        nested_stem = Path(entry_name).stem
        indent_note = f"*Contents of nested archive `{entry_name}`:*"

        try:
            # ── Read nested ZIP bytes from parent ─────────────────────────
            raw_bytes = parent_zf.read(entry_name)
            if not raw_bytes:
                return "*(Nested ZIP is empty — skipped)*"

            nested_buf = io.BytesIO(raw_bytes)

            if not zipfile.is_zipfile(nested_buf):
                return f"*(Not a valid ZIP: `{entry_name}`)*"

            nested_buf.seek(0)

            try:
                nested_zf = zipfile.ZipFile(nested_buf, 'r')
            except zipfile.BadZipFile as e:
                return f"*(Corrupt nested ZIP `{entry_name}`: {e})*"

            with nested_zf:
                # Encryption check
                inner_entries = nested_zf.infolist()
                if inner_entries and inner_entries[0].flag_bits & 0x1:
                    return f"*(Nested ZIP `{entry_name}` is password-protected — skipped)*"

                inner_names = nested_zf.namelist()
                if not inner_names:
                    return f"*(Nested ZIP `{entry_name}` is empty)*"

                logger.info(
                    f"  {'  ' * current_depth}  📦 Extracting nested ZIP: {entry_name} "
                    f"({len(inner_names)} entries, depth={current_depth + 1})"
                )

                # ── Build folder tree for nested archive ──────────────────
                nested_tree = self._build_folder_tree(inner_names)
                nested_lines: List[str] = [indent_note, ""]

                # ── Walk nested tree ──────────────────────────────────────
                def _walk_nested(tree: Dict, zf: zipfile.ZipFile, depth: int) -> None:
                    for node_name in sorted(tree.keys()):
                        node = tree[node_name]

                        if node['type'] == 'folder':
                            folder_path = node['path']
                            h = '#' * min(depth + 3, 6)
                            label = '/ (root)' if not folder_path else folder_path + '/'
                            nested_lines.append(f"\n{h} 📁 {label}\n")
                            logger.info(f"  {'  ' * depth}  📁 Nested folder: {folder_path or '(root)'}")
                            _walk_nested(node['children'], zf, depth + 1)

                        else:
                            inner_entry = node['path']

                            if self._should_skip_entry(inner_entry):
                                continue

                            suffix = Path(inner_entry).suffix.lower()
                            fh = '#' * min(depth + 4, 6)

                            if suffix in ZIP_EXTENSIONS:
                                # Further nesting
                                nested_lines.append(f"\n{fh} 📦 `{Path(inner_entry).name}`\n")
                                if depth + 1 >= MAX_NESTED_ZIP_DEPTH:
                                    nested_lines.append(
                                        f"*(Max nesting depth {MAX_NESTED_ZIP_DEPTH} reached — skipped)*"
                                    )
                                else:
                                    deeper_md = self._extract_nested_zip(
                                        zf, inner_entry, image_output_dir,
                                        f"{parent_stem}/{nested_stem}", depth + 1
                                    )
                                    nested_lines.append(deeper_md)

                            elif suffix in TEXT_EXTENSIONS:
                                nested_lines.append(f"\n{fh} 📄 `{Path(inner_entry).name}`\n")
                                content, ok = self._read_text_entry(zf, inner_entry)
                                if ok and content:
                                    nested_lines.append(f"```{suffix.lstrip('.')}\n{content}\n```")
                                    logger.info(f"  {'  ' * depth}  ✅ text {inner_entry}: {len(content):,} chars")
                                else:
                                    nested_lines.append("*(Could not read or empty file)*")

                            elif suffix in IMAGE_EXTENSIONS:
                                nested_lines.append(f"\n{fh} 🖼️ `{Path(inner_entry).name}`\n")
                                img_md, ok = self._describe_image_entry(
                                    zf, inner_entry, image_output_dir,
                                    f"{parent_stem}/{nested_stem}"
                                )
                                nested_lines.append(img_md)
                                logger.info(f"  {'  ' * depth}  {'✅' if ok else '⚠️'} image {inner_entry}")

                            elif suffix in DOC_EXTENSIONS:
                                nested_lines.append(f"\n{fh} 📑 `{Path(inner_entry).name}`\n")
                                tmp_path = self._write_temp_entry(zf, inner_entry, image_output_dir)
                                if not tmp_path:
                                    nested_lines.append("*(Could not extract to temp file)*")
                                else:
                                    try:
                                        sub_md = self._delegate_to_sub_extractor(tmp_path, suffix)
                                        if sub_md:
                                            nested_lines.append(sub_md)
                                            logger.info(f"  {'  ' * depth}  ✅ doc {inner_entry}: {len(sub_md):,} chars")
                                        else:
                                            nested_lines.append("*(Sub-extractor returned no content)*")
                                    except Exception as e:
                                        nested_lines.append(f"*(Sub-extractor failed: {e})*")
                                    finally:
                                        try:
                                            os.unlink(tmp_path)
                                        except Exception:
                                            pass

                            else:
                                nested_lines.append(f"\n{fh} ⏭️ `{Path(inner_entry).name}`\n")
                                nested_lines.append(
                                    f"*(Unsupported type — skipped: `{suffix or 'no extension'}`)*"
                                )

                            nested_lines.append("")

                _walk_nested(nested_tree, nested_zf, depth=current_depth + 1)

            return '\n'.join(nested_lines)

        except Exception as e:
            logger.error(f"❌ Failed to extract nested ZIP {entry_name}: {e}")
            traceback.print_exc()
            return f"*(Error extracting nested ZIP `{entry_name}`: {e})*"

    # ── Image description ─────────────────────────────────────────────────────

    def _describe_image_entry(
        self,
        zf: zipfile.ZipFile,
        entry_name: str,
        image_output_dir: str,
        zip_stem: str,
    ) -> Tuple[str, bool]:
        """
        Read an image entry from the open ZipFile, deduplicate,
        describe via Vision LLM, save to disk.
        Mirrors DocxExtractor._process_images_python_docx() image loop.

        Returns (markdown_string, success_bool).
        """
        try:
            image_bytes = zf.read(entry_name)

            if not image_bytes:
                return "*(Empty image file)*", False

            # Deduplication
            img_hash = self.image_cache.compute_hash(image_bytes)
            if img_hash in getattr(self, '_seen_hashes', set()):
                self._duplicates_skipped += 1
                logger.debug(f"  ⏭ Duplicate image skipped: {entry_name}")
                return "*(Duplicate image — skipped)*", False

            if not hasattr(self, '_seen_hashes'):
                self._seen_hashes: set = set()
            self._seen_hashes.add(img_hash)

            # Check cache
            cached_desc = self.image_cache.get(image_bytes)
            if cached_desc:
                description = cached_desc
                from_cache  = True
                success     = True
            else:
                ext        = Path(entry_name).suffix.lower().lstrip('.')
                media_type = self._get_media_type(ext)
                base64_img = base64.b64encode(image_bytes).decode('utf-8')

                success, description = self.image_service.describe_image_from_base64(
                    base64_img,
                    media_type,
                    context=f"ZIP archive: {zip_stem}",
                    page_number=0
                )
                from_cache = False

                if success:
                    self.image_cache.put(image_bytes, description)

            # Save image to output directory
            safe_name  = re.sub(r'[^\w.\-]', '_', Path(entry_name).name)
            image_path = os.path.join(image_output_dir, safe_name)
            with open(image_path, 'wb') as f:
                f.write(image_bytes)

            cache_str = "cached" if from_cache else "described"
            md = (
                f"**[Image Description]**: {description}\n"
                f"*(File: `{entry_name}`, {cache_str})*"
            )
            return md, success

        except Exception as e:
            logger.warning(f"  ⚠️ Error describing image {entry_name}: {e}")
            return f"*(Image description failed: {e})*", False

    # ── Text entry reader ─────────────────────────────────────────────────────

    def _read_text_entry(
        self, zf: zipfile.ZipFile, entry_name: str
    ) -> Tuple[str, bool]:
        """
        Read a text entry from the open ZipFile.
        Respects MAX_TEXT_ENTRY_BYTES; tries UTF-8 then latin-1.
        """
        try:
            raw = zf.read(entry_name)
            if len(raw) > MAX_TEXT_ENTRY_BYTES:
                logger.warning(
                    f"  ⚠️ {entry_name} is {len(raw):,} bytes — truncating to {MAX_TEXT_ENTRY_BYTES:,}"
                )
                raw = raw[:MAX_TEXT_ENTRY_BYTES]

            try:
                return raw.decode('utf-8'), True
            except UnicodeDecodeError:
                return raw.decode('latin-1', errors='replace'), True

        except Exception as e:
            logger.warning(f"  ⚠️ Could not read {entry_name}: {e}")
            return "", False

    # ── Temp file helper ──────────────────────────────────────────────────────

    def _write_temp_entry(
        self,
        zf: zipfile.ZipFile,
        entry_name: str,
        output_dir: str,
    ) -> Optional[str]:
        """
        Extract a single ZIP entry to a temp file in output_dir.
        Returns the temp file path, or None on failure.
        """
        try:
            data     = zf.read(entry_name)
            suffix   = Path(entry_name).suffix
            safe_name = re.sub(r'[^\w.\-]', '_', Path(entry_name).name)
            tmp_path  = os.path.join(output_dir, f"__tmp_{safe_name}")

            with open(tmp_path, 'wb') as f:
                f.write(data)
            return tmp_path
        except Exception as e:
            logger.warning(f"  ⚠️ Could not write temp file for {entry_name}: {e}")
            return None

    # ── Sub-extractor delegation ──────────────────────────────────────────────

    def _delegate_to_sub_extractor(self, tmp_path: str, suffix: str) -> Optional[str]:
        """
        Delegate to a format-specific extractor for nested documents.
        Returns extracted markdown string, or None if unavailable.

        Supported:
          .pdf   → PDFExtractor
          .docx  → DocxExtractor
          .xlsx / .xlsm → XlsxExtractor (or Docling fallback)
          .pptx  → PptxExtractor (or Docling fallback)
          .msg   → MsgExtractor (or Docling fallback)
          .epub  → EpubExtractor (or Docling fallback)
        """
        try:
            if suffix == '.pdf':
                from app.services.extraction.extractors.pdf_extractor import PDFExtractor
                extractor = PDFExtractor()
                md, success, _ = extractor.extract(tmp_path)
                return md if success else None

            elif suffix == '.docx':
                from app.services.extraction.extractors.docx_extractor import DocxExtractor
                extractor = DocxExtractor()
                md, success, _ = extractor.extract(tmp_path)
                return md if success else None

            elif suffix in {'.xlsx', '.xlsm'}:
                # Try dedicated XlsxExtractor first, fall back to Docling
                try:
                    from app.services.extraction.extractors.excel_extractor import ExcelExtractor
                    extractor = ExcelExtractor()
                    md, success, _ = extractor.extract(tmp_path)
                    if success:
                        return md
                except (ImportError, Exception) as e:
                    logger.info(f"  ExcelExtractor unavailable ({e}), trying Docling for {suffix}")
                return self._delegate_to_docling(tmp_path, suffix)

            elif suffix == '.pptx':
                try:
                    from app.services.extraction.extractors.pptx_extractor import PptxExtractor
                    extractor = PptxExtractor()
                    md, success, _ = extractor.extract(tmp_path)
                    if success:
                        return md
                except (ImportError, Exception) as e:
                    logger.info(f"  PptxExtractor unavailable ({e}), trying Docling for {suffix}")
                return self._delegate_to_docling(tmp_path, suffix)

            elif suffix == '.msg':
                try:
                    from app.services.extraction.extractors.msg_extractor import MSGExtractor
                    extractor = MSGExtractor()
                    md, success, _ = extractor.extract(tmp_path)
                    if success:
                        return md
                except (ImportError, Exception) as e:
                    logger.info(f"  MSGExtractor unavailable ({e}), trying Docling for {suffix}")
                return self._delegate_to_docling(tmp_path, suffix)

            elif suffix == '.epub':
                try:
                    from app.services.extraction.extractors.epub_extractor import EPUBExtractor
                    extractor = EPUBExtractor()
                    md, success, _ = extractor.extract(tmp_path)
                    if success:
                        return md
                except (ImportError, Exception) as e:
                    logger.info(f"  EPUBExtractor unavailable ({e}), trying Docling for {suffix}")
                return self._delegate_to_docling(tmp_path, suffix)

        except ImportError as e:
            logger.warning(f"  Sub-extractor not available for {suffix}: {e}")
            return self._delegate_to_docling(tmp_path, suffix)
        except Exception as e:
            logger.warning(f"  Sub-extractor error for {suffix}: {e}")

        return None

    def _delegate_to_docling(self, tmp_path: str, suffix: str) -> Optional[str]:
        """
        Fallback: convert a temp file through Docling.
        Used when a dedicated extractor is not available.
        """
        try:
            from docling.document_converter import DocumentConverter
            converter = DocumentConverter()
            doc_result = converter.convert(tmp_path)
            md_text = doc_result.document.export_to_markdown()
            if md_text.strip():
                logger.info(f"  ✅ Docling fallback OK for {suffix}: {len(md_text):,} chars")
                return md_text
            logger.warning(f"  ⚠️ Docling returned empty markdown for {suffix}")
        except ImportError:
            logger.info("  Docling not available — no fallback for sub-extractor")
        except Exception as e:
            logger.warning(f"  Docling fallback failed for {suffix}: {e}")
        return None

    # ── Image directory helper ────────────────────────────────────────────────

    def _create_image_directory_user_isolated(
        self,
        file_path: str,
        zip_stem: str,
        user_id: str,
    ) -> str:
        """
        Create user-isolated image output directory.
        Mirrors DocxExtractor._create_image_directory_user_isolated().
        """
        base_dir  = Path(file_path).parent
        image_dir = base_dir / "extracted_images" / zip_stem
        image_dir.mkdir(parents=True, exist_ok=True)
        return str(image_dir)

    # ── Skip filter ───────────────────────────────────────────────────────────

    @staticmethod
    def _should_skip_entry(entry_name: str) -> bool:
        """
        Return True for entries that should be silently skipped:
          - macOS __MACOSX metadata entries
          - Hidden files (dot-files)
          - Windows thumbs / desktop.ini artefacts
          - Zero-length name
        """
        name = Path(entry_name).name
        if not name:
            return True
        if '__MACOSX' in entry_name or entry_name.startswith('.'):
            return True
        if name.startswith('.'):
            return True
        if name.lower() in {'thumbs.db', 'desktop.ini', '.ds_store'}:
            return True
        return False

    # ── MIME helpers ──────────────────────────────────────────────────────────

    def _get_media_type(self, extension: str) -> str:
        """Map file extension to MIME type — mirrors DocxExtractor._get_media_type()."""
        return {
            'jpg'  : 'image/jpeg',
            'jpeg' : 'image/jpeg',
            'png'  : 'image/png',
            'gif'  : 'image/gif',
            'webp' : 'image/webp',
            'bmp'  : 'image/bmp',
        }.get(extension.lower(), 'image/jpeg')

    # ── BaseExtractor.get_metadata ────────────────────────────────────────────

    def get_metadata(self, file_path: str) -> Dict:
        """Basic file metadata — mirrors DocxExtractor pattern."""
        path = Path(file_path)
        return {
            'filename'      : path.name,
            'size_bytes'    : path.stat().st_size,
            'extension'     : path.suffix.lower(),
        }

    # ── Extraction stats ──────────────────────────────────────────────────────

    def get_extraction_stats(self) -> Dict:
        """Return extraction statistics."""
        return {
            'duplicates_skipped': self._duplicates_skipped,
        }