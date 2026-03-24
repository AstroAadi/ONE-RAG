"""
ZIP extractor that delegates each file inside the archive to the existing
extractor registry and returns one markdown payload per file.
"""

from __future__ import annotations

import logging
import re
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Dict, List, Optional, Tuple

try:
    from ..base_extractor import BaseExtractor
except ImportError:
    from base_extractor import BaseExtractor

logger = logging.getLogger(__name__)

PLAIN_TEXT_EXTENSIONS = {
    ".txt",
    ".text",
    ".md",
    ".markdown",
    ".csv",
    ".tsv",
    ".json",
    ".jsonl",
    ".ndjson",
    ".xml",
    ".html",
    ".htm",
    ".yaml",
    ".yml",
    ".log",
    ".ini",
    ".cfg",
    ".conf",
    ".properties",
    ".sql",
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".java",
    ".c",
    ".cpp",
    ".h",
    ".hpp",
    ".cs",
    ".go",
    ".rs",
    ".sh",
    ".bat",
    ".ps1",
    ".toml",
    ".env",
    ".vtt",
}

DOCLING_FALLBACK_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".pptx",
    ".xlsx",
    ".xlsm",
    ".html",
    ".xml",
    ".md",
    ".txt",
}

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}

MAX_TEXT_ENTRY_BYTES = 1024 * 1024
MAX_NESTED_ZIP_DEPTH = 5
SAFE_PATH_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1F]')


class ZIPExtractor(BaseExtractor):
    def _get_supported_formats(self) -> List[str]:
        return ["zip"]

    def validate_file(self, file_path: str) -> Tuple[bool, str]:
        path = Path(file_path)

        if not path.exists():
            return False, f"File does not exist: {file_path}"
        if path.suffix.lower() != ".zip":
            return False, f"Not a ZIP file: {path.suffix}"
        if path.stat().st_size == 0:
            return False, "ZIP file is empty"

        if not zipfile.is_zipfile(file_path):
            return False, "File is not a valid ZIP archive"

        try:
            with zipfile.ZipFile(file_path, "r") as zf:
                if not zf.infolist():
                    return False, "ZIP archive contains no entries"
        except zipfile.BadZipFile as exc:
            return False, f"Corrupt ZIP archive: {exc}"

        return True, "Valid ZIP archive"

    def extract(self, file_path: str) -> Tuple[str, bool, Dict]:
        file_chunks, ok, metadata = self.extract_chunks(file_path)
        if not ok:
            return "", False, metadata

        archive_name = metadata["archive_name"]
        manifest = self._build_archive_manifest(archive_name, file_chunks, metadata)

        return manifest, True, metadata

    def extract_chunks(self, file_path: str) -> Tuple[List[Dict], bool, Dict]:
        is_valid, message = self.validate_file(file_path)
        if not is_valid:
            return [], False, {"error": message}

        archive_name = Path(file_path).stem

        try:
            file_chunks = self._extract_chunks_from_zip(
                file_path=file_path,
                archive_name=archive_name,
                source_prefix="",
                depth=0,
            )

            if not file_chunks:
                return [], False, {
                    "error": "No supported or readable files were found in the ZIP archive",
                    "archive_name": archive_name,
                }

            metadata = self._build_archive_metadata(file_path, archive_name, file_chunks)
            logger.info(
                "ZIP extraction complete: %s -> %s file(s)",
                Path(file_path).name,
                len(file_chunks),
            )
            return file_chunks, True, metadata
        except Exception as exc:
            logger.error("ZIP extraction failed for %s: %s", file_path, exc, exc_info=True)
            return [], False, {"error": str(exc), "archive_name": archive_name}

    def _extract_chunks_from_zip(
        self,
        file_path: str,
        archive_name: str,
        source_prefix: str,
        depth: int,
    ) -> List[Dict]:
        if depth >= MAX_NESTED_ZIP_DEPTH:
            logger.warning("Skipping nested ZIP beyond max depth: %s", file_path)
            return []

        with tempfile.TemporaryDirectory(prefix=f"zip_extract_{Path(file_path).stem}_") as temp_dir:
            root_dir = self._extract_archive_to_temp_tree(file_path, temp_dir)
            return self._build_file_chunks_from_directory(
                root_dir=root_dir,
                archive_name=archive_name,
                source_prefix=source_prefix,
                depth=depth,
            )

    def _extract_archive_to_temp_tree(self, file_path: str, temp_dir: str) -> Path:
        archive_root = Path(temp_dir) / "archive"
        archive_root.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(file_path, "r") as zf:
            for info in zf.infolist():
                safe_relative = self._normalize_zip_entry_path(info.filename)
                if safe_relative is None:
                    logger.warning("Skipping unsafe ZIP entry: %s", info.filename)
                    continue

                target_path = archive_root.joinpath(*safe_relative.parts)

                if info.is_dir():
                    target_path.mkdir(parents=True, exist_ok=True)
                    continue

                target_path.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info, "r") as src, open(target_path, "wb") as dst:
                    dst.write(src.read())

        return archive_root

    def _build_file_chunks_from_directory(
        self,
        root_dir: Path,
        archive_name: str,
        source_prefix: str,
        depth: int,
    ) -> List[Dict]:
        chunks: List[Dict] = []

        for file_path in sorted(path for path in root_dir.rglob("*") if path.is_file()):
            relative_source = file_path.relative_to(root_dir).as_posix()
            if source_prefix:
                relative_source = f"{source_prefix.rstrip('/')}/{relative_source}"

            if self._should_skip_entry(relative_source):
                continue

            entry_chunks = self._extract_file_entry(
                file_path=file_path,
                source_path=relative_source,
                archive_name=archive_name,
                depth=depth,
            )

            for entry_chunk in entry_chunks:
                entry_chunk["chunk_index"] = len(chunks)
                chunks.append(entry_chunk)

        return chunks

    def _extract_file_entry(
        self,
        file_path: Path,
        source_path: str,
        archive_name: str,
        depth: int,
    ) -> List[Dict]:
        suffix = file_path.suffix.lower()

        if suffix == ".zip":
            logger.info("Entering nested ZIP: %s", source_path)
            return self._extract_chunks_from_zip(
                file_path=str(file_path),
                archive_name=archive_name,
                source_prefix=source_path,
                depth=depth + 1,
            )

        content = ""
        success = False
        metadata: Dict = {}
        method = ""

        if self._has_registered_extractor(suffix):
            content, success, metadata, method = self._extract_with_registered_extractor(file_path)

        if (not success or not content.strip()) and (suffix in PLAIN_TEXT_EXTENSIONS or self._looks_like_text(file_path)):
            content, success = self._read_text_file(file_path)
            metadata = {
                "total_pages": 1 if content.strip() else 0,
                "word_count": self._estimate_word_count(content),
                "images_extracted": 0,
                "tables_extracted": 0,
            }
            method = "native_text"
            if success and content.strip() and suffix not in {".md", ".markdown"}:
                if suffix == ".json":
                    content = self._format_json_markdown(source_path, content)
                else:
                    content = self._format_plain_text_markdown(source_path, suffix, content)

        if not success or not content.strip():
            logger.warning("No content extracted from ZIP entry: %s", source_path)
            return []

        word_count = metadata.get("word_count")
        if not isinstance(word_count, int) or word_count <= 0:
            word_count = self._estimate_word_count(content)

        total_pages = metadata.get("total_pages")
        if not isinstance(total_pages, int):
            total_pages = 0

        image_count = metadata.get("images_extracted", metadata.get("image_count", 0))
        if not isinstance(image_count, int):
            image_count = 0
        if image_count == 0 and suffix in IMAGE_EXTENSIONS:
            image_count = 1

        table_count = metadata.get("tables_extracted", metadata.get("table_count", 0))
        if not isinstance(table_count, int):
            table_count = 0

        markdown_relative_path = self.source_path_to_markdown_path(source_path).as_posix()

        logger.info("ZIP extracted %s via %s", source_path, method)

        return [{
            "content": content,
            "source_path": source_path,
            "markdown_relative_path": markdown_relative_path,
            "file_name": file_path.name,
            "file_type": suffix or "no_extension",
            "chunk_index": 0,
            "archive_name": archive_name,
            "extraction_method": method,
            "total_pages": total_pages,
            "word_count": word_count,
            "images_extracted": image_count,
            "tables_extracted": table_count,
        }]

    def _extract_with_registered_extractor(
        self,
        file_path: Path,
    ) -> Tuple[str, bool, Dict, str]:
        extractor_factory = self._get_extractor_factory()
        extractor = extractor_factory.get_extractor(file_path.suffix, config=self.config)

        if not extractor:
            return "", False, {"error": f"No extractor registered for {file_path.suffix}"}, "unavailable"

        try:
            content, success, metadata = extractor.extract(str(file_path))
            metadata = metadata or {}

            if success and content and content.strip():
                return content, True, metadata, extractor.__class__.__name__

            fallback_content = self._try_docling_fallback(file_path)
            if fallback_content:
                fallback_metadata = {
                    **metadata,
                    "word_count": self._estimate_word_count(fallback_content),
                }
                return fallback_content, True, fallback_metadata, "docling_fallback"

            return content or "", False, metadata, extractor.__class__.__name__
        except Exception as exc:
            logger.warning("Extractor failed for %s: %s", file_path, exc)
            fallback_content = self._try_docling_fallback(file_path)
            if fallback_content:
                return fallback_content, True, {
                    "word_count": self._estimate_word_count(fallback_content),
                }, "docling_fallback"
            return "", False, {"error": str(exc)}, extractor.__class__.__name__

    def _try_docling_fallback(self, file_path: Path) -> str:
        if file_path.suffix.lower() not in DOCLING_FALLBACK_EXTENSIONS:
            return ""

        try:
            from docling.document_converter import DocumentConverter

            result = DocumentConverter().convert(str(file_path))
            markdown = result.document.export_to_markdown()
            return markdown.strip()
        except Exception as exc:
            logger.debug("Docling fallback failed for %s: %s", file_path, exc)
            return ""

    def _get_extractor_factory(self):
        try:
            from app.services.extraction.extractor_factory import (
                ExtractorFactory,
                auto_register_extractors,
            )
        except ImportError:
            from ..extractor_factory import ExtractorFactory, auto_register_extractors

        if not ExtractorFactory.get_supported_formats() and not ExtractorFactory.get_unavailable_formats():
            auto_register_extractors()

        return ExtractorFactory

    def _has_registered_extractor(self, suffix: str) -> bool:
        clean_suffix = suffix.lower().lstrip(".")
        if not clean_suffix:
            return False

        try:
            return self._get_extractor_factory().is_supported(clean_suffix)
        except Exception as exc:
            logger.debug("Could not inspect extractor registry for %s: %s", suffix, exc)
            return False

    def _build_archive_metadata(
        self,
        file_path: str,
        archive_name: str,
        file_chunks: List[Dict],
    ) -> Dict:
        total_pages = sum(chunk.get("total_pages", 0) for chunk in file_chunks)
        word_count = sum(chunk.get("word_count", 0) for chunk in file_chunks)
        image_count = sum(chunk.get("images_extracted", 0) for chunk in file_chunks)
        table_count = sum(chunk.get("tables_extracted", 0) for chunk in file_chunks)

        files_by_type: Dict[str, int] = {}
        for chunk in file_chunks:
            file_type = chunk.get("file_type", "unknown") or "unknown"
            files_by_type[file_type] = files_by_type.get(file_type, 0) + 1

        return {
            "filename": Path(file_path).name,
            "extension": ".zip",
            "size_bytes": Path(file_path).stat().st_size,
            "archive_name": archive_name,
            "files_extracted": len(file_chunks),
            "file_chunks": file_chunks,
            "total_pages": total_pages,
            "word_count": word_count,
            "images_extracted": image_count,
            "tables_extracted": table_count,
            "files_by_type": files_by_type,
            "extraction_method": "per_file_delegation",
        }

    def _build_archive_manifest(
        self,
        archive_name: str,
        file_chunks: List[Dict],
        metadata: Dict,
    ) -> str:
        lines = [
            f"# Archive: {archive_name}",
            "",
            f"**Files Extracted**: {metadata.get('files_extracted', len(file_chunks))}",
            f"**Total Pages**: {metadata.get('total_pages', 0)}",
            f"**Word Count**: {metadata.get('word_count', 0)}",
            f"**Images Extracted**: {metadata.get('images_extracted', 0)}",
            f"**Tables Extracted**: {metadata.get('tables_extracted', 0)}",
            "",
            "## Files",
            "",
        ]

        for chunk in file_chunks:
            lines.append(
                f"- `{chunk['source_path']}` -> `{chunk['markdown_relative_path']}` "
                f"({chunk['extraction_method']})"
            )

        return "\n".join(lines).strip()

    def _read_text_file(self, file_path: Path) -> Tuple[str, bool]:
        try:
            raw = file_path.read_bytes()
            if len(raw) > MAX_TEXT_ENTRY_BYTES:
                logger.warning(
                    "Truncating large text file inside ZIP: %s (%s bytes)",
                    file_path.name,
                    len(raw),
                )
                raw = raw[:MAX_TEXT_ENTRY_BYTES]

            try:
                return raw.decode("utf-8"), True
            except UnicodeDecodeError:
                return raw.decode("latin-1", errors="replace"), True
        except Exception as exc:
            logger.warning("Could not read text entry %s: %s", file_path, exc)
            return "", False

    def _looks_like_text(self, file_path: Path) -> bool:
        try:
            sample = file_path.read_bytes()[:4096]
        except Exception:
            return False

        if not sample:
            return False
        if b"\x00" in sample:
            return False

        printable = sum(
            1
            for byte in sample
            if byte in (9, 10, 13) or 32 <= byte <= 126
        )
        return (printable / len(sample)) >= 0.85

    def _format_plain_text_markdown(self, source_path: str, suffix: str, content: str) -> str:
        language = suffix.lstrip(".")
        return f"# File: {source_path}\n\n```{language}\n{content}\n```"

    def _format_json_markdown(self, source_path: str, content: str) -> str:
        """
        convert JSON content to pretty-printed markdown code block
        """
        import json
        try:
            data= json.loads(content)
        except Exception:
            return f'# File: {source_path}\n\n```json\n{content}\n```'
        
        lines = [f"# File: {source_path}",""]

        def format_item(obj, indent=0):
            prefix = "  " * indent
            if isinstance(obj, dict):
                for k,v in obj.items():
                    if isinstance(v, (dict, list)):
                        lines.append(f"{prefix}- **{k}:**")
                        format_item(v, indent + 1)
                    else:
                        lines.append(f"{prefix}- **{k}:** {v}")

            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    lines.append(f"{prefix}- Item {i + 1}:")
                    format_item(item,indent + 1)
            else:
                lines.append(f"{prefix}- {obj}")
        format_item(data)
        return "\n".join(lines)
                    

    def get_metadata(self, file_path: str) -> Dict:
        path = Path(file_path)
        return {
            "filename": path.name,
            "size_bytes": path.stat().st_size,
            "extension": path.suffix.lower(),
        }

    @staticmethod
    def source_path_to_markdown_path(source_path: str) -> Path:
        raw_path = source_path.replace("\\", "/").strip("/")
        if not raw_path:
            return Path("untitled.md")

        pure_path = PurePosixPath(raw_path)
        safe_parts = [ZIPExtractor._sanitize_path_part(part) for part in pure_path.parts if part not in {"", "."}]

        if not safe_parts:
            return Path("untitled.md")

        final_name = safe_parts[-1]
        final_stem = Path(final_name).stem if Path(final_name).suffix else final_name
        safe_parts[-1] = f"{final_stem}.md"

        return Path(*safe_parts)

    @staticmethod
    def _sanitize_path_part(value: str) -> str:
        cleaned = SAFE_PATH_CHARS.sub("_", value).strip()
        return cleaned or "untitled"

    @staticmethod
    def _normalize_zip_entry_path(raw_name: str) -> Optional[PurePosixPath]:
        normalized = raw_name.replace("\\", "/").strip("/")
        if not normalized:
            return None

        path = PurePosixPath(normalized)
        if path.is_absolute():
            return None
        if any(part in {"", ".", ".."} for part in path.parts):
            return None
        if path.parts and ":" in path.parts[0]:
            return None

        return path

    @staticmethod
    def _should_skip_entry(entry_name: str) -> bool:
        name = Path(entry_name).name
        lowered = entry_name.lower()

        if not name:
            return True
        if "__macosx" in lowered:
            return True
        if name.startswith("."):
            return True
        if name.lower() in {"thumbs.db", "desktop.ini", ".ds_store"}:
            return True
        return False

    @staticmethod
    def _estimate_word_count(text: str) -> int:
        return len(re.findall(r"\b\w+\b", text or ""))
    

    


