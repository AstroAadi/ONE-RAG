"""
Enhanced Chunking Service with 4 Strategies
=============================================

Strategies:
1. Default      – fixed-size with hardcoded 300-token / 30-overlap settings
2. Fixed-size   – user-specified token size + overlap
3. Semantic     – similarity-guided grouping with structure awareness
4. Hierarchical – overlapping parent windows → fixed-size child windows
5. No-chunking  – whole document as one chunk
6. JSON         – structural-recursive: one parent per JSON entry, children
                  formed by recursively decomposing the JSON tree (never a
                  sliding window).  Each child carries a breadcrumb header.

Bugs fixed vs. original
────────────────────────
FIX-1  _chunk_semantic entire body was indented inside the `if not text.strip()`
       block, making it dead code — function silently returned None for every
       non-empty document.

FIX-2  _chunk_semantic referenced undefined locals `max_tokens` and
       `overlap_tokens`; replaced with `self.target_chunk_tokens` and
       `self.chunk_overlap_tokens` (with breakpoint_threshold → sim_threshold).

FIX-3  _extract_overlap_tail and the inner duplicate _is_semantically_similar
       were indented as nested functions inside _chunk_semantic; promoted to
       proper class methods.

FIX-4  _merge_small_chunks used undefined variable `token_ceiling` in both the
       forward pass and backward pass; replaced with `max_tokens` parameter that
       is now passed explicitly by every caller.

FIX-5  _merge_small_chunks signature was missing the `max_tokens` parameter;
       added with a sensible default (self.max_chunk_tokens).

FIX-6  _chunk_by_sentence called _merge_small_chunks without the required
       max_tokens argument; fixed.

FIX-7  Duplicate definition of _is_semantically_similar (one broken copy inside
       _chunk_semantic, one correct copy at class level); removed the broken copy.

FIX-8  _is_semantically_similar inner copy compared `>= threshold` where
       threshold could be None, causing TypeError; the correct class-level
       version uses a default of 0.25 and is now the only definition.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    logging.warning("tiktoken not installed – install with: pip install tiktoken")

try:
    import nltk
    NLTK_AVAILABLE = True
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)
except ImportError:
    NLTK_AVAILABLE = False
    logging.warning("nltk not installed – install with: pip install nltk")

from app.core.config import get_settings

logger   = logging.getLogger(__name__)
settings = get_settings()


# ─────────────────────────────────────────────────────────────────────────────
# Enum
# ─────────────────────────────────────────────────────────────────────────────

class ChunkingStrategy(str, Enum):
    """Mirrors schemas.py exactly (including NO_CHUNKING)."""
    DEFAULT      = "default"
    FIXED_SIZE   = "fixed_size"
    HIERARCHICAL = "hierarchical"
    SEMANTIC     = "semantic"
    NO_CHUNKING  = "no_chunking"
    JSON         = "json"


# ─────────────────────────────────────────────────────────────────────────────
# Token counter
# ─────────────────────────────────────────────────────────────────────────────

class TokenCounter:
    """tiktoken-based counter with 4-chars-per-token character fallback."""

    def __init__(self, model: str = "cl100k_base") -> None:
        self.encoding = None
        if TIKTOKEN_AVAILABLE:
            try:
                self.encoding = tiktoken.get_encoding(model)
                logger.info(f"✓ Token counter initialised: {model}")
            except Exception as exc:
                logger.warning(f"Failed to load tiktoken encoding: {exc}")
        if not self.encoding:
            logger.warning("Using character-based token estimation (4 chars ≈ 1 token)")

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        if self.encoding:
            try:
                return len(self.encoding.encode(text))
            except Exception:
                pass
        return max(1, len(text) // 4)

    def get_tokens(self, text: str) -> List[int]:
        if not text:
            return []
        if self.encoding:
            try:
                return self.encoding.encode(text)
            except Exception:
                pass
        return []

    def decode_tokens(self, tokens: List[int]) -> str:
        if not tokens:
            return ""
        if self.encoding:
            try:
                return self.encoding.decode(tokens)
            except Exception:
                pass
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# Main service
# ─────────────────────────────────────────────────────────────────────────────

class EnhancedChunkingService:
    """
    Chunking service supporting default, fixed_size, semantic, hierarchical,
    and no_chunking strategies.
    """

    def __init__(
        self,
        target_chunk_tokens:  int = 300,
        chunk_overlap_tokens: int = 20,
        tokenizer_model:      str = "cl100k_base",
        min_chunk_tokens:     int = 100,
        max_chunk_tokens:     int = 1024,
    ) -> None:
        self.target_chunk_tokens  = target_chunk_tokens
        self.chunk_overlap_tokens = chunk_overlap_tokens
        self.tokenizer_model      = tokenizer_model
        self.min_chunk_tokens     = min_chunk_tokens
        self.max_chunk_tokens     = max_chunk_tokens
        self.token_counter        = TokenCounter(model=tokenizer_model)

        logger.info(
            f"✓ EnhancedChunkingService initialised:  "
            f"target={target_chunk_tokens}t  overlap={chunk_overlap_tokens}t  "
            f"min={min_chunk_tokens}t  max={max_chunk_tokens}t  "
            f"tokenizer={tokenizer_model}"
        )

    # ──────────────────────────────────────────────────────────────────────
    # PUBLIC ENTRY POINT
    # ──────────────────────────────────────────────────────────────────────

    def split_document_into_chunks(
        self,
        markdown_text:        str,
        structured_data:      Optional[Dict] = None,   # reserved / unused
        base_metadata:        Optional[Dict] = None,
        strategy:             ChunkingStrategy = ChunkingStrategy.FIXED_SIZE,
        # Semantic params
        max_buffer_size:      Optional[int]   = None,
        breakpoint_threshold: Optional[float] = None,
        # Hierarchical params
        parent_chunk_size:    Optional[int]   = None,
        child_chunk_size:     Optional[int]   = None,
        child_chunk_overlap:  Optional[int]   = None,
        # ZIP per-file chunking
        # Pass the `file_chunks` list returned by ZIPExtractor.extract_chunks()
        # so that each file inside the ZIP is chunked independently.
        # When provided, every strategy operates on individual files:
        #   NO_CHUNKING  → exactly 1 chunk per file  (4 files → 4 chunks)
        #   any other    → strategy applied per file, results concatenated
        zip_file_chunks:      Optional[List[Dict]] = None,
        # JSON structural chunking params
        # Used only when strategy == ChunkingStrategy.JSON.
        #   json_child_tokens   – max tokens per child chunk  (default: 100)
        #   json_context_prefix – prepend breadcrumb header to every child
        json_child_tokens:    Optional[int]  = None,
        json_context_prefix:  bool           = True,
    ) -> List[Dict[str, Any]]:
        """
        Split *markdown_text* into chunks using *strategy*.

        Hierarchical strategy returns BOTH parent chunks (chunk_type="parent")
        AND child chunks (chunk_type="child") in a flat list — parents first,
        then children in order.

        Each child carries:
            metadata["parent_chunk_id"]  → chunk_id of the owning parent
        Each parent carries:
            metadata["child_chunk_ids"]  → list of its children's chunk_ids

        ZIP behaviour
        ─────────────
        When *zip_file_chunks* is supplied (non-empty list of per-file dicts
        from ZIPExtractor.extract_chunks()), each file inside the archive is
        chunked independently and the results are combined into a flat list.
        This guarantees that NO_CHUNKING yields exactly one chunk per file.
        """
        if isinstance(strategy, str):
            try:
                strategy = ChunkingStrategy(strategy.lower())
            except ValueError:
                logger.warning(
                    f"Invalid chunking strategy '{strategy}'. Falling back to 'default'."
                )
                strategy = ChunkingStrategy.DEFAULT

        if base_metadata is None:
            base_metadata = {}

        # ── ZIP: per-file chunking ────────────────────────────────────────────
        if zip_file_chunks:
            logger.info(
                f"ZIP mode: chunking {len(zip_file_chunks)} file(s) individually "
                f"using '{strategy.value}' strategy"
            )
            return self._chunk_zip_files(
                zip_file_chunks=zip_file_chunks,
                base_metadata=base_metadata,
                strategy=strategy,
                max_buffer_size=max_buffer_size,
                breakpoint_threshold=breakpoint_threshold,
                parent_chunk_size=parent_chunk_size,
                child_chunk_size=child_chunk_size,
                child_chunk_overlap=child_chunk_overlap,
                json_child_tokens=json_child_tokens,
                json_context_prefix=json_context_prefix,
            )

        # ── Normal (non-ZIP) path ─────────────────────────────────────────────
        if not markdown_text or not markdown_text.strip():
            logger.warning("Empty document provided")
            return []

        logger.info(
            f"Chunking document using '{strategy.value}' "
            f"({len(markdown_text):,} chars) ..."
        )

        if strategy == ChunkingStrategy.DEFAULT:
            chunks = self._chunk_fixed_size(markdown_text, base_metadata)

        elif strategy == ChunkingStrategy.FIXED_SIZE:
            chunks = self._chunk_fixed_size(markdown_text, base_metadata)

        elif strategy == ChunkingStrategy.HIERARCHICAL:
            chunks = self._chunk_hierarchical(
                markdown_text,
                base_metadata,
                max_parent_tokens=parent_chunk_size,
                max_child_tokens=child_chunk_size,
                child_overlap_tokens=child_chunk_overlap,
            )

        elif strategy == ChunkingStrategy.SEMANTIC:
            chunks = self._chunk_semantic(
                markdown_text,
                base_metadata,
                max_buffer_size=max_buffer_size,
                breakpoint_threshold=breakpoint_threshold,
            )

        elif strategy == ChunkingStrategy.NO_CHUNKING:
            chunks = self._chunk_no_chunking(markdown_text, base_metadata)

        elif strategy == ChunkingStrategy.JSON:
            chunks = self._chunk_json(
                markdown_text,
                base_metadata,
                child_tokens=json_child_tokens,
                context_prefix=json_context_prefix,
            )

        else:
            logger.warning(f"Unknown strategy '{strategy}' – falling back to fixed_size")
            chunks = self._chunk_fixed_size(markdown_text, base_metadata)

        if not chunks:
            logger.warning("No chunks created")
            return []

        logger.info(f"✅ Created {len(chunks)} chunk(s) using '{strategy.value}'")
        return chunks

    # ──────────────────────────────────────────────────────────────────────
    # ZIP per-file dispatcher
    # ──────────────────────────────────────────────────────────────────────

    def _chunk_zip_files(
        self,
        zip_file_chunks:      List[Dict],
        base_metadata:        Dict,
        strategy:             ChunkingStrategy,
        max_buffer_size:      Optional[int]   = None,
        breakpoint_threshold: Optional[float] = None,
        parent_chunk_size:    Optional[int]   = None,
        child_chunk_size:     Optional[int]   = None,
        child_chunk_overlap:  Optional[int]   = None,
        json_child_tokens:    Optional[int]   = None,
        json_context_prefix:  bool            = True,
    ) -> List[Dict[str, Any]]:
        """
        Apply *strategy* independently to each file inside a ZIP archive.

        Each element of *zip_file_chunks* is a dict produced by
        ZIPExtractor.extract_chunks() and must contain at minimum:
            'content'           – extracted text / markdown for that file
            'source_path'       – e.g. "docs/report.pdf"
            'file_name'         – e.g. "report.pdf"
            'file_type'         – e.g. ".pdf"
            'extraction_method' – e.g. "docling"
            'archive_name'      – ZIP stem

        Every chunk produced here gets extra metadata keys:
            'zip_source_file'   – source_path of the originating file
            'zip_file_name'     – file_name of the originating file
            'zip_archive'       – archive_name

        NO_CHUNKING yields exactly one chunk per file regardless of size.
        All other strategies apply their normal logic per file.
        """
        all_chunks: List[Dict[str, Any]] = []
        global_chunk_index = 1

        for file_entry in zip_file_chunks:
            content    = file_entry.get('content', '')
            source_path = file_entry.get('source_path', '')
            file_name  = file_entry.get('file_name', '')
            file_type  = file_entry.get('file_type', '')
            archive    = file_entry.get('archive_name', '')
            method     = file_entry.get('extraction_method', '')

            if not content or not content.strip():
                logger.debug(f"ZIP chunking: skipping empty file '{source_path}'")
                continue

            # Merge per-file identity into metadata so every chunk knows
            # which file inside the ZIP it came from.
            file_metadata = {
                **base_metadata,
                'zip_source_file': source_path,
                'zip_file_name':   file_name,
                'zip_archive':     archive,
                'zip_file_type':   file_type,
                'zip_extraction_method': method,
            }

            logger.info(
                f"  ZIP chunk: '{source_path}' "
                f"({len(content):,} chars, strategy='{strategy.value}')"
            )

            if strategy == ChunkingStrategy.NO_CHUNKING:
                file_chunks_out = self._chunk_no_chunking(content, file_metadata)

            elif strategy in (ChunkingStrategy.DEFAULT, ChunkingStrategy.FIXED_SIZE):
                file_chunks_out = self._chunk_fixed_size(content, file_metadata)

            elif strategy == ChunkingStrategy.HIERARCHICAL:
                file_chunks_out = self._chunk_hierarchical(
                    content, file_metadata,
                    max_parent_tokens=parent_chunk_size,
                    max_child_tokens=child_chunk_size,
                    child_overlap_tokens=child_chunk_overlap,
                )

            elif strategy == ChunkingStrategy.SEMANTIC:
                file_chunks_out = self._chunk_semantic(
                    content, file_metadata,
                    max_buffer_size=max_buffer_size,
                    breakpoint_threshold=breakpoint_threshold,
                )

            elif strategy == ChunkingStrategy.JSON:
                file_chunks_out = self._chunk_json(
                    content, file_metadata,
                    child_tokens=json_child_tokens,
                    context_prefix=json_context_prefix,
                )

            else:
                logger.warning(
                    f"Unknown strategy '{strategy}' for ZIP file '{source_path}' "
                    f"– falling back to fixed_size"
                )
                file_chunks_out = self._chunk_fixed_size(content, file_metadata)

            # Re-index chunk_index values to be globally unique across all files
            for chunk in file_chunks_out:
                chunk['metadata']['chunk_index']        = global_chunk_index
                chunk['metadata']['zip_file_chunk_index'] = chunk['metadata'].get(
                    'chunk_index', global_chunk_index
                )
                global_chunk_index += 1

            all_chunks.extend(file_chunks_out)
            logger.info(
                f"  ✅ '{source_path}' → {len(file_chunks_out)} chunk(s)"
            )

        logger.info(
            f"✅ ZIP chunking complete: {len(zip_file_chunks)} file(s) → "
            f"{len(all_chunks)} total chunk(s) using '{strategy.value}'"
        )
        return all_chunks

    # ──────────────────────────────────────────────────────────────────────
    # STRATEGY: JSON  (structural-recursive)
    # ──────────────────────────────────────────────────────────────────────

    def _chunk_json(
        self,
        text:           str,
        base_metadata:  Dict,
        child_tokens:   Optional[int] = None,
        context_prefix: bool          = True,
    ) -> List[Dict[str, Any]]:
        """
        JSON structural-recursive chunking.

        Guarantee: exactly ONE parent per JSON entry (the full object).
        Children are produced by recursively walking the JSON tree —
        never by sliding a character / token window across serialized text.

        Algorithm per entry
        ───────────────────
        1.  The full serialized entry  →  parent chunk
        2.  Collect top-level key→value pairs (or list elements).
        3.  For each item:
              a. item fits (≤ child_tokens)  →  accumulate with siblings.
              b. accumulated group would overflow  →  flush as one child,
                 then start a new group.
              c. item is itself too large  →  flush accumulator, recurse
                 deeper into the item's sub-keys.
        4.  Every child is prefixed with a self-describing breadcrumb:

              Action Network > properties > codex
              ────────────────────────────────────
              {
                "categories": ["Sales", "Marketing"],
                ...
              }

        Input
        ─────
        *text* must be either:
          • a JSON array    – each element becomes one parent
          • a JSON object   – treated as a single entry
          • a JSON string   – serialized version of either of the above

        If *text* cannot be parsed as JSON, falls back to fixed-size chunking.
        """
        effective_child_tokens = child_tokens if child_tokens is not None else 100

        # ── Parse ─────────────────────────────────────────────────────────
        try:
            data = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            logger.warning(
                "JSON strategy: input is not valid JSON — "
                "falling back to fixed_size chunking"
            )
            return self._chunk_fixed_size(text, base_metadata)

        # ── Normalise to a flat list of entries ───────────────────────────
        if isinstance(data, list):
            entries = data
        elif isinstance(data, dict):
            # Check for common envelope keys
            entries = None
            for key in ("data", "results", "items", "records",
                        "entries", "content", "payload", "rows", "list"):
                val = data.get(key)
                if isinstance(val, list) and val:
                    logger.info(f"JSON strategy: envelope key '{key}' detected")
                    entries = val
                    break
            if entries is None:
                entries = [data]
        else:
            entries = [data]

        if not entries:
            logger.warning("JSON strategy: no entries found in input")
            return []

        all_chunks: List[Dict[str, Any]] = []
        parent_index = 1

        for entry in entries:
            # Derive a human display name for breadcrumbs
            display_name = self._json_display_name(entry, parent_index)
            entry_identity = self._json_entry_identity(entry)

            # ── Parent: full serialized entry ──────────────────────────────
            parent_text = self._json_build_parent_text(
                entry,
                display_name=display_name,
                entry_identity=entry_identity,
            )
            if not parent_text.strip():
                continue

            parent_chunk = self._create_chunk(
                parent_text,
                parent_index,
                base_metadata,
                chunk_type="parent",
            )
            parent_chunk["metadata"]["json_entry_name"] = display_name
            if entry_identity:
                parent_chunk["metadata"]["json_entry_id"] = entry_identity

            # ── Children: recursive structural decomposition ───────────────
            children: List[Dict[str, Any]] = []
            child_counter = [parent_index * 10_000]   # mutable via list

            self._json_split_node(
                node           = entry,
                path           = [],
                display_name   = display_name,
                entry_identity = entry_identity,
                base_metadata  = base_metadata,
                children       = children,
                counter        = child_counter,
                child_tokens   = effective_child_tokens,
                context_prefix = context_prefix,
            )

            # Wire parent ↔ children
            parent_chunk["metadata"]["child_chunk_ids"] = [
                c["chunk_id"] for c in children
            ]
            for child in children:
                child["metadata"]["parent_chunk_id"] = parent_chunk["chunk_id"]

            all_chunks.append(parent_chunk)
            all_chunks.extend(children)

            logger.debug(
                f"  JSON entry {parent_index} '{display_name}': "
                f"1 parent + {len(children)} children"
            )
            parent_index += 1

        n_parents  = sum(1 for c in all_chunks if c["metadata"].get("chunk_type") == "parent")
        n_children = len(all_chunks) - n_parents
        logger.info(
            f"JSON structural chunking → {n_parents} parent(s) + {n_children} child(ren)  "
            f"(child_tokens={effective_child_tokens}, "
            f"context_prefix={context_prefix})"
        )
        return all_chunks

    def _json_split_node(
        self,
        node:           Any,
        path:           List[str],
        display_name:   str,
        entry_identity: Optional[str],
        base_metadata:  Dict,
        children:       List[Dict[str, Any]],
        counter:        List[int],
        child_tokens:   int,
        context_prefix: bool,
    ) -> None:
        """
        Recursively walk a JSON node and emit leaf children.

        Groups consecutive small siblings into one child until adding the
        next sibling would exceed child_tokens or the sibling is itself
        too large (in which case we recurse deeper into it).
        """
        # Collect items: dict → key/value pairs, list → index/element pairs
        if isinstance(node, dict):
            items = self._json_ordered_items(node)
        elif isinstance(node, list):
            items = [(str(i), v) for i, v in enumerate(node)]
        else:
            # Scalar at unexpected depth — emit directly
            self._json_emit_child(
                node, path, display_name, entry_identity, base_metadata,
                children, counter, child_tokens, context_prefix,
            )
            return

        if not items:
            return

        accumulator: List[Tuple[str, Any]] = []
        acc_tokens = 0

        def flush() -> None:
            nonlocal acc_tokens
            if not accumulator:
                return
            field_names = [label for label, _ in accumulator]
            if len(accumulator) == 1:
                label, val = accumulator[0]
                if isinstance(node, dict):
                    emit_obj  = {label: val}
                    emit_path = path + [label]
                else:
                    emit_obj  = val
                    emit_path = path + [label]
            else:
                if isinstance(node, dict):
                    emit_obj  = {lbl: v for lbl, v in accumulator}
                    emit_path = path
                else:
                    emit_obj  = [v for _, v in accumulator]
                    emit_path = path + [
                        f"{accumulator[0][0]}..{accumulator[-1][0]}"
                    ]
            self._json_emit_child(
                emit_obj, emit_path, display_name, entry_identity, base_metadata,
                children, counter, child_tokens, context_prefix, field_names,
            )
            accumulator.clear()
            acc_tokens = 0

        for label, value in items:
            serialized   = json.dumps(value, ensure_ascii=False)
            item_tokens  = self.token_counter.count_tokens(serialized)

            if item_tokens > child_tokens:
                # Too large for one child — flush current group, recurse deeper
                flush()
                if isinstance(value, (dict, list)):
                    self._json_split_node(
                        node           = value,
                        path           = path + [str(label)],
                        display_name   = display_name,
                        entry_identity = entry_identity,
                        base_metadata  = base_metadata,
                        children       = children,
                        counter        = counter,
                        child_tokens   = child_tokens,
                        context_prefix = context_prefix,
                    )
                else:
                    self._json_emit_large_scalar(
                        value=value,
                        path=path + [str(label)],
                        display_name=display_name,
                        entry_identity=entry_identity,
                        base_metadata=base_metadata,
                        children=children,
                        counter=counter,
                        child_tokens=child_tokens,
                        context_prefix=context_prefix,
                    )
            elif acc_tokens + item_tokens > child_tokens and accumulator:
                # Adding this would overflow — flush first, then start fresh
                flush()
                accumulator.append((str(label), value))
                acc_tokens = item_tokens
            else:
                accumulator.append((str(label), value))
                acc_tokens += item_tokens

        flush()

    def _json_emit_child(
        self,
        obj:            Any,
        path:           List[str],
        display_name:   str,
        entry_identity: Optional[str],
        base_metadata:  Dict,
        children:       List[Dict[str, Any]],
        counter:        List[int],
        child_tokens:   int,
        context_prefix: bool,
        field_names:    Optional[List[str]] = None,
    ) -> None:
        """Serialize obj, optionally prepend breadcrumb, emit as a child chunk."""
        raw_body = json.dumps(obj, indent=2, ensure_ascii=False)
        if not raw_body or not raw_body.strip():
            return
        breadcrumb = self._json_breadcrumb(display_name, path)
        body = self._json_build_child_text(
            obj=obj,
            raw_body=raw_body,
            display_name=display_name,
            entry_identity=entry_identity,
            breadcrumb=breadcrumb,
            context_prefix=context_prefix,
            field_names=field_names,
        )

        counter[0] += 1
        child = self._create_chunk(
            body,
            counter[0],
            base_metadata,
            chunk_type="child",
        )
        child["metadata"]["path"] = path
        child["metadata"]["breadcrumb"] = breadcrumb
        child["metadata"]["json_entry_name"] = display_name
        child["metadata"]["json_path_depth"] = len(path)
        if field_names:
            child["metadata"]["field_names"] = field_names
        if entry_identity:
            child["metadata"]["json_entry_id"] = entry_identity
        children.append(child)

    @staticmethod
    def _json_breadcrumb(display_name: str, path: List[str]) -> str:
        """Build a human-readable path like 'Action Network > properties > codex'."""
        parts = [display_name] if display_name else []
        parts.extend(str(p) for p in path)
        return " > ".join(parts)

    @staticmethod
    def _json_entry_identity(entry: Any) -> Optional[str]:
        """Extract a stable identity field so every child carries record context."""
        if not isinstance(entry, dict):
            return None
        for key in ("book_id", "id", "uuid", "key", "isbn", "_source_url"):
            value = entry.get(key)
            if value is not None:
                return str(value)
        return None

    @staticmethod
    def _json_display_name(entry: Any, fallback_index: int) -> str:
        """Derive a display name from a JSON entry for use in breadcrumbs."""
        if isinstance(entry, dict):
            return (
                (entry.get("properties") or {}).get("displayName")
                or entry.get("name")
                or entry.get("title")
                or entry.get("label")
                or entry.get("key")
                or (str(entry.get("id")) if entry.get("id") is not None else None)
                or (str(entry.get("uuid")) if entry.get("uuid") is not None else None)
                or f"Entry {fallback_index}"
            )
        return f"Entry {fallback_index}"

    def _json_build_parent_text(
        self,
        entry: Any,
        display_name: str,
        entry_identity: Optional[str],
    ) -> str:
        raw_body = json.dumps(entry, indent=2, ensure_ascii=False)
        semantic_lines = self._json_semantic_lines(entry, max_lines=14, max_depth=3)

        sections: List[str] = []
        if display_name:
            sections.append(f"Record: {display_name}")
        if entry_identity and entry_identity != display_name:
            sections.append(f"Record ID: {entry_identity}")
        if semantic_lines:
            sections.append("Key facts:\n" + "\n".join(semantic_lines))
        sections.append("JSON:\n" + raw_body)
        return "\n\n".join(section for section in sections if section)

    def _json_build_child_text(
        self,
        obj: Any,
        raw_body: str,
        display_name: str,
        entry_identity: Optional[str],
        breadcrumb: str,
        context_prefix: bool,
        field_names: Optional[List[str]] = None,
    ) -> str:
        semantic_lines = self._json_semantic_lines(obj, max_lines=10, max_depth=3)
        sections: List[str] = []

        if display_name:
            sections.append(f"Record: {display_name}")
        if entry_identity and entry_identity != display_name:
            sections.append(f"Record ID: {entry_identity}")
        if context_prefix and breadcrumb and breadcrumb != display_name:
            sections.append(f"Path: {breadcrumb}")
        if field_names:
            sections.append(
                "Fields: " + ", ".join(field_names[:8]) +
                (f" (+{len(field_names) - 8} more)" if len(field_names) > 8 else "")
            )
        if semantic_lines:
            sections.append("Facts:\n" + "\n".join(semantic_lines))
        sections.append("JSON:\n" + raw_body)
        return "\n\n".join(section for section in sections if section)

    def _json_emit_large_scalar(
        self,
        value: Any,
        path: List[str],
        display_name: str,
        entry_identity: Optional[str],
        base_metadata: Dict,
        children: List[Dict[str, Any]],
        counter: List[int],
        child_tokens: int,
        context_prefix: bool,
    ) -> None:
        if isinstance(value, str):
            parts = self._split_text_for_json_child(value, max(20, child_tokens - 30))
            field_name = path[-1] if path else "value"
            for idx, part in enumerate(parts, start=1):
                segment_label = f"{field_name}_part_{idx}" if len(parts) > 1 else field_name
                self._json_emit_child(
                    obj={segment_label: part},
                    path=path,
                    display_name=display_name,
                    entry_identity=entry_identity,
                    base_metadata=base_metadata,
                    children=children,
                    counter=counter,
                    child_tokens=child_tokens,
                    context_prefix=context_prefix,
                    field_names=[field_name],
                )
            return

        self._json_emit_child(
            obj=value,
            path=path,
            display_name=display_name,
            entry_identity=entry_identity,
            base_metadata=base_metadata,
            children=children,
            counter=counter,
            child_tokens=child_tokens,
            context_prefix=context_prefix,
            field_names=[path[-1]] if path else None,
        )

    def _json_ordered_items(self, obj: Dict[str, Any]) -> List[Tuple[str, Any]]:
        priority = {
            "book_id": 0,
            "id": 0,
            "uuid": 0,
            "key": 0,
            "isbn": 0,
            "name": 1,
            "title": 1,
            "label": 1,
            "displayName": 1,
            "type": 2,
            "genre": 2,
            "category": 2,
            "categories": 2,
            "rating": 2,
            "score": 2,
            "author": 3,
            "publisher": 3,
            "tags": 3,
            "description": 4,
            "summary": 4,
            "content": 5,
            "text": 5,
            "body": 5,
            "_source_url": 6,
        }
        order_index = {key: idx for idx, key in enumerate(obj.keys())}
        return sorted(
            obj.items(),
            key=lambda item: (priority.get(item[0], 100), order_index[item[0]]),
        )

    def _json_semantic_lines(
        self,
        obj: Any,
        max_lines: int = 10,
        max_depth: int = 3,
    ) -> List[str]:
        lines: List[str] = []

        def visit(value: Any, prefix: str = "", depth: int = 0) -> None:
            if len(lines) >= max_lines:
                return

            if isinstance(value, dict):
                if depth >= max_depth:
                    compact = self._json_compact_value(value)
                    if compact:
                        lines.append(f"{prefix}: {compact}" if prefix else compact)
                    return

                for key, child in self._json_ordered_items(value):
                    next_prefix = f"{prefix}.{key}" if prefix else str(key)
                    visit(child, next_prefix, depth + 1)
                    if len(lines) >= max_lines:
                        return
                return

            if isinstance(value, list):
                if not value:
                    if prefix:
                        lines.append(f"{prefix}: []")
                    return

                if self._json_list_is_scalar(value):
                    scalar_values = [
                        rendered
                        for rendered in (self._json_scalar_to_text(item) for item in value[:6])
                        if rendered is not None
                    ]
                    joined = ", ".join(scalar_values)
                    if len(value) > 6:
                        joined = f"{joined}, ... ({len(value)} items)" if joined else f"{len(value)} items"
                    if prefix:
                        lines.append(f"{prefix}: {joined}")
                    elif joined:
                        lines.append(joined)
                    return

                if depth >= max_depth:
                    if prefix:
                        lines.append(f"{prefix}: {len(value)} items")
                    return

                for idx, item in enumerate(value[:3]):
                    next_prefix = f"{prefix}[{idx}]" if prefix else f"[{idx}]"
                    visit(item, next_prefix, depth + 1)
                    if len(lines) >= max_lines:
                        return
                if len(value) > 3 and prefix and len(lines) < max_lines:
                    lines.append(f"{prefix}: {len(value)} items")
                return

            rendered = self._json_scalar_to_text(value)
            if rendered is not None:
                lines.append(f"{prefix}: {rendered}" if prefix else rendered)

        visit(obj)
        return lines[:max_lines]

    @staticmethod
    def _json_list_is_scalar(value: List[Any]) -> bool:
        return all(not isinstance(item, (dict, list)) for item in value)

    def _json_compact_value(self, value: Any) -> str:
        if isinstance(value, dict):
            keys = list(value.keys())
            if not keys:
                return "{}"
            preview = ", ".join(keys[:4])
            if len(keys) > 4:
                preview += f", +{len(keys) - 4} more"
            return f"object with keys [{preview}]"
        if isinstance(value, list):
            return f"list with {len(value)} items"
        rendered = self._json_scalar_to_text(value)
        return rendered or ""

    @staticmethod
    def _json_scalar_to_text(value: Any, max_length: int = 180) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)):
            return str(value)
        text = re.sub(r"\s+", " ", str(value)).strip()
        if not text:
            return None
        if len(text) > max_length:
            return text[: max_length - 3].rstrip() + "..."
        return text

    def _split_text_for_json_child(self, text: str, max_tokens: int) -> List[str]:
        normalized = re.sub(r"\s+", " ", (text or "")).strip()
        if not normalized:
            return []
        if self.token_counter.count_tokens(normalized) <= max_tokens:
            return [normalized]

        parts: List[str] = []
        current = ""

        def append_piece(piece: str) -> None:
            nonlocal current
            if not piece:
                return
            candidate = f"{current} {piece}".strip() if current else piece
            if self.token_counter.count_tokens(candidate) <= max_tokens:
                current = candidate
                return
            if current:
                parts.append(current)
                current = ""
            if self.token_counter.count_tokens(piece) <= max_tokens:
                current = piece
                return

            words = piece.split()
            buffer = ""
            for word in words:
                candidate_word = f"{buffer} {word}".strip() if buffer else word
                if self.token_counter.count_tokens(candidate_word) <= max_tokens:
                    buffer = candidate_word
                else:
                    if buffer:
                        parts.append(buffer)
                    buffer = word
            if buffer:
                current = buffer

        for sentence in self._split_into_sentences(normalized):
            append_piece(sentence)

        if current:
            parts.append(current)

        return parts or [normalized]

    # ──────────────────────────────────────────────────────────────────────
    # STRATEGY: No-chunking
    # ──────────────────────────────────────────────────────────────────────

    def _chunk_no_chunking(
        self, text: str, base_metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Entire document becomes one single chunk."""
        if not text or not text.strip():
            return []
        return [self._create_chunk(
            text=text,
            chunk_index=1,
            base_metadata=base_metadata,
            chunk_type="full_document",
            position=0,
            total_tokens=self.token_counter.count_tokens(text),
        )]

    # ──────────────────────────────────────────────────────────────────────
    # STRATEGY: Fixed-size  (also used by DEFAULT)
    # ──────────────────────────────────────────────────────────────────────

    def _chunk_fixed_size(
        self, text: str, base_metadata: Dict
    ) -> List[Dict[str, Any]]:

        tokens = self.token_counter.get_tokens(text)
        if not tokens:
            return self._chunk_fixed_size_chars(text, base_metadata)

        stride = max(1, self.target_chunk_tokens - self.chunk_overlap_tokens)

        raw: List[Tuple[int, List[int]]] = []
        position = 0
        while position < len(tokens):
            window = tokens[position: position + self.target_chunk_tokens]
            raw.append((position, window))
            position += stride

        # Merge tiny tail into the previous window
        if len(raw) >= 2 and len(raw[-1][1]) < self.min_chunk_tokens:
            prev_pos, _ = raw[-2]
            last_pos, last_toks = raw[-1]
            merged_toks = tokens[prev_pos: last_pos + len(last_toks)]
            raw[-2] = (prev_pos, merged_toks)
            raw.pop()

        chunks: List[Dict[str, Any]] = []
        for chunk_index, (pos, toks) in enumerate(raw, start=1):
            chunk_text = self.token_counter.decode_tokens(toks)
            if chunk_text.strip():
                chunks.append(self._create_chunk(
                    chunk_text, chunk_index, base_metadata,
                    position=pos, total_tokens=len(tokens),
                ))
        return chunks

    # ──────────────────────────────────────────────────────────────────────
    # STRATEGY: Hierarchical
    # ──────────────────────────────────────────────────────────────────────

    def _chunk_hierarchical(
        self,
        text: str,
        base_metadata: Dict,
        max_parent_tokens: Optional[int] = None,
        max_child_tokens: Optional[int] = None,
        child_overlap_tokens: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Two-level hierarchical chunking.

        Phase 1 – Create overlapping parent windows over the full token sequence.
        Phase 2 – Split each parent into fixed-size overlapping child windows.

        Returns a flat list: parent chunks first, then child chunks in order.

        """
        if not text.strip():
            return []

        # ── Defaults ─────────────────────────────────────────────────────
        if max_parent_tokens is None:
            max_parent_tokens = max(600, self.target_chunk_tokens * 4)
        if max_child_tokens is None:
            max_child_tokens = self.target_chunk_tokens
        if child_overlap_tokens is None:
            child_overlap_tokens = self.chunk_overlap_tokens

        # Safety guards
        max_parent_tokens    = max(max_parent_tokens, max_child_tokens + 1)
        child_overlap_tokens = min(child_overlap_tokens, max_child_tokens // 2)
        parent_overlap_tokens = min(
            int(max_parent_tokens * 0.20), max_parent_tokens // 2
        )

        # ── Phase 1: parent windows ───────────────────────────────────────
        all_tokens = self.token_counter.get_tokens(text)
        if not all_tokens:
            return self._chunk_fixed_size_chars(text, base_metadata)

        parent_stride = max(1, max_parent_tokens - parent_overlap_tokens)
        parent_records: List[Dict[str, Any]] = []
        p_index  = 1
        position = 0

        while position < len(all_tokens):
            end      = min(position + max_parent_tokens, len(all_tokens))
            p_tokens = all_tokens[position:end]
            p_text   = self.token_counter.decode_tokens(p_tokens)
            if p_text.strip():
                # FIX-4: use hash-based chunk_id (same as _create_chunk does)
                p_chunk = self._create_chunk(
                    p_text, p_index, base_metadata,
                    position=position, total_tokens=len(all_tokens),
                    chunk_type="parent",
                )
                parent_records.append(p_chunk)
                p_index += 1
            position += parent_stride

        # ── Phase 2: children for each parent ────────────────────────────
        all_output: List[Dict[str, Any]] = list(parent_records)   # parents first
        child_counter = 1

        for parent in parent_records:
            parent_id   = parent["chunk_id"]
            parent_text = parent["text"]

            children = self._split_into_children(
                parent_text,
                max_child_tokens,
                child_overlap_tokens,
                base_metadata,
                parent_id=parent_id,
                child_index_start=child_counter,
            )

            # Back-fill child IDs into parent metadata
            parent["metadata"]["child_chunk_ids"] = [c["chunk_id"] for c in children]

            all_output.extend(children)
            child_counter += len(children)

        n_parents  = len(parent_records)
        n_children = len(all_output) - n_parents
        logger.info(
            f"Hierarchical → {n_parents} parents, {n_children} children  "
            f"(parent={max_parent_tokens}t/{parent_overlap_tokens}t-overlap, "
            f"child={max_child_tokens}t/{child_overlap_tokens}t-overlap)"
        )
        return all_output

    def _split_into_children(
        self,
        text: str,
        max_child_tokens: int,
        child_overlap_tokens: int,
        base_metadata: Dict,
        parent_id: str,
        child_index_start: int = 1,
    ) -> List[Dict[str, Any]]:
        """
        Produce fixed-size overlapping child chunks from one parent's text.
        """
        tokens = self.token_counter.get_tokens(text)
        if not tokens:
            return []

        stride = max(1, max_child_tokens - child_overlap_tokens)

        raw: List[Tuple[int, List[int]]] = []
        position = 0
        while position < len(tokens):
            window = tokens[position: position + max_child_tokens]
            raw.append((position, window))
            position += stride

        # FIX-2: merge tiny tail into previous window
        if len(raw) >= 2 and len(raw[-1][1]) < self.min_chunk_tokens:
            prev_pos, _ = raw[-2]
            last_pos, last_toks = raw[-1]
            merged_toks = tokens[prev_pos: last_pos + len(last_toks)]
            raw[-2] = (prev_pos, merged_toks)
            raw.pop()

        children: List[Dict[str, Any]] = []
        for local_i, (pos, toks) in enumerate(raw):
            chunk_text = self.token_counter.decode_tokens(toks)
            if not chunk_text.strip():
                continue
            child = self._create_chunk(
                chunk_text,
                child_index_start + local_i,
                base_metadata,
                position=pos,
                total_tokens=len(tokens),
                chunk_type="child",
            )
            # FIX-3 & FIX-4: always stamp the parent reference
            child["metadata"]["parent_chunk_id"] = parent_id
            children.append(child)

        return children

    # ──────────────────────────────────────────────────────────────────────
    # STRATEGY: Semantic
    # ──────────────────────────────────────────────────────────────────────

    def _chunk_semantic(
        self,
        text:                 str,
        base_metadata:        Dict,
        max_buffer_size:      Optional[int]   = None,
        breakpoint_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Semantic chunking with full structure awareness."""

        # FIX-1: guard at the TOP of the method body, not inside it.
        # Previously the entire implementation below was indented one level
        # too deep (inside `if not text.strip(): return []`), making it dead
        # code — the function returned None for every non-empty document.
        if not text.strip():
            return []

        # FIX-2: original code referenced undefined locals `max_tokens` and
        # `overlap_tokens`. These don't exist as parameters; map correctly from
        # instance attributes and the actual parameters received.
        effective_max_tokens  = self.target_chunk_tokens
        effective_overlap     = self.chunk_overlap_tokens
        effective_buffer_size = max_buffer_size   # None = unlimited

        # FIX-8: always resolve to a concrete float — passing None into
        # _is_semantically_similar caused: TypeError: '>=' not supported
        # between 'float' and 'NoneType'.
        sim_threshold: float = breakpoint_threshold if breakpoint_threshold is not None else 0.25

        units = self._split_into_structured_units(text)
        if not units:
            return self._chunk_by_sentence(text, base_metadata)

        raw_chunks:    List[Dict[str, Any]] = []
        chunk_index    = 1
        cur_parts:     List[str] = []
        cur_text       = ""
        cur_tokens     = 0
        pending_prefix = ""
        overlap_tail   = ""

        def flush() -> None:
            nonlocal chunk_index, cur_parts, cur_text, cur_tokens, overlap_tail
            if not cur_parts:
                return
            chunk_text = " ".join(cur_parts)
            raw_chunks.append(self._create_chunk(
                chunk_text, chunk_index, base_metadata,
                chunk_type="semantic",
            ))
            chunk_index += 1
            if effective_overlap and effective_overlap > 0:
                overlap_tail = self._extract_overlap_tail(chunk_text, effective_overlap)
            else:
                overlap_tail = ""
            cur_parts.clear()
            cur_text = ""
            cur_tokens = 0  # FIX: reset cur_tokens in flush

        def _start_new_chunk_with_overlap() -> None:
            nonlocal cur_parts, cur_text, cur_tokens
            if overlap_tail:
                o_tokens   = self.token_counter.count_tokens(overlap_tail)
                cur_parts  = [overlap_tail]
                cur_text   = overlap_tail
                cur_tokens = o_tokens
            else:
                cur_parts  = []
                cur_text   = ""
                cur_tokens = 0

        def _buffer_is_full() -> bool:
            return (
                effective_buffer_size is not None
                and len(cur_parts) >= effective_buffer_size
            )

        for unit in units:
            unit_type   = unit["type"]
            unit_text   = unit["text"]
            unit_tokens = self.token_counter.count_tokens(unit_text)

            if unit_type in ("heading", "code_block", "list"):
                flush()
                if unit_tokens >= self.min_chunk_tokens:
                    full_text = (
                        (pending_prefix + " " + unit_text).strip()
                        if pending_prefix else unit_text
                    )
                    raw_chunks.append(self._create_chunk(
                        full_text, chunk_index, base_metadata,
                        chunk_type=unit_type,
                    ))
                    chunk_index   += 1
                    pending_prefix = ""
                    overlap_tail   = (
                        self._extract_overlap_tail(full_text, effective_overlap)
                        if effective_overlap else ""
                    )
                else:
                    pending_prefix = (pending_prefix + " " + unit_text).strip()
                continue

            if pending_prefix and not cur_parts:
                _start_new_chunk_with_overlap()
                p_text     = (pending_prefix + " " + cur_text).strip() if cur_text else pending_prefix
                p_tokens   = self.token_counter.count_tokens(p_text)
                cur_parts  = [p_text]
                cur_text   = p_text
                cur_tokens = p_tokens
                pending_prefix = ""

            if not cur_parts and overlap_tail:
                _start_new_chunk_with_overlap()

            fits    = cur_tokens + unit_tokens <= effective_max_tokens
            similar = (not cur_parts) or self._is_semantically_similar(
                cur_text, unit_text, threshold=sim_threshold
            )

            if fits and similar and not _buffer_is_full():
                cur_parts.append(unit_text)
                cur_text   += " " + unit_text
                cur_tokens += unit_tokens
            elif fits and not similar:
                flush()
                _start_new_chunk_with_overlap()
                cur_parts.append(unit_text)
                cur_text   = (cur_text + " " + unit_text).strip() if cur_text else unit_text
                cur_tokens += unit_tokens
            else:
                flush()
                _start_new_chunk_with_overlap()
                cur_parts.append(unit_text)
                cur_text   = (cur_text + " " + unit_text).strip() if cur_text else unit_text
                cur_tokens += unit_tokens

        if pending_prefix:
            cur_parts.append(pending_prefix)
            cur_tokens += self.token_counter.count_tokens(pending_prefix)
        flush()

        merged = self._merge_small_chunks(raw_chunks, base_metadata, effective_max_tokens)

        logger.info(
            f"Semantic chunking → {len(merged)} chunk(s) "
            f"(max_tokens={effective_max_tokens}, overlap={effective_overlap}, "
            f"buffer_size={effective_buffer_size}, sim_threshold={sim_threshold})"
        )
        return merged

    # ──────────────────────────────────────────────────────────────────────
    # FIX-3: _extract_overlap_tail promoted to a proper class method.
    # Was previously nested inside _chunk_semantic (unreachable).
    # ──────────────────────────────────────────────────────────────────────

    def _extract_overlap_tail(self, text: str, overlap_tokens: int) -> str:
        """
        Return the trailing `overlap_tokens` worth of text.
        Splits on whitespace so we never cut mid-word.
        """
        if not text or not overlap_tokens:
            return ""
        words      = text.split()
        tail_words: List[str] = []
        tail_tokens = 0
        for word in reversed(words):
            word_tokens = self.token_counter.count_tokens(word)
            if tail_tokens + word_tokens > overlap_tokens:
                break
            tail_words.insert(0, word)
            tail_tokens += word_tokens
        return " ".join(tail_words)

    # ──────────────────────────────────────────────────────────────────────
    # FIX-4 + FIX-5: _merge_small_chunks promoted to a proper method with
    # the `max_tokens` parameter added.  `token_ceiling` was undefined in the
    # original — it is now consistently `max_tokens`.
    # ──────────────────────────────────────────────────────────────────────

    def _merge_small_chunks(
        self,
        chunks:       List[Dict[str, Any]],
        base_metadata: Dict,
        max_tokens:   Optional[int] = None,   # FIX-5: was missing, token_ceiling undefined
    ) -> List[Dict[str, Any]]:
        """Two-pass chunk merge to eliminate chunks < min_chunk_tokens."""

        # FIX-4: token_ceiling was never defined; use max_tokens (with fallback)
        token_ceiling = max_tokens if max_tokens is not None else self.max_chunk_tokens

        def _forward_pass(source: List[Dict]) -> List[Dict]:
            result: List[Dict] = []
            i = 0
            while i < len(source):
                cur = source[i]
                if (
                    cur["token_count"] < self.min_chunk_tokens
                    and i + 1 < len(source)
                ):
                    nxt = source[i + 1]
                    combined_text   = cur["text"] + " " + nxt["text"]
                    combined_tokens = cur["token_count"] + nxt["token_count"]
                    if combined_tokens <= token_ceiling:
                        result.append(self._create_chunk(
                            combined_text,
                            len(result) + 1,
                            base_metadata,
                            chunk_type="semantic_merged",
                        ))
                        i += 2
                        continue
                result.append(cur)
                i += 1
            return result

        after_forward = _forward_pass(chunks)

        # Backward pass for trailing tiny chunk
        if (
            len(after_forward) >= 2
            and after_forward[-1]["token_count"] < self.min_chunk_tokens
        ):
            last = after_forward.pop()
            prev = after_forward[-1]
            combined_text   = prev["text"] + " " + last["text"]
            combined_tokens = prev["token_count"] + last["token_count"]
            if combined_tokens <= token_ceiling:
                after_forward[-1] = self._create_chunk(
                    combined_text,
                    prev["metadata"]["chunk_index"],
                    base_metadata,
                    chunk_type="semantic_merged",
                )
            else:
                after_forward.append(last)

        # Re-index sequentially after merges
        for new_idx, chunk in enumerate(after_forward, start=1):
            chunk["metadata"]["chunk_index"] = new_idx

        return after_forward

    # ──────────────────────────────────────────────────────────────────────
    # Helper: sentence-based grouping (semantic fallback)
    # ──────────────────────────────────────────────────────────────────────

    def _chunk_by_sentence(
        self, text: str, base_metadata: Dict
    ) -> List[Dict[str, Any]]:
        """Groups sentences until target_chunk_tokens is reached."""
        sentences = self._split_into_sentences(text)
        if not sentences:
            return self._chunk_fixed_size(text, base_metadata)

        chunks:     List[Dict[str, Any]] = []
        chunk_index = 1
        cur_parts:  List[str] = []
        cur_tokens  = 0

        for sentence in sentences:
            s_tokens = self.token_counter.count_tokens(sentence)

            if s_tokens > self.max_chunk_tokens:
                if cur_parts:
                    chunks.append(self._create_chunk(
                        " ".join(cur_parts), chunk_index, base_metadata,
                    ))
                    chunk_index += 1
                    cur_parts, cur_tokens = [], 0
                sub = self._chunk_fixed_size(sentence, base_metadata)
                for sc in sub:
                    sc["metadata"]["chunk_index"] = chunk_index
                    chunks.append(sc)
                    chunk_index += 1
                continue

            if cur_tokens + s_tokens > self.target_chunk_tokens and cur_parts:
                chunks.append(self._create_chunk(
                    " ".join(cur_parts), chunk_index, base_metadata,
                ))
                chunk_index += 1
                cur_parts, cur_tokens = [], 0

            cur_parts.append(sentence)
            cur_tokens += s_tokens

        if cur_parts:
            chunks.append(self._create_chunk(
                " ".join(cur_parts), chunk_index, base_metadata,
            ))

        # FIX-6: pass max_tokens so token_ceiling is defined inside the method
        return self._merge_small_chunks(chunks, base_metadata, self.target_chunk_tokens)

    # ──────────────────────────────────────────────────────────────────────
    # Structural unit parser
    # ──────────────────────────────────────────────────────────────────────

    def _split_into_structured_units(self, text: str) -> List[Dict[str, str]]:
        """Parse markdown into typed units: heading | code_block | list | paragraph."""
        units:      List[Dict[str, str]] = []
        heading_re  = re.compile(r"^#{1,6}\s+.+$")
        list_re     = re.compile(r"^\s*([-*+]|\d+\.)\s+.+")
        lines       = text.split("\n")
        cur_text    = ""
        cur_type    = "paragraph"
        i           = 0

        while i < len(lines):
            line = lines[i].rstrip()

            if heading_re.match(line):
                if cur_text.strip():
                    units.append({"type": cur_type, "text": cur_text.strip()})
                units.append({"type": "heading", "text": line.strip()})
                cur_text, cur_type = "", "paragraph"
                i += 1
                continue

            if line.strip().startswith("```"):
                if cur_text.strip():
                    units.append({"type": cur_type, "text": cur_text.strip()})
                code_lines = [lines[i]]
                i += 1
                while i < len(lines) and not lines[i].strip().startswith("```"):
                    code_lines.append(lines[i])
                    i += 1
                if i < len(lines):
                    code_lines.append(lines[i])
                    i += 1
                units.append({"type": "code_block", "text": "\n".join(code_lines)})
                cur_text = ""
                continue

            if list_re.match(line):
                if cur_text.strip() and cur_type != "list":
                    units.append({"type": cur_type, "text": cur_text.strip()})
                    cur_text = ""
                cur_type  = "list"
                cur_text += line + "\n"
                i += 1
                continue

            if not line.strip():
                if cur_text.strip():
                    units.append({"type": cur_type, "text": cur_text.strip()})
                    cur_text, cur_type = "", "paragraph"
                i += 1
                continue

            if cur_type != "paragraph":
                if cur_text.strip():
                    units.append({"type": cur_type, "text": cur_text.strip()})
                cur_text, cur_type = "", "paragraph"
            cur_text += line + " "
            i += 1

        if cur_text.strip():
            units.append({"type": cur_type, "text": cur_text.strip()})

        return units

    # ──────────────────────────────────────────────────────────────────────
    # Utility helpers
    # ──────────────────────────────────────────────────────────────────────

    def _split_into_sentences(self, text: str) -> List[str]:
        if NLTK_AVAILABLE:
            try:
                return [s.strip() for s in nltk.sent_tokenize(text) if s.strip()]
            except Exception as exc:
                logger.warning(f"NLTK sentence tokenisation failed: {exc}")
        pattern = r"(?<=[.!?])\s+(?=[A-Z])"
        return [s.strip() for s in re.split(pattern, text) if s.strip()]

    # FIX-7: removed duplicate broken definition that was nested inside
    # _chunk_semantic. This single class-level definition is the correct one.
    def _is_semantically_similar(
        self, text1: str, text2: str, threshold: float = 0.25
    ) -> bool:
        """Jaccard similarity over 4+-character words."""
        w1 = set(re.findall(r"\b\w{4,}\b", text1.lower()))
        w2 = set(re.findall(r"\b\w{4,}\b", text2.lower()))
        if not w1 or not w2:
            return False
        union = len(w1 | w2)
        return (len(w1 & w2) / union) >= threshold if union else False

    def _create_chunk(
        self,
        text:         str,
        chunk_index:  int,
        base_metadata: Dict,
        position:     Optional[int] = None,
        total_tokens: Optional[int] = None,
        chunk_type:   str = "text",
    ) -> Dict[str, Any]:
        page_number = self._extract_page_number(text)
        token_count = self.token_counter.count_tokens(text)
        chunk: Dict[str, Any] = {
            "chunk_id":    self._generate_chunk_id(
                text,
                chunk_index,
                chunk_type,
                base_metadata,
            ),
            "text":        text,
            "char_count":  len(text),
            "token_count": token_count,
            "metadata": {
                "chunk_index": chunk_index,
                "chunk_type":  chunk_type,
                "page_number": page_number,
                **base_metadata,
            },
        }
        if position is not None and total_tokens is not None:
            chunk["metadata"]["token_position"]      = position
            chunk["metadata"]["total_tokens_in_doc"] = total_tokens
        return chunk

    def _extract_page_number(self, text: str) -> int:
        matches = re.findall(r"##\s*Page\s+(\d+)", text, re.IGNORECASE)
        return int(matches[0]) if matches else 1

    def _generate_chunk_id(
        self,
        text: str,
        chunk_index: int,
        chunk_type: str = "text",
        base_metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        base_metadata = base_metadata or {}
        file_scope = (
            base_metadata.get("file_id")
            or base_metadata.get("storage_path")
            or base_metadata.get("filename")
            or ""
        )
        hash_input = (
            f"{file_scope}|{text[:100].strip()}|{len(text)}|"
            f"{chunk_index}|{chunk_type}"
        )
        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]

    def _chunk_fixed_size_chars(
        self, text: str, base_metadata: Dict
    ) -> List[Dict[str, Any]]:
        """Character-based fallback when tiktoken is unavailable."""
        target_chars  = self.target_chunk_tokens  * 4
        overlap_chars = self.chunk_overlap_tokens * 4
        stride_chars  = max(1, target_chars - overlap_chars)
        min_chars     = self.min_chunk_tokens * 4

        raw: List[Tuple[int, str]] = []
        pos = 0
        while pos < len(text):
            raw.append((pos, text[pos: pos + target_chars]))
            pos += stride_chars

        if len(raw) >= 2 and len(raw[-1][1]) < min_chars:
            prev_pos, _ = raw[-2]
            last_pos, last_text = raw[-1]
            raw[-2] = (prev_pos, text[prev_pos: last_pos + len(last_text)])
            raw.pop()

        chunks: List[Dict[str, Any]] = []
        for chunk_index, (_, chunk_text) in enumerate(raw, start=1):
            if chunk_text.strip():
                chunks.append(self._create_chunk(chunk_text, chunk_index, base_metadata))
        return chunks

    # ──────────────────────────────────────────────────────────────────────
    # Stats
    # ──────────────────────────────────────────────────────────────────────

    def get_chunk_stats(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not chunks:
            return {
                "total_chunks": 0, "avg_tokens": 0,
                "min_tokens": 0,   "max_tokens": 0, "total_tokens": 0,
            }
        tc = [c["token_count"] for c in chunks]
        return {
            "total_chunks":   len(chunks),
            "avg_tokens":     sum(tc) / len(tc),
            "min_tokens":     min(tc),
            "max_tokens":     max(tc),
            "total_tokens":   sum(tc),
            "target_tokens":  self.target_chunk_tokens,
            "overlap_tokens": self.chunk_overlap_tokens,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Backward-compatibility alias
# ─────────────────────────────────────────────────────────────────────────────
SemanticChunkingService = EnhancedChunkingService
