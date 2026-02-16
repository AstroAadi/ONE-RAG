"""
Enhanced Chunking Service with 4 Strategies
=============================================

Strategies:
1. Default      – fixed-size with hardcoded 300-token / 30-overlap settings
2. Fixed-size   – user-specified token size + overlap
3. Semantic     – similarity-guided grouping with structure awareness
4. Hierarchical – overlapping parent windows → fixed-size child windows
5. No-chunking  – whole document as one chunk

Bug-fixes applied in this version
──────────────────────────────────
FIX-1  (Fixed-size)    Last chunk tail can be < min_chunk_tokens.
                        → When the final window is shorter than min_chunk_tokens
                          it is merged into the previous window instead of being
                          emitted as a tiny stub.

FIX-2  (Hierarchical)  _recursive_split_children silently dropped fragments
                        smaller than min_chunk_tokens, causing data loss.
                        → Replaced the recursive text-split approach with a
                          straightforward token-window approach (_split_into_children)
                          that applies the same tail-merge guard as FIX-1.
                          No text is ever silently discarded.

FIX-3  (Hierarchical)  depth-exceeded fallback called _chunk_fixed_size() which
                        produced chunks with no parent_chunk_id.
                        → The new _split_into_children always sets parent_chunk_id
                          on every child chunk it produces.

FIX-4  (Hierarchical)  Parent chunk_id was a bare "P001" label.  Parent chunks
                        were never saved to the DB, making every
                        child.metadata["parent_chunk_id"] reference an orphaned ID.
                        → Parent chunk_id now uses the same SHA-256 hash as leaf
                          chunks (stable & reproducible).
                        → Parent chunks ARE now included in the return list
                          (chunk_type="parent") so callers can persist them.
                        → Each parent carries metadata["child_chunk_ids"].
                        → Each child carries metadata["parent_chunk_id"].

FIX-5  (Semantic)      Merge pass was forward-only: a trailing tiny chunk with
                        no forward neighbour stayed small.
                        → Added a backward pass after the forward pass so the
                          last chunk is merged into the previous one when it fits.

FIX-6  (Semantic)      Small structural units (headings, short code blocks,
                        short lists < min_chunk_tokens) were silently discarded.
                        → They are now carried as a "pending_prefix" and
                          prepended to the next content unit, so no text is lost.

FIX-7  (All)           chunk_id could collide when two chunks shared the same
                        first 100 chars at the same local chunk_index.
                        → Hash now also incorporates full text length and
                          chunk_type, making collisions practically impossible.
"""

from __future__ import annotations

import hashlib
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

logger = logging.getLogger(__name__)
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
    and no_chunking strategies with all known logical bugs resolved.
    """

    def __init__(
        self,
        target_chunk_tokens: int = 300,
        chunk_overlap_tokens: int = 20,
        tokenizer_model: str = "cl100k_base",
        min_chunk_tokens: int = 100,
        max_chunk_tokens: int = 1024,
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
        markdown_text: str,
        structured_data: Optional[Dict] = None,   # reserved / unused
        base_metadata: Optional[Dict] = None,
        strategy: ChunkingStrategy = ChunkingStrategy.FIXED_SIZE,
        # Semantic params
        max_buffer_size: Optional[int] = None,
        breakpoint_threshold: Optional[float] = None,
        # Hierarchical params
        parent_chunk_size: Optional[int] = None,
        child_chunk_size: Optional[int] = None,
        child_chunk_overlap: Optional[int] = None,
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
        """
        if not markdown_text or not markdown_text.strip():
            logger.warning("Empty document provided")
            return []

        if base_metadata is None:
            base_metadata = {}

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

        else:
            logger.warning(f"Unknown strategy '{strategy}' – falling back to fixed_size")
            chunks = self._chunk_fixed_size(markdown_text, base_metadata)

        if not chunks:
            logger.warning("No chunks created")
            return []

        logger.info(f"✅ Created {len(chunks)} chunk(s) using '{strategy.value}'")
        return chunks

    # ──────────────────────────────────────────────────────────────────────
    # STRATEGY: No-chunking
    # ──────────────────────────────────────────────────────────────────────

    def _chunk_no_chunking(
        self, text: str, base_metadata: Dict
    ) -> List[Dict[str, Any]]:
        """Whole document as a single chunk."""
        return [self._create_chunk(text, 1, base_metadata, chunk_type="full_document")]

    # ──────────────────────────────────────────────────────────────────────
    # STRATEGY: Fixed-size  (also used by DEFAULT)
    # ──────────────────────────────────────────────────────────────────────

    def _chunk_fixed_size(
        self, text: str, base_metadata: Dict
    ) -> List[Dict[str, Any]]:
        """
        Token-based sliding-window chunking with overlap.

        FIX-1: Tail chunk smaller than min_chunk_tokens is merged into
               the previous chunk so no stub is ever emitted.
        """
        tokens = self.token_counter.get_tokens(text)
        if not tokens:
            return self._chunk_fixed_size_chars(text, base_metadata)

        stride = max(1, self.target_chunk_tokens - self.chunk_overlap_tokens)

        # Collect (start_position, token_slice) windows
        raw: List[Tuple[int, List[int]]] = []
        position = 0
        while position < len(tokens):
            window = tokens[position: position + self.target_chunk_tokens]
            raw.append((position, window))
            position += stride

        # FIX-1: merge tiny tail into the previous window
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

        FIX-2: No text is silently dropped.
        FIX-3: Every child chunk carries parent_chunk_id.
        FIX-4: Parent chunk_id is a SHA-256 hash (stable, reproducible).
               Parents are included in the output so callers can persist them.
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

        FIX-2: tail-merge guard prevents tiny stub children.
        FIX-3: parent_chunk_id is set on every chunk produced.
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
        text: str,
        base_metadata: Dict,
        max_buffer_size: Optional[int] = None,
        breakpoint_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Semantic chunking with full structure awareness.

        FIX-5: Two-pass (forward + backward) merge so trailing tiny chunks
               are absorbed into the previous chunk.
        FIX-6: Small structural units (heading / list / code_block that are
               < min_chunk_tokens) are carried forward as a pending prefix
               instead of being silently discarded.
        """
        if not text.strip():
            return []

        sim_threshold = (
            breakpoint_threshold if breakpoint_threshold is not None else 0.25
        )

        units = self._split_into_structured_units(text)
        if not units:
            return self._chunk_by_sentence(text, base_metadata)

        raw_chunks:    List[Dict[str, Any]] = []
        chunk_index    = 1
        cur_parts:     List[str] = []
        cur_text       = ""
        cur_tokens     = 0
        pending_prefix = ""           # FIX-6: carries small structural text forward

        def flush() -> None:
            nonlocal chunk_index, cur_parts, cur_text, cur_tokens
            if not cur_parts:
                return
            raw_chunks.append(self._create_chunk(
                " ".join(cur_parts), chunk_index, base_metadata,
                chunk_type="semantic",
            ))
            chunk_index += 1
            cur_parts, cur_text, cur_tokens = [], "", 0

        for unit in units:
            unit_type   = unit["type"]
            unit_text   = unit["text"]
            unit_tokens = self.token_counter.count_tokens(unit_text)

            # ── Structural boundary ─────────────────────────────────────
            if unit_type in ("heading", "code_block", "list"):
                flush()
                if unit_tokens >= self.min_chunk_tokens:
                    # Large enough to stand alone
                    full_text = (
                        (pending_prefix + " " + unit_text).strip()
                        if pending_prefix
                        else unit_text
                    )
                    raw_chunks.append(self._create_chunk(
                        full_text, chunk_index, base_metadata,
                        chunk_type=unit_type,
                    ))
                    chunk_index   += 1
                    pending_prefix = ""
                else:
                    # FIX-6: too small → carry forward, do NOT discard
                    pending_prefix = (pending_prefix + " " + unit_text).strip()
                continue

            # ── Normal paragraph unit ───────────────────────────────────
            # Inject any pending prefix as the first part of a new chunk
            if pending_prefix and not cur_parts:
                p_tokens  = self.token_counter.count_tokens(pending_prefix)
                cur_parts  = [pending_prefix]
                cur_text   = pending_prefix
                cur_tokens = p_tokens
                pending_prefix = ""

            fits = cur_tokens + unit_tokens <= self.target_chunk_tokens
            similar = (not cur_parts) or self._is_semantically_similar(
                cur_text, unit_text, threshold=sim_threshold
            )

            if fits and similar:
                cur_parts.append(unit_text)
                cur_text   += " " + unit_text
                cur_tokens += unit_tokens
            elif fits and not similar:
                flush()
                cur_parts  = [unit_text]
                cur_text   = unit_text
                cur_tokens = unit_tokens
            else:                        # doesn't fit
                flush()
                cur_parts  = [unit_text]
                cur_text   = unit_text
                cur_tokens = unit_tokens

        # Flush any remaining pending prefix that was never consumed
        if pending_prefix:
            cur_parts.append(pending_prefix)
            cur_tokens += self.token_counter.count_tokens(pending_prefix)
        flush()

        # ── Two-pass merge (FIX-5) ──────────────────────────────────────
        merged = self._merge_small_chunks(raw_chunks, base_metadata)

        logger.info(
            f"Semantic chunking → {len(merged)} chunk(s) "
            f"(sim_threshold={sim_threshold})"
        )
        return merged

    def _merge_small_chunks(
        self,
        chunks: List[Dict[str, Any]],
        base_metadata: Dict,
    ) -> List[Dict[str, Any]]:
        """
        Two-pass chunk merge to eliminate chunks < min_chunk_tokens.

        Pass 1 (forward)  – merge a small chunk with the next one.
        Pass 2 (backward) – FIX-5: merge a trailing small chunk with the
                            previous one (catches the case where the last
                            chunk has no forward neighbour).
        """

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
                    if combined_tokens <= self.target_chunk_tokens:
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

        # FIX-5: backward pass for trailing tiny chunk
        if (
            len(after_forward) >= 2
            and after_forward[-1]["token_count"] < self.min_chunk_tokens
        ):
            last = after_forward.pop()
            prev = after_forward[-1]
            combined_text   = prev["text"] + " " + last["text"]
            combined_tokens = prev["token_count"] + last["token_count"]
            if combined_tokens <= self.target_chunk_tokens:
                after_forward[-1] = self._create_chunk(
                    combined_text,
                    prev["metadata"]["chunk_index"],
                    base_metadata,
                    chunk_type="semantic_merged",
                )
            else:
                after_forward.append(last)   # cannot merge – keep as is

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

        chunks:    List[Dict[str, Any]] = []
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

        return self._merge_small_chunks(chunks, base_metadata)

    # ──────────────────────────────────────────────────────────────────────
    # Structural unit parser
    # ──────────────────────────────────────────────────────────────────────

    def _split_into_structured_units(self, text: str) -> List[Dict[str, str]]:
        """
        Parse markdown into typed units:
            heading | code_block | list | paragraph
        """
        units:      List[Dict[str, str]] = []
        heading_re  = re.compile(r"^#{1,6}\s+.+$")
        list_re     = re.compile(r"^\s*([-*+]|\d+\.)\s+.+")
        lines       = text.split("\n")
        cur_text    = ""
        cur_type    = "paragraph"
        i           = 0

        while i < len(lines):
            line = lines[i].rstrip()

            # Heading
            if heading_re.match(line):
                if cur_text.strip():
                    units.append({"type": cur_type, "text": cur_text.strip()})
                units.append({"type": "heading", "text": line.strip()})
                cur_text, cur_type = "", "paragraph"
                i += 1
                continue

            # Fenced code block
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

            # List item
            if list_re.match(line):
                if cur_text.strip() and cur_type != "list":
                    units.append({"type": cur_type, "text": cur_text.strip()})
                    cur_text = ""
                cur_type  = "list"
                cur_text += line + "\n"
                i += 1
                continue

            # Blank line – paragraph boundary
            if not line.strip():
                if cur_text.strip():
                    units.append({"type": cur_type, "text": cur_text.strip()})
                    cur_text, cur_type = "", "paragraph"
                i += 1
                continue

            # Normal paragraph text
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
        text: str,
        chunk_index: int,
        base_metadata: Dict,
        position: Optional[int] = None,
        total_tokens: Optional[int] = None,
        chunk_type: str = "text",
    ) -> Dict[str, Any]:
        page_number = self._extract_page_number(text)
        token_count = self.token_counter.count_tokens(text)
        chunk: Dict[str, Any] = {
            "chunk_id":    self._generate_chunk_id(text, chunk_index, chunk_type),
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
        self, text: str, chunk_index: int, chunk_type: str = "text"
    ) -> str:
        """
        FIX-7: Hash incorporates full text length + chunk_type to prevent
        collisions between chunks sharing the same first 100 characters.
        """
        hash_input = f"{text[:100].strip()}|{len(text)}|{chunk_index}|{chunk_type}"
        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]

    def _chunk_fixed_size_chars(
        self, text: str, base_metadata: Dict
    ) -> List[Dict[str, Any]]:
        """
        Character-based fallback when tiktoken is unavailable.
        Applies the same FIX-1 tail-merge guard as the token version.
        """
        target_chars  = self.target_chunk_tokens  * 4
        overlap_chars = self.chunk_overlap_tokens * 4
        stride_chars  = max(1, target_chars - overlap_chars)
        min_chars     = self.min_chunk_tokens * 4

        raw: List[Tuple[int, str]] = []
        pos = 0
        while pos < len(text):
            raw.append((pos, text[pos: pos + target_chars]))
            pos += stride_chars

        # FIX-1 (char version): merge tiny tail
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
                "min_tokens": 0, "max_tokens": 0, "total_tokens": 0,
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
# Backward-compatibility aliases
# ─────────────────────────────────────────────────────────────────────────────
SemanticChunkingService = EnhancedChunkingService