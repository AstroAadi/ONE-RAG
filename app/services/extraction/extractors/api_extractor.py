"""
api_extractor.py  —  fully self-contained, no external model files needed
────────────────────────────────────────────────────────────────────────────────
Single file contains:
  1. Pydantic v2 input models  (enums, auth sub-models, pagination, APISourceInput)
  2. APIExtractor class        (fetch → normalize → markdown, mirrors ExcelExtractor)
  3. Interactive CLI           (URL and all other params taken at runtime)

Dependencies:
  pip install pydantic requests pandas

Usage:
  python api_extractor.py --url https://api.example.com/users
  python api_extractor.py --url https://api.example.com/employees \\
      --auth basic --username admin \\
      --pagination page_number \\
      --page-param page --page-size-param per_page --page-size 100 \\
      --json-path "$.data" --output out.md
"""

from __future__ import annotations

# ── stdlib ────────────────────────────────────────────────────────────────────
import argparse
import getpass
import json
import logging
import os
import time
from datetime import datetime
from enum import Enum
from typing import Annotated, Any, Dict, List, Literal, Optional, Tuple, Union
from urllib.parse import urlparse

# ── third-party ───────────────────────────────────────────────────────────────
import pandas as pd
import requests
from requests.auth import HTTPBasicAuth

from pydantic import (
    BaseModel,
    Field,
    SecretStr,
    field_validator,
    model_validator,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — INPUT MODELS
# ══════════════════════════════════════════════════════════════════════════════

class HttpMethod(str, Enum):
    GET    = "GET"
    POST   = "POST"
    PUT    = "PUT"
    PATCH  = "PATCH"
    DELETE = "DELETE"


class AuthType(str, Enum):
    NONE    = "none"
    BASIC   = "basic"
    BEARER  = "bearer"
    API_KEY = "api_key"
    OAUTH2  = "oauth2"


class PaginationType(str, Enum):
    NONE        = "none"
    PAGE_NUMBER = "page_number"   # UI label: "Page Number"
    OFFSET      = "offset"
    CURSOR      = "cursor"


class SyncFrequency(str, Enum):
    """
    Maps to the Sync Frequency dropdown visible in the UI.
    Controls how often the background auto-sync job re-fetches the API.
    """
    HOURLY  = "hourly"    #  every 60 min
    DAILY   = "daily"     #  every 24 h  (UI default)
    WEEKLY  = "weekly"    #  every 7 days
    MONTHLY = "monthly"   #  every 30 days
    MANUAL  = "manual"    #  only on explicit trigger


# ── Auth sub-models ───────────────────────────────────────────────────────────

class NoAuth(BaseModel):
    type: Literal["none"] = "none"


class BasicAuth(BaseModel):
    """Maps to Authentication Type = Basic + Username / Password fields."""
    type:     Literal["basic"] = "basic"
    username: str       = Field(..., min_length=1, description="Username shown in the form")
    password: SecretStr = Field(..., description="Password — stored as secret, never logged")


class BearerAuth(BaseModel):
    type:  Literal["bearer"] = "bearer"
    token: SecretStr = Field(...)


class ApiKeyAuth(BaseModel):
    type:        Literal["api_key"] = "api_key"
    header_name: str       = Field(default="X-API-Key")
    api_key:     SecretStr = Field(...)


class OAuth2Auth(BaseModel):
    type:          Literal["oauth2"] = "oauth2"
    token_url:     str       = Field(...)
    client_id:     str       = Field(..., min_length=1)
    client_secret: SecretStr = Field(...)
    scope:         str       = Field(default="")


AuthConfig = Annotated[
    Union[NoAuth, BasicAuth, BearerAuth, ApiKeyAuth, OAuth2Auth],
    Field(discriminator="type"),
]


# ── Query parameter item ──────────────────────────────────────────────────────

class QueryParam(BaseModel):
    """
    One entry from the 'Add Query Parameters' section in the UI.
    These are appended to the URL as ?key=value on every request.
    Stored separately from pagination params so they are always sent.
    """
    key:   str = Field(..., min_length=1, description="Query param key")
    value: str = Field(...,               description="Query param value")


# ── Pagination config ─────────────────────────────────────────────────────────

class PaginationConfig(BaseModel):
    """
    Maps to the Pagination Type section in the UI:
      Pagination Type          → pagination_type
      Page Parameter Name      → page_param_name
      Page Size Parameter Name → page_size_param_name
      Page Size Value          → page_size_value
      Start Page               → start_page
    """
    pagination_type:      PaginationType = PaginationType.NONE
    page_param_name:      Optional[str]  = Field(default=None,
                              description="UI: Page Parameter Name  e.g. 'page'")
    page_size_param_name: Optional[str]  = Field(default=None,
                              description="UI: Page Size Parameter Name  e.g. 'per_page'")
    page_size_value:      Optional[int]  = Field(default=None, ge=1, le=10_000,
                              description="UI: Page Size Value")
    start_page:           int            = Field(default=1, ge=0,
                              description="UI: Start Page (0 or 1)")

    @model_validator(mode="after")
    def _check_params(self) -> "PaginationConfig":
        if self.pagination_type != PaginationType.NONE:
            missing = [
                name for name, val in [
                    ("page_param_name",      self.page_param_name),
                    ("page_size_param_name", self.page_size_param_name),
                    ("page_size_value",      self.page_size_value),
                ]
                if not val
            ]
            if missing:
                raise ValueError(f"Required when pagination is enabled: {missing}")
        return self


# ── Auto-sync config ──────────────────────────────────────────────────────────

class AutoSyncConfig(BaseModel):
    """
    Maps to the Auto Sync toggle + Sync Frequency dropdown in the UI.

    When enabled, a scheduler re-fetches the API and upserts new records
    into the knowledge base at the specified frequency.
    """
    enabled:   bool          = Field(default=False,
                                  description="UI: Auto Sync toggle")
    frequency: SyncFrequency = Field(default=SyncFrequency.DAILY,
                                  description="UI: Sync Frequency dropdown")

    @property
    def interval_seconds(self) -> Optional[int]:
        """Convert frequency enum to seconds for a scheduler."""
        return {
            SyncFrequency.HOURLY:  3_600,
            SyncFrequency.DAILY:   86_400,
            SyncFrequency.WEEKLY:  604_800,
            SyncFrequency.MONTHLY: 2_592_000,
            SyncFrequency.MANUAL:  None,
        }[self.frequency]


# ── Top-level source input ────────────────────────────────────────────────────

class APISourceInput(BaseModel):
    """
    Complete input model for the API source form.

    UI field           →  model field
    ─────────────────────────────────────────────────
    Source Name        →  source_name
    HTTP Method        →  http_method
    URL                →  url
    Authentication     →  auth  (discriminated union)
    Username/Password  →  auth.username / auth.password
    Pagination Type    →  pagination.pagination_type
    Page Param Name    →  pagination.page_param_name
    Page Size Param    →  pagination.page_size_param_name
    Page Size Value    →  pagination.page_size_value
    Start Page         →  pagination.start_page
    JSON Path          →  json_path
    Add Headers        →  headers       (dict key→value)
    Add Query Params   →  query_params  (list of QueryParam)
    Auto Sync toggle   →  auto_sync.enabled
    Sync Frequency     →  auto_sync.frequency
    """

    # ── Identity ──────────────────────────────────────────────────────────────
    source_name:  str = Field(..., min_length=1, max_length=255,
                              description="UI: Source Name")

    # ── Request ───────────────────────────────────────────────────────────────
    http_method:  HttpMethod = Field(default=HttpMethod.GET,
                                     description="UI: HTTP Method dropdown")
    url:          str        = Field(..., description="UI: URL field")

    # UI: Add Headers — key/value pairs appended to every request
    headers:      Dict[str, str]  = Field(default_factory=dict,
                                          description="UI: Add Headers")

    # UI: Add Query Parameters — appended as ?key=value on every request
    query_params: List[QueryParam] = Field(default_factory=list,
                                           description="UI: Add Query Parameters")

    request_body: Optional[Any]   = Field(default=None,
                                          description="Request body (POST/PUT/PATCH only)")

    # ── Auth ──────────────────────────────────────────────────────────────────
    auth: AuthConfig = Field(default_factory=NoAuth,
                             description="UI: Authentication Type + credentials")

    # ── Pagination ────────────────────────────────────────────────────────────
    pagination: PaginationConfig = Field(default_factory=PaginationConfig,
                                         description="UI: Pagination section")

    # ── Data extraction ───────────────────────────────────────────────────────
    json_path: Optional[str] = Field(default=None,
                                     description="UI: JSON Path editor  e.g. '$.data.items'")

    # ── Auto-sync ─────────────────────────────────────────────────────────────
    auto_sync: AutoSyncConfig = Field(default_factory=AutoSyncConfig,
                                      description="UI: Auto Sync toggle + Sync Frequency")

    # ── Validators ────────────────────────────────────────────────────────────
    @field_validator("url", mode="before")
    @classmethod
    def _fix_url(cls, v: str) -> str:
        """Prepend https:// if the user omitted the scheme."""
        v = str(v).strip()
        if v and not v.startswith(("http://", "https://")):
            v = "https://" + v
        return v

    @model_validator(mode="after")
    def _no_body_on_get(self) -> "APISourceInput":
        if self.request_body is not None and self.http_method == HttpMethod.GET:
            raise ValueError("request_body must not be set for GET requests")
        return self

    def merged_query_params(self) -> Dict[str, str]:
        """
        Merge the query_params list into a plain dict for use in requests.
        Called by _fetch_one() so both manual and pagination params are sent.
        """
        return {qp.key: qp.value for qp in self.query_params}


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — EXTRACTOR CLASS
# ══════════════════════════════════════════════════════════════════════════════

class APIExtractor:
    """
    Fetches data from an HTTP API and produces the same
    (markdown, success, metadata) tuple as ExcelExtractor.
    """

    def __init__(self, config: Optional[Dict] = None):
        cfg = config or {}
        self.max_rows_per_table: int   = cfg.get("max_rows_per_table", 500)
        self.max_pages:          int   = cfg.get("max_pages",          50)
        self.request_timeout:    int   = cfg.get("request_timeout",    30)
        self.max_retries:        int   = cfg.get("max_retries",        3)
        self.retry_backoff:      float = cfg.get("retry_backoff",      1.5)
        self._oauth2_cache: Dict[str, Tuple[str, float]] = {}

    # ── Public ────────────────────────────────────────────────────────────────

    def extract(self, source: APISourceInput) -> Tuple[str, bool, Dict]:
        try:
            ok, msg = self._validate(source)
            if not ok:
                return "", False, {"error": msg}

            logger.info(f"🌐  {source.source_name}  →  {source.url}")
            session, auth_obj = self._build_session(source)
            raw_pages, fetch_meta = self._fetch_all_pages(source, session, auth_obj)

            if not raw_pages:
                return "", False, {**fetch_meta, "error": "API returned no data"}

            pages_data = [
                self._process_page(raw, idx, source)
                for idx, raw in enumerate(raw_pages, 1)
            ]

            markdown   = self._build_markdown(pages_data, source)
            total_rows = sum(p["row_count"] for p in pages_data)

            metadata = {
                "extractor":         "APIExtractor",
                "source_name":       source.source_name,
                "url":               source.url,
                "http_method":       source.http_method.value,
                "auth_type":         source.auth.type,
                "pagination_type":   source.pagination.pagination_type.value,
                "total_pages":       len(pages_data),
                "total_rows":        total_rows,
                "extraction_method": "requests+pandas",
                "auto_sync_enabled":    source.auto_sync.enabled,
                "auto_sync_frequency":  source.auto_sync.frequency.value,
                "auto_sync_interval_s": source.auto_sync.interval_seconds,
                **fetch_meta,
            }
            logger.info(f"✅  {len(pages_data)} page(s), {total_rows} row(s)")
            return markdown, True, metadata

        except Exception as e:
            logger.error(f"❌  {e}", exc_info=True)
            return "", False, {"error": str(e)}

    # ── Validation ────────────────────────────────────────────────────────────

    def _validate(self, source: APISourceInput) -> Tuple[bool, str]:
        parsed = urlparse(source.url)
        if parsed.scheme not in ("http", "https"):
            return False, f"Bad URL scheme: '{parsed.scheme}'"
        return True, "ok"

    # ── Auth ──────────────────────────────────────────────────────────────────

    def _build_session(self, source: APISourceInput) -> Tuple[requests.Session, Any]:
        session = requests.Session()
        session.headers.update({"Content-Type": "application/json"})
        session.headers.update(source.headers)

        auth     = source.auth
        auth_obj = None

        if isinstance(auth, BasicAuth):
            auth_obj = HTTPBasicAuth(auth.username, auth.password.get_secret_value())

        elif isinstance(auth, BearerAuth):
            session.headers["Authorization"] = f"Bearer {auth.token.get_secret_value()}"

        elif isinstance(auth, ApiKeyAuth):
            session.headers[auth.header_name] = auth.api_key.get_secret_value()

        elif isinstance(auth, OAuth2Auth):
            session.headers["Authorization"] = f"Bearer {self._oauth2_token(auth)}"

        return session, auth_obj

    def _oauth2_token(self, auth: OAuth2Auth) -> str:
        cached = self._oauth2_cache.get(auth.token_url)
        if cached:
            token, exp = cached
            if time.time() < exp - 30:
                return token
        resp = requests.post(
            auth.token_url,
            data={
                "grant_type":    "client_credentials",
                "client_id":     auth.client_id,
                "client_secret": auth.client_secret.get_secret_value(),
                "scope":         auth.scope,
            },
            timeout=self.request_timeout,
        )
        resp.raise_for_status()
        body  = resp.json()
        token = body["access_token"]
        self._oauth2_cache[auth.token_url] = (token, time.time() + int(body.get("expires_in", 3600)))
        return token

    # ── Pagination loop ───────────────────────────────────────────────────────

    def _fetch_all_pages(
        self, source: APISourceInput,
        session: requests.Session, auth_obj: Any,
    ) -> Tuple[List[Any], Dict]:
        pag   = source.pagination
        ptype = pag.pagination_type
        pages: List[Any] = []
        meta  = {"pages_fetched": 0, "http_errors": []}

        def _add(raw, params_desc=""):
            items = self._extract_items(raw, source.json_path)
            if items:
                pages.append(raw)
                meta["pages_fetched"] += 1
                logger.info(f"  {params_desc}: {len(items)} record(s)")
            return items

        if ptype == PaginationType.NONE:
            raw, ok, err = self._fetch_one(source, session, auth_obj, {})
            if ok: _add(raw, "single request")
            else: meta["http_errors"].append(err)

        elif ptype == PaginationType.PAGE_NUMBER:
            page = pag.start_page
            for _ in range(self.max_pages):
                params = {pag.page_param_name: page, pag.page_size_param_name: pag.page_size_value}
                raw, ok, err = self._fetch_one(source, session, auth_obj, params)
                if not ok: meta["http_errors"].append(err); break
                items = _add(raw, f"page={page}")
                if not items or len(items) < (pag.page_size_value or 1): break
                page += 1

        elif ptype == PaginationType.OFFSET:
            offset, limit = pag.start_page, pag.page_size_value or 100
            for _ in range(self.max_pages):
                params = {pag.page_param_name: offset, pag.page_size_param_name: limit}
                raw, ok, err = self._fetch_one(source, session, auth_obj, params)
                if not ok: meta["http_errors"].append(err); break
                items = _add(raw, f"offset={offset}")
                if not items or len(items) < limit: break
                offset += limit

        elif ptype == PaginationType.CURSOR:
            cursor: Optional[str] = None
            for _ in range(self.max_pages):
                params: Dict[str, Any] = {pag.page_size_param_name: pag.page_size_value}
                if cursor: params[pag.page_param_name] = cursor
                raw, ok, err = self._fetch_one(source, session, auth_obj, params)
                if not ok: meta["http_errors"].append(err); break
                items = _add(raw, f"cursor={cursor!r}")
                if not items: break
                cursor = self._next_cursor(raw)
                if not cursor: break

        return pages, meta

    def _fetch_one(
        self, source: APISourceInput, session: requests.Session,
        auth_obj: Any, params: Dict[str, Any],
    ) -> Tuple[Any, bool, str]:
        url    = source.url
        method = source.http_method.value
        kwargs: Dict[str, Any] = {"timeout": self.request_timeout, "auth": auth_obj}

        # Merge UI "Add Query Parameters" with pagination params
        base_qp = source.merged_query_params()

        if method == "GET":
            kwargs["params"] = {**base_qp, **params}
        else:
            body = dict(source.request_body or {})
            body.update(params)
            kwargs["json"]   = body
            # Still append static query params on the URL for non-GET
            if base_qp:
                kwargs["params"] = base_qp

        for attempt in range(1, self.max_retries + 1):
            try:
                resp = session.request(method, url, **kwargs)
                resp.raise_for_status()
                return resp.json(), True, ""
            except requests.HTTPError as e:
                if resp.status_code in (429, 503) and attempt < self.max_retries:
                    wait = self.retry_backoff ** attempt
                    logger.warning(f"  HTTP {resp.status_code} — retry {attempt} in {wait:.1f}s")
                    time.sleep(wait)
                    continue
                err = f"HTTP {resp.status_code}: {e}"
                logger.error(f"  ❌ {err}")
                return None, False, err
            except Exception as e:
                logger.error(f"  ❌ {e}")
                return None, False, str(e)
        return None, False, "Max retries exceeded"

    # ── JSON helpers ──────────────────────────────────────────────────────────

    def _extract_items(self, raw: Any, json_path: Optional[str]) -> List[Dict]:
        if not raw:
            return []
        if not json_path:
            return raw if isinstance(raw, list) else [raw]
        node = raw
        for key in json_path.lstrip("$.").split("."):
            node = node.get(key) if isinstance(node, dict) else None
            if node is None:
                logger.warning(f"json_path '{json_path}' not found")
                return []
        return node if isinstance(node, list) else [node]

    def _next_cursor(self, raw: Any) -> Optional[str]:
        if not isinstance(raw, dict):
            return None
        for key in ("next_cursor", "cursor", "next_page_token", "nextPageToken"):
            if raw.get(key):
                return str(raw[key])
        for wrapper in ("meta", "pagination", "page_info"):
            sub = raw.get(wrapper, {})
            if isinstance(sub, dict):
                for key in ("next_cursor", "cursor", "next_page_token", "nextPageToken"):
                    if sub.get(key):
                        return str(sub[key])
        return None

    # ── Page processing ───────────────────────────────────────────────────────

    def _process_page(self, raw: Any, idx: int, source: APISourceInput) -> Dict:
        try:
            items = self._extract_items(raw, source.json_path)
            if not items:
                return self._empty_page(idx)

            df = pd.json_normalize(items)
            if len(df) > self.max_rows_per_table:
                logger.warning(f"Page {idx}: {len(df)} rows — capping at {self.max_rows_per_table}")
                df_out, truncated = df.head(self.max_rows_per_table), True
            else:
                df_out, truncated = df, False

            return {
                "page_idx":     idx,
                "dataframe":    df_out,
                "row_count":    len(df),
                "column_count": len(df.columns),
                "column_stats": self._column_stats(df),
                "truncated":    truncated,
                "has_data":     not df.empty,
            }
        except Exception as e:
            logger.error(f"Error processing page {idx}: {e}")
            return {**self._empty_page(idx), "error": str(e)}

    def _empty_page(self, idx: int) -> Dict:
        return {
            "page_idx": idx, "dataframe": pd.DataFrame(),
            "row_count": 0, "column_count": 0,
            "column_stats": {}, "truncated": False, "has_data": False,
        }

    # ── Column statistics (identical to ExcelExtractor) ───────────────────────

    def _column_stats(self, df: pd.DataFrame) -> Dict:
        stats: Dict = {}
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                s = df[col].dropna()
                if len(s) == 0:
                    continue
                stats[str(col)] = {
                    "count":  int(s.count()),
                    "sum":    float(round(s.sum(),    4)),
                    "mean":   float(round(s.mean(),   4)),
                    "median": float(round(s.median(), 4)),
                    "min":    float(s.min()),
                    "max":    float(s.max()),
                    "std":    float(round(s.std(), 4)) if len(s) > 1 else 0.0,
                }
            elif df[col].dtype == object or pd.api.types.is_string_dtype(df[col]):
                s  = df[col].dropna().astype(str)
                vc = s.value_counts()
                stats[str(col)] = {
                    "count":         int(s.count()),
                    "unique_values": int(s.nunique()),
                    "top_values":    vc.head(10).index.tolist(),
                }
        return stats

    # ── Markdown builder (mirrors ExcelExtractor._build_xlsx_markdown) ────────

    def _build_markdown(self, pages_data: List[Dict], source: APISourceInput) -> str:
        lines: List[str] = [
            f"# Document: {source.source_name}",
            f"**Format**: API ({source.http_method.value})",
            f"**URL**: {source.url}",
            f"**Auth**: {source.auth.type}",
            f"**Pagination**: {source.pagination.pagination_type.value}",
            f"**Total Pages**: {len(pages_data)}",
            f"**Extracted At**: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}",
            f"**Auto Sync**: {'enabled — ' + source.auto_sync.frequency.value if source.auto_sync.enabled else 'disabled'}",
        ]
        if source.query_params:
            qp_str = ", ".join(f"{q.key}={q.value}" for q in source.query_params)
            lines.append(f"**Query Params**: {qp_str}")
        if source.headers:
            hdr_str = ", ".join(f"{k}: {v}" for k, v in source.headers.items())
            lines.append(f"**Headers**: {hdr_str}")
        lines.append("")

        for page in pages_data:
            idx = page["page_idx"]
            lines.append(f"## Page {idx}\n")

            if "error" in page:
                lines.append(f"*(Error: {page['error']})*\n")
                continue
            if not page["has_data"]:
                lines.append("*(No data on this page)*\n")
                continue

            df = page["dataframe"]
            if page["truncated"]:
                lines.append(
                    f"**Note**: Showing first {self.max_rows_per_table} "
                    f"of {page['row_count']} rows\n"
                )

            lines.append(self._df_to_markdown(df))
            lines.append("")
            lines.append(
                f"**Page Statistics**: {page['row_count']} rows × "
                f"{page['column_count']} columns"
            )
            lines.append("")

            col_stats = page.get("column_stats", {})
            if col_stats:
                lines.append("### Column Statistics\n")
                for col, st in col_stats.items():
                    if "sum" in st:
                        lines.append(
                            f"- **{col}**: count={st['count']}, sum={st['sum']}, "
                            f"mean={st['mean']}, median={st['median']}, "
                            f"min={st['min']}, max={st['max']}, std={st['std']}"
                        )
                    else:
                        top = ", ".join(str(v) for v in st.get("top_values", [])[:5])
                        lines.append(
                            f"- **{col}**: {st['count']} values, "
                            f"{st['unique_values']} unique — top: {top}"
                        )
                lines.append("")

        return "\n".join(lines)

    def _df_to_markdown(self, df: pd.DataFrame) -> str:
        if df.empty:
            return "*(No data)*"
        df = df.fillna("")
        headers = [str(h) for h in df.columns]
        rows = (
            ["| " + " | ".join(headers) + " |"]
            + ["| " + " | ".join(["---"] * len(headers)) + " |"]
            + ["| " + " | ".join(str(v) for v in row) + " |"
               for row in df.itertuples(index=False)]
        )
        return "\n".join(rows)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — CLI
# ══════════════════════════════════════════════════════════════════════════════

def _secret(label: str, env_var: str) -> str:
    val = os.environ.get(env_var, "")
    return val if val else getpass.getpass(f"{label}: ")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="api_extractor.py",
        description="Fetch any HTTP API and extract its data to markdown",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # Core
    p.add_argument("--url", required=True,
                   help="API endpoint URL  (scheme auto-prepended if missing)")
    p.add_argument("--source-name", "--source_name", dest="source_name", default=None,
                   help="Label for this source  (default: derived from URL)")
    p.add_argument("--method", default="GET",
                   choices=["GET","POST","PUT","PATCH","DELETE"])
    p.add_argument("--json-path", "--json_path", dest="json_path", default=None,
                   help="JSONPath to record array  e.g. '$.data.items'")
    p.add_argument("--headers", default=None,
                   help='Extra headers as JSON  e.g. \'{"X-Tenant":"acme"}\'')
    p.add_argument("--body", default=None,
                   help="Request body as JSON (POST/PUT/PATCH only)")

    # Auth
    ag = p.add_argument_group("Authentication")
    ag.add_argument("--auth", default="none",
                    choices=["none","basic","bearer","api_key","oauth2"])
    ag.add_argument("--username",   default=None)
    ag.add_argument("--password",   default=None,
                    help="Basic password  (omit → prompted / env API_PASSWORD)")
    ag.add_argument("--token",      default=None,
                    help="Bearer token  (or env API_TOKEN)")
    ag.add_argument("--api-key",    "--api_key",    dest="api_key",      default=None,
                    help="API key  (or env API_KEY)")
    ag.add_argument("--api-key-header","--api_key_header", dest="api_key_header",
                    default="X-API-Key")
    ag.add_argument("--token-url",  "--token_url",  dest="token_url",    default=None)
    ag.add_argument("--client-id",  "--client_id",  dest="client_id",    default=None)
    ag.add_argument("--client-secret","--client_secret", dest="client_secret", default=None,
                    help="OAuth2 secret  (omit → prompted / env OAUTH2_SECRET)")
    ag.add_argument("--scope", default="")

    # Pagination
    pg = p.add_argument_group("Pagination")
    pg.add_argument("--pagination", default="none",
                    choices=["none","page_number","offset","cursor"])
    pg.add_argument("--page-param",      "--page_param",      dest="page_param",      default=None)
    pg.add_argument("--page-size-param", "--page_size_param", dest="page_size_param", default=None)
    pg.add_argument("--page-size",       "--page_size",       dest="page_size",
                    type=int, default=100)
    pg.add_argument("--start-page",      "--start_page",      dest="start_page",
                    type=int, default=1)

    # Limits
    lm = p.add_argument_group("Limits")
    lm.add_argument("--max-pages", "--max_pages", dest="max_pages",  type=int, default=50)
    lm.add_argument("--max-rows",  "--max_rows",  dest="max_rows",   type=int, default=500)
    lm.add_argument("--timeout",                                      type=int, default=30)

    # Auto Sync  (UI: Auto Sync toggle + Sync Frequency dropdown)
    sy = p.add_argument_group("Auto Sync")
    sy.add_argument(
        "--auto-sync", "--auto_sync", dest="auto_sync",
        action="store_true", default=False,
        help="Enable auto-sync  (UI: Auto Sync toggle ON)",
    )
    sy.add_argument(
        "--sync-frequency", "--sync_frequency", dest="sync_frequency",
        default="daily",
        choices=["hourly", "daily", "weekly", "monthly", "manual"],
        help="Re-fetch interval  (UI: Sync Frequency dropdown, default: daily)",
    )

    # Query Parameters  (UI: Add Query Parameters)
    p.add_argument(
        "--query-params", "--query_params", dest="query_params",
        default=None,
        help=(
            'Static query params as JSON array  (UI: Add Query Parameters)\n'
            '  e.g. \'[{"key":"format","value":"json"},{"key":"lang","value":"en"}]\''
        ),
    )

    # Output
    p.add_argument("--output", "-o", default=None,
                   help="Save markdown to file  (default: print to stdout)")
    return p


def main():
    args = build_parser().parse_args()

    # Source name default
    if not args.source_name:
        _p = urlparse(args.url if "://" in args.url else "https://" + args.url)
        args.source_name = _p.netloc + (_p.path.rstrip("/") or "")

    # ── Auth ──────────────────────────────────────────────────────────────────
    if args.auth == "none":
        auth_cfg = NoAuth()

    elif args.auth == "basic":
        if not args.username:
            args.username = input("Username: ").strip()
        auth_cfg = BasicAuth(
            username=args.username,
            password=args.password or _secret("Password", "API_PASSWORD"),
        )

    elif args.auth == "bearer":
        auth_cfg = BearerAuth(token=args.token or _secret("Bearer token", "API_TOKEN"))

    elif args.auth == "api_key":
        auth_cfg = ApiKeyAuth(
            header_name=args.api_key_header,
            api_key=args.api_key or _secret("API key", "API_KEY"),
        )

    elif args.auth == "oauth2":
        if not args.token_url:
            args.token_url = input("OAuth2 token URL: ").strip()
        if not args.client_id:
            args.client_id = input("Client ID: ").strip()
        auth_cfg = OAuth2Auth(
            token_url     = args.token_url,
            client_id     = args.client_id,
            client_secret = args.client_secret or _secret("Client secret", "OAUTH2_SECRET"),
            scope         = args.scope,
        )

    # ── Pagination ────────────────────────────────────────────────────────────
    pag_type = PaginationType(args.pagination)
    if pag_type != PaginationType.NONE:
        if not args.page_param:
            args.page_param = input(
                "Page parameter name (e.g. page / offset / cursor): "
            ).strip() or None
        if not args.page_size_param:
            args.page_size_param = input(
                "Page-size parameter name (e.g. per_page / limit): "
            ).strip() or None

    pag_cfg = PaginationConfig(
        pagination_type      = pag_type,
        page_param_name      = args.page_param,
        page_size_param_name = args.page_size_param,
        page_size_value      = args.page_size if pag_type != PaginationType.NONE else None,
        start_page           = args.start_page,
    )

    # ── Query params (UI: Add Query Parameters) ───────────────────────────────
    qp_list: List[QueryParam] = []
    if hasattr(args, "query_params") and args.query_params:
        raw_qp = json.loads(args.query_params)
        qp_list = [QueryParam(key=q["key"], value=q["value"]) for q in raw_qp]

    # ── Auto-sync (UI: Auto Sync toggle + Sync Frequency) ─────────────────────
    auto_sync_cfg = AutoSyncConfig(
        enabled   = getattr(args, "auto_sync", False),
        frequency = SyncFrequency(getattr(args, "sync_frequency", "daily")),
    )

    # ── Build source + run ────────────────────────────────────────────────────
    source = APISourceInput(
        source_name  = args.source_name,
        http_method  = args.method,
        url          = args.url,
        auth         = auth_cfg,
        pagination   = pag_cfg,
        json_path    = args.json_path,
        headers      = json.loads(args.headers) if args.headers else {},
        query_params = qp_list,
        request_body = json.loads(args.body) if args.body else None,
        auto_sync    = auto_sync_cfg,
    )

    print(f"\n{'─'*62}")
    print(f"  Source     : {source.source_name}")
    print(f"  URL        : {source.url}")
    print(f"  Method     : {source.http_method.value}")
    print(f"  Auth       : {source.auth.type}")
    print(f"  Paging     : {source.pagination.pagination_type.value}")
    if source.json_path:
        print(f"  JSON path  : {source.json_path}")
    if source.query_params:
        print(f"  Query params: {source.merged_query_params()}")
    if source.headers:
        print(f"  Headers    : {list(source.headers.keys())}")
    print(f"  Auto Sync  : {'✅ ' + source.auto_sync.frequency.value if source.auto_sync.enabled else '❌ off'}")
    print(f"{'─'*62}\n")

    extractor = APIExtractor(config={
        "max_pages":          args.max_pages,
        "max_rows_per_table": args.max_rows,
        "request_timeout":    args.timeout,
    })

    markdown, success, metadata = extractor.extract(source)

    print(f"{'✅' if success else '❌'}  Success : {success}")
    print(f"   Pages    : {metadata.get('total_pages', 0)}")
    print(f"   Rows     : {metadata.get('total_rows', 0)}")
    print(f"   Auto Sync: {metadata.get('auto_sync_enabled')} / {metadata.get('auto_sync_frequency')}")
    if metadata.get("http_errors"):
        print(f"   Errors   : {metadata['http_errors']}")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(markdown)
        print(f"\n  Saved → {args.output}")
    else:
        print("\n" + "─"*62)
        print(markdown[:4000])
        if len(markdown) > 4000:
            print(f"\n... ({len(markdown)} total chars)")


if __name__ == "__main__":
    main()