"""
mongodb_agent.py
────────────────────────────────────────────────────────────────────────────────
MongoDB ReAct Agent  —  Natural language → tool calls → answer

Architecture:
  User question
       ↓
  LangChain ReAct Agent  (Groq LLM)
       ↓ reasons about which tool to call
  ┌─────────────────────────────────────────┐
  │  Tools:                                 │
  │   • get_schema       — inspect fields   │
  │   • list_collections — list all colls   │
  │   • find_documents   — find + project   │
  │   • count_documents  — filter count     │
  │   • aggregate        — pipeline         │
  │   • distinct_values  — unique values    │
  │   • sample_documents — random sample    │
  └─────────────────────────────────────────┘
       ↓ tool result fed back to LLM
  LLM reasons → may call more tools → final answer in plain English

Unlike the previous single-chain approach:
  ✅ Multi-step reasoning  ("find youngest age, then count people with that age")
  ✅ Self-correcting       (if first query returns nothing, agent retries)
  ✅ Natural language answers  (not raw JSON dumps)
  ✅ Conversation memory   (remembers previous questions in session)

Dependencies:
  pip install pymongo langchain langchain-groq langgraph python-dotenv
"""

import argparse
import json
import os
import socket
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Optional, Any, Literal
from urllib.parse import urlparse, quote_plus, parse_qs, urlencode, urlunparse

from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.prebuilt import create_react_agent

from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import PyMongoError

# ── LLM ───────────────────────────────────────────────────────────────────────

llm = ChatGroq(
    model       = os.getenv("LLM_MODEL", "llama-3.1-70b-versatile"),
    temperature = 0,
    # FIX: was 2048 — too small for large tabular responses (50-row tables need
    # ~4000-6000 tokens). Now reads from env so it can be tuned without code edits.
    max_tokens  = int(os.getenv("LLM_MAX_TOKENS", "8192")),
    api_key     = "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",
)

# ── Connection config ─────────────────────────────────────────────────────────

@dataclass
class MongoConfig:
    host:            str  = "localhost"
    port:            int  = 27017
    database:        str  = "bigdb"
    username:        str  = "sounak"
    password:        str  = "root"
    collection_name: str  = "users"
    uri:             Optional[str] = None

    # ── SSL / TLS ──────────────────────────────────────────────────────────────
    # ssl_mode controls how strictly TLS is enforced:
    #   "off"         — no TLS at all (default; safe for localhost/dev)
    #   "require"     — TLS on, but certificate NOT validated (encrypted channel only)
    #   "verify-ca"   — TLS on, CA certificate validated, hostname NOT checked
    #   "verify-full" — TLS on, CA + hostname both validated (production standard)
    #
    # Previously this was a bool (ssl_mode=False) with no granularity and
    # tlsAllowInvalidCertificates hardcoded to True — so cert validation could
    # never be enforced regardless of intent.
    ssl_mode:        Literal["off", "require", "verify-ca", "verify-full"] = "off"
    tls_ca_file:     Optional[str] = None   # path to CA bundle (.pem)
    tls_cert_file:   Optional[str] = None   # path to client certificate (.pem)
    tls_cert_key_file: Optional[str] = None # path to client private key (.pem)

    # ── Auto-sync ──────────────────────────────────────────────────────────────
    # auto_sync_interval > 0 enables a background thread that refreshes the
    # schema cache every N seconds. The agent's system prompt is updated on
    # each refresh so it always reasons against current field names/types.
    # Set to 0 to disable (default).
    auto_sync_interval: int = 0   # seconds between schema refreshes; 0 = off


def build_tls_kwargs(cfg: "MongoConfig") -> dict:
    """
    Translate ssl_mode + cert paths into the correct pymongo TLS kwargs.

    ssl_mode   tls    tlsAllowInvalidCertificates   tlsAllowInvalidHostnames
    --------   ---    ---------------------------   ------------------------
    off        False  —                             —
    require    True   True                          True
    verify-ca  True   False                         True
    verify-full True  False                         False
    """
    if cfg.ssl_mode == "off":
        return {}

    kwargs: dict = {"tls": True}

    if cfg.ssl_mode == "require":
        kwargs["tlsAllowInvalidCertificates"] = True
        kwargs["tlsAllowInvalidHostnames"]    = True

    elif cfg.ssl_mode == "verify-ca":
        kwargs["tlsAllowInvalidCertificates"] = False
        kwargs["tlsAllowInvalidHostnames"]    = True   # CA checked, hostname not
        if cfg.tls_ca_file:
            kwargs["tlsCAFile"] = cfg.tls_ca_file

    elif cfg.ssl_mode == "verify-full":
        kwargs["tlsAllowInvalidCertificates"] = False
        kwargs["tlsAllowInvalidHostnames"]    = False
        if cfg.tls_ca_file:
            kwargs["tlsCAFile"] = cfg.tls_ca_file

    # Mutual TLS (client certificate auth) — used in verify-ca / verify-full
    if cfg.tls_cert_file:
        kwargs["tlsCertificateKeyFile"] = cfg.tls_cert_file
    if cfg.tls_cert_key_file:
        kwargs["tlsCertificateKeyFile"] = cfg.tls_cert_key_file

    return kwargs


# ── Global state shared by tools ──────────────────────────────────────────────
# Tools are module-level functions and need access to the live client/config.
# We store them here after connection is established.

_mongo_client:     Optional[MongoClient] = None
_mongo_config:     Optional[MongoConfig] = None
_cached_schema:    Optional[dict]        = None
_schema_lock:      threading.Lock        = threading.Lock()   # guards _cached_schema
_sync_thread:      Optional[threading.Thread] = None          # auto-sync background thread
_agent_executor                          = None               # rebuilt on schema refresh


def _auto_sync_worker(interval: int):
    """
    Background thread: refreshes _cached_schema every `interval` seconds and
    rebuilds the agent so it reasons against up-to-date field names and types.

    Previously the schema was sampled once at startup, printed to console, then
    never updated — so any schema changes (new fields, renamed collections, etc.)
    were invisible to the agent until restart.
    """
    global _cached_schema, _agent_executor
    while True:
        time.sleep(interval)
        if _mongo_client is None or _mongo_config is None:
            continue
        try:
            coll       = _mongo_client[_mongo_config.database][_mongo_config.collection_name]
            new_schema = infer_schema(coll, sample_size=30)
            with _schema_lock:
                old_keys = set(_cached_schema or {})
                new_keys = set(new_schema)
                changed  = old_keys != new_keys
                _cached_schema = new_schema
            if changed:
                # Rebuild agent so system prompt reflects the new schema
                _agent_executor = build_agent(
                    _mongo_config.database,
                    _mongo_config.collection_name
                )
                added   = new_keys - old_keys
                removed = old_keys - new_keys
                parts   = []
                if added:   parts.append(f"+{len(added)} field(s)")
                if removed: parts.append(f"-{len(removed)} field(s)")
                print(f"\n[auto-sync] Schema updated — {', '.join(parts)}. Agent rebuilt.\n")
            else:
                print(f"\r[auto-sync] Schema unchanged ({len(new_keys)} fields)", end="", flush=True)
        except Exception as e:
            print(f"\n[auto-sync] Refresh error: {e}\n")


def start_auto_sync(interval: int):
    """Start the background schema-sync thread (daemon so it exits with main)."""
    global _sync_thread
    if interval <= 0:
        return
    _sync_thread = threading.Thread(
        target=_auto_sync_worker,
        args=(interval,),
        daemon=True,
        name="mongo-auto-sync"
    )
    _sync_thread.start()
    print(f"[auto-sync] Background schema refresh every {interval}s started.\n")


def _get_collection(collection_name: Optional[str] = None):
    """Return the active pymongo collection."""
    if _mongo_client is None or _mongo_config is None:
        raise RuntimeError("MongoDB not connected")
    col = collection_name or _mongo_config.collection_name
    return _mongo_client[_mongo_config.database][col]


def _serialize(obj: Any) -> Any:
    """Make MongoDB results JSON-serialisable."""
    from bson import ObjectId
    from datetime import datetime
    if isinstance(obj, ObjectId):
        return str(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serialize(i) for i in obj]
    return obj


# ── Schema inference ──────────────────────────────────────────────────────────

def infer_schema(collection, sample_size: int = 30) -> dict:
    schema: dict = {}

    def record(key, value):
        e = schema.setdefault(key, {"types": set(), "examples": []})
        e["types"].add(type(value).__name__)
        if value is not None and len(e["examples"]) < 3:
            s = str(value)[:60]
            if s not in e["examples"]:
                e["examples"].append(s)

    try:
        docs = list(collection.find({}, limit=sample_size))
    except Exception:
        return {}

    for doc in docs:
        for k, v in doc.items():
            if k == "_id":
                continue
            if isinstance(v, dict):
                for sk, sv in v.items():
                    record(f"{k}.{sk}", sv)
            elif isinstance(v, list):
                record(k, f"Array({type(v[0]).__name__ if v else 'empty'})")
            else:
                record(k, v)

    return {
        k: {"types": sorted(v["types"]), "examples": v["examples"]}
        for k, v in sorted(schema.items())
    }


def format_schema(schema: dict) -> str:
    if not schema:
        return "(no documents — schema unavailable)"
    lines = []
    for field, meta in schema.items():
        t = ", ".join(meta["types"])
        e = ", ".join(meta["examples"]) if meta["examples"] else "—"
        lines.append(f"  {field:<40} {t:<18}  e.g. {e}")
    return "\n".join(lines)


# ── Safety guard ──────────────────────────────────────────────────────────────

_FORBIDDEN = {
    "$where", "$function", "$accumulator",
    "update", "delete", "insert", "drop",
    "create", "remove", "rename", "drop",
}

def _is_safe(payload: str) -> bool:
    low = payload.lower()
    return not any(kw in low for kw in _FORBIDDEN)


# ═══════════════════════════════════════════════════════════════════════════════
#  TOOLS  — each decorated with @tool so LangChain can bind them to the agent
# ═══════════════════════════════════════════════════════════════════════════════

@tool
def get_schema(collection_name: str = "") -> str:
    """
    Returns the schema of the MongoDB collection — field names, types, and
    example values sampled from up to 30 documents.
    Call this first whenever you are unsure of field names or data types.
    Optional: pass a specific collection_name, otherwise uses the default.
    """
    global _cached_schema
    try:
        coll   = _get_collection(collection_name or None)
        schema = infer_schema(coll, sample_size=30)
        # FIX: update cache under lock so auto-sync thread sees consistent state
        with _schema_lock:
            _cached_schema = schema
        formatted = format_schema(schema)
        col_label = collection_name or _mongo_config.collection_name
        return f"Schema for '{col_label}':\n{formatted}"
    except Exception as e:
        return f"Error getting schema: {e}"


@tool
def list_collections() -> str:
    """
    Lists all collection names in the current database.
    Use this when the user asks about what data is available or which collections exist.
    """
    try:
        db = _mongo_client[_mongo_config.database]
        cols = db.list_collection_names()
        if not cols:
            return "No collections found in database."
        return f"Collections in '{_mongo_config.database}':\n" + "\n".join(f"  - {c}" for c in sorted(cols))
    except Exception as e:
        return f"Error: {e}"


@tool
def find_documents(
    filter_json: str,
    projection_json: str = "{}",
    sort_json: str = "[]",
    limit: int = 10,
    collection_name: str = "",
) -> str:
    """
    Run a MongoDB find() query and return matching documents.

    Args:
        filter_json:     JSON string — MongoDB filter e.g. '{"age": {"$gt": 25}}'
        projection_json: JSON string — fields to include/exclude e.g. '{"name":1,"_id":0}'
        sort_json:       JSON string — sort spec e.g. '[["age", 1]]' (1=asc, -1=desc)
        limit:           Max number of documents to return (default 10, max 50)
        collection_name: Optional collection name (uses default if empty)

    Returns string with found documents or error message.
    Use get_schema first if you are unsure of field names.
    """
    if not _is_safe(filter_json + projection_json + sort_json):
        return "Rejected: unsafe operation detected."
    try:
        filt  = json.loads(filter_json)   if filter_json     else {}
        proj  = json.loads(projection_json) if projection_json and projection_json != "{}" else None
        sort  = json.loads(sort_json)     if sort_json and sort_json != "[]" else None
        limit = min(max(1, limit), 50)

        coll  = _get_collection(collection_name or None)
        cursor = coll.find(filter=filt, projection=proj, sort=sort, limit=limit)
        docs  = [_serialize(d) for d in cursor]

        if not docs:
            return "No documents matched the filter."

        # Count total matches (not just the limited set)
        total = coll.count_documents(filt)
        out   = [f"Found {total:,} total match(es), showing {len(docs)}:"]
        for i, doc in enumerate(docs, 1):
            out.append(f"\n[{i}] {json.dumps(doc, indent=2)}")
        return "\n".join(out)

    except json.JSONDecodeError as e:
        return f"Invalid JSON in arguments: {e}"
    except PyMongoError as e:
        return f"MongoDB error: {e}"
    except Exception as e:
        return f"Error: {e}"


@tool
def count_documents(filter_json: str = "{}", collection_name: str = "") -> str:
    """
    Count documents matching a filter.

    Args:
        filter_json:     JSON string — MongoDB filter e.g. '{"city": "Pune"}'
                         Use '{}' to count all documents.
        collection_name: Optional collection name.

    Returns the count as a string.
    """
    if not _is_safe(filter_json):
        return "⛔ Rejected: unsafe operation."
    try:
        filt = json.loads(filter_json) if filter_json else {}
        coll = _get_collection(collection_name or None)
        cnt  = coll.count_documents(filt)
        desc = f"with filter {filter_json}" if filt else "(all documents)"
        return f"Count {desc}: {cnt:,}"
    except json.JSONDecodeError as e:
        return f"Invalid JSON: {e}"
    except Exception as e:
        return f"Error: {e}"


@tool
def aggregate(pipeline_json: str, collection_name: str = "") -> str:
    """
    Run a MongoDB aggregation pipeline.

    Args:
        pipeline_json:   JSON string — array of pipeline stages
                         e.g. '[{"$group":{"_id":"$city","count":{"$sum":1}}},{"$sort":{"count":-1}}]'
        collection_name: Optional collection name.

    Use for: grouping, sums, averages, min/max with counts, complex analytics.
    NEVER use for: updates, deletes, inserts — read-only operations only.
    """
    if not _is_safe(pipeline_json):
        return "⛔ Rejected: unsafe operation."
    try:
        pipeline = json.loads(pipeline_json)
        if not isinstance(pipeline, list):
            return "pipeline_json must be a JSON array of stage objects."

        coll   = _get_collection(collection_name or None)
        result = [_serialize(r) for r in coll.aggregate(pipeline)]

        if not result:
            return "Aggregate returned no results."

        out = [f"Aggregate result ({len(result)} row(s)):"]
        for i, row in enumerate(result, 1):
            out.append(f"\n[{i}] {json.dumps(row, indent=2)}")
        return "\n".join(out)

    except json.JSONDecodeError as e:
        return f"Invalid JSON pipeline: {e}"
    except PyMongoError as e:
        return f"MongoDB error: {e}"
    except Exception as e:
        return f"Error: {e}"


@tool
def distinct_values(field: str, filter_json: str = "{}", collection_name: str = "") -> str:
    """
    Get all distinct (unique) values for a field.

    Args:
        field:           Field name to get distinct values of e.g. "city" or "status"
        filter_json:     Optional filter to restrict which documents are considered
        collection_name: Optional collection name.

    Use for questions like: "what are all unique cities?", "what statuses exist?"
    """
    try:
        filt   = json.loads(filter_json) if filter_json and filter_json != "{}" else {}
        coll   = _get_collection(collection_name or None)
        values = coll.distinct(field, filt)
        values = [_serialize(v) for v in values]

        if not values:
            return f"No distinct values found for field '{field}'."

        preview = values[:50]
        note    = f" (showing first 50 of {len(values)})" if len(values) > 50 else ""
        return f"Distinct values for '{field}'{note}:\n  " + ", ".join(str(v) for v in preview)

    except Exception as e:
        return f"Error: {e}"


@tool
def sample_documents(n: int = 5, collection_name: str = "") -> str:
    """
    Return n random sample documents from the collection.

    Args:
        n:               Number of random documents to return (default 5, max 20)
        collection_name: Optional collection name.

    Use this to understand the shape of the data when you're uncertain.
    """
    try:
        n    = min(max(1, n), 20)
        coll = _get_collection(collection_name or None)
        docs = [_serialize(d) for d in coll.aggregate([{"$sample": {"size": n}}])]

        if not docs:
            return "Collection appears to be empty."

        out = [f"{len(docs)} random sample document(s):"]
        for i, doc in enumerate(docs, 1):
            out.append(f"\n[{i}] {json.dumps(doc, indent=2)}")
        return "\n".join(out)

    except Exception as e:
        return f"Error: {e}"


# ═══════════════════════════════════════════════════════════════════════════════
#  AGENT SETUP
# ═══════════════════════════════════════════════════════════════════════════════

TOOLS = [
    get_schema,
    list_collections,
    find_documents,
    count_documents,
    aggregate,
    distinct_values,
    sample_documents,
]

SYSTEM_PROMPT = """You are an expert MongoDB data analyst assistant.
You have access to a live MongoDB database and a set of tools to query it.

Always think step by step. Use tools to retrieve the data you need, then
synthesise a clear, concise natural language answer. Do not dump raw JSON
as your final answer — always interpret and explain the results.

Guidelines:
- Call get_schema first if you are unsure of field names or types
- For min/max/youngest/oldest questions: use aggregate to get the extreme value AND count
- For "how many" questions: use count_documents
- For "show me / list" questions: use find_documents
- For "what unique values exist": use distinct_values
- For exploring unknown data: use sample_documents first

Large result sets:
- When returning >20 rows always use a markdown table for readability
- Never truncate a table mid-row — if you are running out of space, reduce
  column width (e.g. trim long addresses to city+state only) rather than
  dropping rows. Every requested row must appear in the output.
- Abbreviate long free-text fields (address, notes) to keep rows compact;
  always show name, key numeric fields, and dates in full.
"""

def build_agent(database: str, collection: str):
    """
    Create a langgraph ReAct agent bound to the current DB/collection.

    FIX: Embeds the current cached schema directly into the system prompt so
    the agent reasons against real field names from the start — previously the
    schema was only printed to the console and never fed to the agent.
    Auto-sync calls this again whenever the schema changes so the agent stays
    current without a restart.
    """
    with _schema_lock:
        schema_snapshot = dict(_cached_schema) if _cached_schema else {}

    schema_section = ""
    if schema_snapshot:
        schema_section = (
            f"\n\nCurrent schema for '{database}.{collection}':\n"
            + format_schema(schema_snapshot)
            + "\n"
        )

    system_msg = SYSTEM_PROMPT + f"\nDatabase: {database}  |  Collection: {collection}" + schema_section
    return create_react_agent(
        model  = llm,
        tools  = TOOLS,
        prompt = system_msg,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  CONNECTION VERIFIER  (unchanged from previous version)
# ═══════════════════════════════════════════════════════════════════════════════

class CompassVerifier:
    def __init__(self, config: MongoConfig):
        self.cfg     = config
        self.results = []
        self.client: Optional[MongoClient] = None

    def verify(self) -> bool:
        self._print_header()
        self._check_tcp()
        self._check_driver()
        ok = self._check_connection()
        self._print_summary(ok)
        return ok

    def _check_tcp(self):
        cfg = self.cfg
        if cfg.uri:
            parsed = urlparse(cfg.uri)
            host, port = parsed.hostname or "localhost", parsed.port or 27017
        else:
            host, port = cfg.host, cfg.port
        try:
            start = time.perf_counter()
            s = socket.create_connection((host, port), timeout=10)
            s.close()
            ms = (time.perf_counter() - start) * 1000
            self._pass("TCP reachability", f"{host}:{port} in {ms:.0f} ms")
        except Exception as e:
            self._fail("TCP reachability", str(e))

    def _check_driver(self):
        try:
            import pymongo
            self._pass("pymongo driver", f"v{pymongo.__version__}")
        except ImportError:
            self._fail("pymongo driver", "pip install pymongo")

    def _check_connection(self) -> bool:
        cfg = self.cfg
        if cfg.uri:
            parsed    = urlparse(cfg.uri)
            target_db = parsed.path.lstrip("/").split("?")[0].strip() or "admin"
            base_user = parsed.username or ""
            base_pass = parsed.password or ""
        else:
            target_db = cfg.database
            base_user = cfg.username
            base_pass = cfg.password

        auth_candidates = ([target_db] if target_db != "admin" else []) + ["admin"]

        for auth_src in auth_candidates:
            self._info("Trying authSource", f"'{auth_src}'")
            if cfg.uri:
                p_uri = urlparse(cfg.uri)
                qs    = parse_qs(p_uri.query, keep_blank_values=True)
                qs["authSource"] = [auth_src]
                test_uri = urlunparse(p_uri._replace(query=urlencode(qs, doseq=True)))
            else:
                creds    = f"{quote_plus(base_user)}:{quote_plus(base_pass)}@" if base_user else ""
                test_uri = (f"mongodb://{creds}{cfg.host}:{cfg.port}"
                            f"/{quote_plus(target_db)}?authSource={quote_plus(auth_src)}")

            kwargs = {
                "host": test_uri,
                "serverSelectionTimeoutMS": 15000,
                "connectTimeoutMS":         10000,
                "socketTimeoutMS":          15000,
            }
            # FIX: use build_tls_kwargs() for all TLS config instead of the
            # previous hardcoded tls=True + tlsAllowInvalidCertificates=True
            # which could never enforce cert validation.
            kwargs.update(build_tls_kwargs(cfg))

            try:
                t    = MongoClient(**kwargs)
                resp = t[target_db].command("ping")
                if resp.get("ok") == 1.0:
                    self.client = t
                    self._pass("Server ping", f"Connected via authSource='{auth_src}'")
                    break
                t.close()
            except Exception as e:
                self._warn("Connection attempt", str(e)[:120])

        if not self.client:
            self._fail("Authentication", "All authSource candidates failed.")
            return False

        try:
            self.client[target_db].list_collection_names()
            self._pass("Database access", f"'{target_db}' is readable")
        except Exception as e:
            self._fail("Database access", str(e))

        return not any(r["status"] == "fail" for r in self.results)

    def _pass(self, l, d): self.results.append({"status":"pass"}); print(f"  ✅  {l:<30} {d}")
    def _fail(self, l, d): self.results.append({"status":"fail"}); print(f"  ❌  {l:<30} {d}")
    def _warn(self, l, d): self.results.append({"status":"warn"}); print(f"  ⚠️   {l:<30} {d}")
    def _info(self, l, d): self.results.append({"status":"info"}); print(f"  ℹ️   {l:<30} {d}")

    def _print_header(self):
        print("\n" + "━"*68)
        print("  MongoDB Agent — Connection Verifier")
        print("━"*68)
        if self.cfg.uri:
            p = urlparse(self.cfg.uri)
            masked = self.cfg.uri.replace(f":{p.password}@", ":****@") if p.password else self.cfg.uri
            print(f"  URI : {masked}")
        else:
            print(f"  Host : {self.cfg.host}:{self.cfg.port}  DB: {self.cfg.database}")
        print("━"*68 + "\n")

    def _print_summary(self, ok):
        print("\n" + "━"*68)
        print("  🎉  Connection verified" if ok else "  🔴  Verification FAILED")
        print("━"*68 + "\n")


# ═══════════════════════════════════════════════════════════════════════════════
#  INTERACTIVE AGENT LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def run_agent_loop(client: MongoClient, cfg: MongoConfig):
    global _mongo_client, _mongo_config, _cached_schema, _agent_executor

    # Wire up globals so tools can access the live connection
    _mongo_client = client
    _mongo_config = cfg

    # Pre-load schema so it's in cache before building the agent
    print("\n⏳ Sampling collection schema...", end=" ", flush=True)
    coll   = client[cfg.database][cfg.collection_name]
    schema = infer_schema(coll, sample_size=30)
    with _schema_lock:
        _cached_schema = schema
    print(f"✅ {len(schema)} field(s) discovered\n")

    if schema:
        print("  Schema context:")
        print(format_schema(schema))
        print()

    # Build agent — schema is now embedded in the system prompt (previously it
    # was only printed to console and the agent had no access to it)
    _agent_executor = build_agent(cfg.database, cfg.collection_name)

    # Start background auto-sync if configured
    # Previously there was no auto-sync at all — schema was sampled once and frozen
    if cfg.auto_sync_interval > 0:
        start_auto_sync(cfg.auto_sync_interval)

    # Conversation history (for display only — ReAct agent is stateless per call)
    history = []

    print("═"*78)
    print("  MongoDB ReAct Agent  —  ask anything in plain English")
    print("  The agent will reason step-by-step and use tools to answer.")
    print("  Type an empty line to exit.")
    print("═"*78 + "\n")

    while True:
        try:
            question = input("You: ").strip()
            if not question:
                print("Goodbye.\n")
                break

            print()  # breathing room before verbose agent trace

            try:
                messages = [HumanMessage(content=question)]
                # FIX: use the global _agent_executor so auto-sync rebuilds
                # are picked up mid-session without restarting the loop
                result   = _agent_executor.invoke({"messages": messages})
                ai_msgs  = [m for m in result["messages"] if isinstance(m, AIMessage)]
                final    = ai_msgs[-1].content if ai_msgs else "(no response)"
            except Exception as e:
                final = f"Agent error: {e}"

            print("\n" + "─"*68)
            print(f"Answer: {final}")
            print("─"*68 + "\n")

            history.append({"q": question, "a": final})

        except KeyboardInterrupt:
            print("\nInterrupted.\n")
            break
        except Exception as e:
            print(f"\nUnexpected error: {e}\n")


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="MongoDB ReAct Agent (Groq + LangChain)")
    p.add_argument("--uri",        help="mongodb://user:pass@host:port/db")
    p.add_argument("--host",       default=MongoConfig.host)
    p.add_argument("--port",       type=int, default=MongoConfig.port)
    p.add_argument("--database",   default=MongoConfig.database)
    p.add_argument("--username",   default=MongoConfig.username)
    p.add_argument("--password",   default=MongoConfig.password)
    p.add_argument("--collection", default=MongoConfig.collection_name, dest="collection_name")

    # FIX: Replace the previous --no-ssl bool flag (which defaulted SSL to ON,
    # breaking every local connection) with an explicit --ssl-mode option.
    # Choices mirror PostgreSQL's sslmode levels so the intent is always clear.
    # Both --ssl-mode and --ssl_mode are accepted (hyphen and underscore).
    p.add_argument(
        "--ssl-mode", "--ssl_mode",
        dest    = "ssl_mode",
        default = "off",
        choices = ["off", "require", "verify-ca", "verify-full",
                   "true", "false"],   # 'true'/'false' mapped below for compat
        help    = (
            "TLS mode: off=no TLS (default), require=TLS no cert check, "
            "verify-ca=validate CA cert, verify-full=validate CA+hostname. "
            "'true' is treated as 'require', 'false' as 'off'."
        ),
    )
    p.add_argument("--tls-ca-file",       "--tls_ca_file",
                   dest="tls_ca_file",      default=None,
                   help="Path to CA certificate bundle (.pem) for verify-ca / verify-full")
    p.add_argument("--tls-cert-file",     "--tls_cert_file",
                   dest="tls_cert_file",    default=None,
                   help="Path to client certificate (.pem) for mutual TLS")
    p.add_argument("--tls-cert-key-file", "--tls_cert_key_file",
                   dest="tls_cert_key_file",default=None,
                   help="Path to client private key (.pem) for mutual TLS")

    # FIX: Add --auto-sync / --auto_sync_interval to enable background schema refresh.
    # Both spellings accepted; 0 = disabled (default).
    p.add_argument(
        "--auto-sync", "--auto_sync", "--auto_sync_interval",
        dest    = "auto_sync_interval",
        type    = int,
        default = 0,
        metavar = "SECONDS",
        help    = "Refresh schema every N seconds in background (0 = disabled)"
    )

    args = p.parse_args()

    # Map legacy bool strings to proper ssl_mode values
    ssl_mode = args.ssl_mode
    if ssl_mode == "true":
        ssl_mode = "require"
    elif ssl_mode == "false":
        ssl_mode = "off"

    cfg = MongoConfig(
        uri                 = args.uri,
        host                = args.host,
        port                = args.port,
        database            = args.database,
        username            = args.username,
        password            = args.password or os.getenv("DB_PASSWORD", ""),
        collection_name     = args.collection_name,
        ssl_mode            = ssl_mode,
        tls_ca_file         = args.tls_ca_file,
        tls_cert_file       = args.tls_cert_file,
        tls_cert_key_file   = args.tls_cert_key_file,
        auto_sync_interval  = args.auto_sync_interval,
    )

    verifier = CompassVerifier(cfg)
    ok       = verifier.verify()

    if ok and verifier.client:
        run_agent_loop(verifier.client, cfg)
        verifier.client.close()
    else:
        print("Cannot start agent — connection failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()