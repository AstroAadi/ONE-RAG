"""
verify_mongo_nl_query.py
MongoDB connection verifier + Natural Language → MongoDB Query (using langchain_groq)

After successful connection verification, enters interactive mode where
you can ask questions in natural language → Groq converts to MongoDB query → execute

Dependencies:
    pip install pymongo langchain langchain-groq python-dotenv
"""

import argparse
import json
import os
import socket
import sys
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any
from urllib.parse import urlparse, quote_plus, parse_qs, urlencode, urlunparse

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from pymongo import MongoClient
from pymongo.errors import OperationFailure, ServerSelectionTimeoutError, PyMongoError

from dotenv import load_dotenv

load_dotenv()  # reads GROQ_API_KEY, GROQ_MODEL from .env

# ────────────────────────────────────────────────────────────────
#  LLM Configuration using langchain_groq
# ────────────────────────────────────────────────────────────────

llm = ChatGroq(
    model=os.getenv("LLM_MODEL", 'llama-3.1-70b-versatile'),
    temperature=0.15,
    max_tokens=800,
    api_key="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",
)    

json_parser = JsonOutputParser()

# ────────────────────────────────────────────────────────────────
#  Schema inference  — samples the live collection to extract
#  field names, value types, and representative example values.
#  This context is injected into every LLM prompt so the model
#  generates accurate field names instead of guessing.
# ────────────────────────────────────────────────────────────────

def infer_schema(collection, sample_size: int = 30) -> dict:
    """
    Sample up to `sample_size` documents and return:
      { "field_name": {"types": ["str", ...], "examples": ["val1", ...]} }
    Flattens one level of nested documents (e.g. address.city).
    """
    schema: dict = {}

    def record(key: str, value):
        entry = schema.setdefault(key, {"types": set(), "examples": []})
        entry["types"].add(type(value).__name__)
        if value is not None and len(entry["examples"]) < 3:
            s = str(value)[:60]
            if s not in entry["examples"]:
                entry["examples"].append(s)

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


def format_schema_for_prompt(schema: dict) -> str:
    """
    Render the schema as a compact table for the system prompt.

    Example:
      - age          : int          | e.g. 28, 34, 41
      - email        : str          | e.g. alice@x.com
      - status       : str          | e.g. active, inactive
    """
    if not schema:
        return "  (schema unavailable — collection may be empty)"
    lines = []
    for field, meta in schema.items():
        types_str    = ", ".join(meta["types"])
        examples_str = ", ".join(meta["examples"]) if meta["examples"] else "—"
        lines.append(f"  - {field:<40} {types_str:<18} | e.g. {examples_str}")
    return "\n".join(lines)


def build_chain(schema_context: str):
    """Build a schema-aware LangChain chain. Called after connecting."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert MongoDB query generator.
You will receive a natural language question and must respond **only** with valid JSON.

Collection name: {collection_name}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
COLLECTION SCHEMA  (use ONLY these exact field names)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""" + schema_context + """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STRICT RULES:
- Only generate READ operations: find, count, aggregate
- NEVER generate update, delete, insert, drop, create, remove
- Use ONLY the field names listed above — do not invent or assume any others
- Match field name casing EXACTLY as listed in the schema
- For nested fields listed as "address.city", use dot notation in filters
- Use projection to limit returned fields
- Use sort/limit when appropriate
- For grouping/stats/complex logic, use aggregate pipeline
- Return pure JSON only — no markdown, no explanation

Response format:

find:
{{
  "operation": "find",
  "filter": {{ "status": "active", "age": {{ "$gt": 25 }} }},
  "projection": {{ "name": 1, "email": 1, "_id": 0 }},
  "sort": [["createdAt", -1]],
  "limit": 10
}}

count:
{{
  "operation": "count",
  "filter": {{ "city": "Pune" }}
}}

aggregate:
{{
  "operation": "aggregate",
  "pipeline": [
    {{ "$match": {{ "status": "completed" }} }},
    {{ "$group": {{ "_id": "$category", "total": {{ "$sum": "$amount" }} }} }},
    {{ "$sort": {{ "total": -1 }} }},
    {{ "$limit": 5 }}
  ]
}}

{format_instructions}"""),
        ("human", "{question}"),
    ])
    return prompt | llm | json_parser


# Placeholder — replaced with schema-aware chain in interactive_query_mode()
chain = build_chain("  (schema not yet loaded)")

# ────────────────────────────────────────────────────────────────

@dataclass
class CompassConfig:
    host:            str  = "localhost"
    port:            int  = 27017
    database:        str  = "bigdb"
    username:        str  = "tejas"
    password:        str  = "root"
    collection_name: str  = "users"
    ssl_mode:        bool = False
    uri:             Optional[str] = None


class CompassVerifier:
    def __init__(self, config: CompassConfig):
        self.cfg = config
        self.results = []
        self.client: Optional[MongoClient] = None
        self.winning_auth: Optional[str] = None

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
            self._pass("TCP reachability", f"{host}:{port} reachable in {ms:.0f} ms")
        except Exception as e:
            self._fail("TCP reachability", str(e))

    def _check_driver(self):
        try:
            import pymongo
            self._pass("pymongo driver", f"Version {pymongo.__version__}")
        except ImportError:
            self._fail("pymongo driver", "Not installed — run: pip install pymongo")

    def _check_connection(self) -> bool:
        cfg = self.cfg

        if cfg.uri:
            parsed = urlparse(cfg.uri)
            target_db = parsed.path.lstrip("/").split("?")[0].strip() or "admin"
            base_user = parsed.username or ""
            base_pass = parsed.password or ""
        else:
            target_db = cfg.database
            base_user = cfg.username
            base_pass = cfg.password

        auth_candidates = [target_db] if target_db != "admin" else []
        auth_candidates.append("admin")

        for auth_src in auth_candidates:
            self._info("Trying authSource", f"'{auth_src}'")

            if cfg.uri:
                p_uri = urlparse(cfg.uri)
                qs = parse_qs(p_uri.query, keep_blank_values=True)
                qs["authSource"] = [auth_src]
                test_uri = urlunparse(p_uri._replace(query=urlencode(qs, doseq=True)))
            else:
                creds = f"{quote_plus(base_user)}:{quote_plus(base_pass)}@" if base_user else ""
                test_uri = f"mongodb://{creds}{cfg.host}:{cfg.port}/{quote_plus(target_db)}?authSource={quote_plus(auth_src)}"

            kwargs = {
                "host": test_uri,
                "serverSelectionTimeoutMS": 15000,
                "connectTimeoutMS": 10000,
                "socketTimeoutMS": 15000,
            }
            if cfg.ssl_mode:
                kwargs["tls"] = True
                kwargs["tlsAllowInvalidCertificates"] = True

            try:
                t = MongoClient(**kwargs)
                resp = t[target_db].command("ping")
                if resp.get("ok") == 1.0:
                    self.client = t
                    self.winning_auth = auth_src
                    self._pass("Server ping", f"Connected via authSource='{auth_src}'")
                    break
                t.close()
            except Exception as e:
                self._warn("Connection attempt", str(e)[:120])

        if not self.client:
            self._fail("Authentication", "No successful authSource found.")
            return False

        try:
            db = self.client[target_db]
            db.list_collection_names()
            self._pass("Database access", f"'{target_db}' is readable")
        except Exception as e:
            self._fail("Database access", str(e))

        return len([r for r in self.results if r["status"] == "fail"]) == 0

    def _pass(self, label, detail):
        self.results.append({"status": "pass"})
        print(f"  ✅  {label:<30} {detail}")

    def _fail(self, label, detail):
        self.results.append({"status": "fail"})
        print(f"  ❌  {label:<30} {detail}")

    def _warn(self, label, detail):
        self.results.append({"status": "warn"})
        print(f"  ⚠️   {label:<30} {detail}")

    def _info(self, label, detail):
        self.results.append({"status": "info"})
        print(f"  ℹ️   {label:<30} {detail}")

    def _print_header(self):
        print("\n" + "━"*68)
        print("  MongoDB Connection Verifier + NL Query (Groq + LangChain)")
        print("━"*68)
        if self.cfg.uri:
            masked = self.cfg.uri.replace(f":{urlparse(self.cfg.uri).password}@", ":****@") if urlparse(self.cfg.uri).password else self.cfg.uri
            print(f"  URI : {masked}")
        else:
            print(f"  Host     : {self.cfg.host}:{self.cfg.port}")
            print(f"  Database : {self.cfg.database}")
            print(f"  User     : {self.cfg.username or '(none)'}")
        print("━"*68 + "\n")

    def _print_summary(self, ok):
        print("\n" + "━"*68)
        if ok:
            print("  🎉  Connection verified successfully")
        else:
            print("  🔴  Verification FAILED")
        print("━"*68 + "\n")


# ────────────────────────────────────────────────────────────────
#  LLM → MongoDB Query Generation
# ────────────────────────────────────────────────────────────────

def normalize_llm_response(raw: Dict) -> Dict:
    """
    The LLM sometimes wraps the result in an extra key, e.g.:
      {"aggregate": {"operation": "aggregate", "pipeline": [...]}}
    instead of the flat expected form:
      {"operation": "aggregate", "pipeline": [...]}

    This function unwraps such nesting so execute_mongo_query always
    receives a consistent flat dict with an "operation" key.
    """
    # Already flat and valid
    if "operation" in raw:
        return raw

    # Wrapped under a known operation key  e.g. {"aggregate": {...}}
    for op_key in ("find", "count", "aggregate"):
        if op_key in raw and isinstance(raw[op_key], dict):
            inner = raw[op_key]
            if "operation" not in inner:
                inner["operation"] = op_key   # inject missing key
            return inner

    # Wrapped under a single unknown key  e.g. {"query": {"operation": ...}}
    if len(raw) == 1:
        inner = next(iter(raw.values()))
        if isinstance(inner, dict) and "operation" in inner:
            return inner

    # Cannot normalise — return as-is and let execute handle the error
    return raw


def llm_generate_mongodb_query(
    question: str,
    collection_name: str,
    active_chain=None,          # schema-aware chain built after connection
) -> Dict[str, Any]:
    """
    Invoke the LLM with the user question.
    `active_chain` is the schema-aware chain built by build_chain(); falls
    back to the global placeholder chain if not supplied.
    """
    selected_chain = active_chain if active_chain is not None else chain
    try:
        raw = selected_chain.invoke({
            "question": question,
            "collection_name": collection_name,
            "format_instructions": json_parser.get_format_instructions(),
        })

        result = normalize_llm_response(raw)

        print("\n[Generated MongoDB query]")
        print(json.dumps(result, indent=2, default=str))
        return result

    except Exception as e:
        print(f"[LLM Error] {type(e).__name__}: {str(e)}")
        return {"operation": "find", "filter": {}}


def is_safe_query(params: Dict) -> bool:
    text = json.dumps(params).lower()
    forbidden = ["$where", "$function", "$accumulator", "update", "delete", "insert", "drop", "create", "remove"]
    return not any(kw in text for kw in forbidden)


def execute_mongo_query(client: MongoClient, cfg: CompassConfig, params: Dict) -> Dict:
    db = client[cfg.database]
    coll = db[cfg.collection_name]

    try:
        op = params.get("operation", "find")

        if op == "find":
            cursor = coll.find(
                filter=params.get("filter", {}),
                projection=params.get("projection"),
                sort=params.get("sort"),
                limit=params.get("limit", 25)
            )
            docs = list(cursor)
            return {"type": "documents", "count": len(docs), "data": docs}

        elif op == "count":
            cnt = coll.count_documents(params.get("filter", {}))
            return {"type": "count", "value": cnt}

        elif op == "aggregate":
            result = list(coll.aggregate(params.get("pipeline", [])))
            return {"type": "aggregate", "data": result}

        else:
            return {"error": f"Unsupported operation: {op}"}

    except PyMongoError as e:
        return {"error": str(e)}


def interactive_query_mode(client: MongoClient, cfg: CompassConfig):
    # ── Step 1: Infer schema from the live collection ─────────────────────────
    print("\n⏳ Sampling collection to infer schema...", end=" ", flush=True)
    collection   = client[cfg.database][cfg.collection_name]
    schema       = infer_schema(collection, sample_size=30)
    schema_text  = format_schema_for_prompt(schema)

    if schema:
        print(f"✅ {len(schema)} fields discovered\n")
        print("  Schema context passed to LLM:")
        print(schema_text)
    else:
        print("⚠️  No documents found — LLM will operate without schema context\n")

    # ── Step 2: Build a schema-aware chain ────────────────────────────────────
    active_chain = build_chain(schema_text)

    print("\n" + "═"*78)
    print("  Natural Language → MongoDB Query Mode  (type empty line to exit)")
    print(f"  Database: {cfg.database}   Collection: {cfg.collection_name}")
    print("═"*78 + "\n")

    while True:
        try:
            q = input("Question: ").strip()
            if not q:
                print("Exiting.\n")
                break

            print("\n→ Processing...\n")

            params = llm_generate_mongodb_query(q, cfg.collection_name, active_chain=active_chain)

            if not is_safe_query(params):
                print("  Rejected: unsafe operation detected.\n")
                continue

            result = execute_mongo_query(client, cfg, params)

            if "error" in result:
                print(f"  ❌ Error: {result['error']}\n")

            elif result["type"] == "count":
                print(f"  ✅ Count: {result['value']}\n")

            elif result["type"] == "documents":
                docs = result["data"]
                if not docs:
                    print("  ℹ️  No documents matched the query.\n")
                else:
                    print(f"  ✅ Found {result['count']} document(s):\n")
                    for i, doc in enumerate(docs, 1):
                        print(f"  {i}. {json.dumps(doc, indent=2, default=str)}\n")

            elif result["type"] == "aggregate":
                data = result["data"]
                if not data:
                    print("  ℹ️  Aggregate returned no results.\n")
                elif len(data) == 1 and len(data[0]) <= 3:
                    # Single-row scalar result (e.g. total rainfall) — print cleanly
                    row = {k: v for k, v in data[0].items() if k != "_id"}
                    for k, v in row.items():
                        label = k.replace("_", " ").replace("-", " ").title()
                        print(f"  ✅ {label}: {v}\n")
                else:
                    print(f"  ✅ Aggregate result ({len(data)} row(s)):\n")
                    for i, row in enumerate(data, 1):
                        print(f"  {i}. {json.dumps(row, indent=2, default=str)}\n")

            else:
                print(f"  ⚠️  Unknown result type: {result.get('type')}\n")

        except KeyboardInterrupt:
            print("\nInterrupted.")
            break
        except Exception as e:
            print(f"  Unexpected error: {str(e)}\n")


def main():
    parser = argparse.ArgumentParser(description="MongoDB verifier + Groq NL→MongoDB query")
    parser.add_argument("--uri", help="mongodb://... URI")
    parser.add_argument("--host", default=CompassConfig.host)
    parser.add_argument("--port", type=int, default=CompassConfig.port)
    parser.add_argument("--database", default=CompassConfig.database)
    parser.add_argument("--username", default=CompassConfig.username)
    parser.add_argument("--password", default=CompassConfig.password)
    parser.add_argument("--collection", default=CompassConfig.collection_name, dest="collection_name")
    parser.add_argument("--no-ssl", action="store_true")
    args = parser.parse_args()

    cfg = CompassConfig(
        uri=args.uri,
        host=args.host,
        port=args.port,
        database=args.database,
        username=args.username,
        password=args.password or os.getenv("DB_PASSWORD", ""),
        collection_name=args.collection_name,
        ssl_mode=not args.no_ssl,
    )

    verifier = CompassVerifier(cfg)
    success = verifier.verify()

    if success and verifier.client:
        print("\nConnection verified → entering Groq-powered query mode\n")
        interactive_query_mode(verifier.client, cfg)
        verifier.client.close()
    else:
        print("Cannot start query mode — connection verification failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
 