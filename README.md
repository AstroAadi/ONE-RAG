# ONE-RAG

FastAPI-based retrieval-augmented generation API.

## Run locally on Windows

Use Python 3.12. Install `uv` in PowerShell:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Open a new PowerShell window, go to the repository root, then create a short-path
virtual environment and install the runtime dependencies:

```powershell
$venv = "$env:USERPROFILE\venvs\one-rag"
New-Item -ItemType Directory -Force (Split-Path $venv) | Out-Null
uv venv --python 3.12 $venv
uv pip install --python "$venv\Scripts\python.exe" -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env` and set:

- `HUGGINGFACE_API_TOKEN`: a Hugging Face token with permission to use the configured models.
- `QDRANT_URL` and `QDRANT_API_KEY`: the URL and API key for a reachable Qdrant instance.

Both services are required by the current API routes. The sample file intentionally contains no live credentials. Do not commit `.env`.

`DATABASE_URL` is required and must point to PostgreSQL. The application normalizes PostgreSQL URL variants (including `postgres://` and URLs naming other drivers) to psycopg2. Do not commit `.env` or print its contents. The startup configuration summary never prints database credentials.

SPLADE sparse embeddings are disabled by default because the configured `naver/splade-v3` model is gated and requires separate Hugging Face access. Dense Qdrant retrieval remains available; set `ENABLE_SPARSE_EMBEDDINGS=true` only after access to that model is granted.

Start the API from the repository root:

```powershell
$venv = "$env:USERPROFILE\venvs\one-rag"
uv run --python "$venv\Scripts\python.exe" -m uvicorn app.main:app --reload
```

Open <http://127.0.0.1:8000/docs> for the API documentation, or <http://127.0.0.1:8000/health> for the database health check. API routes that require a user also require the `X-User-ID` header.

Optional: install `locust` if you want to run the load-test script, or
`unstructured[pdf]` if you want the PDF extractor's Unstructured fallback:

```powershell
$venv = "$env:USERPROFILE\venvs\one-rag"
uv pip install --python "$venv\Scripts\python.exe" locust "unstructured[pdf]"
```

OCR also requires the Tesseract executable to be installed separately; the
Python `pytesseract` dependency is already in `requirements.txt`.
