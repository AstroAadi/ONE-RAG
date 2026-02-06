# locustfile.py
# Run from D:\ONE-RAG with:
# python -m locust -f locustfile.py --host=http://localhost:8000

import random
from pathlib import Path
from locust import HttpUser, task, between, tag, constant_throughput

# ────────────────────────────────────────────────
# CONFIGURATION
# ────────────────────────────────────────────────

USER_ID = "123"  # ← change to your real test user ID

# Real test files — use DIFFERENT types & correct paths
TEST_FILES = {
    "pdf1": r"D:/Downloads/sample_kubernetes_doc.pdf",
    "pdf2": r"D:/Downloads/ICPC World Finals problemset.pdf",
    # Add more files here when you have them
    # "txt":  r"D:/path/to/sample.txt",
    # "html": r"D:/path/to/sample.html",
}

# Check files exist at startup
for name, p in TEST_FILES.items():
    if not Path(p).is_file():
        print(f"ERROR: Missing test file → {p}")
        exit(1)

# ────────────────────────────────────────────────
# Locust User Class
# ────────────────────────────────────────────────

class RAGApiUser(HttpUser):
    host = "http://localhost:8000"
    wait_time = between(1, 3)          # realistic user think time

    def on_start(self):
        """Executed once per virtual user at start"""
        self.client.headers = {
            "x-user-id": USER_ID,
            "accept": "application/json"
        }

    @tag("upload")
    @task(weight=6)   # 6× more uploads than queries
    def upload_document(self):
        """Simulate uploading a random document"""
        file_key = random.choice(list(TEST_FILES.keys()))
        file_path = TEST_FILES[file_key]

        try:
            with open(file_path, "rb") as f:
                # Correct MIME types
                mime = {
                    "pdf1": "application/pdf",
                    "pdf2": "application/pdf",
                    # add others: "txt": "text/plain", etc.
                }.get(file_key, "application/octet-stream")

                files = {"file": (Path(file_path).name, f, mime)}

                self.client.post(
                    "/api/upload/",
                    files=files,
                    name="/api/upload/"   # groups stats
                )

        except Exception as e:
            self.environment.events.request.fire(
                request_type="POST",
                name="/api/upload/",
                response_time=0,
                response_length=0,
                exception=e,
                context={}
            )

    @tag("query")
    @task(weight=2)   # 2× less than uploads
    def make_query(self):
        """Simulate a typical query"""
        queries = [
            "What is Kubernetes?",
            "Explain pods and nodes",
            "Summarize the document",
            "What are the main components?",
            "How does container orchestration work?"
        ]
        payload = {
            "query": random.choice(queries),
            "top_k": 5
        }
        self.client.post(
            "/api/query/",
            json=payload,
            name="/api/query/"
        )

    @tag("reset")
    @task(weight=1)   # very rare
    def reset_knowledge_base(self):
        """Rarely reset (careful – deletes data!)"""
        self.client.delete(
            "/api/knowledge-base/reset",
            name="/api/knowledge-base/reset"
        )