# locustfile.py
# Usage: python -m locust -f locustfile.py --host=http://localhost:8000

import random
from pathlib import Path
from locust import HttpUser, task, between, tag

# ────────────────────────────────────────────────
# CONFIG
# ────────────────────────────────────────────────

HOST = "http://localhost:8000"
USER_ID = "./uploads/123"  # fixed test user

# Real test files (adjust paths!)
TEST_FILES = {
    "pdf":   r"D:/Downloads/sample_kubernetes_doc.pdf"
}

# Check files exist
for name, p in TEST_FILES.items():
    if not Path(p).is_file():
        print(f"WARNING: Missing test file: {p}")

# ────────────────────────────────────────────────
# Locust User Class
# ────────────────────────────────────────────────

class RAGApiUser(HttpUser):
    host = HOST
    wait_time = between(1, 5)  # 1–5 seconds between tasks

    def on_start(self):
        """Run once per virtual user"""
        self.client.headers = {
            "x-user-id": USER_ID,
            "accept": "application/json"
        }

    @tag("upload")
    @task(6)  # 6× more uploads than queries (integer only)
    def upload_document(self):
        """Simulate file upload"""
        file_type = random.choice(list(TEST_FILES.keys()))
        file_path = TEST_FILES[file_type]

        if not Path(file_path).exists():
            self.environment.events.request.fire(
                request_type="POST",
                name="/api/upload/",
                response_time=0,
                response_length=0,
                exception=Exception(f"Missing test file: {file_path}"),
                context={}
            )
            return

        try:
            with open(file_path, "rb") as f:
                files = {"file": (Path(file_path).name, f, f"application/{file_type}")}
                self.client.post(
                    "/api/upload/",
                    files=files,
                    name="/api/upload/"  # for stats grouping
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
    @task(2)  # 2× less frequent than uploads
    def make_query(self):
        """Simulate a query request"""
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
    @task(1)  # rare – only 1/9 of tasks
    def reset_knowledge_base(self):
        """Rarely trigger reset (careful – deletes user data!)"""
        self.client.delete(
            "/api/knowledge-base/reset",
            name="/api/knowledge-base/reset"
        )