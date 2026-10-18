"""
Embedding Service

Dense embeddings with two interchangeable backends:

  "api"   - HuggingFace Inference Providers (hosted, requires credits)
  "local" - sentence-transformers running in-process (free, downloads the
            model once, then works offline)
  "auto"  - try the API, and permanently switch to local if the API rejects
            the request for a reason retrying cannot fix (no credits, bad
            token, model not served)

Both backends run the same model id, so vectors stay comparable and switching
backends does NOT require re-indexing.

A failed embedding never returns a placeholder vector: a zero vector would be
silently unretrievable, so errors are raised to the caller instead.
"""

from langchain_huggingface import HuggingFaceEndpointEmbeddings
from huggingface_hub import InferenceClient
import logging
import threading
from typing import List, Optional
import time
import numpy as np

from app.core.config import settings

logger = logging.getLogger(__name__)


# HTTP statuses that will never succeed on retry with the same request.
_NON_RETRYABLE_STATUS = {400, 401, 402, 403, 404, 422}

# Known output dimensions, used only as a fallback when the model is not
# loaded locally (the local model reports its own dimension).
_KNOWN_DIMENSIONS = {
    "BAAI/bge-large-en-v1.5": 1024,
    "BAAI/bge-base-en-v1.5": 768,
    "BAAI/bge-small-en-v1.5": 384,
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "sentence-transformers/all-mpnet-base-v2": 768,
    "sentence-transformers/paraphrase-mpnet-base-v2": 768,
    "intfloat/e5-large-v2": 1024,
}


class EmbeddingBackendUnavailable(Exception):
    """Raised when no embedding backend can serve a request."""


def _status_code(error: Exception) -> Optional[int]:
    """Best-effort extraction of an HTTP status code from a client error."""
    response = getattr(error, "response", None)
    if response is not None and getattr(response, "status_code", None):
        return int(response.status_code)

    status = getattr(error, "status_code", None)
    if status:
        return int(status)

    # huggingface_hub / httpx often only put the code in the message
    message = str(error)
    for code in _NON_RETRYABLE_STATUS | {429, 500, 503}:
        if f"'{code} " in message or f" {code} " in message:
            return code

    return None


def _is_non_retryable(error: Exception) -> bool:
    return _status_code(error) in _NON_RETRYABLE_STATUS


class OpenSourceEmbeddings:
    """Dense embeddings with an API backend and a local fallback."""

    # The local model is expensive to load and identical for every instance,
    # so it is shared process-wide.
    _local_model = None
    _local_model_id: Optional[str] = None
    _local_lock = threading.Lock()

    def __init__(self, model_id: str = None):
        """
        Initialize embeddings service.

        When `model_id` is omitted the active model is read from settings on
        every call, so changing it at runtime (via /api/settings/models/
        embedding) takes effect in every existing instance rather than only in
        newly constructed ones.
        """
        self._model_id_override = model_id
        self.api_token = settings.huggingface_api_token
        self.backend = settings.embedding_backend

        self.client = InferenceClient(token=self.api_token)

        # Kept for compatibility with callers that expect a LangChain object
        try:
            self.langchain_embeddings = HuggingFaceEndpointEmbeddings(
                model=self.model_id,
                huggingfacehub_api_token=self.api_token
            )
        except Exception as e:
            logger.warning(
                f"LangChain embeddings init failed, using InferenceClient only: {e}"
            )
            self.langchain_embeddings = None

        self.max_retries = settings.max_retries
        self.retry_delay = settings.retry_delay_seconds

        # Set once the API proves unusable, so we stop paying the retry cost
        # on every subsequent call.
        self._api_disabled_reason: Optional[str] = None

        if self.backend == "local":
            # Load eagerly so a misconfiguration surfaces at startup
            self._load_local_model()
            logger.info(f"✅ Embedding service initialized (local): {self.model_id}")
        else:
            logger.info(
                f"✅ Embedding service initialized ({self.backend} backend): "
                f"{self.model_id}"
            )

    @property
    def model_id(self) -> str:
        """The model this instance embeds with (follows settings by default)."""
        return self._model_id_override or settings.embedding_model_id

    # ==================== Backend selection ====================

    def _load_local_model(self):
        """Load (once) the sentence-transformers model for in-process embedding."""
        with self._local_lock:
            if (
                OpenSourceEmbeddings._local_model is not None
                and OpenSourceEmbeddings._local_model_id == self.model_id
            ):
                return OpenSourceEmbeddings._local_model

            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:
                raise EmbeddingBackendUnavailable(
                    "Local embeddings need sentence-transformers installed "
                    "(pip install sentence-transformers)"
                ) from e

            logger.info(
                f"Loading local embedding model '{self.model_id}' "
                f"(first run downloads it, this can take a few minutes)..."
            )

            try:
                model = SentenceTransformer(
                    self.model_id,
                    cache_folder=str(settings.model_cache_dir),
                )
            except Exception as e:
                raise EmbeddingBackendUnavailable(
                    f"Could not load local embedding model "
                    f"'{self.model_id}': {e}"
                ) from e

            OpenSourceEmbeddings._local_model = model
            OpenSourceEmbeddings._local_model_id = self.model_id

            logger.info(
                f"✅ Local embedding model ready: {self.model_id} "
                f"(dimension {model.get_sentence_embedding_dimension()})"
            )

            return model

    def _disable_api(self, reason: str):
        """Record that the hosted API cannot serve requests."""
        if self._api_disabled_reason:
            return

        self._api_disabled_reason = reason

        if self.backend == "auto":
            logger.warning(
                f"HuggingFace Inference API is unusable ({reason}). "
                f"Falling back to local embeddings for '{self.model_id}'."
            )
        else:
            logger.error(f"HuggingFace Inference API is unusable ({reason}).")

    @property
    def _use_local(self) -> bool:
        if self.backend == "local":
            return True
        if self.backend == "api":
            return False
        return self._api_disabled_reason is not None

    # ==================== Public API ====================

    def embed_query(self, text: str) -> List[float]:
        """Generate an embedding for a single query string."""
        return self._embed_batch([text])[0]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for several texts.

        Raises on failure rather than substituting zero vectors, which would
        index chunks that can never be retrieved.
        """
        if not texts:
            return []

        return self._embed_batch(texts)

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed texts with the active backend, falling back when allowed."""
        truncated = [
            text[:settings.embedding_max_length] if len(text) > settings.embedding_max_length
            else text
            for text in texts
        ]

        if self._use_local:
            return self._embed_local(truncated)

        try:
            return self._embed_api(truncated)
        except Exception as api_error:
            # On "auto", any API failure is recoverable by running the same
            # model locally. Retry the whole batch so a partially embedded
            # batch is never mixed across backends.
            if self.backend == "auto":
                self._disable_api(str(api_error)[:120])
                logger.info("Retrying this batch with local embeddings...")
                try:
                    return self._embed_local(truncated)
                except Exception as local_error:
                    raise EmbeddingBackendUnavailable(
                        f"Both embedding backends failed. "
                        f"API: {api_error}. Local: {local_error}"
                    ) from local_error

            raise

    # ==================== Local backend ====================

    def _embed_local(self, texts: List[str]) -> List[List[float]]:
        """Embed in-process with sentence-transformers."""
        model = self._load_local_model()

        vectors = model.encode(
            texts,
            batch_size=16,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return [np.asarray(vector).flatten().tolist() for vector in vectors]

    # ==================== Hosted API backend ====================

    def _embed_api(self, texts: List[str]) -> List[List[float]]:
        """Embed via the HuggingFace Inference API, one text at a time."""
        embeddings = []

        for index, text in enumerate(texts):
            embeddings.append(self._embed_with_client(text))

            if (index + 1) % 10 == 0:
                logger.debug(f"Embedded {index + 1}/{len(texts)} documents")

            # Be gentle with rate limits
            if index < len(texts) - 1:
                time.sleep(0.1)

        return embeddings

    def _embed_with_client(self, text: str, retry_count: int = 0) -> List[float]:
        """Generate a single embedding using InferenceClient."""
        try:
            result = self.client.feature_extraction(
                text=text,
                model=self.model_id
            )

            return self._normalize_api_result(result)

        except Exception as e:
            # A 402/401/403 will never succeed on retry, so fail immediately
            # instead of burning the full retry budget on every chunk.
            if _is_non_retryable(e):
                status = _status_code(e)
                self._disable_api(f"HTTP {status}")
                raise EmbeddingBackendUnavailable(
                    f"Embedding API rejected the request with HTTP {status}: {e}"
                ) from e

            if retry_count < self.max_retries:
                logger.warning(
                    f"Embedding failed, retrying "
                    f"({retry_count + 1}/{self.max_retries})..."
                )
                time.sleep(self.retry_delay * (retry_count + 1))
                return self._embed_with_client(text, retry_count + 1)

            self._disable_api(str(e)[:120])
            raise EmbeddingBackendUnavailable(
                f"Embedding failed after {self.max_retries} retries: {e}"
            ) from e

    @staticmethod
    def _normalize_api_result(result) -> List[float]:
        """Flatten the various shapes feature_extraction can return."""
        if isinstance(result, np.ndarray):
            return result.flatten().tolist()

        if isinstance(result, list):
            if result and isinstance(result[0], (list, np.ndarray)):
                first = result[0]
                if isinstance(first, np.ndarray):
                    return first.flatten().tolist()
                return list(first)
            return list(result)

        raise ValueError(f"Unexpected embedding format: {type(result)}")

    # ==================== Introspection ====================

    def _get_embedding_dimension(self) -> int:
        """
        Output dimension of the current model.

        Prefers the loaded local model, which is authoritative, and otherwise
        falls back to the known-models table.
        """
        if (
            OpenSourceEmbeddings._local_model is not None
            and OpenSourceEmbeddings._local_model_id == self.model_id
        ):
            dimension = OpenSourceEmbeddings._local_model.get_sentence_embedding_dimension()
            if dimension:
                return int(dimension)

        if self.model_id in _KNOWN_DIMENSIONS:
            return _KNOWN_DIMENSIONS[self.model_id]

        # Unknown model: determine the dimension empirically.
        try:
            return len(self.embed_query("dimension probe"))
        except Exception as e:
            logger.warning(
                f"Could not determine embedding dimension for "
                f"'{self.model_id}', defaulting to 768: {e}"
            )
            return 768

    def is_available(self) -> tuple[bool, str]:
        """
        Report whether embeddings can currently be produced.

        Used by health checks and by the indexing endpoint so it can return an
        actionable error instead of a generic failure.
        """
        try:
            self.embed_query("availability probe")
            backend = "local" if self._use_local else "api"
            return True, f"Embeddings available via {backend} backend"
        except Exception as e:
            return False, str(e)

    def update_model(self, new_model_id: str) -> bool:
        """
        Switch this instance to a different embedding model.

        Note that instances created without an explicit model_id follow
        `settings.embedding_model_id`, so the usual way to switch models
        globally is to update that setting.
        """
        try:
            previous = self.model_id
            self._model_id_override = new_model_id

            self.client = InferenceClient(token=self.api_token)
            self._api_disabled_reason = None

            try:
                self.langchain_embeddings = HuggingFaceEndpointEmbeddings(
                    model=self.model_id,
                    huggingfacehub_api_token=self.api_token
                )
            except Exception as e:
                logger.warning(f"LangChain embeddings reinit failed: {e}")
                self.langchain_embeddings = None

            # Drop the cached local model so the next call loads the new one
            with self._local_lock:
                if OpenSourceEmbeddings._local_model_id != new_model_id:
                    OpenSourceEmbeddings._local_model = None
                    OpenSourceEmbeddings._local_model_id = None

            logger.info(
                f"✅ Embedding model updated: {previous} -> {new_model_id}"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to update embedding model: {e}")
            return False
