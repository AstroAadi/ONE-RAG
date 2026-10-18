"""
Vector Store Service - Qdrant with Open Source Embeddings
Uses HuggingFace API for embeddings instead of AWS Bedrock
"""

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, Filter, 
    FieldCondition, MatchValue, SparseVector,
    SparseVectorParams, SparseIndexParams
)
from typing import List, Dict, Any, Tuple, Optional, Literal
import logging
import uuid
from datetime import datetime
import hashlib

from app.core.config import settings
from app.services.embedding_service import OpenSourceEmbeddings
from app.core.exceptions import (
    VectorStoreException, 
    EmbeddingError, 
    SearchError
)

logger = logging.getLogger(__name__)


def string_to_uuid(text: str) -> str:
    """Convert any string to a valid UUID v5"""
    namespace = uuid.UUID('6ba7b810-9dad-11d1-80b4-00c04fd430c8')
    return str(uuid.uuid5(namespace, text))


class SPLADEEmbedder:
    """SPLADE sparse embedder (kept from original)"""
    
    def __init__(self, enabled: bool = settings.enable_sparse_embeddings):
        if not enabled:
            self.model = None
            self.available = False
            logger.info("SPLADE embeddings disabled; dense retrieval will be used")
            return

        try:
            from sentence_transformers import SparseEncoder
            model_name = "naver/splade-v3"
            logger.info(f"Loading SPLADE model: {model_name}...")
            self.model = SparseEncoder(model_name)
            logger.info(f"✅ SPLADE model loaded successfully")
            self.available = True
        except Exception as e:
            logger.error(f"Failed to load SPLADE model: {e}")
            logger.warning("SPLADE unavailable - sparse search will fall back to dense retrieval")
            self.model = None
            self.available = False
    
    def _sparse_matrix_to_qdrant(self, sparse_output) -> SparseVector:
        """Convert SPLADE output to Qdrant SparseVector"""
        try:
            import torch
            
            if isinstance(sparse_output, torch.Tensor):
                if sparse_output.is_sparse:
                    sparse_tensor = sparse_output.coalesce()
                    indices_tensor = sparse_tensor.indices()
                    values_tensor = sparse_tensor.values()
                    
                    if indices_tensor.dim() == 2:
                        indices_numpy = indices_tensor[-1].cpu().numpy()
                    else:
                        indices_numpy = indices_tensor.cpu().numpy()
                    
                    values_numpy = values_tensor.cpu().numpy()
                    
                    if len(indices_numpy) == 0 or len(values_numpy) == 0:
                        return SparseVector(indices=[0], values=[0.0])
                    
                    indices = [int(idx) for idx in indices_numpy]
                    values = [float(val) for val in values_numpy]
                    
                    return SparseVector(indices=indices, values=values)
                else:
                    if sparse_output.dim() > 1:
                        sparse_output = sparse_output.squeeze()
                    
                    numpy_vector = sparse_output.cpu().numpy()
                    nonzero_indices = numpy_vector.nonzero()[0]
                    nonzero_values = numpy_vector[nonzero_indices]
                    
                    if len(nonzero_indices) == 0:
                        return SparseVector(indices=[0], values=[0.0])
                    
                    indices = [int(idx) for idx in nonzero_indices]
                    values = [float(val) for val in nonzero_values]
                    
                    return SparseVector(indices=indices, values=values)
            else:
                import scipy.sparse
                if scipy.sparse.issparse(sparse_output):
                    coo = sparse_output.tocoo()
                    indices = [int(idx) for idx in coo.col.tolist()]
                    values = [float(val) for val in coo.data.tolist()]
                    
                    if len(indices) == 0:
                        return SparseVector(indices=[0], values=[0.0])
                    
                    return SparseVector(indices=indices, values=values)
                else:
                    return SparseVector(indices=[0], values=[0.0])
                
        except Exception as e:
            logger.error(f"Error converting sparse output: {e}")
            return SparseVector(indices=[0], values=[0.0])
    
    def embed_documents(self, texts: List[str]) -> List[SparseVector]:
        """Generate SPLADE sparse embeddings"""
        if not self.available:
            return [SparseVector(indices=[0], values=[0.0]) for _ in texts]
        
        try:
            sparse_matrices = self.model.encode(texts, convert_to_tensor=False)
            sparse_vectors = []
            
            for sparse_matrix in sparse_matrices:
                sparse_vector = self._sparse_matrix_to_qdrant(sparse_matrix)
                sparse_vectors.append(sparse_vector)
            
            return sparse_vectors
            
        except Exception as e:
            logger.error(f"SPLADE embedding error: {e}")
            return [SparseVector(indices=[0], values=[0.0]) for _ in texts]
    
    def embed_query(self, text: str) -> SparseVector:
        """Generate SPLADE sparse embedding for query"""
        if not self.available:
            return SparseVector(indices=[0], values=[0.0])
        
        try:
            sparse_matrices = self.model.encode([text], convert_to_tensor=False)
            
            if len(sparse_matrices) == 0:
                return SparseVector(indices=[0], values=[0.0])
            
            return self._sparse_matrix_to_qdrant(sparse_matrices[0])
                
        except Exception as e:
            logger.error(f"SPLADE query embedding error: {e}")
            return SparseVector(indices=[0], values=[0.0])


class VectorStoreService:
    """Qdrant-based vector store with open-source embeddings"""
    
    def __init__(self):
        """Initialize Qdrant client and embeddings"""
        try:
            # Initialize Qdrant Cloud client
            self.client = QdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key,
                timeout=60,
                prefer_grpc=False
            )
            
            # Initialize open-source dense embeddings (via API)
            self.dense_embeddings = OpenSourceEmbeddings()
            
            # Initialize sparse embeddings (LOCAL SPLADE)
            self.sparse_embeddings = SPLADEEmbedder(settings.enable_sparse_embeddings)
            
            # Collection name
            self.collection_name = settings.qdrant_collection_name
            
            # Initialize collection
            self._initialize_collection()
            
            logger.info(f"✅ Qdrant Vector Store initialized (collection: {self.collection_name})")
            logger.info(f"✅ Dense embeddings: {settings.embedding_model_id} (HuggingFace API)")
            logger.info(f"✅ Sparse embeddings: {'SPLADE (local)' if self.sparse_embeddings.available else 'Disabled'}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant: {e}", exc_info=True)
            raise VectorStoreException(f"Failed to initialize vector store: {str(e)}")
    
    def _initialize_collection(self):
        """
        Create the collection if missing, and reconcile its vector dimension
        with the active embedding model.

        A mismatch between the two makes every search fail with an opaque
        Qdrant "Vector dimension error", so it is resolved here: an empty
        collection is rebuilt at the correct size, and a populated one raises
        with instructions rather than failing later on every request.
        """
        try:
            collections = self.client.get_collections().collections
            collection_names = [col.name for col in collections]

            dense_dim = self.dense_embeddings._get_embedding_dimension()

            if self.collection_name in collection_names:
                existing_dim = self.get_dense_vector_size()

                if existing_dim is not None and existing_dim != dense_dim:
                    indexed = self.count_chunks()

                    if indexed > 0:
                        raise VectorStoreException(
                            f"Collection '{self.collection_name}' stores "
                            f"{existing_dim}-dimensional vectors but the active "
                            f"embedding model '{self.dense_embeddings.model_id}' "
                            f"produces {dense_dim}. The {indexed} indexed "
                            f"chunk(s) would be unsearchable. Either set "
                            f"EMBEDDING_MODEL_ID back to the model that "
                            f"produced {existing_dim}-dimensional vectors, or "
                            f"delete the collection and re-index."
                        )

                    logger.warning(
                        f"Collection '{self.collection_name}' has dimension "
                        f"{existing_dim} but the active model needs {dense_dim}. "
                        f"The collection is empty, so it will be rebuilt."
                    )
                    self.client.delete_collection(
                        collection_name=self.collection_name
                    )
                    collection_names.remove(self.collection_name)
                else:
                    logger.info(
                        f"✅ Using existing Qdrant collection: "
                        f"{self.collection_name} (dimension {existing_dim})"
                    )
                    # Backfill any indexes added after the collection was created
                    self._create_payload_indexes()
                    return

            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense": VectorParams(
                        size=dense_dim,
                        distance=Distance.COSINE
                    )
                },
                sparse_vectors_config={
                    "sparse": SparseVectorParams(
                        index=SparseIndexParams()
                    )
                }
            )

            self._create_payload_indexes()

            logger.info(f"✅ Created Qdrant collection: {self.collection_name}")
            logger.info(f"  - Dense vector size: {dense_dim}")

        except VectorStoreException:
            raise
        except Exception as e:
            logger.error(f"Error initializing collection: {e}", exc_info=True)
            raise VectorStoreException(f"Failed to initialize collection: {str(e)}")
    
    # Payload fields that are filtered on and therefore need an index
    _INDEXED_PAYLOAD_FIELDS = ("user_id", "file_id", "chunk_id", "knowledge_base_id")
    
    def _create_payload_indexes(self):
        """
        Create indexes on the payload fields used for filtering.
        
        Each index is created independently so that an already-existing index
        does not prevent the remaining ones from being created. This also lets
        the method run safely against a pre-existing collection.
        """
        created = []
        
        for field_name in self._INDEXED_PAYLOAD_FIELDS:
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field_name,
                    field_schema="keyword"
                )
                created.append(field_name)
            except Exception as e:
                # Most commonly "index already exists" - not an error
                logger.debug(f"Payload index for '{field_name}' not created: {e}")
        
        if created:
            logger.info(f"✅ Created payload indexes for: {', '.join(created)}")
    
    def add_document_chunks(
        self,
        chunks: List[Dict[str, Any]],
        file_id: str,
        filename: str,
        structured_data: Optional[Dict] = None
    ) -> int:
        """Add document chunks to Qdrant"""
        if not chunks:
            logger.warning(f"No chunks provided for {file_id}")
            return 0
        
        try:
            points = []
            texts_for_embedding = []
            
            for chunk in chunks:
                text = chunk.get('text', '')
                chunk_id = chunk.get('chunk_id', '')
                metadata = chunk.get('metadata', {})
                
                if not text or not chunk_id:
                    continue
                
                if 'user_id' not in metadata:
                    raise VectorStoreException(f"Cannot add chunk without user_id: {chunk_id}")
                
                payload = {
                    'file_id': str(metadata.get('file_id', file_id)),
                    'user_id': str(metadata.get('user_id')),
                    'knowledge_base_id': str(metadata.get('knowledge_base_id', '')),
                    'filename': str(metadata.get('filename', filename)),
                    'chunk_type': str(metadata.get('chunk_type', 'text')),
                    'page_number': int(metadata.get('page_number', 0)),
                    'chunk_index': int(chunk.get('chunk_index', metadata.get('chunk_index', 0))),
                    'chunk_id': str(chunk_id),
                    'text': text,
                    'created_at': datetime.utcnow().isoformat()
                }
                
                for key, value in metadata.items():
                    if key not in payload and value is not None:
                        payload[key] = str(value) if not isinstance(value, (int, float, bool)) else value
                
                texts_for_embedding.append(text)
                points.append({
                    'id': chunk_id,
                    'payload': payload
                })
            
            if not points:
                return 0
            
            # Truncate texts
            max_embedding_length = settings.embedding_max_length
            truncated_texts = []
            
            for text in texts_for_embedding:
                if len(text) > max_embedding_length:
                    truncated_texts.append(text[:max_embedding_length])
                else:
                    truncated_texts.append(text)
            
            # Generate dense embeddings via API
            logger.info(f"Generating dense embeddings for {len(truncated_texts)} chunks...")
            dense_embeddings = self.dense_embeddings.embed_documents(truncated_texts)
            
            # The embedding service falls back to a zero vector when a call
            # fails, which would index chunks that can never be retrieved.
            # Refuse the write instead of reporting a false success.
            degenerate = [
                index for index, embedding in enumerate(dense_embeddings)
                if not embedding or not any(embedding)
            ]
            if degenerate:
                raise VectorStoreException(
                    f"Embedding failed for {len(degenerate)} of "
                    f"{len(dense_embeddings)} chunks (the embedding service "
                    f"returned empty vectors). Nothing was indexed; resolve the "
                    f"embedding error and retry."
                )
            
            # Generate sparse embeddings
            logger.info(f"Generating sparse embeddings for {len(truncated_texts)} chunks...")
            sparse_embeddings = self.sparse_embeddings.embed_documents(truncated_texts)
            
            # Create Qdrant points
            qdrant_points = []
            for point, dense_emb, sparse_emb in zip(points, dense_embeddings, sparse_embeddings):
                point_uuid = string_to_uuid(point['id'])
                
                qdrant_points.append(
                    PointStruct(
                        id=point_uuid,
                        vector={
                            "dense": dense_emb,
                            "sparse": sparse_emb
                        },
                        payload=point['payload']
                    )
                )
            
            # Upload to Qdrant
            batch_size = 100
            for i in range(0, len(qdrant_points), batch_size):
                batch = qdrant_points[i:i + batch_size]
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch
                )
            
            logger.info(f"✅ Added {len(qdrant_points)} chunks to Qdrant")
            
            return len(qdrant_points)
            
        except Exception as e:
            logger.error(f"Error adding chunks: {str(e)}", exc_info=True)
            raise VectorStoreException(f"Failed to add document chunks: {str(e)}")
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        user_id: Optional[str] = None,
        file_id: Optional[str] = None,
        search_type: Literal["dense", "sparse", "hybrid"] = "hybrid",
        knowledge_base_id: Optional[str] = None
    ) -> Tuple[List[str], List[Dict[str, Any]], List[float]]:
        """Enhanced search method"""
        try:
            if search_type == "dense":
                results = self._dense_search(query, top_k, user_id, file_id, knowledge_base_id)
            elif search_type == "sparse":
                results = self._sparse_search(query, top_k, user_id, file_id, knowledge_base_id)
            else:
                results = self._hybrid_search(query, top_k, user_id, file_id, knowledge_base_id)
            
            documents, metadatas, scores = results
            
            # Ensure chunk_id and file_id are in metadata
            for i, meta in enumerate(metadatas):
                if 'chunk_id' not in meta:
                    meta['chunk_id'] = f"unknown_{hash(documents[i]) % 10000}"
                if 'file_id' not in meta:
                    meta['file_id'] = 'unknown'
            
            logger.info(f"Search completed: type={search_type}, results={len(documents)}")
            
            return documents, metadatas, scores
            
        except Exception as e:
            logger.error(f"Search error: {e}", exc_info=True)
            raise SearchError(f"Search failed: {str(e)}")
    
    def _dense_search(
        self,
        query: str,
        top_k: int,
        user_id: Optional[str] = None,
        file_id: Optional[str] = None,
        knowledge_base_id: Optional[str] = None
    ) -> Tuple[List[str], List[Dict[str, Any]], List[float]]:
        """Dense vector search"""
        try:
            query_embedding = self.dense_embeddings.embed_query(query)
            query_filter = self._build_filter(user_id, file_id, knowledge_base_id)
            
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_embedding,
                using="dense",
                query_filter=query_filter,
                limit=top_k,
                with_payload=True
            )
            
            return self._process_results(results.points)
            
        except Exception as e:
            logger.error(f"Dense search error: {e}")
            raise SearchError(f"Dense search failed: {str(e)}")
    
    def _sparse_search(
        self,
        query: str,
        top_k: int,
        user_id: Optional[str] = None,
        file_id: Optional[str] = None,
        knowledge_base_id: Optional[str] = None
    ) -> Tuple[List[str], List[Dict[str, Any]], List[float]]:
        """Sparse vector search"""
        try:
            if not self.sparse_embeddings.available:
                logger.warning("Sparse search not available, using dense")
                return self._dense_search(query, top_k, user_id, file_id, knowledge_base_id)
            
            query_sparse = self.sparse_embeddings.embed_query(query)
            query_filter = self._build_filter(user_id, file_id, knowledge_base_id)
            
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_sparse,
                using="sparse",
                query_filter=query_filter,
                limit=top_k,
                with_payload=True
            )
            
            return self._process_results(results.points)
            
        except Exception as e:
            logger.error(f"Sparse search error: {e}")
            raise SearchError(f"Sparse search failed: {str(e)}")
    
    def _hybrid_search(
        self,
        query: str,
        top_k: int,
        user_id: Optional[str] = None,
        file_id: Optional[str] = None,
        knowledge_base_id: Optional[str] = None
    ) -> Tuple[List[str], List[Dict[str, Any]], List[float]]:
        """Hybrid search using RRF"""
        try:
            if not self.sparse_embeddings.available:
                return self._dense_search(query, top_k, user_id, file_id, knowledge_base_id)
            
            retrieval_k = min(top_k * 3, 50)
            
            dense_docs, dense_metas, dense_scores = self._dense_search(
                query, retrieval_k, user_id, file_id, knowledge_base_id
            )
            
            sparse_docs, sparse_metas, sparse_scores = self._sparse_search(
                query, retrieval_k, user_id, file_id, knowledge_base_id
            )
            
            fused_results = self._reciprocal_rank_fusion(
                dense_results=list(zip(dense_docs, dense_metas, dense_scores)),
                sparse_results=list(zip(sparse_docs, sparse_metas, sparse_scores)),
                k=60
            )
            
            fused_results = fused_results[:top_k]
            
            if not fused_results:
                return [], [], []
            
            documents = [r[0] for r in fused_results]
            metadatas = [r[1] for r in fused_results]
            scores = [r[2] for r in fused_results]
            
            return documents, metadatas, scores
            
        except Exception as e:
            logger.error(f"Hybrid search error: {e}")
            raise SearchError(f"Hybrid search failed: {str(e)}")
    
    def _reciprocal_rank_fusion(
        self,
        dense_results: List[Tuple[str, Dict, float]],
        sparse_results: List[Tuple[str, Dict, float]],
        k: int = 60
    ) -> List[Tuple[str, Dict, float]]:
        """Reciprocal Rank Fusion algorithm"""
        rrf_scores = {}
        
        for rank, (doc, meta, score) in enumerate(dense_results, start=1):
            chunk_id = meta.get('chunk_id', '')
            if chunk_id not in rrf_scores:
                rrf_scores[chunk_id] = {
                    'doc': doc,
                    'meta': meta,
                    'rrf_score': 0.0
                }
            rrf_scores[chunk_id]['rrf_score'] += 1.0 / (k + rank)
        
        for rank, (doc, meta, score) in enumerate(sparse_results, start=1):
            chunk_id = meta.get('chunk_id', '')
            if chunk_id not in rrf_scores:
                rrf_scores[chunk_id] = {
                    'doc': doc,
                    'meta': meta,
                    'rrf_score': 0.0
                }
            rrf_scores[chunk_id]['rrf_score'] += 1.0 / (k + rank)
        
        sorted_results = sorted(
            rrf_scores.values(),
            key=lambda x: x['rrf_score'],
            reverse=True
        )
        
        fused = []
        for item in sorted_results:
            normalized_score = min(item['rrf_score'] / 0.05, 1.0)
            fused.append((item['doc'], item['meta'], normalized_score))
        
        return fused
    
    def _build_filter(
        self,
        user_id: Optional[str] = None,
        file_id: Optional[str] = None,
        knowledge_base_id: Optional[str] = None,
        chunk_id: Optional[str] = None
    ) -> Optional[Filter]:
        """Build Qdrant filter"""
        must_conditions = []
        
        if user_id:
            must_conditions.append(
                FieldCondition(key="user_id", match=MatchValue(value=user_id))
            )
        
        if file_id:
            must_conditions.append(
                FieldCondition(key="file_id", match=MatchValue(value=file_id))
            )
        
        if knowledge_base_id:
            must_conditions.append(
                FieldCondition(
                    key="knowledge_base_id",
                    match=MatchValue(value=knowledge_base_id)
                )
            )
        
        if chunk_id:
            must_conditions.append(
                FieldCondition(key="chunk_id", match=MatchValue(value=chunk_id))
            )
        
        if must_conditions:
            return Filter(must=must_conditions)
        
        return None
    
    def _process_results(self, results) -> Tuple[List[str], List[Dict[str, Any]], List[float]]:
        """Process Qdrant search results"""
        documents = []
        metadatas = []
        scores = []
        
        for result in results:
            payload = result.payload
            
            text = payload.get('text', '')
            documents.append(text)
            
            metadata = {k: v for k, v in payload.items() if k != 'text'}
            metadatas.append(metadata)
            
            scores.append(float(result.score))
        
        return documents, metadatas, scores
    
    # Include other methods: delete_document_by_file_id, get_collection_stats, etc.
    # (Same as original, just without AWS dependencies)
    
    def delete_document_by_file_id(self, file_id: str, user_id: Optional[str] = None) -> bool:
        """Delete all chunks for a file"""
        try:
            must_conditions = [
                FieldCondition(key="file_id", match=MatchValue(value=file_id))
            ]
            if user_id:
                must_conditions.append(
                    FieldCondition(key="user_id", match=MatchValue(value=user_id))
                )
            
            query_filter = Filter(must=must_conditions)
            
            result = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=query_filter,
                limit=10000,
                with_payload=False,
                with_vectors=False
            )
            
            points_to_delete = result[0]
            
            if not points_to_delete:
                return False
            
            point_ids = [point.id for point in points_to_delete]
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=point_ids
            )
            
            logger.info(f"Deleted {len(point_ids)} chunks for file {file_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting document: {str(e)}")
            raise VectorStoreException(f"Failed to delete document: {str(e)}")
    
    def get_dense_vector_size(self) -> Optional[int]:
        """
        The dense vector dimension the collection was created with.

        Returns None if it cannot be determined. Used to reject an embedding
        model switch that would produce vectors the collection cannot store.
        """
        try:
            info = self.client.get_collection(collection_name=self.collection_name)
            vectors = info.config.params.vectors

            # Named vector configs come back as a dict
            if isinstance(vectors, dict):
                dense = vectors.get("dense")
                return int(dense.size) if dense is not None else None

            return int(vectors.size) if vectors is not None else None

        except Exception as e:
            logger.warning(f"Could not read collection vector size: {e}")
            return None

    def count_chunks(self, user_id: Optional[str] = None) -> int:
        """Number of indexed chunks, optionally for a single user."""
        try:
            result = self.client.count(
                collection_name=self.collection_name,
                count_filter=self._build_filter(user_id=user_id),
                exact=True,
            )
            return int(result.count)
        except Exception as e:
            logger.warning(f"Could not count chunks: {e}")
            return 0

    def delete_by_knowledge_base(
        self,
        knowledge_base_id: str,
        user_id: Optional[str] = None
    ) -> int:
        """
        Delete every chunk belonging to a knowledge base.

        Returns the number of deleted points (0 if the KB had no chunks).
        """
        try:
            query_filter = self._build_filter(
                user_id=user_id,
                knowledge_base_id=knowledge_base_id
            )

            deleted = 0
            while True:
                points, _ = self.client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=query_filter,
                    limit=1000,
                    with_payload=False,
                    with_vectors=False
                )

                if not points:
                    break

                self.client.delete(
                    collection_name=self.collection_name,
                    points_selector=[point.id for point in points]
                )
                deleted += len(points)

            logger.info(
                f"Deleted {deleted} chunks for knowledge base {knowledge_base_id}"
            )
            return deleted

        except Exception as e:
            logger.error(f"Error deleting knowledge base chunks: {str(e)}")
            raise VectorStoreException(
                f"Failed to delete knowledge base chunks: {str(e)}"
            )

    def get_chunk_by_id(
        self,
        chunk_id: str,
        user_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve a single chunk with its full text and metadata.

        The point ID in Qdrant is a UUID derived from the chunk_id, so the
        lookup is a direct retrieve rather than a scan. The user_id is still
        verified against the payload so one user cannot read another's chunk.

        Returns None when the chunk does not exist or is not owned by user_id.
        """
        try:
            point_uuid = string_to_uuid(chunk_id)

            points = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[point_uuid],
                with_payload=True,
                with_vectors=False
            )

            if not points:
                return None

            payload = points[0].payload or {}

            # Enforce user isolation
            if user_id and str(payload.get('user_id', '')) != str(user_id):
                logger.warning(
                    f"Access denied for chunk {chunk_id}: owner mismatch"
                )
                return None

            text = payload.get('text', '')

            reserved = {
                'text', 'chunk_id', 'file_id', 'user_id', 'filename',
                'chunk_type', 'page_number', 'chunk_index', 'created_at',
                'updated_at', 'edited'
            }

            return {
                'chunk_id': str(payload.get('chunk_id', chunk_id)),
                'file_id': str(payload.get('file_id', 'unknown')),
                'text': text,
                'filename': str(payload.get('filename', 'unknown')),
                'user_id': str(payload.get('user_id', user_id or 'unknown')),
                'chunk_type': str(payload.get('chunk_type', 'text')),
                'page_number': int(payload.get('page_number', 0) or 0),
                'char_count': len(text),
                'chunk_index': int(payload.get('chunk_index', 0) or 0),
                'created_at': str(payload.get('created_at', '')),
                'updated_at': payload.get('updated_at'),
                'edited': bool(payload.get('edited', False)),
                'metadata': {
                    key: value for key, value in payload.items()
                    if key not in reserved
                }
            }

        except Exception as e:
            logger.error(f"Error retrieving chunk {chunk_id}: {str(e)}")
            raise VectorStoreException(f"Failed to retrieve chunk: {str(e)}")

    def update_chunk(
        self,
        chunk_id: str,
        file_id: str,
        new_text: str,
        user_id: str,
        preserve_metadata: bool = True
    ) -> bool:
        """
        Replace a chunk's text and re-embed it in place.

        The point keeps its ID, so the chunk stays addressable and all
        references to it remain valid. Both dense and sparse vectors are
        regenerated so search stays consistent with the new text.
        """
        if not new_text or not new_text.strip():
            raise VectorStoreException("Cannot update a chunk to empty text")

        try:
            existing = self.get_chunk_by_id(chunk_id=chunk_id, user_id=user_id)

            if not existing:
                logger.warning(f"Chunk not found for update: {chunk_id}")
                return False

            if existing['file_id'] != file_id:
                raise VectorStoreException(
                    f"Chunk {chunk_id} belongs to file {existing['file_id']}, "
                    f"not {file_id}"
                )

            point_uuid = string_to_uuid(chunk_id)

            # Rebuild the payload
            if preserve_metadata:
                payload = dict(existing['metadata'])
            else:
                payload = {}

            payload.update({
                'chunk_id': chunk_id,
                'file_id': file_id,
                'user_id': user_id,
                'filename': existing['filename'],
                'chunk_type': existing['chunk_type'],
                'page_number': existing['page_number'],
                'chunk_index': existing['chunk_index'],
                'created_at': existing['created_at'] or datetime.utcnow().isoformat(),
                'updated_at': datetime.utcnow().isoformat(),
                'edited': True,
                'text': new_text
            })

            # Re-embed, truncating to the embedding model's limit
            truncated = new_text[:settings.embedding_max_length]

            dense_embedding = self.dense_embeddings.embed_documents([truncated])[0]

            # A zero vector means the embedding call failed; overwriting the
            # stored vector with it would silently make the chunk unretrievable.
            if not dense_embedding or not any(dense_embedding):
                raise VectorStoreException(
                    "Embedding failed for the edited text (the embedding "
                    "service returned an empty vector). The chunk was left "
                    "unchanged; resolve the embedding error and retry."
                )

            sparse_embedding = self.sparse_embeddings.embed_documents([truncated])[0]

            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    PointStruct(
                        id=point_uuid,
                        vector={
                            "dense": dense_embedding,
                            "sparse": sparse_embedding
                        },
                        payload=payload
                    )
                ]
            )

            logger.info(
                f"✅ Updated chunk {chunk_id} "
                f"({existing['char_count']} -> {len(new_text)} chars)"
            )
            return True

        except VectorStoreException:
            raise
        except Exception as e:
            logger.error(f"Error updating chunk {chunk_id}: {str(e)}", exc_info=True)
            raise VectorStoreException(f"Failed to update chunk: {str(e)}")

    def get_collection_stats(
        self,
        user_id: Optional[str] = None,
        knowledge_base_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get collection statistics, optionally scoped to one knowledge base"""
        try:
            query_filter = self._build_filter(
                user_id=user_id,
                knowledge_base_id=knowledge_base_id
            )
            
            all_points = []
            offset = None
            
            while True:
                result = self.client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=query_filter,
                    limit=1000,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False
                )
                
                points, next_offset = result
                all_points.extend(points)
                
                if next_offset is None:
                    break
                offset = next_offset
            
            if not all_points:
                return {
                    'total_documents': 0,
                    'total_chunks': 0,
                    'documents': []
                }
            
            file_ids = {}
            for point in all_points:
                fid = point.payload.get('file_id', 'unknown')
                if fid not in file_ids:
                    file_ids[fid] = {
                        'file_id': fid,
                        'filename': point.payload.get('filename', 'unknown'),
                        'chunk_count': 0
                    }
                file_ids[fid]['chunk_count'] += 1
            
            return {
                'total_documents': len(file_ids),
                'total_chunks': len(all_points),
                'documents': list(file_ids.values()),
                'user_id': user_id
            }
            
        except Exception as e:
            logger.error(f"Error getting stats: {str(e)}")
            return {
                'total_documents': 0,
                'total_chunks': 0,
                'documents': [],
                'error': str(e)
            }