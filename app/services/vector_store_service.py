"""
Vector Store Service - CORRECTED VERSION
Dense embeddings: AWS Bedrock
Sparse embeddings: HuggingFace SPLADE model only
"""

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, Filter, 
    FieldCondition, MatchValue, SparseVector, NamedVector,
    SparseVectorParams, SparseIndexParams
)
from typing import List, Dict, Any, Tuple, Optional, Literal, Union
import logging
import uuid
from datetime import datetime
import hashlib
from concurrent.futures import ThreadPoolExecutor
import time

from app.core.config import settings
from app.core.exceptions import (
    VectorStoreException, 
    EmbeddingError, 
    SearchError,
    NoDocumentsFoundError
)
from app.services.embedding_service import BedrockEmbeddingService
from app.services.splade_service import GPUSPLADEEmbedder

logger = logging.getLogger(__name__)


def string_to_uuid(text: str) -> str:
    """Convert any string to a valid UUID v5"""
    namespace = uuid.UUID('6ba7b810-9dad-11d1-80b4-00c04fd430c8')
    return str(uuid.uuid5(namespace, text))


# ==================== EMBEDDING CACHE ====================

class EmbeddingCache:
    """LRU cache for embeddings to reduce API calls"""
    
    def __init__(self, max_size: int = 10000, ttl_seconds: int = 3600):
        self.cache = {}
        self.max_size = max_size
        self.ttl = ttl_seconds
        self.access_count = {}
    
    def _make_key(self, text: str, embedding_type: str) -> str:
        text_sample = text[:500] if len(text) > 500 else text
        text_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
        return f"{embedding_type}:{text_hash}:{text_sample}"
    
    def get(self, text: str, embedding_type: str = 'dense'):
        key = self._make_key(text, embedding_type)
        
        if key in self.cache:
            embedding, timestamp = self.cache[key]
            
            if time.time() - timestamp > self.ttl:
                del self.cache[key]
                return None
            
            self.access_count[key] = self.access_count.get(key, 0) + 1
            logger.debug(f"✓ Embedding cache hit: {embedding_type}")
            return embedding
        
        return None
    
    def set(self, text: str, embedding: Union[List[float], SparseVector], embedding_type: str = 'dense'):
        key = self._make_key(text, embedding_type)
        
        if len(self.cache) >= self.max_size:
            self._evict_lru()
        
        self.cache[key] = (embedding, time.time())
        self.access_count[key] = 1
    
    def _evict_lru(self):
        if not self.access_count:
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][1])
        else:
            oldest_key = min(self.access_count.keys(), key=lambda k: self.access_count[k])
        
        del self.cache[oldest_key]
        if oldest_key in self.access_count:
            del self.access_count[oldest_key]
    
    def clear(self):
        self.cache.clear()
        self.access_count.clear()


# ==================== VECTOR STORE SERVICE ====================

class VectorStoreService:
    """
    Multi-tenant Vector Store Service
    
    Dense Embeddings: AWS Bedrock (Cohere, Titan, etc.)
    Sparse Embeddings: HuggingFace SPLADE
    
    Features:
    - Knowledge Base isolation
    - User isolation
    - Hybrid search (dense + sparse)
    - Caching
    - Lazy initialization with graceful degradation
    """
    
    def __init__(self, embedding_model_id: Optional[str] = None):
        """Initialize Qdrant with Bedrock dense + SPLADE sparse
        
        Args:
            embedding_model_id: Optional AWS Bedrock embedding model ID.
                              If not provided, uses default from settings.
        """
        try:
            logger.info("Attempting to connect to Qdrant...")
            self.client = QdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key,
                timeout=30,
                prefer_grpc=False
            )
            
            # Use AWS Bedrock for dense embeddings with specified or default model
            self.embedding_model_id = embedding_model_id or settings.bedrock_embedding_model_id
            self.dense_embeddings = BedrockEmbeddingService(model_id=self.embedding_model_id)
            
            # Use HuggingFace SPLADE for sparse embeddings
            try:
                self.sparse_embeddings = GPUSPLADEEmbedder(
                    model_name=settings.splade_model_name
                )
                logger.info("✓ SPLADE sparse embeddings initialized")
            except Exception as e:
                logger.warning(f"SPLADE initialization failed: {e}")
                logger.warning("Continuing with dense-only search")
                self.sparse_embeddings = None
            
            self.embedding_cache = EmbeddingCache(max_size=10000, ttl_seconds=3600)
            
            # Try to initialize collection
            try:
                self._initialize_collection()
            except Exception as e:
                logger.warning(f"Collection initialization deferred: {e}")
                logger.info("Vector store will attempt lazy initialization on first use")
                return
            
            self._initialized = True
            logger.info(f"✅ Multi-tenant Vector Store initialized")
            logger.info(f"  - Dense: AWS Bedrock ({self.embedding_model_id})")
            logger.info(f"  - Sparse: {'HuggingFace SPLADE' if self.sparse_embeddings else 'DISABLED'}")
            logger.info(f"  - KB isolation: ENABLED")
            
        except Exception as e:
            self._init_error = str(e)
            logger.warning(f"⚠️  Qdrant initialization deferred: {e}")
            logger.info("Vector store will attempt lazy initialization on first use")
    
    def _ensure_initialized(self):
        """Ensure the vector store is initialized before operations"""
        if self._initialized and self.client:
            return
        
        self._try_initialize()
        
        if not self._initialized or not self.client:
            raise VectorStoreException(
                f"Vector store not available: {self._init_error or 'Unknown error'}"
            )
    
    def _initialize_collection(self):
        """Create collection with dense + sparse vectors"""
        try:
            collections = self.client.get_collections().collections
            collection_names = [col.name for col in collections]
            
            if self.collection_name not in collection_names:
                dense_dim = self.dense_embeddings.get_embedding_dimension()
                
                vectors_config = {
                    "dense": VectorParams(size=dense_dim, distance=Distance.COSINE)
                }
                
                sparse_config = None
                if self.sparse_embeddings:
                    sparse_config = {
                        "sparse": SparseVectorParams(index=SparseIndexParams())
                    }
                
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=vectors_config,
                    sparse_vectors_config=sparse_config
                )
                
                self._create_payload_indexes()
                
                logger.info(f"✓ Created collection: {self.collection_name}")
            else:
                logger.info(f"✓ Using existing collection: {self.collection_name}")
                
        except Exception as e:
            logger.error(f"Error initializing collection: {e}", exc_info=True)
            raise VectorStoreException(f"Failed to initialize collection: {str(e)}")
    
    def _create_payload_indexes(self):
        """Create indexes for multi-tenant queries"""
        try:
            # Index user_id (critical for isolation)
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="user_id",
                field_schema="keyword"
            )
            
            # Index knowledge_base_id (critical for KB queries)
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="knowledge_base_id",
                field_schema="keyword"
            )
            
            # Index file_id (for file-specific queries)
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="file_id",
                field_schema="keyword"
            )
            
            # Index chunk_id (for direct retrieval)
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="chunk_id",
                field_schema="keyword"
            )
            
            logger.info("✓ Created payload indexes (user_id, knowledge_base_id, file_id, chunk_id)")
            
        except Exception as e:
            logger.warning(f"Could not create payload indexes: {e}")
    
    def _get_cached_embedding(
        self, 
        text: str, 
        embedding_type: Literal['dense', 'sparse'] = 'dense'
    ) -> Union[List[float], SparseVector]:
        """Get embedding with caching"""
        cached = self.embedding_cache.get(text, embedding_type)
        if cached is not None:
            return cached
        
        if embedding_type == 'dense':
            # Use AWS Bedrock
            embedding = self.dense_embeddings.embed_query(text)
        else:
            # Use HuggingFace SPLADE
            if self.sparse_embeddings:
                embedding = self.sparse_embeddings.embed_query(text)
            else:
                embedding = SparseVector(indices=[0], values=[0.0])
        
        self.embedding_cache.set(text, embedding, embedding_type)
        return embedding
    
    def add_document_chunks(
        self,
        chunks: List[Dict[str, Any]],
        file_id: str,
        filename: str,
        knowledge_base_id: str,
        knowledge_base_name: str,
        user_id: str,
        structured_data: Optional[Dict] = None
    ) -> int:
        """
        Add document chunks to Qdrant with KB metadata
        
        Uses:
        - AWS Bedrock for dense embeddings
        - HuggingFace SPLADE for sparse embeddings
        """
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
                    logger.warning(f"Skipping invalid chunk: {chunk_id}")
                    continue
                
                # Build payload with KB isolation
                payload = {
                    # Multi-tenant fields (REQUIRED)
                    'user_id': str(user_id),
                    'knowledge_base_id': str(knowledge_base_id),
                    'knowledge_base_name': str(knowledge_base_name),
                    
                    # File identification
                    'file_id': str(file_id),
                    'filename': str(filename),
                    
                    # Chunk data
                    'chunk_id': str(chunk_id),
                    'text': text,
                    'chunk_type': str(metadata.get('chunk_type', 'text')),
                    'chunk_index': int(chunk.get('chunk_index', metadata.get('chunk_index', 0))),
                    'page_number': int(metadata.get('page_number', 0)),
                    
                    # Timestamps
                    'created_at': datetime.utcnow().isoformat()
                }
                
                # Add any additional metadata
                for key, value in metadata.items():
                    if key not in payload and value is not None:
                        payload[key] = str(value) if not isinstance(value, (int, float, bool)) else value
                
                texts_for_embedding.append(text)
                points.append({'id': chunk_id, 'payload': payload})
            
            if not points:
                logger.warning(f"No valid chunks for {file_id}")
                return 0
            
            # Truncate texts
            max_len = settings.embedding_max_length
            truncated_texts = [text[:max_len] if len(text) > max_len else text for text in texts_for_embedding]
            
            # Generate embeddings
            logger.info(f"Generating embeddings for {len(truncated_texts)} chunks...")
            
            # Dense: AWS Bedrock
            dense_embeddings = self.dense_embeddings.embed_documents(truncated_texts)
            
            # Sparse: HuggingFace SPLADE
            if self.sparse_embeddings:
                sparse_embeddings = self.sparse_embeddings.embed_documents(truncated_texts)
            else:
                sparse_embeddings = [SparseVector(indices=[0], values=[0.0]) for _ in truncated_texts]
            
            # Create points
            qdrant_points = []
            for point, dense_emb, sparse_emb in zip(points, dense_embeddings, sparse_embeddings):
                point_uuid = string_to_uuid(point['id'])
                
                qdrant_points.append(
                    PointStruct(
                        id=point_uuid,
                        vector={"dense": dense_emb, "sparse": sparse_emb},
                        payload=point['payload']
                    )
                )
            
            # Upload in batches
            batch_size = 100
            for i in range(0, len(qdrant_points), batch_size):
                batch = qdrant_points[i:i + batch_size]
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch
                )
            
            logger.info(f"✓ Added {len(qdrant_points)} chunks to KB '{knowledge_base_name}' ({knowledge_base_id})")
            
            return len(qdrant_points)
            
        except Exception as e:
            logger.error(f"Error adding chunks: {str(e)}", exc_info=True)
            raise VectorStoreException(f"Failed to add chunks: {str(e)}")
    
    # ==================== SEARCH METHODS WITH KB ISOLATION ====================
    
    def _build_filter(
        self,
        user_id: str,
        knowledge_base_id: str,
        file_id: Optional[str] = None
    ) -> Filter:
        """Build Qdrant filter with MANDATORY KB isolation"""
        must_conditions = [
            FieldCondition(key="user_id", match=MatchValue(value=user_id)),
            FieldCondition(key="knowledge_base_id", match=MatchValue(value=knowledge_base_id))
        ]
        
        if file_id:
            must_conditions.append(
                FieldCondition(key="file_id", match=MatchValue(value=file_id))
            )
        
        return Filter(must=must_conditions)
    
    def _process_results(self, results) -> Tuple[List[str], List[Dict], List[float]]:
        """Process Qdrant results"""
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
    
    def _search_with_vector(
        self,
        query_vector: Union[List[float], SparseVector],
        vector_name: str,
        top_k: int,
        user_id: str,
        knowledge_base_id: str,
        file_id: Optional[str]
    ) -> Tuple[List[str], List[Dict], List[float]]:
        """Unified search with KB filtering"""
        try:
            query_filter = self._build_filter(user_id, knowledge_base_id, file_id)
            
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                using=vector_name,
                query_filter=query_filter,
                limit=top_k,
                with_payload=True
            )
            
            return self._process_results(results.points)
            
        except Exception as e:
            logger.error(f"{vector_name} search error: {e}")
            raise SearchError(f"{vector_name} search failed: {str(e)}")
    
    def _dense_search(
        self,
        query: str,
        top_k: int,
        user_id: str,
        knowledge_base_id: str,
        file_id: Optional[str] = None
    ) -> Tuple[List[str], List[Dict], List[float]]:
        """Dense search using AWS Bedrock embeddings"""
        query_vector = self._get_cached_embedding(query, 'dense')
        return self._search_with_vector(query_vector, "dense", top_k, user_id, knowledge_base_id, file_id)
    
    def _sparse_search(
        self,
        query: str,
        top_k: int,
        user_id: str,
        knowledge_base_id: str,
        file_id: Optional[str] = None
    ) -> Tuple[List[str], List[Dict], List[float]]:
        """Sparse search using HuggingFace SPLADE"""
        if not self.sparse_embeddings:
            logger.warning("Sparse unavailable, using dense")
            return self._dense_search(query, top_k, user_id, knowledge_base_id, file_id)
        
        query_vector = self._get_cached_embedding(query, 'sparse')
        return self._search_with_vector(query_vector, "sparse", top_k, user_id, knowledge_base_id, file_id)
    
    def _hybrid_search(
        self,
        query: str,
        top_k: int,
        user_id: str,
        knowledge_base_id: str,
        file_id: Optional[str] = None
    ) -> Tuple[List[str], List[Dict], List[float]]:
        """Hybrid search combining Bedrock dense + SPLADE sparse"""
        if not self.sparse_embeddings:
            return self._dense_search(query, top_k, user_id, knowledge_base_id, file_id)
        
        retrieval_k = min(top_k * 2, 30)
        
        # Parallel searches
        with ThreadPoolExecutor(max_workers=2) as executor:
            dense_future = executor.submit(
                self._dense_search, query, retrieval_k, user_id, knowledge_base_id, file_id
            )
            sparse_future = executor.submit(
                self._sparse_search, query, retrieval_k, user_id, knowledge_base_id, file_id
            )
            
            dense_docs, dense_metas, dense_scores = dense_future.result()
            sparse_docs, sparse_metas, sparse_scores = sparse_future.result()
        
        # RRF fusion
        fused = self._reciprocal_rank_fusion(
            list(zip(dense_docs, dense_metas, dense_scores)),
            list(zip(sparse_docs, sparse_metas, sparse_scores)),
            k=60
        )
        
        fused = fused[:top_k]
        
        if not fused:
            return [], [], []
        
        documents = [r[0] for r in fused]
        metadatas = [r[1] for r in fused]
        scores = [r[2] for r in fused]
        
        return documents, metadatas, scores
    
    def _reciprocal_rank_fusion(
        self,
        dense_results: List[Tuple],
        sparse_results: List[Tuple],
        k: int = 60
    ) -> List[Tuple]:
        """RRF algorithm for combining dense and sparse results"""
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
    
    def search(
        self,
        query: str,
        user_id: str,
        knowledge_base_id: str,
        top_k: int = 5,
        file_id: Optional[str] = None,
        search_type: Literal["dense", "sparse", "hybrid"] = "hybrid"
    ) -> Tuple[List[str], List[Dict], List[float]]:
        """
        Main search method with MANDATORY KB isolation
        
        Dense: AWS Bedrock
        Sparse: HuggingFace SPLADE
        """
        try:
            if search_type == "dense":
                results = self._dense_search(query, top_k, user_id, knowledge_base_id, file_id)
            elif search_type == "sparse":
                results = self._sparse_search(query, top_k, user_id, knowledge_base_id, file_id)
            else:
                results = self._hybrid_search(query, top_k, user_id, knowledge_base_id, file_id)
            
            documents, metadatas, scores = results
            
            # Ensure metadata completeness
            for meta in metadatas:
                if 'chunk_id' not in meta:
                    meta['chunk_id'] = 'unknown'
                if 'file_id' not in meta:
                    meta['file_id'] = 'unknown'
                if 'knowledge_base_id' not in meta:
                    meta['knowledge_base_id'] = knowledge_base_id
            
            logger.info(
                f"Search: KB={knowledge_base_id}, query='{query[:50]}...', "
                f"type={search_type}, results={len(documents)}"
            )
            
            return documents, metadatas, scores
            
        except Exception as e:
            logger.error(f"Search error: {e}", exc_info=True)
            raise SearchError(f"Search failed: {str(e)}")
    
    # ==================== OTHER METHODS (unchanged from original) ====================
    
    def get_chunk_by_id(
        self,
        chunk_id: str,
        user_id: str,
        knowledge_base_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Get chunk with ownership verification"""
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
            
            payload = points[0].payload
            
            # Verify user ownership
            if payload.get('user_id') != user_id:
                logger.warning(f"Access denied: chunk {chunk_id} belongs to different user")
                return None
            
            # Verify KB ownership if provided
            if knowledge_base_id and payload.get('knowledge_base_id') != knowledge_base_id:
                logger.warning(f"Access denied: chunk {chunk_id} belongs to different KB")
                return None
            
            return {
                "chunk_id": payload.get("chunk_id", chunk_id),
                "file_id": payload.get("file_id", ""),
                "knowledge_base_id": payload.get("knowledge_base_id", ""),
                "knowledge_base_name": payload.get("knowledge_base_name", ""),
                "text": payload.get("text", ""),
                "filename": payload.get("filename", ""),
                "user_id": payload.get("user_id", ""),
                "chunk_type": payload.get("chunk_type", "text"),
                "page_number": payload.get("page_number", 0),
                "char_count": len(payload.get("text", "")),
                "chunk_index": payload.get("chunk_index", 0),
                "created_at": payload.get("created_at", ""),
                "metadata": {k: v for k, v in payload.items() 
                           if k not in ['text', 'chunk_id', 'file_id', 'knowledge_base_id']}
            }
            
        except Exception as e:
            logger.error(f"Error retrieving chunk {chunk_id}: {e}", exc_info=True)
            raise VectorStoreException(f"Failed to retrieve chunk: {str(e)}")
    
    def update_chunk(
        self,
        chunk_id: str,
        file_id: str,
        new_text: str,
        user_id: str,
        knowledge_base_id: str,
        preserve_metadata: bool = True
    ) -> bool:
        """Update chunk with KB verification"""
        try:
            existing = self.get_chunk_by_id(chunk_id, user_id, knowledge_base_id)
            
            if not existing:
                raise VectorStoreException(f"Chunk not found or access denied: {chunk_id}")
            
            if existing['file_id'] != file_id:
                raise VectorStoreException("File ID mismatch")
            
            if existing['knowledge_base_id'] != knowledge_base_id:
                raise VectorStoreException("Knowledge base ID mismatch")
            
            truncated_text = new_text[:settings.embedding_max_length] if len(new_text) > settings.embedding_max_length else new_text
            
            # Generate new embeddings
            dense_vector = self._get_cached_embedding(truncated_text, 'dense')
            
            if self.sparse_embeddings:
                sparse_vector = self._get_cached_embedding(truncated_text, 'sparse')
            else:
                sparse_vector = SparseVector(indices=[0], values=[0.0])
            
            # Prepare payload
            if preserve_metadata:
                updated_payload = existing['metadata'].copy()
                updated_payload.update({
                    "text": new_text,
                    "chunk_id": chunk_id,
                    "file_id": file_id,
                    "knowledge_base_id": knowledge_base_id,
                    "user_id": user_id,
                    "char_count": len(new_text),
                    "updated_at": datetime.utcnow().isoformat(),
                    "edited": True
                })
            else:
                updated_payload = {
                    "text": new_text,
                    "chunk_id": chunk_id,
                    "file_id": file_id,
                    "knowledge_base_id": knowledge_base_id,
                    "user_id": user_id,
                    "char_count": len(new_text),
                    "updated_at": datetime.utcnow().isoformat(),
                    "edited": True
                }
            
            point_uuid = string_to_uuid(chunk_id)
            
            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    PointStruct(
                        id=point_uuid,
                        vector={"dense": dense_vector, "sparse": sparse_vector},
                        payload=updated_payload
                    )
                ]
            )
            
            logger.info(f"✓ Updated chunk: {chunk_id} in KB {knowledge_base_id}")
            
            return True
            
        except VectorStoreException:
            raise
        except Exception as e:
            logger.error(f"Error updating chunk: {e}", exc_info=True)
            raise VectorStoreException(f"Failed to update chunk: {str(e)}")
    
    def delete_document_by_file_id(
        self, 
        file_id: str, 
        user_id: str,
        knowledge_base_id: str
    ) -> bool:
        """Delete all chunks for a file with KB verification"""
        try:
            must_conditions = [
                FieldCondition(key="file_id", match=MatchValue(value=file_id)),
                FieldCondition(key="user_id", match=MatchValue(value=user_id)),
                FieldCondition(key="knowledge_base_id", match=MatchValue(value=knowledge_base_id))
            ]
            
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
            self.client.delete(collection_name=self.collection_name, points_selector=point_ids)
            
            logger.info(f"Deleted {len(point_ids)} chunks for file {file_id} in KB {knowledge_base_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error deleting document: {e}", exc_info=True)
            raise VectorStoreException(f"Failed to delete: {str(e)}")
    
    def delete_knowledge_base(
        self,
        knowledge_base_id: str,
        user_id: str
    ) -> int:
        """Delete all chunks for a knowledge base"""
        try:
            must_conditions = [
                FieldCondition(key="knowledge_base_id", match=MatchValue(value=knowledge_base_id)),
                FieldCondition(key="user_id", match=MatchValue(value=user_id))
            ]
            
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
                return 0
            
            point_ids = [point.id for point in points_to_delete]
            self.client.delete(collection_name=self.collection_name, points_selector=point_ids)
            
            logger.info(f"Deleted {len(point_ids)} chunks for KB {knowledge_base_id}")
            
            return len(point_ids)
            
        except Exception as e:
            logger.error(f"Error deleting KB: {e}", exc_info=True)
            raise VectorStoreException(f"Failed to delete KB: {str(e)}")
    
    def get_collection_stats(
        self, 
        user_id: Optional[str] = None,
        knowledge_base_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get collection statistics with optional KB filter"""
        try:
            query_filter = None
            must_conditions = []
            
            if user_id:
                must_conditions.append(FieldCondition(key="user_id", match=MatchValue(value=user_id)))
            
            if knowledge_base_id:
                must_conditions.append(FieldCondition(key="knowledge_base_id", match=MatchValue(value=knowledge_base_id)))
            
            if must_conditions:
                query_filter = Filter(must=must_conditions)
            
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
                    'documents': [],
                    'user_id': user_id,
                    'knowledge_base_id': knowledge_base_id
                }
            
            file_ids = {}
            for point in all_points:
                fid = point.payload.get('file_id', 'unknown')
                if fid not in file_ids:
                    file_ids[fid] = {
                        'file_id': fid,
                        'filename': point.payload.get('filename', 'unknown'),
                        'knowledge_base_id': point.payload.get('knowledge_base_id', 'unknown'),
                        'chunk_count': 0
                    }
                file_ids[fid]['chunk_count'] += 1
            
            return {
                'total_documents': len(file_ids),
                'total_chunks': len(all_points),
                'documents': list(file_ids.values()),
                'user_id': user_id,
                'knowledge_base_id': knowledge_base_id
            }
            
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {
                'total_documents': 0, 
                'total_chunks': 0, 
                'documents': [], 
                'error': str(e)
            }