"""
Vector Store Service - CORRECTED VERSION
Dense embeddings: AWS Bedrock
Sparse embeddings: HuggingFace SPLADE model only

Single collection strategy: all data (file chunks AND database rows) is stored
in the one shared collection defined by settings.qdrant_collection_name
('one-rag-collection'). Multi-tenancy is enforced via payload filters on
user_id and knowledge_base_id.
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
    
    Single shared collection for all data sources.
    KB isolation via payload filters on user_id + knowledge_base_id.
    """
    
    def __init__(self, embedding_model_id: Optional[str] = None):
        self._initialized = False
        self._init_error = None

        try:
            self.client = QdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key,
                timeout=30,
                prefer_grpc=False
            )
            self.collection_name = settings.qdrant_collection_name
            self.embedding_model_id = embedding_model_id or settings.bedrock_embedding_model_id
            self.dense_embeddings = BedrockEmbeddingService(model_id=self.embedding_model_id)
            
            try:
                self.sparse_embeddings = GPUSPLADEEmbedder(model_name=settings.splade_model_name)
                logger.info("✓ SPLADE sparse embeddings initialized")
            except Exception as e:
                logger.error(f"SPLADE initialization failed: {str(e)}", exc_info=True)
                logger.warning("Continuing with dense-only search")
                self.sparse_embeddings = None
            
            self.embedding_cache = EmbeddingCache(max_size=10000, ttl_seconds=3600)
            
            try:
                self._initialize_collection()
            except Exception as e:
                logger.warning(f"Collection initialization deferred: {e}")
                logger.info("Vector store will attempt lazy initialization on first use")
                return
            
            self._initialized = True
            logger.info(f"✅ Multi-tenant Vector Store initialized")
            logger.info(f"  - Collection: {self.collection_name}")
            logger.info(f"  - Dense: AWS Bedrock ({self.embedding_model_id})")
            logger.info(f"  - Sparse: {'HuggingFace SPLADE' if self.sparse_embeddings else 'DISABLED'}")
            logger.info(f"  - KB isolation: ENABLED (payload filter)")
            
        except Exception as e:
            self._init_error = str(e)
            logger.warning(f"⚠️  Qdrant initialization deferred: {e}")
            logger.info("Vector store will attempt lazy initialization on first use")
    
    def _ensure_initialized(self):
        if self._initialized and self.client:
            return
        self._try_initialize()
        if not self._initialized or not self.client:
            raise VectorStoreException(
                f"Vector store not available: {self._init_error or 'Unknown error'}"
            )
    
    def _initialize_collection(self):
        """
        Ensure the shared collection exists with the correct schema.

        SAFETY RULE: Never delete a collection that already contains data.
        Deletion is only allowed when the collection is empty AND the schema
        is wrong (missing sparse vectors).  If the collection has data and the
        schema looks wrong, we log a warning and continue — wiping live data
        is never acceptable.
        """
        try:
            dense_dim = self.dense_embeddings.get_embedding_dimension()
            vectors_config = {"dense": VectorParams(size=dense_dim, distance=Distance.COSINE)}
            sparse_config = None
            if self.sparse_embeddings:
                sparse_config = {"sparse": SparseVectorParams(index=SparseIndexParams())}

            existing_collections = [c.name for c in self.client.get_collections().collections]

            if self.collection_name in existing_collections:
                col = self.client.get_collection(self.collection_name)

                # Safely probe sparse config — attribute absent on some client versions
                sparse_cfg = getattr(col.config, 'sparse_vectors_config', None)
                has_sparse  = sparse_cfg is not None and 'sparse' in sparse_cfg
                needs_sparse = self.sparse_embeddings is not None

                if needs_sparse and not has_sparse:
                    # Schema mismatch detected — only recreate if collection is EMPTY
                    point_count = col.points_count if hasattr(col, 'points_count') else None
                    if point_count is None:
                        # Fallback: ask Qdrant directly
                        point_count = self.client.get_collection(self.collection_name).points_count

                    if point_count == 0:
                        logger.warning(
                            f"⚠️  Collection '{self.collection_name}' is empty and lacks sparse vectors. "
                            f"Recreating with dense + sparse support..."
                        )
                        self.client.delete_collection(self.collection_name)
                        self.client.create_collection(
                            collection_name=self.collection_name,
                            vectors_config=vectors_config,
                            sparse_vectors_config=sparse_config,
                        )
                        self._create_payload_indexes()
                        logger.info(f"✅ Recreated '{self.collection_name}' with dense + sparse")
                    else:
                        # DATA EXISTS — do NOT delete under any circumstances.
                        # Log a warning and continue so existing data is preserved.
                        logger.warning(
                            f"⚠️  Collection '{self.collection_name}' has {point_count} points but "
                            f"lacks sparse vector config. This is likely because some points were "
                            f"indexed before sparse support was added. Continuing without recreation "
                            f"to preserve existing data. New points will include sparse vectors."
                        )
                else:
                    logger.info(f"✓ Collection '{self.collection_name}' exists — reusing as-is")
            else:
                # Does not exist yet — create fresh
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=vectors_config,
                    sparse_vectors_config=sparse_config,
                )
                self._create_payload_indexes()
                logger.info(f"✅ Created collection '{self.collection_name}' with dense + sparse")

        except Exception as e:
            logger.error(f"Failed to initialize collection: {e}", exc_info=True)
            raise VectorStoreException(f"Failed to initialize collection: {str(e)}")
    
    def _create_payload_indexes(self):
        try:
            for field in ("user_id", "knowledge_base_id", "file_id", "chunk_id", "chunk_type"):
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field,
                    field_schema="keyword"
                )
            logger.info("✓ Created payload indexes")
        except Exception as e:
            logger.warning(f"Could not create payload indexes: {e}")
    
    def _get_cached_embedding(
        self,
        text: str,
        embedding_type: Literal['dense', 'sparse'] = 'dense'
    ) -> Union[List[float], SparseVector]:
        cached = self.embedding_cache.get(text, embedding_type)
        if cached is not None:
            return cached
        if embedding_type == 'dense':
            embedding = self.dense_embeddings.embed_query(text)
        else:
            embedding = self.sparse_embeddings.embed_query(text) if self.sparse_embeddings \
                        else SparseVector(indices=[0], values=[0.0])
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
        if not chunks:
            logger.warning(f"No chunks provided for {file_id}")
            return 0
        
        try:
            points = []
            texts_for_embedding = []
            
            for chunk in chunks:
                text     = chunk.get('text', '')
                chunk_id = chunk.get('chunk_id', '')
                metadata = chunk.get('metadata', {})
                if not text or not chunk_id:
                    logger.warning(f"Skipping invalid chunk: {chunk_id}")
                    continue
                
                payload = {
                    'user_id':            str(user_id),
                    'knowledge_base_id':  str(knowledge_base_id),
                    'knowledge_base_name':str(knowledge_base_name),
                    'file_id':            str(file_id),
                    'filename':           str(filename),
                    'chunk_id':           str(chunk_id),
                    'text':               text,
                    'chunk_type':         str(metadata.get('chunk_type', 'text')),
                    'chunk_index':        int(chunk.get('chunk_index', metadata.get('chunk_index', 0))),
                    'page_number':        int(metadata.get('page_number', 0)),
                    'created_at':         datetime.utcnow().isoformat()
                }
                for key, value in metadata.items():
                    if key not in payload and value is not None:
                        payload[key] = str(value) if not isinstance(value, (int, float, bool)) else value
                
                texts_for_embedding.append(text)
                points.append({'id': chunk_id, 'payload': payload})
            
            if not points:
                logger.warning(f"No valid chunks for {file_id}")
                return 0
            
            max_len = settings.embedding_max_length
            truncated = [t[:max_len] if len(t) > max_len else t for t in texts_for_embedding]
            
            logger.info(f"Generating embeddings for {len(truncated)} chunks...")
            dense_embs = self.dense_embeddings.embed_documents(truncated)
            
            if not self.sparse_embeddings:
                raise VectorStoreException("SPLADE not available — cannot add chunks.")
            sparse_embs = self.sparse_embeddings.embed_documents(truncated)
            
            qdrant_points = [
                PointStruct(
                    id=string_to_uuid(p['id']),
                    vector={"dense": dense_embs[i], "sparse": sparse_embs[i]},
                    payload=p['payload']
                )
                for i, p in enumerate(points)
            ]
            
            for i in range(0, len(qdrant_points), 100):
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=qdrant_points[i:i + 100]
                )
            
            logger.info(f"✓ Added {len(qdrant_points)} chunks to KB '{knowledge_base_name}'")
            return len(qdrant_points)
            
        except Exception as e:
            logger.error(f"Error adding chunks: {str(e)}", exc_info=True)
            raise VectorStoreException(f"Failed to add chunks: {str(e)}")
    
    # ==================== SEARCH ====================
    
    def _build_filter(self, user_id, knowledge_base_id, file_id=None):
        must = [
            FieldCondition(key="user_id",           match=MatchValue(value=user_id)),
            FieldCondition(key="knowledge_base_id", match=MatchValue(value=knowledge_base_id)),
        ]
        if file_id:
            must.append(FieldCondition(key="file_id", match=MatchValue(value=file_id)))
        return Filter(must=must)
    
    def _process_results(self, results):
        docs, metas, scores = [], [], []
        for r in results:
            docs.append(r.payload.get('text', ''))
            metas.append({k: v for k, v in r.payload.items() if k != 'text'})
            scores.append(float(r.score))
        return docs, metas, scores
    
    def _search_with_vector(self, query_vector, vector_name, top_k, user_id, knowledge_base_id, file_id):
        try:
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                using=vector_name,
                query_filter=self._build_filter(user_id, knowledge_base_id, file_id),
                limit=top_k,
                with_payload=True
            )
            return self._process_results(results.points)
        except Exception as e:
            logger.error(f"{vector_name} search error: {e}")
            raise SearchError(f"{vector_name} search failed: {str(e)}")
    
    def _dense_search(self, query, top_k, user_id, knowledge_base_id, file_id=None):
        return self._search_with_vector(
            self._get_cached_embedding(query, 'dense'), "dense",
            top_k, user_id, knowledge_base_id, file_id)
    
    def _sparse_search(self, query, top_k, user_id, knowledge_base_id, file_id=None):
        if not self.sparse_embeddings:
            return self._dense_search(query, top_k, user_id, knowledge_base_id, file_id)
        return self._search_with_vector(
            self._get_cached_embedding(query, 'sparse'), "sparse",
            top_k, user_id, knowledge_base_id, file_id)
    
    def _hybrid_search(self, query, top_k, user_id, knowledge_base_id, file_id=None):
        if not self.sparse_embeddings:
            return self._dense_search(query, top_k, user_id, knowledge_base_id, file_id)
        retrieval_k = min(top_k * 2, 30)
        with ThreadPoolExecutor(max_workers=2) as executor:
            df = executor.submit(self._dense_search,  query, retrieval_k, user_id, knowledge_base_id, file_id)
            sf = executor.submit(self._sparse_search, query, retrieval_k, user_id, knowledge_base_id, file_id)
            dd, dm, ds = df.result()
            sd, sm, ss = sf.result()
        fused = self._reciprocal_rank_fusion(
            list(zip(dd, dm, ds)), list(zip(sd, sm, ss)), k=60
        )[:top_k]
        if not fused:
            return [], [], []
        return [r[0] for r in fused], [r[1] for r in fused], [r[2] for r in fused]
    
    def _resolve_hierarchical_chunks(self, documents, metadatas, scores, top_k):
        regular, parent_map = [], {}
        for doc, meta, score in zip(documents, metadatas, scores):
            ct = meta.get('chunk_type', 'text')
            if ct == 'child':
                pid = meta.get('parent_chunk_id')
                if pid:
                    if score > parent_map.get(pid, -1):
                        parent_map[pid] = score
                else:
                    regular.append((doc, meta, score))
            elif ct == 'parent':
                pid = meta.get('chunk_id')
                if pid and pid not in parent_map:
                    parent_map[pid] = score
            else:
                regular.append((doc, meta, score))

        promoted = []
        if parent_map:
            try:
                pts = self.client.retrieve(
                    collection_name=self.collection_name,
                    ids=[string_to_uuid(pid) for pid in parent_map],
                    with_payload=True, with_vectors=False
                )
                for pt in pts:
                    pl = pt.payload
                    promoted.append((
                        pl.get('text', ''),
                        {k: v for k, v in pl.items() if k != 'text'},
                        parent_map.get(pl.get('chunk_id', ''), 0.0)
                    ))
            except Exception as e:
                logger.warning(f"Parent fetch failed: {e}")
                for doc, meta, score in zip(documents, metadatas, scores):
                    if meta.get('chunk_type') == 'child':
                        regular.append((doc, meta, score))

        seen, deduped = set(), []
        for doc, meta, score in sorted(regular + promoted, key=lambda x: x[2], reverse=True):
            cid = meta.get('chunk_id', '')
            if cid not in seen:
                seen.add(cid)
                deduped.append((doc, meta, score))

        deduped = deduped[:top_k]
        if not deduped:
            return [], [], []
        return [r[0] for r in deduped], [r[1] for r in deduped], [r[2] for r in deduped]

    def _reciprocal_rank_fusion(self, dense_results, sparse_results, k=60):
        rrf = {}
        for rank, (doc, meta, _) in enumerate(dense_results, 1):
            cid = meta.get('chunk_id', '')
            rrf.setdefault(cid, {'doc': doc, 'meta': meta, 'score': 0.0})
            rrf[cid]['score'] += 1.0 / (k + rank)
        for rank, (doc, meta, _) in enumerate(sparse_results, 1):
            cid = meta.get('chunk_id', '')
            rrf.setdefault(cid, {'doc': doc, 'meta': meta, 'score': 0.0})
            rrf[cid]['score'] += 1.0 / (k + rank)
        return [
            (v['doc'], v['meta'], min(v['score'] / 0.05, 1.0))
            for v in sorted(rrf.values(), key=lambda x: x['score'], reverse=True)
        ]
    
    def search(
        self,
        query: str,
        user_id: str,
        knowledge_base_id: str,
        top_k: int = 5,
        file_id: Optional[str] = None,
        search_type: Literal["dense", "sparse", "hybrid"] = "hybrid"
    ) -> Tuple[List[str], List[Dict], List[float]]:
        try:
            fetch_k = top_k * 3
            if search_type == "dense":
                docs, metas, scores = self._dense_search(query, fetch_k, user_id, knowledge_base_id, file_id)
            elif search_type == "sparse":
                docs, metas, scores = self._sparse_search(query, fetch_k, user_id, knowledge_base_id, file_id)
            else:
                docs, metas, scores = self._hybrid_search(query, fetch_k, user_id, knowledge_base_id, file_id)
            
            docs, metas, scores = self._resolve_hierarchical_chunks(docs, metas, scores, top_k)
            
            for meta in metas:
                meta.setdefault('chunk_id', 'unknown')
                meta.setdefault('file_id', 'unknown')
                meta.setdefault('knowledge_base_id', knowledge_base_id)
            
            logger.info(f"Search: KB={knowledge_base_id}, query='{query[:50]}...', type={search_type}, results={len(docs)}")
            return docs, metas, scores
        except Exception as e:
            logger.error(f"Search error: {e}", exc_info=True)
            raise SearchError(f"Search failed: {str(e)}")
    
    # ==================== CRUD ====================
    
    def get_chunk_by_id(self, chunk_id, user_id, knowledge_base_id=None):
        try:
            pts = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[string_to_uuid(chunk_id)],
                with_payload=True, with_vectors=False
            )
            if not pts:
                return None
            payload = pts[0].payload
            if payload.get('user_id') != user_id:
                return None
            if knowledge_base_id and payload.get('knowledge_base_id') != knowledge_base_id:
                return None
            return {
                "chunk_id":           payload.get("chunk_id", chunk_id),
                "file_id":            payload.get("file_id", ""),
                "knowledge_base_id":  payload.get("knowledge_base_id", ""),
                "knowledge_base_name":payload.get("knowledge_base_name", ""),
                "text":               payload.get("text", ""),
                "filename":           payload.get("filename", ""),
                "user_id":            payload.get("user_id", ""),
                "chunk_type":         payload.get("chunk_type", "text"),
                "page_number":        payload.get("page_number", 0),
                "char_count":         len(payload.get("text", "")),
                "chunk_index":        payload.get("chunk_index", 0),
                "created_at":         payload.get("created_at", ""),
                "metadata": {k: v for k, v in payload.items()
                             if k not in ['text', 'chunk_id', 'file_id', 'knowledge_base_id']}
            }
        except Exception as e:
            raise VectorStoreException(f"Failed to retrieve chunk: {str(e)}")
    
    def update_chunk(self, chunk_id, file_id, new_text, user_id, knowledge_base_id, preserve_metadata=True):
        try:
            existing = self.get_chunk_by_id(chunk_id, user_id, knowledge_base_id)
            if not existing:
                raise VectorStoreException(f"Chunk not found: {chunk_id}")
            if existing['file_id'] != file_id:
                raise VectorStoreException("File ID mismatch")
            truncated     = new_text[:settings.embedding_max_length]
            dense_vector  = self._get_cached_embedding(truncated, 'dense')
            sparse_vector = self._get_cached_embedding(truncated, 'sparse')
            base = existing['metadata'].copy() if preserve_metadata else {}
            base.update({"text": new_text, "chunk_id": chunk_id, "file_id": file_id,
                         "knowledge_base_id": knowledge_base_id, "user_id": user_id,
                         "char_count": len(new_text), "updated_at": datetime.utcnow().isoformat(),
                         "edited": True})
            self.client.upsert(
                collection_name=self.collection_name,
                points=[PointStruct(id=string_to_uuid(chunk_id),
                                    vector={"dense": dense_vector, "sparse": sparse_vector},
                                    payload=base)]
            )
            return True
        except VectorStoreException:
            raise
        except Exception as e:
            raise VectorStoreException(f"Failed to update chunk: {str(e)}")
    
    def delete_document_by_file_id(self, file_id, user_id, knowledge_base_id):
        try:
            result = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(must=[
                    FieldCondition(key="file_id",           match=MatchValue(value=file_id)),
                    FieldCondition(key="user_id",           match=MatchValue(value=user_id)),
                    FieldCondition(key="knowledge_base_id", match=MatchValue(value=knowledge_base_id)),
                ]),
                limit=10000, with_payload=False, with_vectors=False
            )
            pts = result[0]
            if not pts:
                return False
            self.client.delete(collection_name=self.collection_name,
                               points_selector=[p.id for p in pts])
            logger.info(f"Deleted {len(pts)} chunks for file {file_id}")
            return True
        except Exception as e:
            raise VectorStoreException(f"Failed to delete: {str(e)}")
    
    def delete_knowledge_base(self, knowledge_base_id, user_id):
        try:
            result = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(must=[
                    FieldCondition(key="knowledge_base_id", match=MatchValue(value=knowledge_base_id)),
                    FieldCondition(key="user_id",           match=MatchValue(value=user_id)),
                ]),
                limit=10000, with_payload=False, with_vectors=False
            )
            pts = result[0]
            if not pts:
                return 0
            self.client.delete(collection_name=self.collection_name,
                               points_selector=[p.id for p in pts])
            logger.info(f"Deleted {len(pts)} chunks for KB {knowledge_base_id}")
            return len(pts)
        except Exception as e:
            raise VectorStoreException(f"Failed to delete KB: {str(e)}")
    
    def get_collection_stats(self, user_id=None, knowledge_base_id=None):
        try:
            must = []
            if user_id:
                must.append(FieldCondition(key="user_id", match=MatchValue(value=user_id)))
            if knowledge_base_id:
                must.append(FieldCondition(key="knowledge_base_id", match=MatchValue(value=knowledge_base_id)))
            qf = Filter(must=must) if must else None
            all_pts, offset = [], None
            while True:
                pts, next_off = self.client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=qf, limit=1000, offset=offset,
                    with_payload=True, with_vectors=False
                )
                all_pts.extend(pts)
                if next_off is None:
                    break
                offset = next_off
            if not all_pts:
                return {'total_documents': 0, 'total_chunks': 0, 'documents': [],
                        'user_id': user_id, 'knowledge_base_id': knowledge_base_id}
            file_ids = {}
            for pt in all_pts:
                fid = pt.payload.get('file_id', 'unknown')
                if fid not in file_ids:
                    file_ids[fid] = {'file_id': fid,
                                     'filename': pt.payload.get('filename', 'unknown'),
                                     'knowledge_base_id': pt.payload.get('knowledge_base_id', 'unknown'),
                                     'chunk_count': 0}
                file_ids[fid]['chunk_count'] += 1
            return {'total_documents': len(file_ids), 'total_chunks': len(all_pts),
                    'documents': list(file_ids.values()),
                    'user_id': user_id, 'knowledge_base_id': knowledge_base_id}
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {'total_documents': 0, 'total_chunks': 0, 'documents': [], 'error': str(e)}
