"""
Vector Store Service - Qdrant Cloud Implementation with Hybrid Dense+Sparse Search
- Stores both dense (user-selectable) and sparse (SPLADE) vectors
- Uses LOCAL SPLADE (sentence-transformers) instead of API
- Supports three search types: dense, sparse, hybrid
- Uses Reciprocal Rank Fusion (RRF) for hybrid search
"""

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, Filter, 
    FieldCondition, MatchValue, SparseVector, NamedVector,
    SparseVectorParams, SparseIndexParams
)
from langchain_aws import BedrockEmbeddings
from typing import List, Dict, Any, Tuple, Optional, Literal
import logging
import uuid
from datetime import datetime
import hashlib

from app.core.config import settings
from app.core.exceptions import (
    VectorStoreException, 
    EmbeddingError, 
    SearchError,
    NoDocumentsFoundError
)
from app.core.monitoring import get_logger

logger = get_logger(__name__)


def string_to_uuid(text: str) -> str:
    """Convert any string to a valid UUID v5"""
    namespace = uuid.UUID('6ba7b810-9dad-11d1-80b4-00c04fd430c8')
    return str(uuid.uuid5(namespace, text))


class SPLADEEmbedder:
    """
    Local SPLADE sparse embedder using sentence-transformers
    
    CORRECTED: Properly handles scipy sparse matrix output
    """
    
    def __init__(self):
        """Initialize SPLADE embedder"""
        try:
            from sentence_transformers import SparseEncoder
            
            # Try SPLADE v3 first (recommended)
            model_name = "naver/splade-v3"
            
            logger.info(f"Loading SPLADE model: {model_name}...")
            self.model = SparseEncoder(model_name)
            logger.info(f"✓ SPLADE model loaded successfully: {model_name}")
            
            self.available = True
            
        except Exception as e:
            logger.error(f"Failed to load SPLADE model: {e}")
            logger.warning("⚠️  SPLADE not available - sparse search will be disabled")
            logger.warning("   Install with: pip install sentence-transformers")
            self.model = None
            self.available = False
    
    def _sparse_matrix_to_qdrant(self, sparse_output) -> SparseVector:
        """
        Convert SPLADE output to Qdrant SparseVector
        
        Handles PyTorch sparse tensors (COO format) from sentence-transformers
        
        Args:
            sparse_output: PyTorch sparse tensor in COO format
                
        Returns:
            SparseVector with indices and values
        """
        try:
            import torch
            
            # Check if it's a PyTorch tensor
            if isinstance(sparse_output, torch.Tensor):
                
                # CASE 1: PyTorch Sparse Tensor (COO format)
                if sparse_output.is_sparse:
                    logger.debug(f"Converting PyTorch sparse tensor: shape={sparse_output.shape}")
                    
                    # Get sparse tensor in COO format
                    sparse_tensor = sparse_output.coalesce()  # Ensure it's coalesced
                    
                    # Extract indices and values
                    # For 1D or 2D sparse tensor, indices() returns [2, nnz] or [1, nnz]
                    indices_tensor = sparse_tensor.indices()
                    values_tensor = sparse_tensor.values()
                    
                    # Handle 2D tensor (batch_size=1, vocab_size)
                    if indices_tensor.dim() == 2:
                        # Take the last row (column indices for vocab positions)
                        indices_numpy = indices_tensor[-1].cpu().numpy()
                    else:
                        # 1D tensor
                        indices_numpy = indices_tensor.cpu().numpy()
                    
                    values_numpy = values_tensor.cpu().numpy()
                    
                    # Handle empty vectors
                    if len(indices_numpy) == 0 or len(values_numpy) == 0:
                        logger.warning("Generated empty sparse vector (no non-zero values)")
                        return SparseVector(indices=[0], values=[0.0])
                    
                    # Convert to required types
                    indices = [int(idx) for idx in indices_numpy]
                    values = [float(val) for val in values_numpy]
                    
                    logger.debug(
                        f"Converted sparse tensor: {len(indices)} non-zero elements, "
                        f"max_value={max(values):.4f}, max_idx={max(indices)}"
                    )
                    
                    return SparseVector(indices=indices, values=values)
                
                # CASE 2: Dense PyTorch Tensor (fallback)
                else:
                    logger.debug(f"Converting dense PyTorch tensor: shape={sparse_output.shape}")
                    
                    # Squeeze to remove batch dimensions
                    if sparse_output.dim() > 1:
                        sparse_output = sparse_output.squeeze()
                    
                    numpy_vector = sparse_output.cpu().numpy()
                    
                    # Get non-zero indices and values
                    nonzero_indices = numpy_vector.nonzero()[0]
                    nonzero_values = numpy_vector[nonzero_indices]
                    
                    if len(nonzero_indices) == 0:
                        logger.warning("Generated empty sparse vector (no non-zero values)")
                        return SparseVector(indices=[0], values=[0.0])
                    
                    indices = [int(idx) for idx in nonzero_indices]
                    values = [float(val) for val in nonzero_values]
                    
                    return SparseVector(indices=indices, values=values)
            
            # CASE 3: scipy sparse matrix (legacy support)
            else:
                import scipy.sparse
                
                if scipy.sparse.issparse(sparse_output):
                    logger.debug("Converting scipy sparse matrix")
                    
                    coo = sparse_output.tocoo()
                    indices = coo.col.tolist()
                    values = coo.data.tolist()
                    
                    if len(indices) == 0 or len(values) == 0:
                        logger.warning("Generated empty sparse vector from sparse matrix")
                        return SparseVector(indices=[0], values=[0.0])
                    
                    indices = [int(idx) for idx in indices]
                    values = [float(val) for val in values]
                    
                    return SparseVector(indices=indices, values=values)
                else:
                    logger.warning(
                        f"Unexpected sparse output type: {type(sparse_output)}, "
                        f"using fallback"
                    )
                    return SparseVector(indices=[0], values=[0.0])
                
        except Exception as e:
            logger.error(f"Error converting sparse output: {e}")
            import traceback
            traceback.print_exc()
            return SparseVector(indices=[0], values=[0.0])
    
    def embed_documents(self, texts: List[str]) -> List[SparseVector]:
        """
        Generate SPLADE sparse embeddings for multiple texts
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of SparseVector objects
        """
        if not self.available:
            logger.debug("SPLADE not available, returning empty sparse vectors")
            return [SparseVector(indices=[0], values=[0.0]) for _ in texts]
        
        try:
            # Generate sparse embeddings using sentence-transformers
            # This returns a list of scipy sparse matrices
            sparse_matrices = self.model.encode(texts, convert_to_tensor=False)
            
            logger.debug(f"Generated {len(sparse_matrices)} sparse embeddings")
            
            # Convert each sparse matrix to Qdrant SparseVector format
            sparse_vectors = []
            for i, sparse_matrix in enumerate(sparse_matrices):
                try:
                    sparse_vector = self._sparse_matrix_to_qdrant(sparse_matrix)
                    sparse_vectors.append(sparse_vector)
                    
                    # Log first few for debugging
                    if i < 3:
                        logger.debug(
                            f"Sparse vector {i}: {len(sparse_vector.indices)} non-zero elements, "
                            f"first 5 indices: {sparse_vector.indices[:5]}"
                        )
                except Exception as e:
                    logger.warning(f"Error processing sparse matrix {i}: {e}")
                    sparse_vectors.append(SparseVector(indices=[0], values=[0.0]))
            
            return sparse_vectors
            
        except Exception as e:
            logger.error(f"SPLADE embedding error: {e}")
            import traceback
            traceback.print_exc()
            # Return empty sparse vectors as fallback
            return [SparseVector(indices=[0], values=[0.0]) for _ in texts]
    
    def embed_query(self, text: str) -> SparseVector:
        """
        Generate SPLADE sparse embedding for a single text
        
        Args:
            text: Text to embed
            
        Returns:
            SparseVector object with indices and values
        """
        if not self.available:
            logger.debug("SPLADE not available, returning empty sparse vector")
            return SparseVector(indices=[0], values=[0.0])
        
        try:
            # Generate sparse embedding for single text
            # Returns a list with one sparse matrix
            sparse_matrices = self.model.encode([text], convert_to_tensor=False)
            
            if len(sparse_matrices) == 0:
                logger.warning("No sparse embedding generated")
                return SparseVector(indices=[0], values=[0.0])
            
            # Convert the first (and only) sparse matrix
            sparse_vector = self._sparse_matrix_to_qdrant(sparse_matrices[0])
            
            logger.debug(
                f"Query sparse vector: {len(sparse_vector.indices)} non-zero elements"
            )
            
            return sparse_vector
                
        except Exception as e:
            logger.error(f"SPLADE query embedding error: {e}")
            import traceback
            traceback.print_exc()
            return SparseVector(indices=[0], values=[0.0])


class VectorStoreService:
    """Qdrant-based vector store service with hybrid dense+sparse search"""
    
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
            
            # Initialize dense embeddings (user-selectable model)
            self.dense_embeddings = BedrockEmbeddings(
                model_id=settings.embedding_model_id,
                region_name=settings.aws_region,
                credentials_profile_name=settings.aws_profile_name if settings.aws_profile_name else None
            )
            
            # Initialize sparse embeddings (LOCAL SPLADE)
            self.sparse_embeddings = SPLADEEmbedder()
            
            # Collection name
            self.collection_name = settings.qdrant_collection_name
            
            # Initialize collection
            self._initialize_collection()
            
            logger.info(f"✓ Qdrant Vector Store initialized (collection: {self.collection_name})")
            logger.info(f"✓ Dense embeddings: {settings.embedding_model_id}")
            logger.info(f"✓ Sparse embeddings: {'SPLADE (local)' if self.sparse_embeddings.available else 'Disabled'}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant: {e}", exc_info=True)
            raise VectorStoreException(f"Failed to initialize vector store: {str(e)}")
        
    def _initialize_collection(self):
        """Create collection if it doesn't exist with both dense and sparse vectors"""
        try:
            collections = self.client.get_collections().collections
            collection_names = [col.name for col in collections]
            
            if self.collection_name not in collection_names:
                dense_dim = self._get_embedding_dimension()
                
                # Create collection with named vectors (dense + sparse)
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
                
                logger.info(f"✓ Created Qdrant collection: {self.collection_name}")
                logger.info(f"  - Dense vector size: {dense_dim}")
                logger.info(f"  - Sparse vector: SPLADE (local)")
            else:
                logger.info(f"✓ Using existing Qdrant collection: {self.collection_name}")
                
        except Exception as e:
            logger.error(f"Error initializing collection: {e}", exc_info=True)
            raise VectorStoreException(f"Failed to initialize collection: {str(e)}")
        
    def _get_embedding_dimension(self) -> int:
        """Get embedding dimension"""
        try:
            test_embedding = self.dense_embeddings.embed_query("test")
            dim = len(test_embedding)
            logger.info(f"Detected embedding dimension: {dim}")
            return dim
        except Exception as e:
            logger.warning(f"Could not determine embedding dimension: {e}, using default 1024")
            return 1024
    
    def _create_payload_indexes(self):
        """Create indexes on payload fields for faster filtering"""
        try:
            # Index user_id for user isolation
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="user_id",
                field_schema="keyword"
            )
            
            # Index file_id for file-specific queries
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="file_id",
                field_schema="keyword"
            )
            
            # ADD THIS: Index chunk_id for direct chunk retrieval
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="chunk_id",
                field_schema="keyword"
            )
            
            logger.info("✓ Created payload indexes for user_id, file_id, and chunk_id")
            
        except Exception as e:
            logger.warning(f"Could not create payload indexes: {e}")

    def add_document_chunks(
        self,
        chunks: List[Dict[str, Any]],
        file_id: str,
        filename: str,
        structured_data: Optional[Dict] = None
    ) -> int:
        """Add document chunks to Qdrant with both dense and sparse vectors"""
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
                
                if 'user_id' not in metadata:
                    logger.error(f"Chunk {chunk_id} missing user_id!")
                    raise VectorStoreException(
                        f"Cannot add chunk without user_id: {chunk_id}"
                    )
                
                payload = {
                    'file_id': str(metadata.get('file_id', file_id)),
                    'user_id': str(metadata.get('user_id')),
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
                logger.warning(f"No valid chunks to add for {file_id}")
                return 0
            
            # Truncate texts to fit embedding limits
            max_embedding_length = settings.embedding_max_length
            truncated_texts = []
            truncated_count = 0
            
            for i, text in enumerate(texts_for_embedding):
                if len(text) > max_embedding_length:
                    truncated = text[:max_embedding_length]
                    truncated_texts.append(truncated)
                    truncated_count += 1
                else:
                    truncated_texts.append(text)
            
            if truncated_count > 0:
                logger.warning(
                    f"Truncated {truncated_count}/{len(texts_for_embedding)} texts to fit {max_embedding_length} char limit"
                )
            
            # Generate dense embeddings
            logger.info(f"Generating dense embeddings for {len(truncated_texts)} chunks...")
            dense_embeddings = self.dense_embeddings.embed_documents(truncated_texts)
            
            # Generate sparse embeddings (LOCAL SPLADE)
            logger.info(f"Generating sparse embeddings (SPLADE local) for {len(truncated_texts)} chunks...")
            sparse_embeddings = self.sparse_embeddings.embed_documents(truncated_texts)
            
            # Create Qdrant points with both dense and sparse vectors
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
                logger.debug(f"Uploaded batch {i//batch_size + 1}/{(len(qdrant_points)-1)//batch_size + 1}")
            
            logger.info(
                f"✓ Added {len(qdrant_points)} chunks with dense+sparse vectors to Qdrant",
                extra={"file_id": file_id, "user_id": points[0]['payload']['user_id']}
            )
            
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
    search_type: Literal["dense", "sparse", "hybrid"] = "hybrid"
) -> Tuple[List[str], List[Dict[str, Any]], List[float]]:
        """
        Enhanced search method that returns full chunk metadata
        
        MODIFIED to include:
        - chunk_id in metadata
        - file_id in metadata  
        - Full text in documents (not truncated)
        
        Args:
            query: Search query
            top_k: Number of results
            user_id: User ID for isolation
            file_id: Optional file ID filter
            search_type: "dense", "sparse", or "hybrid"
            
        Returns:
            Tuple of (documents, metadatas, scores) where metadatas includes
            chunk_id, file_id, and all other original metadata
        """
        try:
            # Perform search based on type
            # FIXED: Pass user_id and file_id strings, NOT query_filter
            if search_type == "dense":
                results = self._dense_search(query, top_k, user_id, file_id)
            elif search_type == "sparse":
                results = self._sparse_search(query, top_k, user_id, file_id)
            else:  # hybrid
                results = self._hybrid_search(query, top_k, user_id, file_id)
            
            # Unpack results (they're already processed tuples)
            documents, metadatas, scores = results
            
            # Ensure chunk_id and file_id are in metadata
            for i, meta in enumerate(metadatas):
                if 'chunk_id' not in meta:
                    logger.warning(f"chunk_id missing from metadata, generating fallback")
                    meta['chunk_id'] = f"unknown_{hash(documents[i]) % 10000}"
                
                if 'file_id' not in meta:
                    logger.warning(f"file_id missing from metadata")
                    meta['file_id'] = 'unknown'
            
            logger.info(
                f"Search completed: query='{query[:50]}...', "
                f"type={search_type}, results={len(documents)}"
            )
            
            return documents, metadatas, scores
            
        except Exception as e:
            logger.error(f"Search error: {e}", exc_info=True)
            raise SearchError(f"Search failed: {str(e)}")
    
    def _dense_search(
        self,
        query: str,
        top_k: int,
        user_id: Optional[str] = None,
        file_id: Optional[str] = None
    ) -> Tuple[List[str], List[Dict[str, Any]], List[float]]:
        """Dense vector search only"""
        try:
            # Generate dense query embedding
            query_embedding = self.dense_embeddings.embed_query(query)
            
            # Build filter
            query_filter = self._build_filter(user_id, file_id)
            
            # Use query_points with named vector
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_embedding,
                using="dense",  # Specify which named vector to use
                query_filter=query_filter,
                limit=top_k,
                with_payload=True
            )
            
            # query_points returns a QueryResponse object with .points attribute
            return self._process_results(results.points)
            
        except Exception as e:
            logger.error(f"Dense search error: {e}", exc_info=True)
            raise SearchError(f"Dense search failed: {str(e)}")
    
    def _sparse_search(
        self,
        query: str,
        top_k: int,
        user_id: Optional[str] = None,
        file_id: Optional[str] = None
    ) -> Tuple[List[str], List[Dict[str, Any]], List[float]]:
        """Sparse vector (SPLADE) search only"""
        try:
            if not self.sparse_embeddings.available:
                logger.warning("Sparse search requested but SPLADE not available, falling back to dense search")
                return self._dense_search(query, top_k, user_id, file_id)
            
            # Generate sparse query embedding
            query_sparse = self.sparse_embeddings.embed_query(query)
            
            # Build filter
            query_filter = self._build_filter(user_id, file_id)
            
            # Use query_points with named sparse vector
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_sparse,
                using="sparse",  # Specify which named sparse vector to use
                query_filter=query_filter,
                limit=top_k,
                with_payload=True
            )
            
            # query_points returns a QueryResponse object with .points attribute
            return self._process_results(results.points)
            
        except Exception as e:
            logger.error(f"Sparse search error: {e}", exc_info=True)
            raise SearchError(f"Sparse search failed: {str(e)}")
    
    def _hybrid_search(
        self,
        query: str,
        top_k: int,
        user_id: Optional[str] = None,
        file_id: Optional[str] = None
    ) -> Tuple[List[str], List[Dict[str, Any]], List[float]]:
        """
        Hybrid search using Reciprocal Rank Fusion (RRF)
        
        Combines dense and sparse results using RRF algorithm
        """
        try:
            if not self.sparse_embeddings.available:
                logger.warning("Hybrid search requested but SPLADE not available, using dense search only")
                return self._dense_search(query, top_k, user_id, file_id)
            
            # Get both dense and sparse results (retrieve more for better fusion)
            retrieval_k = min(top_k * 3, 50)  # Retrieve 3x more for fusion
            
            logger.debug(f"Hybrid search: retrieving {retrieval_k} results from each method")
            
            # Dense search
            dense_docs, dense_metas, dense_scores = self._dense_search(
                query, retrieval_k, user_id, file_id
            )
            
            # Sparse search
            sparse_docs, sparse_metas, sparse_scores = self._sparse_search(
                query, retrieval_k, user_id, file_id
            )
            
            # Apply Reciprocal Rank Fusion
            fused_results = self._reciprocal_rank_fusion(
                dense_results=list(zip(dense_docs, dense_metas, dense_scores)),
                sparse_results=list(zip(sparse_docs, sparse_metas, sparse_scores)),
                k=60  # RRF constant (typical value)
            )
            
            # Take top_k after fusion
            fused_results = fused_results[:top_k]
            
            # Unpack results
            if not fused_results:
                return [], [], []
            
            documents = [r[0] for r in fused_results]
            metadatas = [r[1] for r in fused_results]
            scores = [r[2] for r in fused_results]
            
            logger.info(
                f"Hybrid search returned {len(documents)} results",
                extra={"query": query[:50], "user_id": user_id}
            )
            
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
        """
        Reciprocal Rank Fusion algorithm
        
        RRF score = sum(1 / (k + rank)) for each result list
        
        Args:
            dense_results: List of (doc, metadata, score) from dense search
            sparse_results: List of (doc, metadata, score) from sparse search
            k: RRF constant (typically 60)
            
        Returns:
            Fused results sorted by RRF score
        """
        # Build RRF scores
        rrf_scores = {}
        
        # Process dense results
        for rank, (doc, meta, score) in enumerate(dense_results, start=1):
            chunk_id = meta.get('chunk_id', '')
            if chunk_id not in rrf_scores:
                rrf_scores[chunk_id] = {
                    'doc': doc,
                    'meta': meta,
                    'rrf_score': 0.0,
                    'dense_rank': None,
                    'sparse_rank': None
                }
            rrf_scores[chunk_id]['rrf_score'] += 1.0 / (k + rank)
            rrf_scores[chunk_id]['dense_rank'] = rank
        
        # Process sparse results
        for rank, (doc, meta, score) in enumerate(sparse_results, start=1):
            chunk_id = meta.get('chunk_id', '')
            if chunk_id not in rrf_scores:
                rrf_scores[chunk_id] = {
                    'doc': doc,
                    'meta': meta,
                    'rrf_score': 0.0,
                    'dense_rank': None,
                    'sparse_rank': None
                }
            rrf_scores[chunk_id]['rrf_score'] += 1.0 / (k + rank)
            rrf_scores[chunk_id]['sparse_rank'] = rank
        
        # Sort by RRF score
        sorted_results = sorted(
            rrf_scores.values(),
            key=lambda x: x['rrf_score'],
            reverse=True
        )
        
        # Convert to output format
        fused = []
        for item in sorted_results:
            # Normalize RRF score to 0-1 range for consistency
            # Max RRF score would be ~0.033 (1/60 + 1/60), so normalize
            normalized_score = min(item['rrf_score'] / 0.05, 1.0)
            fused.append((item['doc'], item['meta'], normalized_score))
        
        logger.debug(
            f"RRF fusion: {len(dense_results)} dense + {len(sparse_results)} sparse "
            f"→ {len(fused)} unique results"
        )
        
        return fused
    
    def _build_filter(
        self,
        user_id: Optional[str] = None,
        file_id: Optional[str] = None
    ) -> Optional[Filter]:
        """Build Qdrant filter for user and file isolation"""
        must_conditions = []
        
        if user_id:
            must_conditions.append(
                FieldCondition(key="user_id", match=MatchValue(value=user_id))
            )
        
        if file_id:
            must_conditions.append(
                FieldCondition(key="file_id", match=MatchValue(value=file_id))
            )
        
        if must_conditions:
            return Filter(must=must_conditions)
        
        return None
    
    def _process_results(
        self,
        results
    ) -> Tuple[List[str], List[Dict[str, Any]], List[float]]:
        """Process Qdrant search results"""
        documents = []
        metadatas = []
        scores = []
        
        for result in results:
            payload = result.payload
            
            # Extract text
            text = payload.get('text', '')
            documents.append(text)
            
            # Build metadata dict
            metadata = {k: v for k, v in payload.items() if k != 'text'}
            metadatas.append(metadata)
            
            # Get score
            scores.append(float(result.score))
        
        return documents, metadatas, scores
    
    def delete_document_by_file_id(
        self,
        file_id: str,
        user_id: Optional[str] = None
    ) -> bool:
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
            
            logger.info(
                f"Deleted {len(point_ids)} chunks for file {file_id}",
                extra={"file_id": file_id, "user_id": user_id}
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error deleting document: {str(e)}", exc_info=True)
            raise VectorStoreException(f"Failed to delete document: {str(e)}")
    
    def get_collection_stats(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Get collection statistics"""
        try:
            query_filter = None
            if user_id:
                query_filter = Filter(
                    must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]
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
            logger.error(f"Error getting stats: {str(e)}", exc_info=True)
            return {
                'total_documents': 0,
                'total_chunks': 0,
                'documents': [],
                'error': str(e)
            }
    
    def reset_collection(self) -> bool:
        """Reset entire collection"""
        try:
            self.client.delete_collection(collection_name=self.collection_name)
            
            # Recreate with both dense and sparse vectors
            dense_dim = self._get_embedding_dimension()
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
            
            logger.warning("Collection reset - all data deleted")
            
            return True
            
        except Exception as e:
            logger.error(f"Error resetting collection: {str(e)}", exc_info=True)
            raise VectorStoreException(f"Failed to reset collection: {str(e)}")
    
    @property
    def collection(self):
        """Compatibility property for collection.count()"""
        class CollectionMock:
            def __init__(self, client, collection_name):
                self._client = client
                self._collection_name = collection_name
                self.name = collection_name
            
            def count(self):
                try:
                    info = self._client.get_collection(self._collection_name)
                    return info.points_count
                except:
                    return 0
        
        return CollectionMock(self.client, self.collection_name)
    def get_chunk_by_id(
        self,
        chunk_id: str,
        user_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific chunk by chunk_id with full payload
        
        Args:
            chunk_id: The unique chunk identifier
            user_id: Optional user ID for isolation (recommended)
            
        Returns:
            Dictionary with chunk data including:
            - chunk_id: Unique chunk identifier
            - file_id: Parent file identifier
            - text: Complete chunk text
            - All other metadata fields
            
            Returns None if chunk not found
        """
        try:
            # Build filter conditions
            must_conditions = [
                FieldCondition(key="chunk_id", match=MatchValue(value=chunk_id))
            ]
            
            if user_id:
                must_conditions.append(
                    FieldCondition(key="user_id", match=MatchValue(value=user_id))
                )
            
            query_filter = Filter(must=must_conditions)
            
            # Retrieve the chunk
            result = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=query_filter,
                limit=1,
                with_payload=True,
                with_vectors=False  # Don't need vectors for retrieval
            )
            
            points, _ = result
            
            if not points:
                logger.warning(
                    f"Chunk not found: chunk_id={chunk_id}, user_id={user_id}"
                )
                return None
            
            # Extract full payload
            point = points[0]
            payload = point.payload
            
            # Return structured chunk data
            return {
                "chunk_id": payload.get("chunk_id", chunk_id),
                "file_id": payload.get("file_id", ""),
                "text": payload.get("text", ""),
                "filename": payload.get("filename", ""),
                "user_id": payload.get("user_id", ""),
                "chunk_type": payload.get("chunk_type", "text"),
                "page_number": payload.get("page_number", 0),
                "char_count": payload.get("char_count", 0),
                "chunk_index": payload.get("chunk_index", 0),
                "created_at": payload.get("created_at", ""),
                # Include any other metadata fields
                "metadata": {k: v for k, v in payload.items() 
                           if k not in ['text', 'chunk_id', 'file_id']}
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
        preserve_metadata: bool = True
    ) -> bool:
        """
        Update an existing chunk with new text using upsert
        
        This method:
        1. Retrieves the existing chunk to get its metadata and point ID
        2. Re-embeds the new text (both dense and sparse)
        3. Upserts the chunk, replacing old vectors while preserving metadata
        
        Args:
            chunk_id: Unique chunk identifier
            file_id: Parent file identifier (for validation)
            new_text: New chunk text to embed and store
            user_id: User ID (for isolation and validation)
            preserve_metadata: If True, preserve all existing metadata
            
        Returns:
            True if update successful, raises exception otherwise
        """
        try:
            # Step 1: Retrieve existing chunk to get metadata and validate ownership
            logger.info(f"Updating chunk: chunk_id={chunk_id}, file_id={file_id}")
            
            existing_chunk = self.get_chunk_by_id(chunk_id, user_id)
            
            if not existing_chunk:
                raise VectorStoreException(
                    f"Chunk not found or access denied: {chunk_id}"
                )
            
            # Validate file_id matches
            if existing_chunk['file_id'] != file_id:
                raise VectorStoreException(
                    f"File ID mismatch: expected {existing_chunk['file_id']}, got {file_id}"
                )
            
            # Validate user ownership
            if existing_chunk['user_id'] != user_id:
                raise VectorStoreException(
                    f"Access denied: chunk belongs to different user"
                )
            
            # Step 2: Generate new embeddings
            logger.info(f"Generating new embeddings for chunk {chunk_id}")
            
            # Truncate text if needed
            truncated_text = new_text
            if len(new_text) > settings.embedding_max_length:
                logger.warning(
                    f"Text exceeds max length ({settings.embedding_max_length}), truncating"
                )
                truncated_text = new_text[:settings.embedding_max_length]
            
            # Generate dense embedding
            try:
                dense_vector = self.dense_embeddings.embed_query(truncated_text)
            except Exception as e:
                logger.error(f"Dense embedding failed: {e}")
                raise EmbeddingError(f"Failed to generate dense embedding: {str(e)}")
            
            # Generate sparse embedding
            try:
                sparse_vector = self.sparse_embeddings.embed_query(truncated_text)
            except Exception as e:
                logger.warning(f"Sparse embedding failed: {e}, using empty vector")
                sparse_vector = SparseVector(indices=[0], values=[0.0])
            
            # Step 3: Prepare updated payload
            if preserve_metadata:
                # Keep all existing metadata, only update text and timestamp
                updated_payload = existing_chunk['metadata'].copy()
                updated_payload.update({
                    "text": new_text,  # Store full text (not truncated)
                    "chunk_id": chunk_id,
                    "file_id": file_id,
                    "user_id": user_id,
                    "char_count": len(new_text),
                    "updated_at": datetime.utcnow().isoformat(),
                    "edited": True  # Flag to indicate this chunk was edited
                })
            else:
                # Create minimal payload
                updated_payload = {
                    "text": new_text,
                    "chunk_id": chunk_id,
                    "file_id": file_id,
                    "user_id": user_id,
                    "char_count": len(new_text),
                    "updated_at": datetime.utcnow().isoformat(),
                    "edited": True
                }
            
            # Step 4: Get the point ID for this chunk
            # We need to find the actual Qdrant point ID
            must_conditions = [
                FieldCondition(key="chunk_id", match=MatchValue(value=chunk_id)),
                FieldCondition(key="user_id", match=MatchValue(value=user_id))
            ]
            query_filter = Filter(must=must_conditions)
            
            result = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=query_filter,
                limit=1,
                with_payload=False,
                with_vectors=False
            )
            
            points, _ = result
            if not points:
                raise VectorStoreException(f"Could not find point for chunk {chunk_id}")
            
            point_id = points[0].id
            
            # Step 5: Upsert the chunk (replaces existing vectors and payload)
            logger.info(f"Upserting chunk with point_id={point_id}")
            
            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    PointStruct(
                        id=point_id,  # Use existing point ID to replace
                        vector={
                            "dense": dense_vector,
                            "sparse": sparse_vector
                        },
                        payload=updated_payload
                    )
                ]
            )
            
            logger.info(
                f"✓ Successfully updated chunk: {chunk_id}\n"
                f"  - Point ID: {point_id}\n"
                f"  - New char count: {len(new_text)}\n"
                f"  - Metadata preserved: {preserve_metadata}"
            )
            
            return True
            
        except VectorStoreException:
            raise
        except Exception as e:
            logger.error(f"Error updating chunk {chunk_id}: {e}", exc_info=True)
            raise VectorStoreException(f"Failed to update chunk: {str(e)}")
    
    