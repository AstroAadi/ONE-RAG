"""
Vector Store Service with Hybrid Search
Combines BM25 keyword search with semantic search
"""

from langchain_chroma import Chroma
from langchain_aws import BedrockEmbeddings
from langchain.docstore.document import Document as LangChainDocument
from typing import List, Dict, Tuple, Optional
from app.core.config import settings
import logging
import time
from botocore.exceptions import ClientError
from rank_bm25 import BM25Okapi
import numpy as np

logger = logging.getLogger(__name__)


class EnhancedVectorStoreService:
    """Vector store with hybrid search (BM25 + Semantic)"""
    
    def __init__(self):
        """Initialize vector store and BM25"""
        aws_config = {"region_name": settings.aws_region}

        if hasattr(settings, "aws_profile_name") and settings.aws_profile_name:
            aws_config["credentials_profile_name"] = settings.aws_profile_name

        self.embeddings = BedrockEmbeddings(
            model_id=settings.embedding_model_id,
            **aws_config
        )

        # Single vectorstore for all chunks
        self.vectorstore = Chroma(
            collection_name="kb_chunks",
            embedding_function=self.embeddings,
            persist_directory=settings.chroma_persist_dir
        )
        
        # BM25 index (will be built when documents are added)
        self.bm25_index = None
        self.bm25_docs = []  # Store documents for BM25
        self.bm25_ids = []   # Store IDs for BM25
        
        logger.info("VectorStore with hybrid search initialized")
    
    def _rebuild_bm25_index(self):
        """Rebuild BM25 index from vector store"""
        try:
            collection = self.vectorstore._collection
            all_data = collection.get()
            
            documents = all_data.get('documents', [])
            ids = all_data.get('ids', [])
            
            if documents:
                # Tokenize documents for BM25
                tokenized_docs = [doc.lower().split() for doc in documents]
                self.bm25_index = BM25Okapi(tokenized_docs)
                self.bm25_docs = documents
                self.bm25_ids = ids
                logger.info(f"BM25 index rebuilt with {len(documents)} documents")
            else:
                self.bm25_index = None
                self.bm25_docs = []
                self.bm25_ids = []
                logger.info("No documents for BM25 index")
                
        except Exception as e:
            logger.error(f"Error rebuilding BM25 index: {str(e)}")
            self.bm25_index = None
    
    def _retry_with_backoff(self, func, max_retries=3, base_delay=1):
        """Retry with exponential backoff for API throttling"""
        for attempt in range(max_retries):
            try:
                return func()
            except ClientError as e:
                error_code = e.response.get('Error', {}).get('Code', '')
                
                if error_code in ['ThrottlingException', 'TooManyRequestsException']:
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)
                        logger.warning(f"API throttled, retrying in {delay}s...")
                        time.sleep(delay)
                        continue
                    else:
                        logger.error(f"Max retries reached")
                        raise
                else:
                    raise
            except Exception as e:
                logger.error(f"Error in retry: {e}")
                raise
        
        raise Exception(f"Failed after {max_retries} retries")
    
    def add_document_chunks(
        self, 
        chunks: List[Dict], 
        file_id: str,
        filename: str,
        structured_data: Optional[Dict] = None
    ) -> int:
        """Add chunks to vector store and rebuild BM25 index"""
        try:
            if not chunks:
                logger.warning("No chunks to add")
                return 0
            
            # Prepare documents for vector store
            documents = []
            metadatas = []
            ids = []
            
            for chunk in chunks:
                # Skip if marked as parent (from old system)
                if chunk.get('metadata', {}).get('is_parent', False):
                    continue
                
                text = chunk.get('text', '')
                if not text.strip():
                    continue
                
                doc = LangChainDocument(
                    page_content=text,
                    metadata=chunk.get('metadata', {})
                )
                
                documents.append(doc)
                metadatas.append(chunk.get('metadata', {}))
                ids.append(chunk.get('chunk_id', f"chunk_{len(ids)}"))
            
            if not documents:
                logger.warning("No valid documents to add")
                return 0
            
            # Add to vector store with retry
            def add_docs():
                self.vectorstore.add_documents(
                    documents=documents,
                    ids=ids
                )
                return len(documents)
            
            added_count = self._retry_with_backoff(add_docs)
            logger.info(f"Added {added_count} chunks to vector store")
            
            # Rebuild BM25 index after adding documents
            self._rebuild_bm25_index()
            
            return added_count
            
        except Exception as e:
            logger.error(f"Error adding chunks: {str(e)}")
            raise
    
    def semantic_search(
        self,
        query: str,
        top_k: int = 5,
        file_id: Optional[str] = None
    ) -> Tuple[List[str], List[Dict], List[float]]:
        """
        Pure semantic search using embeddings
        
        Returns:
            Tuple of (documents, metadatas, distances)
        """
        try:
            # Build filter
            where_filter = None
            if file_id:
                where_filter = {"file_id": file_id}
            
            # Perform search with retry
            def do_search():
                if where_filter:
                    return self.vectorstore.similarity_search_with_score(
                        query,
                        k=top_k,
                        filter=where_filter
                    )
                else:
                    return self.vectorstore.similarity_search_with_score(
                        query,
                        k=top_k
                    )
            
            results = self._retry_with_backoff(do_search)
            
            # Parse results
            documents = []
            metadatas = []
            distances = []
            
            for doc, score in results:
                documents.append(doc.page_content)
                metadatas.append(doc.metadata)
                distances.append(score)
            
            return documents, metadatas, distances
            
        except Exception as e:
            logger.error(f"Semantic search error: {str(e)}")
            raise
    
    def keyword_search(
        self,
        query: str,
        top_k: int = 5,
        file_id: Optional[str] = None
    ) -> Tuple[List[str], List[Dict], List[float]]:
        """
        Keyword search using BM25
        
        Returns:
            Tuple of (documents, metadatas, scores)
        """
        try:
            if not self.bm25_index:
                logger.warning("BM25 index not available, rebuilding...")
                self._rebuild_bm25_index()
                
                if not self.bm25_index:
                    logger.warning("No documents for keyword search")
                    return [], [], []
            
            # Tokenize query
            tokenized_query = query.lower().split()
            
            # Get BM25 scores
            bm25_scores = self.bm25_index.get_scores(tokenized_query)
            
            # Get top-k indices
            top_indices = np.argsort(bm25_scores)[::-1][:top_k * 3]  # Get more for filtering
            
            # Get full metadata from vector store
            collection = self.vectorstore._collection
            
            documents = []
            metadatas = []
            scores = []
            
            for idx in top_indices:
                if idx >= len(self.bm25_ids):
                    continue
                    
                chunk_id = self.bm25_ids[idx]
                score = float(bm25_scores[idx])
                
                # Get metadata from vector store
                result = collection.get(
                    ids=[chunk_id],
                    include=['documents', 'metadatas']
                )
                
                if result and result['documents']:
                    doc_text = result['documents'][0]
                    doc_metadata = result['metadatas'][0]
                    
                    # Apply file filter if specified
                    if file_id and doc_metadata.get('file_id') != file_id:
                        continue
                    
                    documents.append(doc_text)
                    metadatas.append(doc_metadata)
                    scores.append(score)
                    
                    if len(documents) >= top_k:
                        break
            
            return documents, metadatas, scores
            
        except Exception as e:
            logger.error(f"Keyword search error: {str(e)}")
            return [], [], []
    
    def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        file_id: Optional[str] = None,
        keyword_weight: float = 0.3,
        semantic_weight: float = 0.7
    ) -> Tuple[List[str], List[Dict], List[float]]:
        """
        Hybrid search combining BM25 keyword and semantic search
        
        Args:
            query: Search query
            top_k: Number of results to return
            file_id: Optional file filter
            keyword_weight: Weight for keyword search (0-1)
            semantic_weight: Weight for semantic search (0-1)
            
        Returns:
            Tuple of (documents, metadatas, combined_scores)
        """
        try:
            # Validate weights
            total_weight = keyword_weight + semantic_weight
            if abs(total_weight - 1.0) > 0.01:
                logger.warning(f"Weights don't sum to 1.0 ({total_weight}), normalizing...")
                keyword_weight = keyword_weight / total_weight
                semantic_weight = semantic_weight / total_weight
            
            logger.info(f"Hybrid search: keyword={keyword_weight:.2f}, semantic={semantic_weight:.2f}")
            
            # Perform both searches (get more results for merging)
            search_k = min(top_k * 3, 30)  # Get 3x results for better fusion
            
            # Semantic search
            sem_docs, sem_metas, sem_distances = self.semantic_search(query, search_k, file_id)
            
            # Keyword search
            kw_docs, kw_metas, kw_scores = self.keyword_search(query, search_k, file_id)
            
            # If one search returns no results, use the other
            if not sem_docs and not kw_docs:
                logger.warning("Both searches returned no results")
                return [], [], []
            elif not sem_docs:
                logger.info("Using keyword search only")
                return kw_docs[:top_k], kw_metas[:top_k], kw_scores[:top_k]
            elif not kw_docs:
                logger.info("Using semantic search only")
                return sem_docs[:top_k], sem_metas[:top_k], sem_distances[:top_k]
            
            # Normalize scores to 0-1 range
            # For semantic: lower distance is better, so invert
            sem_scores_norm = []
            if sem_distances:
                max_dist = max(sem_distances) if max(sem_distances) > 0 else 1.0
                sem_scores_norm = [1.0 - (d / max_dist) for d in sem_distances]
            
            # For keyword: higher score is better
            kw_scores_norm = []
            if kw_scores:
                max_score = max(kw_scores) if max(kw_scores) > 0 else 1.0
                kw_scores_norm = [s / max_score for s in kw_scores]
            
            # Create scoring dictionaries
            semantic_dict = {}
            for doc, meta, score in zip(sem_docs, sem_metas, sem_scores_norm):
                chunk_id = meta.get('chunk_id', doc[:50])  # Use chunk_id or first 50 chars
                semantic_dict[chunk_id] = {
                    'document': doc,
                    'metadata': meta,
                    'score': score
                }
            
            keyword_dict = {}
            for doc, meta, score in zip(kw_docs, kw_metas, kw_scores_norm):
                chunk_id = meta.get('chunk_id', doc[:50])
                keyword_dict[chunk_id] = {
                    'document': doc,
                    'metadata': meta,
                    'score': score
                }
            
            # Combine scores using weighted sum
            all_chunk_ids = set(semantic_dict.keys()) | set(keyword_dict.keys())
            
            combined_results = []
            for chunk_id in all_chunk_ids:
                sem_score = semantic_dict.get(chunk_id, {}).get('score', 0.0)
                kw_score = keyword_dict.get(chunk_id, {}).get('score', 0.0)
                
                # Weighted combination
                combined_score = (semantic_weight * sem_score) + (keyword_weight * kw_score)
                
                # Get document and metadata (prefer semantic if available)
                if chunk_id in semantic_dict:
                    doc = semantic_dict[chunk_id]['document']
                    meta = semantic_dict[chunk_id]['metadata']
                else:
                    doc = keyword_dict[chunk_id]['document']
                    meta = keyword_dict[chunk_id]['metadata']
                
                combined_results.append({
                    'document': doc,
                    'metadata': meta,
                    'combined_score': combined_score,
                    'semantic_score': sem_score,
                    'keyword_score': kw_score
                })
            
            # Sort by combined score (descending)
            combined_results.sort(key=lambda x: x['combined_score'], reverse=True)
            
            # Take top-k
            top_results = combined_results[:top_k]
            
            # Extract results
            final_docs = [r['document'] for r in top_results]
            final_metas = [r['metadata'] for r in top_results]
            final_scores = [r['combined_score'] for r in top_results]
            
            logger.info(f"Hybrid search returned {len(final_docs)} results")
            
            return final_docs, final_metas, final_scores
            
        except Exception as e:
            logger.error(f"Hybrid search error: {str(e)}")
            # Fallback to semantic search
            logger.info("Falling back to semantic search")
            return self.semantic_search(query, top_k, file_id)
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        file_id: Optional[str] = None
    ) -> Tuple[List[str], List[Dict], List[float]]:
        """
        Default search (uses hybrid with default weights)
        """
        return self.hybrid_search(
            query, 
            top_k, 
            file_id,
            keyword_weight=settings.default_keyword_weight,
            semantic_weight=settings.default_semantic_weight
        )
    
    def search_by_metadata(
        self,
        filters: Dict,
        limit: int = 10
    ) -> List[Dict]:
        """Search by metadata filters"""
        try:
            results = self.vectorstore.similarity_search(
                query="",
                k=limit,
                filter=filters
            )
            
            return [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in results
            ]
            
        except Exception as e:
            logger.error(f"Metadata search error: {str(e)}")
            return []
    
    def get_collection_stats(self) -> Dict:
        """Get statistics about the collection"""
        try:
            collection = self.vectorstore._collection
            count = collection.count()
            
            # Get unique file IDs
            all_data = collection.get()
            metadatas = all_data.get('metadatas', [])
            
            file_docs = {}
            for metadata in metadatas:
                file_id = metadata.get('file_id', 'unknown')
                filename = metadata.get('filename', 'unknown')
                
                if file_id not in file_docs:
                    file_docs[file_id] = {
                        'file_id': file_id,
                        'filename': filename,
                        'chunk_count': 0
                    }
                
                file_docs[file_id]['chunk_count'] += 1
            
            return {
                'total_documents': len(file_docs),
                'total_chunks': count,
                'documents': list(file_docs.values()),
                'bm25_indexed': len(self.bm25_docs) if self.bm25_docs else 0
            }
            
        except Exception as e:
            logger.error(f"Error getting stats: {str(e)}")
            return {
                'total_documents': 0,
                'total_chunks': 0,
                'documents': [],
                'bm25_indexed': 0
            }
    
    def delete_document_by_file_id(self, file_id: str) -> bool:
        """Delete all chunks for a file"""
        try:
            collection = self.vectorstore._collection
            
            # Get all IDs for this file
            results = collection.get(
                where={"file_id": file_id}
            )
            
            ids_to_delete = results.get('ids', [])
            
            if ids_to_delete:
                collection.delete(ids=ids_to_delete)
                logger.info(f"Deleted {len(ids_to_delete)} chunks for file {file_id}")
                
                # Rebuild BM25 index
                self._rebuild_bm25_index()
                
                return True
            else:
                logger.warning(f"No chunks found for file {file_id}")
                return False
            
        except Exception as e:
            logger.error(f"Error deleting document: {str(e)}")
            return False
    
    def reset_collection(self) -> bool:
        """Reset entire collection"""
        try:
            collection = self.vectorstore._collection
            
            # Get all IDs
            all_data = collection.get()
            ids = all_data.get('ids', [])
            
            if ids:
                collection.delete(ids=ids)
                logger.info(f"Reset collection - deleted {len(ids)} chunks")
            
            # Clear BM25 index
            self.bm25_index = None
            self.bm25_docs = []
            self.bm25_ids = []
            
            return True
            
        except Exception as e:
            logger.error(f"Error resetting collection: {str(e)}")
            return False