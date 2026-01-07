"""
Improved Vector Store Service
- Proper user isolation via metadata filtering
- Better ChromaDB usage (minimal metadata)
- Separation of concerns (store only embeddings, use SQL for metadata)
- Better error handling
"""

import chromadb
from chromadb.config import Settings as ChromaSettings
from typing import List, Dict, Any, Tuple, Optional
import logging

from rank_bm25 import BM25Okapi

from app.core.config import settings
from app.core.exceptions import (
    VectorStoreException, 
    EmbeddingError, 
    SearchError,
    NoDocumentsFoundError
)
from app.core.monitoring import get_logger

logger = get_logger(__name__)


class VectorStoreService:
    """
    Vector store service with user isolation
    
    Stores document chunks with embeddings in ChromaDB
    Filters by user_id for isolation
    """
    
    def __init__(self):
        """Initialize ChromaDB client and collection"""
        try:
            self.client = chromadb.PersistentClient(
                path=str(settings.chroma_persist_dir),
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=settings.chroma_collection_name,
                metadata={"description": "RAG document chunks with user isolation"}
            )
            
            # BM25 index (in-memory)
            self.bm25_index: Optional[BM25Okapi] = None
            self.bm25_documents: List[str] = []
            self.bm25_metadatas: List[Dict] = []
            
            logger.info(f"Vector store initialized: {self.collection.name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}", exc_info=True)
            raise VectorStoreException(f"Failed to initialize vector store: {str(e)}")
    
    def add_document_chunks(
        self,
        chunks: List[Dict[str, Any]],
        file_id: str,
        filename: str,
        structured_data: Optional[Dict] = None
    ) -> int:
        """
        Add document chunks to vector store
        
        Args:
            chunks: List of chunk dictionaries with 'text', 'chunk_id', and 'metadata'
            file_id: Document file ID
            filename: Document filename
            structured_data: Additional structured data (optional)
            
        Returns:
            Number of chunks added
            
        Note:
            User ID must be in chunk metadata for isolation
        """
        if not chunks:
            logger.warning(f"No chunks provided for {file_id}")
            return 0
        
        try:
            texts = []
            ids = []
            metadatas = []
            
            for chunk in chunks:
                # Extract data
                text = chunk.get('text', '')
                chunk_id = chunk.get('chunk_id', '')
                metadata = chunk.get('metadata', {})
                
                if not text or not chunk_id:
                    logger.warning(f"Skipping invalid chunk: {chunk_id}")
                    continue
                
                # Ensure user_id is in metadata (CRITICAL for isolation)
                if 'user_id' not in metadata:
                    logger.error(f"Chunk {chunk_id} missing user_id!")
                    raise VectorStoreException(
                        f"Cannot add chunk without user_id: {chunk_id}"
                    )
                
                # Prepare minimal metadata for ChromaDB
                # Store only essential fields (ChromaDB has size limits)
                chroma_metadata = {
                    'file_id': str(metadata.get('file_id', file_id)),
                    'user_id': str(metadata.get('user_id')),  # CRITICAL
                    'filename': str(metadata.get('filename', filename)),
                    'chunk_type': str(metadata.get('chunk_type', 'text')),
                    'page_number': int(metadata.get('page_number', 0)),  # Keep as int
                    'chunk_index': int(metadata.get('chunk_index', 0)),  # Keep as int
                    'chunk_id': str(chunk_id)  # Add chunk_id for easier lookup
                }
                
                texts.append(text)
                ids.append(chunk_id)
                metadatas.append(chroma_metadata)
            
            if not texts:
                logger.warning(f"No valid chunks to add for {file_id}")
                return 0
            
            # Add to ChromaDB (will auto-generate embeddings)
            self.collection.add(
                documents=texts,
                ids=ids,
                metadatas=metadatas
            )
            
            # Rebuild BM25 index
            self._rebuild_bm25_index()
            
            logger.info(
                f"Added {len(texts)} chunks to vector store",
                extra={"file_id": file_id, "chunks": len(texts)}
            )
            
            return len(texts)
            
        except VectorStoreException:
            raise
        except Exception as e:
            logger.error(f"Error adding chunks: {str(e)}", exc_info=True)
            raise VectorStoreException(f"Failed to add chunks: {str(e)}")
    
    def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        user_id: Optional[str] = None,
        file_id: Optional[str] = None,
        keyword_weight: float = 0.3,
        semantic_weight: float = 0.7
    ) -> Tuple[List[str], List[Dict], List[float]]:
        """
        Hybrid search combining BM25 and semantic search with user isolation
        
        Args:
            query: Search query
            top_k: Number of results to return
            user_id: User ID for isolation (optional for backward compatibility)
            file_id: Filter by specific file (optional)
            keyword_weight: Weight for BM25 keyword search
            semantic_weight: Weight for semantic search
            
        Returns:
            Tuple of (documents, metadatas, scores)
            
        Raises:
            SearchError: If search fails
            NoDocumentsFoundError: If no documents in collection
        """
        try:
            # Check if collection is empty
            collection_count = self.collection.count()
            if collection_count == 0:
                raise NoDocumentsFoundError()
            
            # Build where filter for user isolation
            where_filter = {}
            if user_id:
                where_filter['user_id'] = user_id
            if file_id:
                where_filter['file_id'] = file_id
            
            # Perform semantic search
            semantic_results = self.collection.query(
                query_texts=[query],
                n_results=min(top_k * 2, 50),  # Get more for hybrid ranking
                where=where_filter if where_filter else None
            )
            
            if not semantic_results or not semantic_results['documents']:
                logger.warning(f"No semantic results for query")
                return [], [], []
            
            # Extract semantic results
            semantic_docs = semantic_results['documents'][0]
            semantic_metas = semantic_results['metadatas'][0]
            semantic_distances = semantic_results['distances'][0]
            
            # Convert distances to similarity scores (cosine distance -> similarity)
            # ChromaDB returns L2 distances, ensure scores are always positive
            semantic_scores = []
            for d in semantic_distances:
                # Convert distance to similarity (0 to 1 range)
                # For L2 distance: similarity = 1 / (1 + distance)
                similarity = max(0.0, min(1.0, 1.0 / (1.0 + abs(d))))
                semantic_scores.append(similarity)
            
            # Perform BM25 keyword search
            if self.bm25_index and keyword_weight > 0:
                keyword_results = self._bm25_search(
                    query, top_k * 2, user_id, file_id
                )
                
                # Combine scores
                final_results = self._combine_search_results(
                    semantic_docs, semantic_metas, semantic_scores,
                    keyword_results,
                    keyword_weight, semantic_weight,
                    top_k
                )
            else:
                # Semantic only
                final_results = list(zip(
                    semantic_docs[:top_k],
                    semantic_metas[:top_k],
                    semantic_scores[:top_k]
                ))
            
            if not final_results:
                logger.info(f"No results for query after filtering")
                return [], [], []
            
            # Unzip results
            documents, metadatas, scores = zip(*final_results)
            
            return list(documents), list(metadatas), list(scores)
            
        except NoDocumentsFoundError:
            raise
        except Exception as e:
            logger.error(f"Search error: {str(e)}", exc_info=True)
            raise SearchError(f"Search failed: {str(e)}")
    
    def _bm25_search(
        self,
        query: str,
        top_k: int,
        user_id: Optional[str] = None,
        file_id: Optional[str] = None
    ) -> List[Tuple[str, Dict, float]]:
        """Perform BM25 keyword search with filtering"""
        if not self.bm25_index:
            return []
        
        # Tokenize query
        tokenized_query = query.lower().split()
        
        # Get BM25 scores
        scores = self.bm25_index.get_scores(tokenized_query)
        
        # Filter and rank
        results = []
        for idx, score in enumerate(scores):
            if idx >= len(self.bm25_documents):
                break
            
            metadata = self.bm25_metadatas[idx]
            
            # Apply filters
            if user_id and metadata.get('user_id') != user_id:
                continue
            if file_id and metadata.get('file_id') != file_id:
                continue
            
            if score > 0:
                results.append((
                    self.bm25_documents[idx],
                    metadata,
                    float(score)
                ))
        
        # Sort by score and return top_k
        results.sort(key=lambda x: x[2], reverse=True)
        return results[:top_k]
    
    def _combine_search_results(
        self,
        semantic_docs: List[str],
        semantic_metas: List[Dict],
        semantic_scores: List[float],
        keyword_results: List[Tuple[str, Dict, float]],
        keyword_weight: float,
        semantic_weight: float,
        top_k: int
    ) -> List[Tuple[str, Dict, float]]:
        """Combine semantic and keyword search results"""
        
        # Create document map for combining
        doc_map = {}
        
        # Add semantic results
        for idx, (doc, meta, score) in enumerate(zip(semantic_docs, semantic_metas, semantic_scores)):
            # Use chunk_id if available, otherwise use index as unique identifier
            chunk_id = str(meta.get('chunk_id', f'semantic_{idx}'))
            doc_map[chunk_id] = {
                'doc': doc,
                'meta': meta,
                'semantic_score': score,
                'keyword_score': 0.0
            }
        
        # Add keyword results
        for idx, (doc, meta, score) in enumerate(keyword_results):
            chunk_id = str(meta.get('chunk_id', f'keyword_{idx}'))
            if chunk_id in doc_map:
                doc_map[chunk_id]['keyword_score'] = score
            else:
                doc_map[chunk_id] = {
                    'doc': doc,
                    'meta': meta,
                    'semantic_score': 0.0,
                    'keyword_score': score
                }
        
        # Normalize scores
        max_semantic = max((v['semantic_score'] for v in doc_map.values()), default=1.0)
        max_keyword = max((v['keyword_score'] for v in doc_map.values()), default=1.0)
        
        # Calculate combined scores
        combined_results = []
        for chunk_id, data in doc_map.items():
            normalized_semantic = data['semantic_score'] / max_semantic if max_semantic > 0 else 0
            normalized_keyword = data['keyword_score'] / max_keyword if max_keyword > 0 else 0
            
            combined_score = (
                semantic_weight * normalized_semantic +
                keyword_weight * normalized_keyword
            )
            
            combined_results.append((
                data['doc'],
                data['meta'],
                combined_score
            ))
        
        # Sort by combined score
        combined_results.sort(key=lambda x: x[2], reverse=True)
        
        return combined_results[:top_k]
    
    def _rebuild_bm25_index(self) -> None:
        """Rebuild BM25 index from all documents"""
        try:
            # Get all documents
            all_docs = self.collection.get()
            
            if not all_docs or not all_docs['documents']:
                self.bm25_index = None
                self.bm25_documents = []
                self.bm25_metadatas = []
                return
            
            self.bm25_documents = all_docs['documents']
            self.bm25_metadatas = all_docs['metadatas']
            
            # Tokenize documents
            tokenized_docs = [doc.lower().split() for doc in self.bm25_documents]
            
            # Create BM25 index
            self.bm25_index = BM25Okapi(tokenized_docs)
            
            logger.info(f"BM25 index rebuilt with {len(self.bm25_documents)} documents")
            
        except Exception as e:
            logger.error(f"Failed to rebuild BM25 index: {e}")
            self.bm25_index = None
    
    def delete_document_by_file_id(
        self,
        file_id: str,
        user_id: Optional[str] = None
    ) -> bool:
        """
        Delete all chunks for a specific file with user verification
        
        Args:
            file_id: File ID to delete
            user_id: User ID for verification (optional)
            
        Returns:
            True if deleted, False if not found
        """
        try:
            # ✅ FIXED - Use $and operator for multiple conditions
            where_filter = {
                "$and": [
                    {"file_id": file_id},
                    {"user_id": user_id}
                ]
            }
            
            results = self.collection.get(where=where_filter)
            
            if not results or not results['ids']:
                return False
            
            # Delete by IDs
            self.collection.delete(ids=results['ids'])
            
            # Rebuild BM25
            self._rebuild_bm25_index()
            
            logger.info(
                f"Deleted {len(results['ids'])} chunks for file {file_id}",
                extra={"file_id": file_id, "user_id": user_id}
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error deleting document: {str(e)}", exc_info=True)
            raise VectorStoreException(f"Failed to delete document: {str(e)}")
    
    def get_collection_stats(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get collection statistics with optional user filtering
        
        Args:
            user_id: Filter by user ID (optional)
            
        Returns:
            Dictionary with statistics
        """
        try:
            # Get all documents with optional filtering
            where_filter = {'user_id': user_id} if user_id else None
            results = self.collection.get(where=where_filter)
            
            if not results or not results['metadatas']:
                return {
                    'total_documents': 0,
                    'total_chunks': 0,
                    'documents': []
                }
            
            # Count unique files
            file_ids = {}
            for meta in results['metadatas']:
                fid = meta.get('file_id', 'unknown')
                if fid not in file_ids:
                    file_ids[fid] = {
                        'file_id': fid,
                        'filename': meta.get('filename', 'unknown'),
                        'chunk_count': 0
                    }
                file_ids[fid]['chunk_count'] += 1
            
            return {
                'total_documents': len(file_ids),
                'total_chunks': len(results['ids']),
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
        """
        Reset entire collection (admin function - use with caution)
        
        Returns:
            True if successful
        """
        try:
            self.client.delete_collection(name=settings.chroma_collection_name)
            
            self.collection = self.client.create_collection(
                name=settings.chroma_collection_name,
                metadata={"description": "RAG document chunks with user isolation"}
            )
            
            self.bm25_index = None
            self.bm25_documents = []
            self.bm25_metadatas = []
            
            logger.warning("Collection reset - all data deleted")
            
            return True
            
        except Exception as e:
            logger.error(f"Error resetting collection: {str(e)}", exc_info=True)
            raise VectorStoreException(f"Failed to reset collection: {str(e)}")