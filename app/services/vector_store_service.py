"""
Enhanced Vector Store Service with Hybrid Search and Re-ranking
Supports multi-vector retrieval, metadata filtering, and contextual compression
"""

from langchain_chroma import Chroma
from langchain_aws import BedrockEmbeddings
from langchain.docstore.document import Document as LangChainDocument
from typing import List, Dict, Tuple, Optional, Any
from app.core.config import settings
import logging
from collections import defaultdict
import json
import time
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


class EnhancedVectorStoreService:
    """
    Advanced vector store with hybrid search capabilities
    """
    
    def __init__(self):
        """Initialize with multi-collection support"""
        aws_config = {
            "region_name": settings.aws_region
        }

        if hasattr(settings, "aws_profile_name") and settings.aws_profile_name:
            aws_config["credentials_profile_name"] = settings.aws_profile_name

        self.embeddings = BedrockEmbeddings(
            model_id=settings.embedding_model_id,
            **aws_config
        )

        # Primary vectorstore for child chunks (detailed retrieval)
        self.child_vectorstore = Chroma(
            collection_name="kb_child_chunks",
            embedding_function=self.embeddings,
            persist_directory=settings.chroma_persist_dir
        )
        
        # Parent vectorstore for broader context
        self.parent_vectorstore = Chroma(
            collection_name="kb_parent_chunks",
            embedding_function=self.embeddings,
            persist_directory=settings.chroma_persist_dir
        )
        
        logger.info("Enhanced VectorStore initialized with parent-child collections")
    
    def _retry_with_backoff(self, func, max_retries=3, base_delay=1):
        """
        Retry function with exponential backoff for API throttling
        
        Args:
            func: Function to retry
            max_retries: Maximum number of retries
            base_delay: Base delay in seconds
        """
        for attempt in range(max_retries):
            try:
                return func()
            except ClientError as e:
                error_code = e.response.get('Error', {}).get('Code', '')
                
                # Retry on throttling errors
                if error_code in ['ThrottlingException', 'TooManyRequestsException', 'ServiceUnavailableException']:
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)  # Exponential backoff
                        logger.warning(f"API throttled, retrying in {delay}s... (attempt {attempt + 1}/{max_retries})")
                        time.sleep(delay)
                        continue
                    else:
                        logger.error(f"Max retries reached for throttling error: {e}")
                        raise
                else:
                    # Non-throttling error, raise immediately
                    logger.error(f"Non-retriable error: {e}")
                    raise
            except Exception as e:
                logger.error(f"Unexpected error in retry wrapper: {e}")
                raise
        
        raise Exception(f"Failed after {max_retries} retries")
    
    def add_document_chunks(
        self, 
        chunks: List[Dict], 
        file_id: str,
        filename: str,
        structured_data: Optional[Dict] = None
    ) -> int:
        """
        Add chunks with parent-child relationships
        
        Args:
            chunks: List of chunk dictionaries with metadata
            file_id: Unique file identifier
            filename: Original filename
            structured_data: Optional structured document data
            
        Returns:
            Number of chunks added
        """
        if not chunks:
            return 0
        
        try:
            child_docs = []
            parent_docs = []
            child_ids = []
            parent_ids = []
            
            for i, chunk in enumerate(chunks):
                metadata = chunk.get('metadata', {})
                
                # Enhance metadata
                enhanced_metadata = {
                    "file_id": file_id,
                    "filename": filename,
                    "chunk_index": metadata.get('chunk_index', i),
                    "char_count": chunk.get('char_count', len(chunk['text'])),
                    "chunk_id": chunk.get('chunk_id', f"{file_id}_chunk_{i}"),
                    "chunk_type": metadata.get('chunk_type', 'text'),
                    "page_number": metadata.get('page_number', 0),
                    "section_hierarchy": str(metadata.get('section_hierarchy', '')),  # Store as string, not JSON
                    "parent_chunk_id": metadata.get('parent_chunk_id', ''),
                    "is_parent": bool(metadata.get('is_parent', False))  # Store as boolean
                }
                
                # Add structured data reference if available (as booleans)
                if structured_data:
                    enhanced_metadata['has_images'] = bool(structured_data.get('statistics', {}).get('images', 0) > 0)
                    enhanced_metadata['has_tables'] = bool(structured_data.get('statistics', {}).get('tables', 0) > 0)
                
                doc = LangChainDocument(
                    page_content=chunk['text'],
                    metadata=enhanced_metadata
                )
                
                # Separate parent and child chunks
                if enhanced_metadata['is_parent']:
                    parent_docs.append(doc)
                    parent_ids.append(enhanced_metadata['chunk_id'])
                else:
                    child_docs.append(doc)
                    child_ids.append(enhanced_metadata['chunk_id'])
            
            # Add to respective vectorstores with retry logic
            if child_docs:
                try:
                    def add_child_docs():
                        self.child_vectorstore.add_documents(documents=child_docs, ids=child_ids)
                        return True
                    
                    self._retry_with_backoff(add_child_docs)
                    logger.info(f"Added {len(child_docs)} child chunks")
                except Exception as e:
                    logger.error(f"Failed to add child chunks after retries: {e}")
                    raise Exception(f"Embedding generation failed for child chunks: {str(e)}")
            
            if parent_docs:
                try:
                    def add_parent_docs():
                        self.parent_vectorstore.add_documents(documents=parent_docs, ids=parent_ids)
                        return True
                    
                    self._retry_with_backoff(add_parent_docs)
                    logger.info(f"Added {len(parent_docs)} parent chunks")
                except Exception as e:
                    logger.error(f"Failed to add parent chunks after retries: {e}")
                    raise Exception(f"Embedding generation failed for parent chunks: {str(e)}")
            
            total_added = len(child_docs) + len(parent_docs)
            logger.info(f"Total chunks added for file {file_id}: {total_added}")
            
            return total_added
            
        except Exception as e:
            logger.error(f"Error adding chunks: {str(e)}")
            raise
    
    def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        file_id: Optional[str] = None,
        search_mode: str = "hybrid",  # 'semantic', 'parent', 'hybrid'
        filters: Optional[Dict] = None
    ) -> Tuple[List[str], List[Dict], List[float]]:
        """
        Advanced hybrid search with multiple strategies
        
        Args:
            query: Search query
            top_k: Number of results
            file_id: Optional file filter
            search_mode: Search strategy
            filters: Additional metadata filters
            
        Returns:
            Tuple of (documents, metadatas, distances)
        """
        try:
            # Build filter dictionary
            filter_dict = self._build_filter_dict(file_id, filters)
            
            if search_mode == "parent":
                # Search only parent chunks for broad context
                results = self._search_parents(query, top_k, filter_dict)
            elif search_mode == "semantic":
                # Search only child chunks for specific details
                results = self._search_children(query, top_k, filter_dict)
            else:
                # Hybrid: Search both and merge results
                results = self._hybrid_search_internal(query, top_k, filter_dict)
            
            documents, metadatas, distances = results
            
            # Post-process: Fetch parent context for child chunks
            enhanced_results = self._enhance_with_parent_context(
                documents, metadatas, distances
            )
            
            logger.info(f"Retrieved {len(enhanced_results[0])} chunks via {search_mode} search")
            return enhanced_results
            
        except Exception as e:
            logger.error(f"Error in hybrid search: {str(e)}")
            raise
    
    def _build_filter_dict(
        self,
        file_id: Optional[str],
        additional_filters: Optional[Dict]
    ) -> Optional[Dict]:
        """Build comprehensive filter dictionary"""
        filter_dict = {}
        
        if file_id:
            filter_dict["file_id"] = file_id
        
        if additional_filters:
            filter_dict.update(additional_filters)
        
        return filter_dict if filter_dict else None
    
    def _search_children(
        self,
        query: str,
        top_k: int,
        filter_dict: Optional[Dict]
    ) -> Tuple[List[str], List[Dict], List[float]]:
        """Search child chunks"""
        results = self.child_vectorstore.similarity_search_with_score(
            query=query,
            k=top_k,
            filter=filter_dict
        )
        
        documents = [doc.page_content for doc, score in results]
        metadatas = [doc.metadata for doc, score in results]
        distances = [score for doc, score in results]
        
        return documents, metadatas, distances
    
    def _search_parents(
        self,
        query: str,
        top_k: int,
        filter_dict: Optional[Dict]
    ) -> Tuple[List[str], List[Dict], List[float]]:
        """Search parent chunks"""
        results = self.parent_vectorstore.similarity_search_with_score(
            query=query,
            k=top_k,
            filter=filter_dict
        )
        
        documents = [doc.page_content for doc, score in results]
        metadatas = [doc.metadata for doc, score in results]
        distances = [score for doc, score in results]
        
        return documents, metadatas, distances
    
    def _hybrid_search_internal(
        self,
        query: str,
        top_k: int,
        filter_dict: Optional[Dict]
    ) -> Tuple[List[str], List[Dict], List[float]]:
        """
        Hybrid search: Combine child and parent results
        """
        # Search both collections
        child_k = max(3, int(top_k * 0.7))  # 70% from children
        parent_k = max(2, int(top_k * 0.3))  # 30% from parents
        
        child_results = self.child_vectorstore.similarity_search_with_score(
            query=query,
            k=child_k,
            filter=filter_dict
        )
        
        parent_results = self.parent_vectorstore.similarity_search_with_score(
            query=query,
            k=parent_k,
            filter=filter_dict
        )
        
        # Merge and re-rank results
        all_results = []
        
        for doc, score in child_results:
            all_results.append({
                'document': doc.page_content,
                'metadata': doc.metadata,
                'score': score,
                'source': 'child'
            })
        
        for doc, score in parent_results:
            all_results.append({
                'document': doc.page_content,
                'metadata': doc.metadata,
                'score': score,
                'source': 'parent'
            })
        
        # Re-rank by score
        all_results.sort(key=lambda x: x['score'])
        
        # Take top_k
        top_results = all_results[:top_k]
        
        documents = [r['document'] for r in top_results]
        metadatas = [r['metadata'] for r in top_results]
        distances = [r['score'] for r in top_results]
        
        return documents, metadatas, distances
    
    def _enhance_with_parent_context(
        self,
        documents: List[str],
        metadatas: List[Dict],
        distances: List[float]
    ) -> Tuple[List[str], List[Dict], List[float]]:
        """
        Enhance child chunks with parent context
        """
        enhanced_docs = []
        enhanced_metas = []
        enhanced_distances = []
        
        for doc, meta, dist in zip(documents, metadatas, distances):
            # Check if this is a child chunk with a parent
            parent_id = meta.get('parent_chunk_id', '')
            
            if parent_id and not meta.get('is_parent', False):
                # Try to fetch parent context
                try:
                    parent_result = self.parent_vectorstore._collection.get(
                        ids=[parent_id],
                        include=['documents', 'metadatas']
                    )
                    
                    if parent_result and parent_result['documents']:
                        parent_doc = parent_result['documents'][0]
                        parent_meta = parent_result['metadatas'][0]
                        
                        # Add parent context to metadata
                        meta['parent_context'] = parent_doc[:500]  # Preview
                        meta['has_parent_context'] = True
                except Exception as e:
                    logger.warning(f"Could not fetch parent context: {e}")
                    meta['has_parent_context'] = False
            
            enhanced_docs.append(doc)
            enhanced_metas.append(meta)
            enhanced_distances.append(dist)
        
        return enhanced_docs, enhanced_metas, enhanced_distances
    
    def search_by_metadata(
        self,
        metadata_filters: Dict[str, Any],
        limit: int = 10
    ) -> List[Dict]:
        """
        Search chunks by metadata only (no semantic search)
        Useful for filtering by page, section, type, etc.
        """
        try:
            collection = self.child_vectorstore._collection
            results = collection.get(
                where=metadata_filters,
                limit=limit,
                include=['documents', 'metadatas']
            )
            
            chunks = []
            if results and results['documents']:
                for doc, meta in zip(results['documents'], results['metadatas']):
                    chunks.append({
                        'text': doc,
                        'metadata': meta
                    })
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error in metadata search: {str(e)}")
            return []
    
    def get_collection_stats(self) -> Dict:
        """Get comprehensive collection statistics"""
        try:
            child_collection = self.child_vectorstore._collection
            parent_collection = self.parent_vectorstore._collection
            
            child_count = child_collection.count()
            parent_count = parent_collection.count()
            
            # Get unique files
            unique_files = {}
            
            if child_count > 0:
                child_data = child_collection.get(include=['metadatas'])
                metadatas = child_data.get('metadatas', [])
                
                for metadata in metadatas:
                    file_id = metadata.get('file_id')
                    filename = metadata.get('filename')
                    if file_id and file_id not in unique_files:
                        unique_files[file_id] = {
                            'file_id': file_id,
                            'filename': filename,
                            'chunk_count': 0,
                            'has_tables': metadata.get('has_tables', False),
                            'has_images': metadata.get('has_images', False)
                        }
                    
                    if file_id in unique_files:
                        unique_files[file_id]['chunk_count'] += 1
            
            documents_list = list(unique_files.values())
            
            return {
                "total_documents": len(documents_list),
                "total_child_chunks": child_count,
                "total_parent_chunks": parent_count,
                "total_chunks": child_count + parent_count,
                "documents": documents_list
            }
            
        except Exception as e:
            logger.error(f"Error getting stats: {str(e)}")
            return {
                "total_documents": 0,
                "total_chunks": 0,
                "documents": []
            }
    
    def delete_document_by_file_id(self, file_id: str) -> bool:
        """Delete all chunks (parent and child) for a file"""
        try:
            success = True
            
            # Delete from child collection
            child_collection = self.child_vectorstore._collection
            child_results = child_collection.get(where={"file_id": file_id})
            
            if child_results and child_results['ids']:
                child_collection.delete(ids=child_results['ids'])
                logger.info(f"Deleted {len(child_results['ids'])} child chunks")
            
            # Delete from parent collection
            parent_collection = self.parent_vectorstore._collection
            parent_results = parent_collection.get(where={"file_id": file_id})
            
            if parent_results and parent_results['ids']:
                parent_collection.delete(ids=parent_results['ids'])
                logger.info(f"Deleted {len(parent_results['ids'])} parent chunks")
            
            return bool(child_results['ids']) or bool(parent_results['ids'])
            
        except Exception as e:
            logger.error(f"Error deleting document: {str(e)}")
            return False
    
    def reset_collection(self) -> bool:
        """Reset both collections"""
        try:
            # Reset child collection
            child_name = self.child_vectorstore._collection.name
            self.child_vectorstore._client.delete_collection(name=child_name)
            
            self.child_vectorstore = Chroma(
                collection_name=child_name,
                embedding_function=self.embeddings,
                persist_directory=settings.chroma_persist_dir
            )
            
            # Reset parent collection
            parent_name = self.parent_vectorstore._collection.name
            self.parent_vectorstore._client.delete_collection(name=parent_name)
            
            self.parent_vectorstore = Chroma(
                collection_name=parent_name,
                embedding_function=self.embeddings,
                persist_directory=settings.chroma_persist_dir
            )
            
            logger.info("Collections reset successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error resetting collections: {str(e)}")
            return False
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict]:
        """Retrieve specific chunk by ID"""
        try:
            # Try child collection first
            child_collection = self.child_vectorstore._collection
            result = child_collection.get(
                ids=[chunk_id],
                include=['documents', 'metadatas']
            )
            
            if result and result['documents']:
                return {
                    'text': result['documents'][0],
                    'metadata': result['metadatas'][0]
                }
            
            # Try parent collection
            parent_collection = self.parent_vectorstore._collection
            result = parent_collection.get(
                ids=[chunk_id],
                include=['documents', 'metadatas']
            )
            
            if result and result['documents']:
                return {
                    'text': result['documents'][0],
                    'metadata': result['metadatas'][0]
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting chunk: {str(e)}")
            return None