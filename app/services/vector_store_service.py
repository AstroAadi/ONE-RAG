"""
Clean Vector Store Service
Simple semantic search without parent-child complexity
"""

from langchain_chroma import Chroma
from langchain_aws import BedrockEmbeddings
from langchain.docstore.document import Document as LangChainDocument
from typing import List, Dict, Tuple, Optional
from app.core.config import settings
import logging
import time
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


class EnhancedVectorStoreService:
    """Simple vector store with semantic search"""
    
    def __init__(self):
        """Initialize vector store"""
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
        
        logger.info("VectorStore initialized")
    
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
        """Add chunks to vector store"""
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
            
            return added_count
            
        except Exception as e:
            logger.error(f"Error adding chunks: {str(e)}")
            raise
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        file_id: Optional[str] = None
    ) -> Tuple[List[str], List[Dict], List[float]]:
        """
        Search vector store
        
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
            
            logger.info(f"Retrieved {len(documents)} chunks")
            
            return documents, metadatas, distances
            
        except Exception as e:
            logger.error(f"Search error: {str(e)}")
            raise
    
    def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        file_id: Optional[str] = None,
        search_mode: str = "semantic"  # Kept for compatibility but only semantic used
    ) -> Tuple[List[str], List[Dict], List[float]]:
        """
        Hybrid search (simplified - just semantic search)
        Kept for backwards compatibility with existing code
        """
        # Just use semantic search - hybrid complexity removed
        return self.search(query, top_k, file_id)
    
    def search_by_metadata(
        self,
        filters: Dict,
        limit: int = 10
    ) -> List[Dict]:
        """Search by metadata filters"""
        try:
            results = self.vectorstore.similarity_search(
                query="",  # Empty query for metadata-only search
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
                'documents': list(file_docs.values())
            }
            
        except Exception as e:
            logger.error(f"Error getting stats: {str(e)}")
            return {
                'total_documents': 0,
                'total_chunks': 0,
                'documents': []
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
            
            return True
            
        except Exception as e:
            logger.error(f"Error resetting collection: {str(e)}")
            return False