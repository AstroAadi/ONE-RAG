import chromadb
from chromadb.config import Settings
from typing import List, Dict, Tuple
from app.core.config import settings
from app.services.bedrock_service import BedrockService
import uuid


class VectorStoreService:
    """Service for managing vector database operations with ChromaDB"""
    
    def __init__(self):
        """Initialize ChromaDB client and Bedrock service"""
        self.chroma_client = chromadb.PersistentClient(
            path=settings.chroma_persist_dir,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        self.bedrock_service = BedrockService()
        self.collection_name = "knowledge_base"
        self._initialize_collection()
    
    def _initialize_collection(self):
        """Initialize or get the ChromaDB collection"""
        try:
            self.collection = self.chroma_client.get_collection(
                name=self.collection_name
            )
        except:
            # Create collection if it doesn't exist
            self.collection = self.chroma_client.create_collection(
                name=self.collection_name,
                metadata={"description": "Knowledge base documents collection"}
            )
    
    def add_document_chunks_to_vector_store(
        self, 
        chunks: List[Dict], 
        file_id: str,
        filename: str
    ) -> int:
        """
        Add document chunks to vector store with embeddings
        
        Args:
            chunks: List of text chunks with metadata
            file_id: Unique file identifier
            filename: Original filename
            
        Returns:
            Number of chunks added
        """
        if not chunks:
            return 0
        
        try:
            # Extract texts from chunks
            texts = [chunk['text'] for chunk in chunks]
            
            # Generate embeddings for all chunks
            embeddings = self.bedrock_service.generate_embeddings_with_cohere(
                texts, 
                input_type="search_document"
            )
            
            # Prepare data for ChromaDB
            ids = []
            metadatas = []
            documents = []
            
            for i, chunk in enumerate(chunks):
                chunk_id = f"{file_id}_chunk_{i}"
                ids.append(chunk_id)
                
                metadata = {
                    "file_id": file_id,
                    "filename": filename,
                    "chunk_index": chunk.get('chunk_index', i),
                    "char_count": chunk.get('char_count', len(chunk['text']))
                }
                metadatas.append(metadata)
                documents.append(chunk['text'])
            
            # Add to ChromaDB
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents
            )
            
            return len(chunks)
            
        except Exception as e:
            print(f"Error adding chunks to vector store: {str(e)}")
            raise
    
    def search_similar_chunks(
        self, 
        query: str, 
        top_k: int = 10
    ) -> Tuple[List[str], List[Dict], List[float]]:
        """
        Search for similar chunks in the vector store
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            Tuple of (documents, metadatas, distances)
        """
        try:
            # Generate query embedding
            query_embedding = self.bedrock_service.generate_single_embedding(
                query,
                input_type="search_query"
            )
            
            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k
            )
            
            documents = results['documents'][0] if results['documents'] else []
            metadatas = results['metadatas'][0] if results['metadatas'] else []
            distances = results['distances'][0] if results['distances'] else []
            
            return documents, metadatas, distances
            
        except Exception as e:
            print(f"Error searching vector store: {str(e)}")
            raise
    
    def get_collection_stats(self) -> Dict:
        """
        Get statistics about the collection
        
        Returns:
            Dictionary with collection statistics
        """
        try:
            count = self.collection.count()
            
            # Get all unique file_ids
            if count > 0:
                all_data = self.collection.get()
                metadatas = all_data.get('metadatas', [])
                
                unique_files = {}
                for metadata in metadatas:
                    file_id = metadata.get('file_id')
                    filename = metadata.get('filename')
                    if file_id and file_id not in unique_files:
                        unique_files[file_id] = filename
                
                documents_list = [
                    {"file_id": fid, "filename": fname} 
                    for fid, fname in unique_files.items()
                ]
            else:
                documents_list = []
            
            return {
                "total_documents": len(documents_list),
                "total_chunks": count,
                "documents": documents_list
            }
            
        except Exception as e:
            print(f"Error getting collection stats: {str(e)}")
            return {
                "total_documents": 0,
                "total_chunks": 0,
                "documents": []
            }
    
    def delete_document_by_file_id(self, file_id: str) -> bool:
        """
        Delete all chunks belonging to a specific file
        
        Args:
            file_id: File identifier
            
        Returns:
            Success status
        """
        try:
            # Get all ids with this file_id
            results = self.collection.get(
                where={"file_id": file_id}
            )
            
            if results and results['ids']:
                self.collection.delete(ids=results['ids'])
                return True
            
            return False
            
        except Exception as e:
            print(f"Error deleting document: {str(e)}")
            return False
    
    def reset_collection(self) -> bool:
        """
        Reset the entire collection (delete all documents)
        
        Returns:
            Success status
        """
        try:
            self.chroma_client.delete_collection(name=self.collection_name)
            self._initialize_collection()
            return True
        except Exception as e:
            print(f"Error resetting collection: {str(e)}")
            return False