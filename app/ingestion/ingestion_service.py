"""
Ingestion Service
Main orchestrator for the data ingestion pipeline
"""

from typing import List, Dict, Any, Optional, Tuple
import logging
from pathlib import Path
from datetime import datetime

from app.core.config import settings
from app.services.extraction.extractor_factory import ExtractorFactory
from app.services.chunking_service import EnhancedChunkingService, ChunkingStrategy
from app.services.vector_store_service import VectorStoreService
from app.utils.markdown_converter import MarkdownConverter

from .base_data_source import BaseDataSource, DataSource
from .file_upload_source import FileUploadSource
from .cloud_storage_source import CloudStorageSource
from .database_source import DatabaseSource
from .api_source import APISource
from .file_tracker import FileTracker

logger = logging.getLogger(__name__)


class IngestionService:
    """
    Main service orchestrating the complete ingestion pipeline:
    1. Discover data sources
    2. Check if already indexed (via FileTracker)
    3. Extract content
    4. Chunk data
    5. Generate embeddings
    6. Store in vector database
    7. Save markdown output locally
    """
    
    def __init__(self):
        """Initialize ingestion service"""
        self.extractor_factory = ExtractorFactory()
        self.chunking_service = self._init_chunking_service()
        self._vector_store = None  # Lazy initialization
        self.file_tracker = None  # Will be initialized when needed
        self.markdown_converter = MarkdownConverter()
        
        logger.info("✅ IngestionService initialized (lazy mode)")
    
    @property
    def vector_store(self) -> VectorStoreService:
        """Get or initialize vector store service"""
        if self._vector_store is None:
            try:
                self._vector_store = VectorStoreService()
            except Exception as e:
                logger.error(f"Failed to initialize vector store: {e}")
                raise
        return self._vector_store
    
    @property
    def tracker(self) -> FileTracker:
        """Get or initialize file tracker"""
        if self.file_tracker is None:
            try:
                self.file_tracker = FileTracker()
            except Exception as e:
                logger.error(f"Failed to initialize file tracker: {e}")
                raise
        return self.file_tracker
    
    def _init_chunking_service(self) -> EnhancedChunkingService:
        """Initialize chunking service with config"""
        return EnhancedChunkingService(
            target_chunk_tokens=settings.target_chunk_tokens,
            chunk_overlap_tokens=settings.chunk_overlap_tokens,
            tokenizer_model=settings.tokenizer_model,
            min_chunk_tokens=settings.min_chunk_tokens,
            max_chunk_tokens=settings.max_chunk_tokens
        )
    
    def get_data_source(self) -> BaseDataSource:
        """
        Get the appropriate data source handler based on config
        
        Returns:
            DataSource handler instance
            
        Raises:
            ValueError if data source not recognized
        """
        source_type = settings.data_source
        
        config = {
            'file_path': settings.file_path,
            'cloud_storage_provider': settings.cloud_storage_provider,
            'cloud_storage_bucket': settings.cloud_storage_bucket,
            'cloud_storage_prefix': settings.cloud_storage_prefix,
            'data_source_db_type': settings.data_source_db_type,
            'data_source_db_url': settings.data_source_db_url,
            'data_source_db_table': settings.data_source_db_table,
            'data_source_db_query': settings.data_source_db_query,
            'data_source_api_url': settings.data_source_api_url,
            'data_source_api_key': settings.data_source_api_key,
            'data_source_api_headers': settings.data_source_api_headers,
        }
        
        if source_type == 'file_upload':
            return FileUploadSource(config)
        elif source_type == 'cloud_storage':
            return CloudStorageSource(config)
        elif source_type == 'database':
            return DatabaseSource(config)
        elif source_type == 'api':
            return APISource(config)
        else:
            raise ValueError(f"Unknown data source type: {source_type}")
    
    def run_ingestion_pipeline(self) -> Dict[str, Any]:
        """
        Execute the complete ingestion pipeline
        
        Returns:
            Dictionary with ingestion results and statistics
        """
        start_time = datetime.utcnow()
        results = {
            "status": "started",
            "start_time": start_time.isoformat(),
            "data_source": settings.data_source,
            "knowledge_base_id": settings.knowledge_base_id,
            "chunking_strategy": settings.chunking_strategy,
            "total_sources_discovered": 0,
            "total_sources_skipped": 0,
            "total_sources_processed": 0,
            "total_sources_failed": 0,
            "total_chunks_created": 0,
            "failed_sources": [],
            "skipped_sources": [],
            "errors": [],
            "warnings": []
        }
        
        try:
            # Get data source handler
            logger.info(f"Loading data source: {settings.data_source}")
            data_source = self.get_data_source()
            
            # Validate configuration
            is_valid, validation_error = data_source.validate_config()
            if not is_valid:
                results["status"] = "error"
                results["errors"].append(f"Data source validation failed: {validation_error}")
                logger.error(f"Data source validation failed: {validation_error}")
                return results
            
            # Discover sources
            logger.info(f"Discovering sources from {settings.data_source}...")
            sources, discovery_errors = data_source.discover_sources()
            
            if discovery_errors:
                if settings.data_source != 'file_upload':
                    # For placeholder sources, this is expected
                    logger.warning(f"Discovery errors: {discovery_errors}")
                    results["warnings"].extend(discovery_errors)
                    results["status"] = "completed"
                    results["end_time"] = datetime.utcnow().isoformat()
                    return results
                else:
                    # For file upload, errors might be critical
                    logger.warning(f"Discovery errors: {discovery_errors}")
                    results["warnings"].extend(discovery_errors)
            
            results["total_sources_discovered"] = len(sources)
            logger.info(f"Discovered {len(sources)} sources")
            
            if not sources:
                results["status"] = "completed"
                results["warnings"].append("No sources discovered")
                results["end_time"] = datetime.utcnow().isoformat()
                return results
            
            # Process each source
            for idx, source in enumerate(sources, 1):
                logger.info(f"Processing source {idx}/{len(sources)}: {source.metadata.get('filename', source.identifier)}")
                
                try:
                    # Check if file already indexed
                    filename = source.metadata.get('filename', source.identifier)
                    is_indexed, index_metadata = self.tracker.is_file_indexed(
                        filename,
                        settings.knowledge_base_id
                    )
                    
                    if is_indexed:
                        logger.info(f"⊘ Skipping already indexed file: {filename}")
                        results["total_sources_skipped"] += 1
                        results["skipped_sources"].append({
                            "filename": filename,
                            "reason": "Already indexed",
                            "indexed_at": index_metadata.get('indexed_timestamp')
                        })
                        continue
                    
                    # Fetch data
                    content, content_type, fetch_metadata = data_source.fetch_data(source)
                    
                    # Extract content
                    extracted_content, extraction_success, extraction_metadata = \
                        self._extract_content(content, source)
                    
                    if not extraction_success:
                        error_msg = extraction_metadata.get('error', 'Unknown extraction error')
                        logger.error(f"✗ Extraction failed for {filename}: {error_msg}")
                        results["total_sources_failed"] += 1
                        results["failed_sources"].append({
                            "filename": filename,
                            "error": error_msg
                        })
                        continue
                    
                    # Chunk content
                    chunks = self._chunk_content(
                        extracted_content,
                        extraction_metadata,
                        source
                    )
                    
                    if not chunks:
                        logger.warning(f"No chunks generated for {filename}")
                        results["total_sources_failed"] += 1
                        results["failed_sources"].append({
                            "filename": filename,
                            "error": "No chunks generated after chunking"
                        })
                        continue
                    
                    # Store chunks in vector database
                    upsert_success, upsert_error = self._store_chunks(
                        chunks,
                        source,
                        extraction_metadata
                    )
                    
                    if not upsert_success:
                        logger.error(f"✗ Failed to store chunks for {filename}: {upsert_error}")
                        results["total_sources_failed"] += 1
                        results["failed_sources"].append({
                            "filename": filename,
                            "error": f"Storage failed: {upsert_error}"
                        })
                        continue
                    
                    # Save markdown output locally
                    self._save_markdown_output(
                        extracted_content,
                        source,
                        extraction_metadata
                    )
                    
                    # Mark file as indexed
                    self.tracker.mark_file_indexed(
                        filename,
                        settings.knowledge_base_id,
                        len(chunks),
                        {
                            'extension': source.metadata.get('extension'),
                            'file_size_bytes': source.metadata.get('size_bytes')
                        }
                    )
                    
                    results["total_sources_processed"] += 1
                    results["total_chunks_created"] += len(chunks)
                    
                    logger.info(f"✓ Successfully processed {filename} ({len(chunks)} chunks)")
                    
                except Exception as e:
                    logger.error(f"✗ Error processing source: {e}", exc_info=True)
                    filename = source.metadata.get('filename', source.identifier)
                    results["total_sources_failed"] += 1
                    results["failed_sources"].append({
                        "filename": filename,
                        "error": str(e)
                    })
                    # Continue with next source
                    continue
            
            # Cleanup
            data_source.cleanup()
            
            results["status"] = "completed"
            results["end_time"] = datetime.utcnow().isoformat()
            
            # Log summary
            logger.info("")
            logger.info("=" * 70)
            logger.info("INGESTION PIPELINE COMPLETED")
            logger.info("=" * 70)
            logger.info(f"Total sources discovered: {results['total_sources_discovered']}")
            logger.info(f"Total sources processed: {results['total_sources_processed']}")
            logger.info(f"Total sources skipped: {results['total_sources_skipped']}")
            logger.info(f"Total sources failed: {results['total_sources_failed']}")
            logger.info(f"Total chunks created: {results['total_chunks_created']}")
            logger.info("=" * 70)
            logger.info("")
            
            return results
            
        except Exception as e:
            logger.error(f"Fatal error in ingestion pipeline: {e}", exc_info=True)
            results["status"] = "error"
            results["errors"].append(str(e))
            results["end_time"] = datetime.utcnow().isoformat()
            return results
    
    def _extract_content(
        self,
        content: bytes,
        source: DataSource
    ) -> Tuple[str, bool, Dict[str, Any]]:
        """
        Extract content from file using appropriate extractor
        
        Args:
            content: Raw file content
            source: DataSource object
            
        Returns:
            Tuple of (extracted_text, success, metadata)
        """
        try:
            # Get file extension
            extension = source.metadata.get('extension', '').lower()
            
            # Get appropriate extractor
            extractor = ExtractorFactory.get_extractor(extension)
            if not extractor:
                return "", False, {'error': f"No extractor found for .{extension}"}
            
            # Create temporary file to extract from
            file_path = source.identifier
            
            # Extract content
            content_text, extraction_success, extraction_metadata = extractor.extract(file_path)
            
            if not extraction_success:
                error = extraction_metadata.get('error', 'Unknown error')
                logger.error(f"Extraction failed: {error}")
                return "", False, extraction_metadata
            
            return content_text, True, extraction_metadata
            
        except Exception as e:
            logger.error(f"Error during content extraction: {e}")
            return "", False, {'error': str(e)}
    
    def _chunk_content(
        self,
        content: str,
        extraction_metadata: Dict[str, Any],
        source: DataSource
    ) -> List[Dict[str, Any]]:
        """
        Chunk extracted content
        
        Args:
            content: Extracted text content
            extraction_metadata: Metadata from extraction
            source: DataSource object
            
        Returns:
            List of chunk dictionaries
        """
        try:
            # Parse chunking strategy
            strategy_str = settings.chunking_strategy.upper()
            strategy = ChunkingStrategy[strategy_str]
            
            # Chunk content
            chunks = self.chunking_service.split_document_into_chunks(
                content,
                base_metadata={
                    'knowledge_base_id': settings.knowledge_base_id,
                    'filename': source.metadata.get('filename'),
                    'source_type': source.source_type,
                    **extraction_metadata
                },
                strategy=strategy
            )
            
            logger.info(f"Chunked content into {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Error during chunking: {e}")
            return []
    
    def _store_chunks(
        self,
        chunks: List[Dict[str, Any]],
        source: DataSource,
        extraction_metadata: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """
        Store chunks in vector database
        
        Args:
            chunks: List of chunk dictionaries
            source: Source data object
            extraction_metadata: Extraction metadata
            
        Returns:
            Tuple of (success, error_message)
        """
        try:
            import hashlib
            
            filename = source.metadata.get('filename', 'unknown')
            
            # Generate unique file_id from filename + timestamp
            file_id_source = f"{filename}_{datetime.utcnow().isoformat()}"
            file_id = hashlib.md5(file_id_source.encode()).hexdigest()
            
            # Store chunks to Qdrant
            chunk_count = self.vector_store.add_document_chunks(
                chunks=chunks,
                file_id=file_id,
                filename=filename,
                knowledge_base_id=settings.knowledge_base_id,
                knowledge_base_name=settings.knowledge_base_id,  # Use KB ID as name
                user_id=settings.knowledge_base_id,  # Use KB ID as user identifier
                structured_data=None
            )
            
            if chunk_count > 0:
                logger.info(f"Stored {chunk_count} chunks in vector database")
                return True, ""
            else:
                return False, "No chunks were stored"
                
        except Exception as e:
            logger.error(f"Error storing chunks: {e}")
            return False, str(e)
    
    def _save_markdown_output(
        self,
        content: str,
        source: DataSource,
        extraction_metadata: Dict[str, Any]
    ) -> None:
        """
        Save extracted content as markdown locally for debugging
        
        Args:
            content: Extracted content text
            source: Source data object
            extraction_metadata: Extraction metadata
        """
        try:
            # Create directory structure: markdown_output/{kb_id}/{filename}
            kb_dir = Path(settings.markdown_dir) / settings.knowledge_base_id
            kb_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate filename
            filename = source.metadata.get('filename', 'document')
            markdown_filename = f"{Path(filename).stem}_extracted.md"
            output_path = kb_dir / markdown_filename
            
            # Write markdown file
            with open(output_path, 'w', encoding='utf-8') as f:
                # Add metadata header
                f.write("# Extracted Content\n\n")
                f.write(f"**Source**: {filename}\n")
                f.write(f"**Imported**: {datetime.utcnow().isoformat()}\n")
                f.write(f"**Knowledge Base**: {settings.knowledge_base_id}\n\n")
                f.write("---\n\n")
                
                # Write content
                f.write(content)
            
            logger.debug(f"Saved markdown output to {output_path}")
            
        except Exception as e:
            logger.warning(f"Failed to save markdown output: {e}")
