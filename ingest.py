#!/usr/bin/env python3
"""
RAG System - Data Ingestion Pipeline
Main entry point for ingesting data into the RAG system

Usage:
    python ingest.py
    
Configuration:
    All settings are read from .env file
    Key variables:
    - DATA_SOURCE: file_upload, cloud_storage, database, or api
    - FILE_PATH: Path to file or directory (for file_upload)
    - KNOWLEDGE_BASE_ID: Identifier for the knowledge base
    - CHUNKING_STRATEGY: default, fixed_size, semantic, hierarchical, no_chunking
    - QDRANT_URL: Vector database URL
    - QDRANT_API_KEY: Vector database API key
"""

import sys
import logging
from pathlib import Path
from datetime import datetime

# Setup path
sys.path.insert(0, str(Path(__file__).parent))

from app.core.config import settings
from app.ingestion.ingestion_service import IngestionService


def setup_logging():
    """Configure logging for the ingestion pipeline"""
    # Create logs directory if it doesn't exist
    log_dir = Path(settings.log_file_path).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, settings.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(settings.log_file_path, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)


def print_banner():
    """Print startup banner"""
    print("\n" + "=" * 70)
    print("RAG SYSTEM - DATA INGESTION PIPELINE")
    print("=" * 70)
    print(f"Start Time: {datetime.utcnow().isoformat()}")
    print(f"Data Source: {settings.data_source}")
    print(f"Knowledge Base: {settings.knowledge_base_id}")
    print(f"Chunking Strategy: {settings.chunking_strategy}")
    print("=" * 70 + "\n")


def print_configuration_summary(logger: logging.Logger):
    """Print configuration summary"""
    logger.info("")
    logger.info("=" * 70)
    logger.info("CONFIGURATION")
    logger.info("=" * 70)
    
    logger.info(f"Data Source: {settings.data_source}")
    logger.info(f"Knowledge Base ID: {settings.knowledge_base_id}")
    logger.info(f"Chunking Strategy: {settings.chunking_strategy}")
    logger.info(f"Qdrant Collection: {settings.qdrant_collection_name}")
    logger.info(f"Markdown Output Dir: {settings.markdown_dir}")
    
    if settings.data_source == 'file_upload':
        logger.info(f"File Path: {settings.file_path}")
    
    logger.info("=" * 70)
    logger.info("")


def validate_configuration(logger: logging.Logger) -> bool:
    """
    Validate configuration before starting ingestion
    
    Args:
        logger: Logger instance
        
    Returns:
        True if configuration is valid, False otherwise
    """
    errors = []
    warnings = []
    
    # Check Qdrant configuration
    if not settings.qdrant_url:
        errors.append("QDRANT_URL not set in .env")
    
    if not settings.qdrant_api_key and settings.qdrant_url:
        errors.append("QDRANT_API_KEY not set in .env")
    
    # Check AWS configuration (for embeddings)
    if not settings.aws_access_key_id or not settings.aws_secret_access_key:
        if not settings.aws_profile_name:
            warnings.append("AWS credentials not fully configured (no key ID/secret or profile)")
    
    # Check data source configuration
    if settings.data_source == 'file_upload':
        if not settings.file_path:
            errors.append("FILE_PATH not set in .env for file_upload source")
        else:
            file_path = Path(settings.file_path)
            if not file_path.exists():
                errors.append(f"FILE_PATH does not exist: {settings.file_path}")
    
    # Print warnings
    if warnings:
        logger.warning(f"Configuration warnings:")
        for warning in warnings:
            logger.warning(f"  ⚠️  {warning}")
    
    # Print errors
    if errors:
        logger.error(f"Configuration errors:")
        for error in errors:
            logger.error(f"  ✗ {error}")
        return False
    
    return True


def main():
    """Main entry point"""
    # Setup logging
    logger = setup_logging()
    
    # Print banner
    print_banner()
    
    try:
        # Log startup
        logger.info("RAG System - Ingestion Pipeline Starting")
        logger.info(f"Python version: {sys.version.split()[0]}")
        
        # Print configuration
        print_configuration_summary(logger)
        
        # Validate configuration
        logger.info("Validating configuration...")
        if not validate_configuration(logger):
            logger.error("Configuration validation failed. Exiting.")
            print("\n❌ Configuration validation failed. Check .env settings and logs.")
            return 1
        
        logger.info("✓ Configuration valid")
        logger.info("")
        
        # Initialize ingestion service
        logger.info("Initializing ingestion service...")
        ingestion_service = IngestionService()
        logger.info("✓ Ingestion service initialized")
        logger.info("")
        
        # Run ingestion pipeline
        logger.info("Starting ingestion pipeline...")
        results = ingestion_service.run_ingestion_pipeline()
        
        # Print results
        print("\n" + "=" * 70)
        print("INGESTION RESULTS")
        print("=" * 70)
        print(f"Status: {results['status'].upper()}")
        print(f"Data Source: {results['data_source']}")
        print(f"Knowledge Base: {results['knowledge_base_id']}")
        print(f"Chunking Strategy: {results['chunking_strategy']}")
        print()
        print(f"Sources Discovered: {results['total_sources_discovered']}")
        print(f"Sources Processed: {results['total_sources_processed']}")
        print(f"Sources Skipped: {results['total_sources_skipped']}")
        print(f"Sources Failed: {results['total_sources_failed']}")
        print(f"Total Chunks Created: {results['total_chunks_created']}")
        
        if results['failed_sources']:
            print()
            print("Failed Sources:")
            for failed in results['failed_sources']:
                print(f"  ✗ {failed['filename']}: {failed['error']}")
        
        if results['skipped_sources']:
            print()
            print("Skipped Sources:")
            for skipped in results['skipped_sources'][:5]:  # Show first 5
                print(f"  ⊘ {skipped['filename']}: {skipped['reason']}")
            if len(results['skipped_sources']) > 5:
                print(f"  ... and {len(results['skipped_sources']) - 5} more")
        
        if results['errors']:
            print()
            print("Errors:")
            for error in results['errors']:
                print(f"  ✗ {error}")
        
        if results['warnings']:
            print()
            print("Warnings:")
            for warning in results['warnings'][:5]:  # Show first 5
                print(f"  ⚠️  {warning}")
            if len(results['warnings']) > 5:
                print(f"  ... and {len(results['warnings']) - 5} more")
        
        print("=" * 70)
        
        # Log results
        logger.info("")
        logger.info("Pipeline execution completed")
        logger.info(f"End time: {results['end_time']}")
        
        # Determine exit code
        if results['status'] == 'completed':
            if results['total_sources_failed'] > 0:
                return 1  # Exit with error if any sources failed
            else:
                return 0  # Success
        else:
            return 1  # Error status
    
    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user")
        print("\n⚠️  Pipeline interrupted by user")
        return 130
    
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        print(f"\n❌ Fatal error: {e}")
        print("Check logs for details")
        return 1


if __name__ == "__main__":
    sys.exit(main())
