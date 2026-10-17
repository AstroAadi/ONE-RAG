"""Selects the appropriate extractor based on file type."""

from typing import Optional, Dict, List, Type
import logging

from .base_extractor import BaseExtractor

logger = logging.getLogger(__name__)


class ExtractorFactory:
    """
    Factory class for creating document extractors
    
    Usage:
        extractor = ExtractorFactory.get_extractor('pdf')
        content, success, metadata = extractor.extract('document.pdf')
    """
    
    # Registry of available extractors
    # Format: {extension: ExtractorClass}
    _extractors: Dict[str, Type[BaseExtractor]] = {}
    
    @classmethod
    def register_extractor(cls, extractor_class: Type[BaseExtractor]) -> None:
        """
        Register a new extractor class
        
        Args:
            extractor_class: Class that inherits from BaseExtractor
            
        Example:
            ExtractorFactory.register_extractor(PDFExtractor)
        """
        # Create temporary instance to get supported formats
        temp_instance = extractor_class()
        formats = temp_instance._get_supported_formats()
        
        for fmt in formats:
            cls._extractors[fmt.lower()] = extractor_class
            logger.info(f"Registered {extractor_class.__name__} for .{fmt} files")
    
    @classmethod
    def register_for_extensions(cls, extensions: List[str], extractor_class: Type[BaseExtractor]) -> None:
        """
        Register an extractor for specific extensions
        
        Args:
            extensions: List of file extensions (without dots)
            extractor_class: Extractor class
        """
        for ext in extensions:
            cls._extractors[ext.lower()] = extractor_class
            logger.info(f"Registered {extractor_class.__name__} for .{ext} files")
    
    @classmethod
    def get_extractor(
        cls, 
        file_extension: str, 
        config: Optional[Dict] = None
    ) -> Optional[BaseExtractor]:
        """
        Get appropriate extractor for file extension
        
        Args:
            file_extension: File extension (with or without dot)
            config: Optional configuration dictionary
            
        Returns:
            Extractor instance or None if format not supported
            
        Example:
            pdf_extractor = ExtractorFactory.get_extractor('pdf')
            docx_extractor = ExtractorFactory.get_extractor('.docx', {'enable_images': True})
        """
        # Clean extension
        ext = file_extension.lower().lstrip('.')
        
        # Get extractor class
        extractor_class = cls._extractors.get(ext)
        
        if extractor_class:
            return extractor_class(config)
        
        logger.warning(f"No extractor found for extension: .{ext}")
        return None
    
    @classmethod
    def get_supported_formats(cls) -> List[str]:
        """
        Get list of all supported file formats
        
        Returns:
            List of supported extensions (without dots)
        """
        return sorted(list(cls._extractors.keys()))
    
    @classmethod
    def is_supported(cls, file_extension: str) -> bool:
        """
        Check if a file format is supported
        
        Args:
            file_extension: File extension (with or without dot)
            
        Returns:
            True if supported, False otherwise
        """
        ext = file_extension.lower().lstrip('.')
        return ext in cls._extractors
    
    @classmethod
    def get_extractor_info(cls) -> Dict[str, str]:
        """
        Get information about all registered extractors
        
        Returns:
            Dictionary mapping extensions to extractor class names
        """
        return {
            ext: extractor_class.__name__
            for ext, extractor_class in cls._extractors.items()
        }
    
    @classmethod
    def clear_registry(cls) -> None:
        """
        Clear all registered extractors
        Useful for testing
        """
        cls._extractors.clear()
        logger.info("Cleared extractor registry")


# Auto-registration helper
def auto_register_extractors():
    """Register extractors for supported document and image formats."""
    # PDF Extractor
    try:
        from .extractors.pdf_extractor import PDFExtractor
        ExtractorFactory.register_extractor(PDFExtractor)
    except ImportError as e:
        logger.warning(f"Could not register PDFExtractor: {e}")
    
    # DOCX Extractor
    try:
        from .extractors.docx_extractor import DocxExtractor
        ExtractorFactory.register_extractor(DocxExtractor)
    except ImportError as e:
        logger.warning(f"Could not register DocxExtractor: {e}")
    
    # PPTX Extractor
    try:
        from .extractors.pptx_extractor import PptxExtractor
        ExtractorFactory.register_extractor(PptxExtractor)
    except ImportError as e:
        logger.warning(f"Could not register PptxExtractor: {e}")
    
    # Image Extractor 
    try:
        from .extractors.image_extractor import ImageExtractor
        ExtractorFactory.register_extractor(ImageExtractor)
    except ImportError as e:
        logger.warning(f"Could not register ImageExtractor: {e}")
    
    # Markdown Extractor
    try:
        from .extractors.markdown_extractor import MarkdownExtractor
        ExtractorFactory.register_extractor(MarkdownExtractor)
    except ImportError as e:
        logger.warning(f"Could not register MarkdownExtractor: {e}")
    
    # Plain Text Extractor
    try:
        from .extractors.text_extractor import PlainTextExtractor
        ExtractorFactory.register_extractor(PlainTextExtractor)
    except ImportError as e:
        logger.warning(f"Could not register PlainTextExtractor: {e}")

    # Log final status
    supported = ExtractorFactory.get_supported_formats()
    logger.info(f"Auto-registration complete. {len(supported)} formats supported: {', '.join(supported)}")