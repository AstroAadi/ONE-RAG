"""
Extractor Factory - Updated Version
Automatically selects the appropriate extractor based on file type
Includes Excel extractor registration
"""

from typing import Optional, Dict, List, Type
from pathlib import Path
from importlib import import_module
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
    _unavailable_extractors: Dict[str, Dict[str, str]] = {}
    
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
        formats = temp_instance.supported_formats
        
        for fmt in formats:
            clean_fmt = fmt.lower().lstrip('.')
            cls._extractors[clean_fmt] = extractor_class
            cls._unavailable_extractors.pop(clean_fmt, None)
            logger.info(f"Registered {extractor_class.__name__} for .{clean_fmt} files")
    
    @classmethod
    def register_for_extensions(cls, extensions: List[str], extractor_class: Type[BaseExtractor]) -> None:
        """
        Register an extractor for specific extensions
        
        Args:
            extensions: List of file extensions (without dots)
            extractor_class: Extractor class
        """
        for ext in extensions:
            clean_ext = ext.lower().lstrip('.')
            cls._extractors[clean_ext] = extractor_class
            cls._unavailable_extractors.pop(clean_ext, None)
            logger.info(f"Registered {extractor_class.__name__} for .{clean_ext} files")

    @classmethod
    def register_unavailable_extensions(
        cls,
        extensions: List[str],
        extractor_name: str,
        reason: str,
        force: bool = False,
    ) -> None:
        """
        Track formats that are known but currently unavailable.

        This preserves the difference between:
        - unsupported formats (no extractor exists)
        - unavailable formats (extractor exists but import/init failed)
        """
        for ext in extensions:
            clean_ext = ext.lower().lstrip('.')
            if not force and clean_ext in cls._extractors:
                continue

            cls._extractors.pop(clean_ext, None)
            cls._unavailable_extractors[clean_ext] = {
                "extractor": extractor_name,
                "reason": reason,
            }
    
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
        # Accept either a bare extension (".pdf") or a file path ("folder/file.pdf")
        raw_value = str(file_extension).strip()
        ext = Path(raw_value).suffix.lower().lstrip('.') if Path(raw_value).suffix else raw_value.lower().lstrip('.')
        
        # Get extractor class
        extractor_class = cls._extractors.get(ext)
        
        if extractor_class:
            try:
                return extractor_class(config)
            except Exception as e:
                cls.register_unavailable_extensions(
                    [ext],
                    extractor_class.__name__,
                    f"Initialization failed: {e}",
                    force=True,
                )
                logger.error(
                    f"Extractor initialization failed for .{ext} "
                    f"({extractor_class.__name__}): {e}"
                )
                return None

        unavailable_info = cls._unavailable_extractors.get(ext)
        if unavailable_info:
            logger.warning(
                f"Extractor unavailable for extension .{ext}: "
                f"{unavailable_info['extractor']} ({unavailable_info['reason']})"
            )
            return None
        
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
    def get_unavailable_formats(cls) -> Dict[str, Dict[str, str]]:
        """
        Get formats that are known but not currently usable.
        """
        return {
            ext: dict(info)
            for ext, info in sorted(cls._unavailable_extractors.items())
        }

    @classmethod
    def get_format_status(cls, file_extension: str) -> Dict[str, Optional[str]]:
        """
        Describe whether a format is available, unavailable, or unsupported.
        """
        raw_value = str(file_extension).strip()
        ext = Path(raw_value).suffix.lower().lstrip('.') if Path(raw_value).suffix else raw_value.lower().lstrip('.')

        if ext in cls._extractors:
            return {
                "extension": ext,
                "status": "available",
                "extractor": cls._extractors[ext].__name__,
                "reason": None,
            }

        unavailable_info = cls._unavailable_extractors.get(ext)
        if unavailable_info:
            return {
                "extension": ext,
                "status": "unavailable",
                "extractor": unavailable_info["extractor"],
                "reason": unavailable_info["reason"],
            }

        return {
            "extension": ext,
            "status": "unsupported",
            "extractor": None,
            "reason": None,
        }

    @classmethod
    def get_availability_report(cls) -> Dict[str, object]:
        """
        Return a summary of currently available and unavailable extractors.
        """
        available_formats = {
            ext: extractor_class.__name__
            for ext, extractor_class in sorted(cls._extractors.items())
        }
        unavailable_formats = cls.get_unavailable_formats()

        return {
            "available_count": len(available_formats),
            "unavailable_count": len(unavailable_formats),
            "known_format_count": len(set(available_formats) | set(unavailable_formats)),
            "available_formats": available_formats,
            "unavailable_formats": unavailable_formats,
        }

    @classmethod
    def validate_registered_extractors(cls) -> Dict[str, object]:
        """
        Instantiate each registered extractor class once so constructor-time
        dependency issues are reflected in the availability report.
        """
        validation_results: Dict[str, Dict[str, object]] = {}
        extensions_by_class: Dict[Type[BaseExtractor], List[str]] = {}

        for ext, extractor_class in list(cls._extractors.items()):
            extensions_by_class.setdefault(extractor_class, []).append(ext)

        for extractor_class, extensions in extensions_by_class.items():
            sorted_extensions = sorted(extensions)
            try:
                extractor_class()
                validation_results[extractor_class.__name__] = {
                    "status": "available",
                    "extensions": sorted_extensions,
                }
            except Exception as e:
                cls.register_unavailable_extensions(
                    sorted_extensions,
                    extractor_class.__name__,
                    f"Initialization failed: {e}",
                    force=True,
                )
                validation_results[extractor_class.__name__] = {
                    "status": "unavailable",
                    "extensions": sorted_extensions,
                    "reason": str(e),
                }
                logger.warning(
                    f"Extractor validation failed for {extractor_class.__name__} "
                    f"({', '.join('.' + ext for ext in sorted_extensions)}): {e}"
                )

        report = cls.get_availability_report()
        report["validation_results"] = validation_results
        return report
    
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
        cls._unavailable_extractors.clear()
        logger.info("Cleared extractor registry")


# Auto-registration helper
def _register_optional_extractor(
    module_path: str,
    class_name: str,
    extensions: List[str],
) -> None:
    try:
        module = import_module(module_path, package=__package__)
        extractor_class = getattr(module, class_name)
        ExtractorFactory.register_for_extensions(extensions, extractor_class)
    except Exception as e:
        ExtractorFactory.register_unavailable_extensions(extensions, class_name, str(e))
        logger.warning(f"Could not register {class_name}: {e}")


def auto_register_extractors():
    """
    Automatically register all available extractors
    Call this function at module initialization
    """
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
    
    # Excel Extractor 
    try:
        from .extractors.excel_extractor import ExcelExtractor
        ExtractorFactory.register_extractor(ExcelExtractor)
    except ImportError as e:
        logger.warning(f"Could not register ExcelExtractor: {e}")
    
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

    # EPUB Extractor
    try:
        from .extractors.epub_extractor import EPUBExtractor
        ExtractorFactory.register_extractor(EPUBExtractor)
    except ImportError as e:
        logger.warning(f"Could not register EPUBExtractor: {e}")
    
    # MSG Extractor
    try:
        from .extractors.msg_extractor import MSGExtractor
        ExtractorFactory.register_extractor(MSGExtractor)
    except ImportError as e:
        logger.warning(f"Could not register MSGExtractor: {e}")
    
    # EML Extractor
    try:
        from .extractors.eml_extractor import EMLExtractor
        ExtractorFactory.register_extractor(EMLExtractor)
    except ImportError as e:
        logger.warning(f"Could not register EMLExtractor: {e}")

    # HTML Extractor
    try:
        from .extractors.html_extractor import HTMLExtractor
        ExtractorFactory.register_extractor(HTMLExtractor)
    except ImportError as e:
        logger.warning(f"Could not register HTMLExtractor: {e}")

    # XML Extractor
    try:
        from .extractors.xml_extractor import XMLExtractor
        ExtractorFactory.register_extractor(XMLExtractor)
    except ImportError as e:
        logger.warning(f"Could not register XMLExtractor: {e}")

    # VTT Extractor
    try:
        from .extractors.vtt_extractor import VTTExtractor
        ExtractorFactory.register_extractor(VTTExtractor)
    except ImportError as e:
        logger.warning(f"Could not register VTTExtractor: {e}")

    # PROPERTIES Extractor
    try:
        from .extractors.properties_extractor import PropertiesExtractor
        ExtractorFactory.register_extractor(PropertiesExtractor)
    except ImportError as e:
        logger.warning(f"Could not register PropertiesExtractor: {e}")
    
    #MDX Extractor
    try:
        from .extractors.mdx_extractor import MDXExtractor
        ExtractorFactory.register_extractor(MDXExtractor)
    except ImportError as e:
        logger.warning(f"Could not register MDXExtractor: {e}")
    
    # XLS Extractor
    # try:
    #     from .extractors.xls_extractor import XLSExtractor
    #     ExtractorFactory.register_extractor(XLSExtractor)
    # except ImportError as e:
    #     logger.warning(f"Could not register XLSExtractor: {e}")

    # CSV / TSV Extractor
    try:
        from .extractors.csv_extractor import CSVExtractor
        ExtractorFactory.register_extractor(CSVExtractor)
    except ImportError as e:
        logger.warning(f"Could not register CSVExtractor: {e}")

    # ZIP Extractor
    try:
        from .extractors.zip_extractor import ZIPExtractor
        ExtractorFactory.register_extractor(ZIPExtractor)
    except ImportError as e:
        logger.warning(f"Could not register ZIPExtractor: {e}")
    
    # JSON Extractor
    try:
        from .extractors.json_extractor import JSONExtractor
        ExtractorFactory.register_extractor(JSONExtractor)
    except ImportError as e:
        logger.warning(f"Could not register JSONExtractor: {e}")
    
    # Log final status
    report = ExtractorFactory.get_availability_report()
    supported = ExtractorFactory.get_supported_formats()
    logger.info(
        f"Auto-registration complete. "
        f"{report['available_count']} available, {report['unavailable_count']} unavailable formats. "
        f"Available: {', '.join(supported)}"
    )

    if report["unavailable_formats"]:
        unavailable_summary = ", ".join(
            f".{ext} ({info['extractor']})"
            for ext, info in report["unavailable_formats"].items()
        )
        logger.warning(f"Unavailable extractor formats: {unavailable_summary}")
