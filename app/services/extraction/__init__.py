"""
Document Extraction Services
Automatically initializes and registers all extractors
"""

from .base_extractor import BaseExtractor
from .extractor_factory import ExtractorFactory, auto_register_extractors
from .processors.caption_detector import CaptionDetector

# Auto-register all available extractors
auto_register_extractors()

__all__ = [
    'BaseExtractor',
    'ExtractorFactory',
    'CaptionDetector',
]