"""
Enhanced Image File Extractor - UPDATED
Uses centralized ImageDescriptionService instead of its own Bedrock client
Extracts text from images using OCR AND generates rich AI descriptions
"""

import os
from pathlib import Path
from typing import Tuple, Dict, List, Optional
import logging
from PIL import Image

try:
    from ..base_extractor import BaseExtractor
except ImportError:
    from base_extractor import BaseExtractor

logger = logging.getLogger(__name__)


class ImageExtractor(BaseExtractor):
    """
    Enhanced extractor for image files
    
    Dual extraction approach:
    1. OCR to extract any text present in the image
    2. AI vision model to generate rich, detailed descriptions (via ImageDescriptionService)
    
    This ensures:
    - Text-based queries can match OCR content
    - Visual/conceptual queries can match AI descriptions
    - Better RAG retrieval for diverse query types
    
    Supports: PNG, JPG, JPEG, GIF, BMP, WEBP, TIFF
    """
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        
        # NEW: Use centralized ImageDescriptionService
        self.enable_ai_description = config.get('enable_ai_description', True) if config else True
        
        if self.enable_ai_description:
            try:
                from app.services.image_description_service import get_image_description_service
                self.description_service = get_image_description_service()
                
                if self.description_service.is_available():
                    logger.info("✓ ImageDescriptionService initialized for standalone images")
                else:
                    logger.warning("ImageDescriptionService unavailable, will use OCR-only mode")
                    self.enable_ai_description = False
                    
            except Exception as e:
                logger.warning(f"Could not initialize ImageDescriptionService: {e}")
                logger.warning("Will use OCR-only mode")
                self.enable_ai_description = False
                self.description_service = None
        else:
            self.description_service = None
    
    def _get_supported_formats(self) -> List[str]:
        return ['png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp', 'tiff', 'tif']
    
    def validate_file(self, file_path: str) -> Tuple[bool, str]:
        """Validate image file"""
        path = Path(file_path)
        
        if not path.exists():
            return False, "File does not exist"
        
        ext = path.suffix.lower().lstrip('.')
        if ext not in self.supported_formats:
            return False, f"Not a supported image format"
        
        # Try to open with PIL to verify it's a valid image
        try:
            img = Image.open(file_path)
            img.verify()
            return True, "Valid image file"
        except Exception as e:
            return False, f"Invalid or corrupted image: {str(e)}"
    
    def extract(self, file_path: str) -> Tuple[str, bool, Dict]:
        """
        Extract text and generate AI description from image
        
        Returns markdown with:
        - Image metadata
        - OCR extracted text (if any)
        - AI-generated rich description (using ImageDescriptionService)
        """
        try:
            is_valid, msg = self.validate_file(file_path)
            if not is_valid:
                return "", False, {'error': msg}
            
            logger.info(f"Extracting from image: {file_path}")
            
            # Open image
            img = Image.open(file_path)
            image_name = Path(file_path).name
            
            # Get image metadata
            width, height = img.size
            format_type = img.format
            mode = img.mode
            
            # 1. Extract text using OCR
            ocr_text = self._extract_text_with_ocr(img)
            
            # 2. Generate AI description (NEW: using ImageDescriptionService)
            ai_description = None
            description_success = False
            
            if self.enable_ai_description and self.description_service:
                description_success, ai_description = self.description_service.describe_image_from_path(
                    file_path,
                    context=f"Standalone image file: {image_name}",
                    page_number=None
                )
                
                if description_success:
                    logger.info(f"✓ Generated AI description ({len(ai_description)} chars)")
                else:
                    logger.warning(f"AI description generation failed")
            
            # Build rich markdown content
            markdown_parts = []
            
            # Header with metadata
            markdown_parts.append(f"# Image: {image_name}\n")
            markdown_parts.append(f"**Dimensions**: {width}x{height}")
            markdown_parts.append(f"**Format**: {format_type}")
            markdown_parts.append(f"**Mode**: {mode}")
            markdown_parts.append(f"**Total Pages**: 1\n")
            markdown_parts.append("---\n")
            
            # Page marker
            markdown_parts.append("## Page 1\n")
            
            # NEW: Add AI description first (most valuable for RAG)
            if ai_description:
                markdown_parts.append("### AI Visual Analysis\n")
                markdown_parts.append(f"**[Image Description]**: {ai_description}")
                markdown_parts.append("*(Standalone image file)*")
                markdown_parts.append("")
            
            # Add OCR text if available
            if ocr_text:
                markdown_parts.append("### Extracted Text (OCR)\n")
                markdown_parts.append(ocr_text)
                markdown_parts.append("")
                logger.info(f"✓ Extracted OCR text ({len(ocr_text)} chars)")
            
            # If neither available
            if not ocr_text and not ai_description:
                markdown_parts.append("*No text detected and AI analysis not available*\n")
                logger.warning("No content extracted from image")
            
            markdown_content = "\n".join(markdown_parts)
            
            # Build metadata
            metadata = self.get_metadata(file_path)
            metadata.update({
                'total_pages': 1,
                'width': width,
                'height': height,
                'format': format_type,
                'mode': mode,
                'ocr_text_length': len(ocr_text) if ocr_text else 0,
                'ai_description_length': len(ai_description) if ai_description else 0,
                'has_ocr': bool(ocr_text),
                'has_ai_description': bool(ai_description),
                'description_success': description_success,
                'extraction_method': 'hybrid' if (ocr_text and ai_description) else ('ocr' if ocr_text else 'ai'),
                'content_type': 'image'
            })
            
            logger.info("✓ Image extraction complete")
            
            return markdown_content, True, metadata
            
        except Exception as e:
            logger.error(f"Image extraction error: {str(e)}")
            import traceback
            traceback.print_exc()
            return "", False, {'error': str(e)}
    
    def _extract_text_with_ocr(self, img: Image.Image) -> Optional[str]:
        """
        Extract text from image using OCR (pytesseract)
        
        Returns:
            Extracted text or None if OCR not available
        """
        try:
            import pytesseract
            
            # Run OCR
            text = pytesseract.image_to_string(img)
            
            if text and text.strip():
                return text.strip()
            else:
                return None
                
        except ImportError:
            logger.warning("pytesseract not installed. Install with: pip install pytesseract")
            logger.warning("Also install Tesseract OCR engine (system-level)")
            return None
        except Exception as e:
            logger.warning(f"OCR failed: {e}")
            return None