"""
Image File Extractor - Simplified
Extracts text from images using OCR and converts to markdown
"""

import os
from pathlib import Path
from typing import Tuple, Dict, List
import logging
from PIL import Image

try:
    from ..base_extractor import BaseExtractor
except ImportError:
    from base_extractor import BaseExtractor

logger = logging.getLogger(__name__)


class ImageExtractor(BaseExtractor):
    """
    Extractor for image files
    
    Simple approach:
    1. Run OCR on the image to extract text
    2. Convert to markdown format
    3. Include image metadata
    
    Supports: PNG, JPG, JPEG, GIF, BMP, WEBP, TIFF
    """
    
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
        """Extract text from image using OCR"""
        try:
            is_valid, msg = self.validate_file(file_path)
            if not is_valid:
                return "", False, {'error': msg}
            
            logger.info(f"Extracting text from image: {file_path}")
            
            # Open image
            img = Image.open(file_path)
            image_name = Path(file_path).name
            
            # Get image metadata
            width, height = img.size
            format_type = img.format
            mode = img.mode
            
            # Run OCR to extract text
            extracted_text = self._extract_text_with_ocr(img)
            
            # Build markdown
            markdown_parts = []
            
            # Header
            markdown_parts.append(f"# Image: {image_name}\n")
            markdown_parts.append(f"**Dimensions**: {width}x{height}")
            markdown_parts.append(f"**Format**: {format_type}")
            markdown_parts.append(f"**Mode**: {mode}")
            markdown_parts.append(f"**Total Pages**: 1\n")
            markdown_parts.append("---\n")
            
            # Page marker (single page for an image)
            markdown_parts.append("## Page 1\n")
            
            # Add extracted text if available
            if extracted_text:
                markdown_parts.append("### Extracted Text (OCR)\n")
                markdown_parts.append(extracted_text)
                markdown_parts.append("")
                logger.info(f"Extracted {len(extracted_text)} characters via OCR")
            else:
                markdown_parts.append("*No text detected in image*\n")
                logger.info("No text found in image")
            
            markdown_content = "\n".join(markdown_parts)
            
            # Build metadata
            metadata = self.get_metadata(file_path)
            metadata.update({
                'total_pages': 1,
                'width': width,
                'height': height,
                'format': format_type,
                'mode': mode,
                'text_extracted': len(extracted_text) if extracted_text else 0,
                'ocr_available': extracted_text is not None,
                'extraction_method': 'OCR'
            })
            
            logger.info("✓ Image text extraction complete")
            
            return markdown_content, True, metadata
            
        except Exception as e:
            logger.error(f"Image extraction error: {str(e)}")
            import traceback
            traceback.print_exc()
            return "", False, {'error': str(e)}
    
    def _extract_text_with_ocr(self, img: Image.Image) -> str:
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