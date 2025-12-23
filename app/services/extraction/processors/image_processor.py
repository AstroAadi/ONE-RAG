"""
Image Processor
Shared utilities for image processing across all extractors
"""

import os
import base64
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from PIL import Image
import io

logger = logging.getLogger(__name__)


class ImageProcessor:
    """
    Utilities for processing images in documents
    
    Handles:
    - Image extraction and saving
    - Image resizing and optimization
    - OCR text extraction
    - Image metadata extraction
    - Format conversion
    """
    
    @staticmethod
    def create_image_directory(base_path: str, document_name: str) -> str:
        """
        Create directory for storing extracted images
        
        Args:
            base_path: Base directory path
            document_name: Name of the document (used for subfolder)
            
        Returns:
            Path to created image directory
        """
        image_dir = os.path.join(
            os.path.dirname(base_path),
            "extracted_images",
            document_name
        )
        os.makedirs(image_dir, exist_ok=True)
        logger.debug(f"Created image directory: {image_dir}")
        return image_dir
    
    @staticmethod
    def save_image_from_base64(
        base64_data: str, 
        output_path: str, 
        image_format: str = 'png'
    ) -> bool:
        """
        Save image from base64 data
        
        Args:
            base64_data: Base64 encoded image data
            output_path: Path to save the image
            image_format: Image format (png, jpg, etc.)
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            # Decode base64
            image_data = base64.b64decode(base64_data)
            
            # Save directly
            with open(output_path, 'wb') as f:
                f.write(image_data)
            
            logger.debug(f"Saved image: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving image: {str(e)}")
            return False
    
    @staticmethod
    def extract_images_from_base64_markdown(
        markdown_text: str,
        output_dir: str,
        base_filename: str = "image"
    ) -> Dict[int, Dict]:
        """
        Extract images from markdown with embedded base64 data
        
        Args:
            markdown_text: Markdown text with base64 images
            output_dir: Directory to save images
            base_filename: Base name for saved images
            
        Returns:
            Dictionary mapping image index to image info
            {1: {'filename': 'image_001.png', 'path': '/path/to/image_001.png', 'format': 'png'}}
        """
        # Pattern to match markdown images with base64 data
        pattern = r'!\[([^\]]*)\]\(data:image/([^;]+);base64,([^)]+)\)'
        matches = re.findall(pattern, markdown_text)
        
        image_mapping = {}
        
        for idx, match in enumerate(matches, start=1):
            try:
                alt_text = match[0]
                image_format = match[1]
                base64_data = match[2]
                
                # Normalize format
                ext = image_format.lower()
                if ext == 'jpeg':
                    ext = 'jpg'
                
                # Generate filename
                filename = f"{base_filename}_{idx:03d}.{ext}"
                output_path = os.path.join(output_dir, filename)
                
                # Save image
                if ImageProcessor.save_image_from_base64(base64_data, output_path, ext):
                    image_mapping[idx] = {
                        'filename': filename,
                        'path': output_path,
                        'format': ext,
                        'alt_text': alt_text
                    }
                    logger.info(f"  ✓ Saved: {filename}")
                
            except Exception as e:
                logger.warning(f"  ⚠ Could not save image {idx}: {e}")
                continue
        
        logger.info(f"Extracted {len(image_mapping)} images")
        return image_mapping
    
    @staticmethod
    def resize_image(
        image_path: str, 
        max_width: int = 1920, 
        max_height: int = 1080,
        quality: int = 85
    ) -> bool:
        """
        Resize image to fit within max dimensions
        
        Args:
            image_path: Path to image file
            max_width: Maximum width in pixels
            max_height: Maximum height in pixels
            quality: JPEG quality (1-100)
            
        Returns:
            True if resized successfully
        """
        try:
            img = Image.open(image_path)
            original_size = img.size
            
            # Calculate new size maintaining aspect ratio
            img.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
            
            # Save resized image
            if img.size != original_size:
                img.save(image_path, quality=quality, optimize=True)
                logger.debug(f"Resized image {image_path}: {original_size} -> {img.size}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error resizing image {image_path}: {str(e)}")
            return False
    
    @staticmethod
    def get_image_metadata(image_path: str) -> Dict:
        """
        Extract metadata from image file
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dictionary with image metadata
        """
        try:
            img = Image.open(image_path)
            
            metadata = {
                'width': img.width,
                'height': img.height,
                'format': img.format,
                'mode': img.mode,
                'size_bytes': os.path.getsize(image_path)
            }
            
            # Get EXIF data if available
            if hasattr(img, '_getexif') and img._getexif():
                metadata['has_exif'] = True
            else:
                metadata['has_exif'] = False
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error getting image metadata: {str(e)}")
            return {}
    
    @staticmethod
    def extract_text_from_image(image_path: str) -> Optional[str]:
        """
        Extract text from image using OCR (requires pytesseract)
        
        Args:
            image_path: Path to image file
            
        Returns:
            Extracted text or None if OCR fails
        """
        try:
            import pytesseract
            
            img = Image.open(image_path)
            text = pytesseract.image_to_string(img)
            
            return text.strip() if text else None
            
        except ImportError:
            logger.warning("pytesseract not installed. OCR unavailable.")
            return None
        except Exception as e:
            logger.error(f"OCR error: {str(e)}")
            return None
    
    @staticmethod
    def convert_image_format(
        input_path: str, 
        output_path: str, 
        target_format: str = 'png'
    ) -> bool:
        """
        Convert image to different format
        
        Args:
            input_path: Input image path
            output_path: Output image path
            target_format: Target format (png, jpg, etc.)
            
        Returns:
            True if converted successfully
        """
        try:
            img = Image.open(input_path)
            
            # Convert RGBA to RGB for JPEG
            if target_format.lower() in ['jpg', 'jpeg'] and img.mode == 'RGBA':
                img = img.convert('RGB')
            
            img.save(output_path, format=target_format.upper())
            logger.debug(f"Converted {input_path} to {target_format}")
            return True
            
        except Exception as e:
            logger.error(f"Error converting image: {str(e)}")
            return False
    
    @staticmethod
    def optimize_image(image_path: str, quality: int = 85) -> bool:
        """
        Optimize image file size
        
        Args:
            image_path: Path to image file
            quality: Quality setting for JPEG (1-100)
            
        Returns:
            True if optimized successfully
        """
        try:
            img = Image.open(image_path)
            original_size = os.path.getsize(image_path)
            
            # Save with optimization
            img.save(image_path, optimize=True, quality=quality)
            
            new_size = os.path.getsize(image_path)
            saved = original_size - new_size
            
            if saved > 0:
                logger.debug(f"Optimized {image_path}: saved {saved} bytes")
            
            return True
            
        except Exception as e:
            logger.error(f"Error optimizing image: {str(e)}")
            return False
    
    @staticmethod
    def is_valid_image(image_path: str) -> bool:
        """
        Check if file is a valid image
        
        Args:
            image_path: Path to image file
            
        Returns:
            True if valid image, False otherwise
        """
        try:
            img = Image.open(image_path)
            img.verify()
            return True
        except Exception:
            return False
    
    @staticmethod
    def get_image_dimensions(image_path: str) -> Tuple[int, int]:
        """
        Get image dimensions
        
        Args:
            image_path: Path to image file
            
        Returns:
            Tuple of (width, height) or (0, 0) if error
        """
        try:
            img = Image.open(image_path)
            return img.size
        except Exception:
            return (0, 0)
    
    @staticmethod
    def image_to_base64(image_path: str) -> Optional[str]:
        """
        Convert image file to base64 string
        
        Args:
            image_path: Path to image file
            
        Returns:
            Base64 encoded string or None if error
        """
        try:
            with open(image_path, 'rb') as f:
                image_data = f.read()
            return base64.b64encode(image_data).decode('utf-8')
        except Exception as e:
            logger.error(f"Error encoding image to base64: {str(e)}")
            return None
    
    @staticmethod
    def create_thumbnail(
        image_path: str, 
        thumbnail_path: str, 
        size: Tuple[int, int] = (200, 200)
    ) -> bool:
        """
        Create thumbnail of image
        
        Args:
            image_path: Original image path
            thumbnail_path: Path to save thumbnail
            size: Thumbnail size (width, height)
            
        Returns:
            True if thumbnail created successfully
        """
        try:
            img = Image.open(image_path)
            img.thumbnail(size, Image.Resampling.LANCZOS)
            img.save(thumbnail_path, optimize=True)
            logger.debug(f"Created thumbnail: {thumbnail_path}")
            return True
        except Exception as e:
            logger.error(f"Error creating thumbnail: {str(e)}")
            return False


class ImageHandler:
    """
    Legacy alias for ImageProcessor
    Maintains backward compatibility
    """
    
    @staticmethod
    def create_image_directory(base_path: str, document_name: str) -> str:
        return ImageProcessor.create_image_directory(base_path, document_name)
    
    @staticmethod
    def extract_and_save_images(
        markdown_with_base64: str,
        image_output_dir: str,
        pdf_stem: str,
        document
    ) -> Dict[int, Dict]:
        """
        Extract and save images with page numbers (for PDF documents)
        
        This is a specialized method for PDF extraction that includes
        page number detection.
        """
        pattern = r'!\[([^\]]*)\]\(data:image/([^;]+);base64,([^)]+)\)'
        matches = re.findall(pattern, markdown_with_base64)
        image_mapping = {}
        
        # Get page hints if available
        page_hints = {}
        try:
            if hasattr(document, 'pictures') and document.pictures:
                for idx, picture in enumerate(document.pictures, start=1):
                    if hasattr(picture, 'prov'):
                        for prov in picture.prov:
                            if hasattr(prov, 'page_no'):
                                page_hints[idx] = prov.page_no + 1
                                break
        except Exception as e:
            logger.debug(f"Could not extract page hints: {e}")
        
        total_pages = len(document.pages) if hasattr(document, 'pages') else 1
        
        for idx, match in enumerate(matches, start=1):
            try:
                image_format = match[1]
                base64_data = match[2]
                
                # Decode image
                image_data = base64.b64decode(base64_data)
                
                ext = image_format.lower()
                if ext == 'jpeg':
                    ext = 'jpg'
                
                # Determine page number
                page_number = page_hints.get(idx)
                if page_number is None:
                    # Estimate page
                    page_number = int((idx / len(matches)) * total_pages) + 1 if len(matches) > 0 else 1
                    page_number = max(1, min(page_number, total_pages))
                
                # Save with page number in filename
                image_filename = f"page_{page_number:02d}_image_{idx:03d}.{ext}"
                image_path = os.path.join(image_output_dir, image_filename)
                
                with open(image_path, 'wb') as f:
                    f.write(image_data)
                
                image_mapping[idx] = {
                    'filename': image_filename,
                    'page_number': page_number,
                    'format': ext
                }
                
                logger.info(f"  ✓ Saved: {image_filename} (page {page_number})")
                
            except Exception as e:
                logger.warning(f"  ⚠ Could not save image {idx}: {e}")
                continue
        
        logger.info(f"Total images: {len(image_mapping)}")
        return image_mapping