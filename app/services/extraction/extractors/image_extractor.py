"""
Enhanced Image File Extractor
Extracts text from images using OCR AND generates rich AI descriptions
for better RAG retrieval and question answering
"""

import os
from pathlib import Path
from typing import Tuple, Dict, List, Optional
import logging
from PIL import Image
import base64
import json
import boto3

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
    2. AI vision model to generate rich, detailed descriptions
    
    This ensures:
    - Text-based queries can match OCR content
    - Visual/conceptual queries can match AI descriptions
    - Better RAG retrieval for diverse query types
    
    Supports: PNG, JPG, JPEG, GIF, BMP, WEBP, TIFF
    """
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        
        # Initialize AWS Bedrock for vision
        self.bedrock_runtime = None
        self.enable_ai_description = config.get('enable_ai_description', True) if config else True
        
        if self.enable_ai_description:
            try:
                # Try to initialize Bedrock
                aws_region = os.getenv('AWS_REGION', 'us-east-1')
                aws_profile = os.getenv('AWS_PROFILE_NAME')
                
                if aws_profile:
                    session = boto3.Session(profile_name=aws_profile, region_name=aws_region)
                    self.bedrock_runtime = session.client('bedrock-runtime')
                else:
                    self.bedrock_runtime = boto3.client('bedrock-runtime', region_name=aws_region)
                
                logger.info("✓ Bedrock initialized for AI image descriptions")
                
            except Exception as e:
                logger.warning(f"Could not initialize Bedrock for AI descriptions: {e}")
                logger.warning("Will use OCR-only mode")
                self.enable_ai_description = False
    
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
        - AI-generated rich description
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
            
            # 2. Generate AI description
            ai_description = None
            if self.enable_ai_description:
                ai_description = self._generate_ai_description(file_path, img)
            
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
            
            # Add AI description first (most valuable for RAG)
            if ai_description:
                markdown_parts.append("### AI Visual Analysis\n")
                markdown_parts.append(ai_description)
                markdown_parts.append("")
                logger.info(f"✓ Generated AI description ({len(ai_description)} chars)")
            
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
    
    def _generate_ai_description(self, file_path: str, img: Image.Image) -> Optional[str]:
        """
        Generate rich AI description using Claude Vision
        
        This description helps answer conceptual questions about:
        - What's in the image (objects, people, scenes)
        - Colors, styles, composition
        - Text content and context
        - Charts/graphs and their insights
        - Relationships between elements
        - Overall purpose/message
        
        Returns:
            Detailed description or None if AI not available
        """
        if not self.bedrock_runtime:
            return None
        
        try:
            # Convert image to base64
            image_bytes = self._image_to_base64_bytes(img)
            
            # Detect image type for media_type
            img_format = img.format.lower() if img.format else 'jpeg'
            media_type_map = {
                'jpeg': 'image/jpeg',
                'jpg': 'image/jpeg',
                'png': 'image/png',
                'gif': 'image/gif',
                'webp': 'image/webp',
                'bmp': 'image/bmp',
                'tiff': 'image/tiff'
            }
            media_type = media_type_map.get(img_format, 'image/jpeg')
            
            # Create prompt for comprehensive description
            prompt = """Please provide a comprehensive description of this image for a knowledge base system. Your description will be used to help answer questions about this image later.

            Include the following in your analysis:

            1. **Main Subject**: What is the primary focus or subject of the image?

            2. **Visual Elements**: 
            - What objects, people, or elements are present?
            - What are the prominent colors and visual style?
            - Describe the composition and layout

            3. **Text Content**: If there is any text in the image (signs, labels, captions, etc.), transcribe it and explain its context

            4. **Data & Information**: If this is a chart, graph, diagram, or infographic:
            - What type of visualization is it?
            - What data or information does it present?
            - What are the key insights or trends?
            - What are the specific values, labels, or data points shown?

            5. **Context & Purpose**: What appears to be the purpose or message of this image?

            6. **Notable Details**: Any other important details that would help answer questions about this image

            Be specific, detailed, and comprehensive. This description will be stored in a vector database and used to answer diverse questions about the image."""

            # Build request body for Claude Vision
            request_body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 2000,
                "temperature": 0.3,  # Lower temperature for more factual descriptions
                "messages": [{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": image_bytes
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }]
            }
            
            # Call Claude Vision via Bedrock
            response = self.bedrock_runtime.invoke_model(
                modelId="us.anthropic.claude-3-5-sonnet-20241022-v2:0",
                body=json.dumps(request_body)
            )
            
            response_body = json.loads(response['body'].read())
            
            if 'content' in response_body and response_body['content']:
                description = response_body['content'][0]['text']
                return description
            else:
                logger.warning("AI vision returned empty response")
                return None
                
        except Exception as e:
            logger.error(f"AI description generation failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _image_to_base64_bytes(self, img: Image.Image) -> str:
        """
        Convert PIL Image to base64 string
        
        Returns:
            Base64 encoded string of image bytes
        """
        import io
        
        # Convert image to RGB if necessary (for formats like PNG with transparency)
        if img.mode in ('RGBA', 'LA', 'P'):
            # Create white background
            background = Image.new('RGB', img.size, (255, 255, 255))
            if img.mode == 'P':
                img = img.convert('RGBA')
            background.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
            img = background
        elif img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Save to bytes
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=95)
        image_bytes = buffer.getvalue()
        
        # Encode to base64
        base64_string = base64.b64encode(image_bytes).decode('utf-8')
        
        return base64_string