"""
Image Description Service
Uses vision-capable LLM to generate textual descriptions of images at ingestion time
"""

import os
import base64
import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import boto3
import json

from app.core.config import settings

logger = logging.getLogger(__name__)


class ImageDescriptionService:
    """
    Service to generate textual descriptions of images using vision LLM
    
    This service is called at document ingestion time to convert images
    into textual descriptions, eliminating the need for vision capabilities
    at query time.
    """
    
    def __init__(self):
        """Initialize Bedrock client for vision model"""
        try:
            if settings.aws_profile_name:
                session = boto3.Session(
                    profile_name=settings.aws_profile_name,
                    region_name=settings.aws_region
                )
                self.bedrock_runtime = session.client('bedrock-runtime')
            else:
                self.bedrock_runtime = boto3.client(
                    'bedrock-runtime',
                    region_name=settings.aws_region,
                    aws_access_key_id=settings.aws_access_key_id if settings.aws_access_key_id else None,
                    aws_secret_access_key=settings.aws_secret_access_key if settings.aws_secret_access_key else None
                )
            
            self.vision_model_id = settings.bedrock_vision_model_id
            self.max_tokens = settings.vision_max_tokens
            self.temperature = settings.vision_temperature
            
            logger.info(f"Image Description Service initialized with model: {self.vision_model_id}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Bedrock client: {e}")
            self.bedrock_runtime = None
    
    def is_available(self) -> bool:
        """Check if vision service is available"""
        return self.bedrock_runtime is not None
    
    def describe_image_from_path(
        self,
        image_path: str,
        context: Optional[str] = None,
        page_number: Optional[int] = None
    ) -> Tuple[bool, str]:
        """
        Generate textual description of an image from file path
        
        Args:
            image_path: Path to the image file
            context: Optional context about the document (e.g., document title, section)
            page_number: Optional page number where image appears
            
        Returns:
            Tuple of (success: bool, description: str)
        """
        try:
            # Read and encode image
            with open(image_path, 'rb') as f:
                image_bytes = f.read()
            
            base64_image = base64.b64encode(image_bytes).decode('utf-8')
            
            # Determine image format
            ext = Path(image_path).suffix.lstrip('.').lower()
            media_type = self._get_media_type(ext)
            
            return self.describe_image_from_base64(
                base64_image,
                media_type,
                context=context,
                page_number=page_number,
                filename=Path(image_path).name
            )
            
        except FileNotFoundError:
            logger.error(f"Image file not found: {image_path}")
            return False, f"[Image file not found: {Path(image_path).name}]"
        except Exception as e:
            logger.error(f"Error reading image {image_path}: {e}")
            return False, f"[Error reading image: {str(e)}]"
    
    def describe_image_from_base64(
        self,
        base64_image: str,
        media_type: str,
        context: Optional[str] = None,
        page_number: Optional[int] = None,
        filename: Optional[str] = None
    ) -> Tuple[bool, str]:
        """
        Generate textual description of an image from base64 data
        
        Args:
            base64_image: Base64-encoded image data
            media_type: MIME type (e.g., 'image/jpeg', 'image/png')
            context: Optional context about the document
            page_number: Optional page number
            filename: Optional filename for error messages
            
        Returns:
            Tuple of (success: bool, description: str)
        """
        if not self.is_available():
            logger.warning("Vision service not available")
            return False, "[Vision service unavailable - image description could not be generated]"
        
        try:
            # Build prompt for image description
            prompt = self._build_description_prompt(context, page_number)
            
            # Prepare request
            content_parts = [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": base64_image
                    }
                },
                {
                    "type": "text",
                    "text": prompt
                }
            ]
            
            request_body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "messages": [{
                    "role": "user",
                    "content": content_parts
                }]
            }
            
            # Call Bedrock
            response = self.bedrock_runtime.invoke_model(
                modelId=self.vision_model_id,
                body=json.dumps(request_body)
            )
            
            response_body = json.loads(response['body'].read())
            
            if 'content' in response_body and response_body['content']:
                description = response_body['content'][0]['text'].strip()
                logger.debug(f"Generated description for image: {description[:100]}...")
                return True, description
            else:
                logger.warning("No content in vision response")
                return False, "[Image description generation failed - no response]"
                
        except Exception as e:
            logger.error(f"Vision API error: {e}")
            import traceback
            traceback.print_exc()
            
            fallback = f"[Image description unavailable: {str(e)}]"
            if filename:
                fallback = f"[Image: {filename} - description unavailable]"
            
            return False, fallback
    
    def describe_multiple_images(
        self,
        image_paths: List[str],
        context: Optional[str] = None,
        page_numbers: Optional[Dict[str, int]] = None
    ) -> Dict[str, str]:
        """
        Generate descriptions for multiple images
        
        Args:
            image_paths: List of image file paths
            context: Optional document context
            page_numbers: Optional dict mapping image paths to page numbers
            
        Returns:
            Dictionary mapping image paths to descriptions
        """
        descriptions = {}
        page_numbers = page_numbers or {}
        
        for image_path in image_paths:
            page_num = page_numbers.get(image_path)
            success, description = self.describe_image_from_path(
                image_path,
                context=context,
                page_number=page_num
            )
            descriptions[image_path] = description
            
            if not success:
                logger.warning(f"Failed to describe image: {image_path}")
        
        return descriptions
    
    def _build_description_prompt(
        self,
        context: Optional[str] = None,
        page_number: Optional[int] = None
    ) -> str:
        """
        Build prompt for image description
        
        The prompt is optimized for generating detailed, searchable descriptions
        that will be embedded in the document and used for RAG retrieval.
        """
        prompt_parts = [
            "Analyze this image and provide a detailed textual description.",
            "",
            "Your description should include:",
            "- What type of visual element it is (diagram, chart, photo, illustration, screenshot, etc.)",
            "- The main subject or content",
            "- Key details, text, labels, or data visible in the image",
            "- Colors, layout, or visual structure if relevant",
            "- Any relationships, trends, or patterns shown",
            "",
            "Be specific and factual. This description will be used for document search and retrieval,",
            "so include relevant keywords and details that someone might search for.",
            "Use markdown formatting where appropriate because this description will be embedded in markdown documents.",
            ""
        ]
        
        if context:
            prompt_parts.extend([
                f"Document context: {context}",
                ""
            ])
        
        if page_number:
            prompt_parts.extend([
                f"This image appears on page {page_number}.",
                ""
            ])
        
        prompt_parts.extend([
            "Provide the description in markdown format, focusing on factual content.",
            "Start your description directly without preamble."
        ])
        
        return "\n".join(prompt_parts)
    
    def _get_media_type(self, extension: str) -> str:
        """Map file extension to MIME type"""
        media_types = {
            'jpg': 'image/jpeg',
            'jpeg': 'image/jpeg',
            'png': 'image/png',
            'gif': 'image/gif',
            'webp': 'image/webp',
            'bmp': 'image/bmp',
            'tiff': 'image/tiff',
            'tif': 'image/tiff'
        }
        return media_types.get(extension.lower(), 'image/jpeg')
    
    def generate_fallback_description(
        self,
        filename: str,
        page_number: Optional[int] = None,
        image_type: Optional[str] = None
    ) -> str:
        """
        Generate a fallback description when vision API is unavailable
        
        Args:
            filename: Image filename
            page_number: Optional page number
            image_type: Optional detected image type (figure, chart, etc.)
            
        Returns:
            Fallback description string
        """
        parts = []
        
        if image_type:
            parts.append(f"[{image_type.capitalize()}:")
        else:
            parts.append("[Image:")
        
        parts.append(filename)
        
        if page_number:
            parts.append(f"on page {page_number}")
        
        parts.append("- AI description unavailable]")
        
        return " ".join(parts)


# Singleton instance
_image_description_service = None


def get_image_description_service() -> ImageDescriptionService:
    """Get singleton instance of ImageDescriptionService"""
    global _image_description_service
    if _image_description_service is None:
        _image_description_service = ImageDescriptionService()
    return _image_description_service