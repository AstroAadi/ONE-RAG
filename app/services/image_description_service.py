"""
Vision Service - Using HuggingFace Router API
Based on working Colab example with direct API calls
"""

import base64
import requests
import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import time
from io import BytesIO
from PIL import Image

from app.core.config import settings

logger = logging.getLogger(__name__)


class OpenSourceVisionService:
    """
    Vision service using HuggingFace Router API
    
    Uses the same pattern as working Colab example
    """
    
    def __init__(self, model_id: str = None):
        """Initialize vision service"""
        self.model_id = model_id or settings.vision_model_id
        self.api_token = settings.huggingface_api_token
        self.api_url = settings.huggingface_api_url  # Router API
        
        self.max_retries = 3
        self.retry_delay = 2
        
        logger.info(f"✅ Vision service initialized: {self.model_id}")
    
    def is_available(self) -> bool:
        """Check if vision service is available"""
        return bool(self.api_token)
    
    def _make_request(self, payload: dict, retry_count: int = 0) -> dict:
        """Make API request with retry logic"""
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=120
            )
            
            # Handle various error codes
            if response.status_code == 503:
                logger.warning("Model is loading, waiting...")
                if retry_count < self.max_retries:
                    time.sleep(self.retry_delay * (retry_count + 1))
                    return self._make_request(payload, retry_count + 1)
                else:
                    raise Exception("Model failed to load after retries")
            
            if response.status_code == 410:
                raise Exception(f"Model {self.model_id} is not available or has been deprecated. Try a different model.")
            
            if response.status_code != 200:
                logger.error(f"HTTP Error {response.status_code}: {response.text}")
                raise Exception(f"API error: {response.status_code}")
            
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Vision API request failed: {e}")
            if retry_count < self.max_retries:
                logger.info(f"Retrying... (attempt {retry_count + 1}/{self.max_retries})")
                time.sleep(self.retry_delay)
                return self._make_request(payload, retry_count + 1)
            else:
                raise Exception(f"Vision API failed: {str(e)}")
    
    def describe_image_from_path(
        self,
        image_path: str,
        context: Optional[str] = None,
        page_number: Optional[int] = None
    ) -> Tuple[bool, str]:
        """Generate description from image file path"""
        try:
            # Read and encode image
            with Image.open(image_path) as img:
                # Convert to RGB if needed
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Resize if too large (to avoid API limits)
                max_size = 1024
                if max(img.size) > max_size:
                    ratio = max_size / max(img.size)
                    new_size = tuple(int(dim * ratio) for dim in img.size)
                    img = img.resize(new_size, Image.Resampling.LANCZOS)
                
                # Convert to base64
                byte_arr = BytesIO()
                img.save(byte_arr, format="PNG")
                image_base64 = base64.b64encode(byte_arr.getvalue()).decode("utf-8")
            
            return self.describe_image_from_base64(
                image_base64,
                context=context,
                page_number=page_number,
                filename=Path(image_path).name
            )
            
        except FileNotFoundError:
            return False, f"[Image file not found: {Path(image_path).name}]"
        except Exception as e:
            logger.error(f"Error reading image: {e}")
            return False, f"[Error reading image: {str(e)}]"
    
    def describe_image_from_base64(
        self,
        base64_image: str,
        media_type: str = "image/png",
        context: Optional[str] = None,
        page_number: Optional[int] = None,
        filename: Optional[str] = None
    ) -> Tuple[bool, str]:
        """Generate description from base64 image"""
        try:
            # Build prompt
            prompt = self._build_description_prompt(context, page_number)
            
            # Format image data URL (same as working Colab example)
            image_data_url = f"data:{media_type};base64,{base64_image}"
            
            # Build payload following working Colab pattern
            payload = {
                "model": self.model_id,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": image_data_url
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ],
                "max_tokens": settings.vision_max_tokens,
                "temperature": settings.vision_temperature
            }
            
            result = self._make_request(payload)
            
            # Extract description from response
            if "choices" in result and len(result["choices"]) > 0:
                description = result["choices"][0]["message"]["content"].strip()
                logger.debug(f"Generated description: {description[:100]}...")
                return True, description
            else:
                logger.warning("No content in vision response")
                return False, "[Image description generation failed]"
                
        except Exception as e:
            logger.error(f"Vision API error: {e}")
            fallback = f"[Image description unavailable: {str(e)}]"
            if filename:
                fallback = f"[Image: {filename} - description unavailable]"
            return False, fallback
    
    def describe_images_batch(
        self,
        images: List[Dict[str, str]],
        context: Optional[str] = None,
        page_number: Optional[int] = None,
        max_batch_size: int = 5
    ) -> List[Dict[str, any]]:
        """Describe multiple images (processes individually)"""
        results = []
        
        for i, img_data in enumerate(images):
            try:
                success, description = self.describe_image_from_base64(
                    img_data['base64'],
                    img_data.get('media_type', 'image/png'),
                    context=context,
                    page_number=page_number
                )
                
                results.append({
                    'success': success,
                    'description': description,
                    'error': None if success else 'Description failed'
                })
                
                # Delay between requests
                if i < len(images) - 1:
                    time.sleep(0.5)
                    
            except Exception as e:
                results.append({
                    'success': False,
                    'description': f'[Error: {str(e)}]',
                    'error': str(e)
                })
        
        return results
    
    def _build_description_prompt(
        self,
        context: Optional[str] = None,
        page_number: Optional[int] = None
    ) -> str:
        """Build prompt for image description"""
        prompt_parts = [
            "Analyze this image and provide a detailed description.",
            ""
        ]
        
        if context:
            prompt_parts.append(f"Context: {context}")
        if page_number:
            prompt_parts.append(f"Page: {page_number}")
        if context or page_number:
            prompt_parts.append("")
        
        prompt_parts.extend([
            "Provide a comprehensive description including:",
            "",
            "For text/code in images:",
            "- Extract ALL text EXACTLY as shown",
            "- Preserve formatting and indentation",
            "",
            "For tables:",
            "- Create markdown table format",
            "- Include all rows and columns",
            "",
            "For charts/diagrams:",
            "- Describe type, title, axes, data points",
            "- List key values and trends",
            "",
            "For photos/screenshots:",
            "- Describe main subject and visual elements",
            "- Extract any visible text",
            "",
            "Be thorough and precise. This will be used for document search."
        ])
        
        return "\n".join(prompt_parts)
    
    def generate_fallback_description(
        self,
        filename: str,
        page_number: Optional[int] = None,
        image_type: Optional[str] = None
    ) -> str:
        """Generate fallback description when API fails"""
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
    
    def update_model(self, new_model_id: str) -> bool:
        """Update to a different vision model"""
        try:
            self.model_id = new_model_id
            logger.info(f"✅ Vision model updated to: {new_model_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to update vision model: {e}")
            return False


# Singleton instance
_vision_service = None

def get_image_description_service() -> OpenSourceVisionService:
    """Get singleton instance"""
    global _vision_service
    if _vision_service is None:
        _vision_service = OpenSourceVisionService()
    return _vision_service