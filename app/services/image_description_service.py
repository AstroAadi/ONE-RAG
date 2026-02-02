"""
Image Description Service - Unified Version
Supports both individual and batch image description
"""

import os
import base64
import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import boto3
import json
import re

from app.core.config import settings

logger = logging.getLogger(__name__)


class ImageDescriptionService:
    """
    Service to generate textual descriptions of images using vision LLM
    
    Features:
    - Individual image description
    - Batch image description (multiple images in one API call)
    - Automatic fallback to individual calls if batch fails
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
    
    # ========== INDIVIDUAL IMAGE DESCRIPTION ==========
    
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
            context: Optional context about the document
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
                logger.debug(f"Generated description: {description[:100]}...")
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
    
    # ========== BATCH IMAGE DESCRIPTION ==========
    
    def describe_images_batch(
        self,
        images: List[Dict[str, str]],
        context: Optional[str] = None,
        page_number: Optional[int] = None,
        max_batch_size: int = 5
    ) -> List[Dict[str, any]]:
        """
        Describe multiple images in a single API call
        
        Args:
            images: List of dicts with 'base64' and 'media_type' keys
            context: Optional document context
            page_number: Optional page number
            max_batch_size: Maximum images per batch (default: 5)
            
        Returns:
            List of result dictionaries with 'success', 'description', 'error'
        """
        if not self.is_available():
            logger.warning("Vision service not available")
            return [{
                'success': False,
                'description': '[Vision service unavailable]',
                'error': 'Service not initialized'
            } for _ in images]
        
        if not images:
            return []
        
        # Split into batches if needed
        if len(images) > max_batch_size:
            logger.debug(f"Splitting {len(images)} images into batches of {max_batch_size}")
            results = []
            
            for i in range(0, len(images), max_batch_size):
                batch = images[i:i + max_batch_size]
                batch_results = self._process_batch(batch, context, page_number)
                results.extend(batch_results)
            
            return results
        else:
            return self._process_batch(images, context, page_number)
    
    def _process_batch(
        self,
        images: List[Dict[str, str]],
        context: Optional[str],
        page_number: Optional[int]
    ) -> List[Dict[str, any]]:
        """
        Process a single batch of images
        
        Args:
            images: List of image dictionaries
            context: Document context
            page_number: Page number
            
        Returns:
            List of result dictionaries
        """
        try:
            # Build prompt for batch
            prompt = self._build_batch_prompt(len(images), context, page_number)
            
            # Build content with multiple images
            content_parts = []
            
            for img_data in images:
                content_parts.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": img_data['media_type'],
                        "data": img_data['base64']
                    }
                })
            
            # Add prompt
            content_parts.append({
                "type": "text",
                "text": prompt
            })
            
            # Prepare request
            request_body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": self.max_tokens * len(images),  # Scale tokens
                "temperature": self.temperature,
                "messages": [{
                    "role": "user",
                    "content": content_parts
                }]
            }
            
            # Call Bedrock
            logger.debug(f"Making batch API call for {len(images)} images")
            
            response = self.bedrock_runtime.invoke_model(
                modelId=self.vision_model_id,
                body=json.dumps(request_body)
            )
            
            response_body = json.loads(response['body'].read())
            
            if 'content' in response_body and response_body['content']:
                full_response = response_body['content'][0]['text'].strip()
                
                # Parse individual descriptions
                descriptions = self._parse_batch_response(full_response, len(images))
                
                if len(descriptions) == len(images):
                    logger.info(f"✓ Batch processed {len(images)} images in one API call")
                    
                    return [{
                        'success': True,
                        'description': desc,
                        'error': None
                    } for desc in descriptions]
                else:
                    logger.warning(
                        f"Batch response mismatch: got {len(descriptions)} descriptions "
                        f"for {len(images)} images"
                    )
                    raise ValueError("Response count mismatch")
            else:
                raise ValueError("No content in response")
                
        except Exception as e:
            logger.warning(f"Batch processing failed: {e}, falling back to individual calls")
            return self._fallback_individual(images, context, page_number)
    
    def _fallback_individual(
        self,
        images: List[Dict[str, str]],
        context: Optional[str],
        page_number: Optional[int]
    ) -> List[Dict[str, any]]:
        """
        Fallback to individual API calls
        
        Args:
            images: List of image dictionaries
            context: Document context
            page_number: Page number
            
        Returns:
            List of result dictionaries
        """
        results = []
        
        for img_data in images:
            try:
                success, description = self.describe_image_from_base64(
                    img_data['base64'],
                    img_data['media_type'],
                    context=context,
                    page_number=page_number
                )
                
                results.append({
                    'success': success,
                    'description': description,
                    'error': None if success else 'Description failed'
                })
                
            except Exception as e:
                results.append({
                    'success': False,
                    'description': f'[Error: {str(e)}]',
                    'error': str(e)
                })
        
        return results
    
    # ========== HELPER METHODS ==========
    
    def _build_description_prompt(
        self,
        context: Optional[str] = None,
        page_number: Optional[int] = None
    ) -> str:
        """Build prompt for single image description"""
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
            "Analyze this image with extreme precision and provide a comprehensive description.",
            "",
            "## CAPTION/LABEL RULES (CRITICAL - FOLLOW EXACTLY)",
            "",
            "**STEP 1: CAPTION DETECTION**",
            "",
            "Look for captions like:",
            "- 'Figure X.Y.Z' or 'Fig. X.Y.Z'",
            "- 'Table X.Y.Z' or 'Tbl. X.Y.Z'",
            "- 'Listing X.Y.Z' or 'Code X.Y.Z'",
            "- ANY numbered label with decimals (e.g., '1.3.1', '2.5.3')",
            "",
            "**CRITICAL RULES:**",
            "- Read caption numbers EXACTLY as written, including ALL decimal points",
            "- '1.3.1' is NOT '1.31' - preserve ALL dots",
            "- If unsure, look multiple times at exact spacing and punctuation",
            "",
            "**STEP 2: APPLY CAPTION RULES**",
            "",
            "**Single element:**",
            "- IF caption exists: Copy EXACTLY → Format: **[Exact Caption]**: Description...",
            "- IF no caption: Start directly with description",
            "",
            "**Multiple elements:**",
            "- IF at least one has caption: Use existing + generate for unlabeled following sequence",
            "  Example: If 'Figure 1.3.1' exists → next becomes 'Figure 1.3.2' (not '1.32')",
            "- IF none have captions: Do NOT generate any",
            "",
            "**NEVER:**",
            "- Change '1.3.1' to '1.31'",
            "- Invent captions when none exist",
            "- Skip describing any element",
            "",
            "## IMAGE TYPE HANDLING",
            "",
            "**1. TEXT-ONLY Images (code, config files, logs, text documents)**",
            "",
            "IF image is >90% text/code:",
            "```",
            "**[Caption if present]**",
            "",
            "Type: [Text Document / Code Snippet / Configuration / Log]",
            "Language/Format: [If code: Python/JavaScript/etc. | If text: specify type]",
            "",
            "CONTENT:",
            "```",
            "[Extract ALL text EXACTLY as shown]",
            "[Preserve ALL formatting, indentation, line breaks]",
            "[Include ALL special characters and symbols]",
            "[No paraphrasing - character-by-character accuracy]",
            "```",
            "",
            "Note: [Brief 1-2 sentence description of what this text/code does]",
            "```",
            "",
            "**CRITICAL for text/code:**",
            "- Copy text character-by-character with 100% accuracy",
            "- Preserve indentation, spacing, line breaks",
            "- Include ALL symbols, brackets, operators",
            "- No summarizing - extract complete content",
            "",
            "---",
            "",
            "**2. TABLE Images**",
            "",
            "IF image shows tabular data:",
            "```",
            "**[Caption if present]**",
            "",
            "| Header1 | Header2 | Header3 | ... |",
            "|---------|---------|---------|-----|",
            "| Value   | Value   | Value   | ... |",
            "| Value   | Value   | Value   | ... |",
            "[Include ALL rows exactly]",
            "```",
            "",
            "**CRITICAL for tables:**",
            "- Extract complete table in markdown format",
            "- Include ALL rows and columns",
            "- Preserve exact values",
            "- No additional description needed",
            "",
            "---",
            "",
            "**3. ALL OTHER Images (charts, diagrams, photos, maps, equations)**",
            "",
            "For visualizations, diagrams, flowcharts, photos, maps, mathematical content:",
            "",
            "Provide COMPREHENSIVE description including:",
            "",
            "**For Data Visualizations (charts/graphs):**",
            "- Type: [Specific chart type]",
            "- Title: [Exact title if present]",
            "- Axes: X-axis: [label, range] | Y-axis: [label, range, units]",
            "- Data Series: [Name, color, ALL visible values]",
            "- Key Data Points: [List ALL specific values]",
            "- Trends: [Detailed observations]",
            "- Insights (3-7): [Specific insights with numbers]",
            "- Source: [Exact source text]",
            "",
            "**For Diagrams/Flowcharts:**",
            "- Type: [Flowchart/Architecture/Network/etc.]",
            "- Components: [List each component with label and position]",
            "- Connections: [All arrows and relationships]",
            "- Flow: [Step-by-step process]",
            "- Text Labels: [Extract ALL visible text]",
            "",
            "**For Photos/Screenshots:**",
            "- Type: [Photograph/Screenshot/etc.]",
            "- Main Subject: [Primary focus]",
            "- Visual Elements: [Key elements with details]",
            "- Text Visible: [Extract ALL text]",
            "- Context: [What it represents]",
            "",
            "**For Mathematical Content:**",
            "- Type: Mathematical Content",
            "- Content: [Extract equations exactly, use LaTeX notation]",
            "- Variables: [Define each variable]",
            "- Explanation: [What it represents]",
            "",
            "**For Maps:**",
            "- Type: [Geographic Map/Heat Map/etc.]",
            "- Coverage: [Region shown]",
            "- Legend: [ALL items with exact labels]",
            "- Locations: [All labeled locations with values]",
            "- Patterns: [Geographic insights]",
            "",
            "## GENERAL REQUIREMENTS",
            "",
            "Your description will be used in a RAG system, so it MUST be:",
            "- **COMPREHENSIVE**: Include every visible detail",
            "- **EXACT**: Copy text/numbers word-for-word",
            "- **SEARCHABLE**: Use terms someone might search for",
            "- **COMPLETE**: Cover all elements thoroughly",
            "",
            "## QUALITY CHECKLIST",
            "",
            "Before finalizing:",
            "- ✅ Captions copied EXACTLY (preserve all dots)",
            "- ✅ Text/code extracted character-by-character (if text-only image)",
            "- ✅ Tables in complete markdown format (if table image)",
            "- ✅ Comprehensive description (if visualization/diagram/photo)",
            "- ✅ ALL visible numbers and text included",
            "- ✅ No elements skipped",
            "",
            "## CRITICAL REMINDERS",
            "",
            "1. **Caption Precision**: '1.3.1' ≠ '1.31' - Read decimal points carefully",
            "2. **Text Accuracy**: For text/code images - copy EXACTLY, no paraphrasing",
            "3. **Table Format**: For tables - markdown format only, no extra description",
            "4. **Everything Else**: For visualizations/diagrams/photos - detailed description required",
            "5. **Completeness**: Extract/describe EVERYTHING visible",
            "",
            "Start your analysis now.",
            ""
        ])
        
        return "\n".join(prompt_parts)
    
    def _parse_batch_response(
        self,
        response_text: str,
        expected_count: int
    ) -> List[str]:
        """Parse batch response into individual descriptions"""
        descriptions = []
        
        # Strategy 1: Parse by "IMAGE N:" markers
        pattern = r'IMAGE\s+(\d+):\s*(.+?)(?=IMAGE\s+\d+:|$)'
        matches = re.findall(pattern, response_text, re.DOTALL | re.IGNORECASE)
        
        if matches:
            matches_sorted = sorted(matches, key=lambda x: int(x[0]))
            descriptions = [match[1].strip() for match in matches_sorted]
        
        # Strategy 2: Split by double newlines
        if len(descriptions) != expected_count:
            descriptions = [
                d.strip() for d in response_text.split('\n\n')
                if d.strip() and not d.strip().startswith('IMAGE')
            ]
        
        # Strategy 3: Split by single newlines with filtering
        if len(descriptions) != expected_count:
            lines = response_text.split('\n')
            descriptions = []
            current_desc = []
            
            for line in lines:
                line = line.strip()
                
                if re.match(r'^IMAGE\s+\d+:', line, re.IGNORECASE):
                    if current_desc:
                        descriptions.append(' '.join(current_desc))
                        current_desc = []
                    line = re.sub(r'^IMAGE\s+\d+:\s*', '', line, flags=re.IGNORECASE)
                
                if line:
                    current_desc.append(line)
            
            if current_desc:
                descriptions.append(' '.join(current_desc))
        
        # Trim to expected count
        descriptions = descriptions[:expected_count]
        
        # Pad if too few
        while len(descriptions) < expected_count:
            descriptions.append("[Description unavailable]")
        
        logger.debug(f"Parsed {len(descriptions)} descriptions from batch response")
        
        return descriptions
    
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
        """Generate fallback description when vision API is unavailable"""
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