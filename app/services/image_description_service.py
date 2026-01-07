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
            "**STEP 1: CAPTION DETECTION (Read very carefully)**",
            "",
            "Examine the image thoroughly for ANY text that looks like:",
            "- 'Figure X.Y.Z' or 'Fig. X.Y.Z' or 'Figure X.Y.Z:'",
            "- 'Table X.Y.Z' or 'Tbl. X.Y.Z'",
            "- ANY numbered label with dots/decimals (e.g., '1.3.1', '2.5.3', 'Fig 1.4.2')",
            "",
            "**CRITICAL CAPTION READING RULES:**",
            "- Read caption numbers EXACTLY as written, including ALL decimal points",
            "- '1.3.1' is NOT '1.31' - these are different numbers",
            "- 'Figure 1.3.1' has THREE parts: 1, 3, and 1 (not two parts: 1 and 31)",
            "- Preserve ALL dots, spaces, and formatting from the original",
            "- If unsure, look multiple times at the exact spacing and punctuation",
            "",
            "**STEP 2: APPLY CAPTION RULES**",
            "",
            "**Single Visual Element (one chart/diagram/table):**",
            "- IF caption exists:",
            "  ✅ Copy it EXACTLY as shown (character-by-character)",
            "  ✅ Format: **Figure [exact number]**: Description...",
            "  ✅ Double-check you copied ALL decimal points",
            "- IF no caption exists:",
            "  ❌ Do NOT invent any caption",
            "  ✅ Start directly with description",
            "",
            "**Multiple Visual Elements (2+ charts/diagrams/tables in same image):**",
            "- IF at least one element has a caption:",
            "  ✅ Use existing captions EXACTLY as shown (preserve all dots)",
            "  ✅ Generate captions ONLY for unlabeled elements:",
            "     • Follow the same numbering pattern",
            "     • Maintain the same decimal structure",
            "     • Example: If 'Figure 1.3.1' exists on left chart:",
            "       → Right chart becomes 'Figure 1.3.2' (not '1.32' or '1.4')",
            "       → Middle chart (if between) would be inserted appropriately",
            "     • Example: If 'Figure 2.5.3' exists on right:",
            "       → Left chart becomes 'Figure 2.5.2' (not '2.52')",
            "       → Chart to right of that becomes 'Figure 2.5.4'",
            "  ✅ Describe EACH element separately and completely",
            "- IF no elements have captions:",
            "  ❌ Do NOT generate any captions",
            "  ✅ Describe as 'Left chart:', 'Right chart:', etc.",
            "",
            "**NEVER:**",
            "- Change '1.3.1' to '1.31' or any other format",
            "- Omit decimal points from caption numbers",
            "- Invent captions when none exist",
            "- Skip describing any visual element",
            "- Assume caption numbers - read them precisely",
            "",
            "## DESCRIPTION COMPLETENESS REQUIREMENTS",
            "",
            "Your description will be used in a RAG (Retrieval-Augmented Generation) system.",
            "Therefore, it MUST be:",
            "",
            "**COMPREHENSIVE**: Include every detail visible in the image",
            "**SEARCHABLE**: Use terms someone might search for",
            "**SPECIFIC**: Include actual numbers, labels, and text",
            "**COMPLETE**: Cover all visual elements thoroughly",
            "",
            "## DESCRIPTION STRUCTURE",
            "",
            "**For EACH visual element, provide:**",
            "",
            "### 1. Chart/Diagram Type Identification",
            "- Exact type: horizontal bar chart, vertical bar chart, line graph, scatter plot, pie chart, stacked area chart, etc.",
            "- Be specific: 'multi-line temporal graph' not just 'line chart'",
            "",
            "### 2. Title and Labels (if present)",
            "- Chart title (word-for-word)",
            "- Axis labels (exact text)",
            "- Legend labels (all items, exact text)",
            "- Any annotations or notes",
            "",
            "### 3. Axis Information",
            "- X-axis: Label, units, range (min to max), scale type",
            "- Y-axis: Label, units, range (min to max), scale type",
            "- For time series: exact start year to end year",
            "",
            "### 4. Data Series (ALL of them)",
            "- Name/label of each series (exact text)",
            "- Color or pattern used",
            "- Data point values (as many as visible)",
            "- Starting value, ending value, peak value, lowest value",
            "- Any labeled points with their exact values",
            "",
            "### 5. Data Values (Comprehensive)",
            "- List ALL visible data points with their values",
            "- Include exact numbers from the chart",
            "- Note any data labels or annotations",
            "- Mention gridline values if they help establish scale",
            "",
            "### 6. Trends and Patterns",
            "- Overall trend direction (increasing, decreasing, stable, cyclical)",
            "- Significant changes (spikes, drops, inflection points)",
            "- Comparative relationships between series",
            "- Notable periods or ranges",
            "- Growth rates or changes (if calculable)",
            "",
            "### 7. Key Insights (3-7 insights minimum)",
            "- Most important takeaways",
            "- Comparisons between elements",
            "- Ranking of values (highest to lowest)",
            "- Significant differences or gaps",
            "- Notable patterns or anomalies",
            "",
            "### 8. Source Attribution",
            "- Copy source text exactly as shown",
            "- Include author, year, report name if visible",
            "",
            "## FORMAT FOR EACH VISUAL ELEMENT",
            "",
            "**[Exact Caption if present]**:",
            "",
            "Type: [Specific chart/diagram type]",
            "",
            "Title: [Exact title text if present]",
            "",
            "Axes:",
            "- X-axis: [Label] ([units]), range: [min] to [max]",
            "- Y-axis: [Label] ([units]), range: [min] to [max]",
            "",
            "Data Series:",
            "1. [Series name] ([color/pattern]):",
            "   - Values: [list all visible values chronologically or categorically]",
            "   - Range: [lowest] to [highest]",
            "   - Notable points: [specific values at key positions]",
            "",
            "2. [Next series name] ([color/pattern]):",
            "   - [Same detailed breakdown]",
            "",
            "Key Data Points:",
            "- [List all specific values visible, e.g., 'United States: 40 models']",
            "- [Include ALL countries/categories with their values]",
            "- [Note any labeled points on the chart]",
            "",
            "Trends:",
            "- [Trend 1 with specific details]",
            "- [Trend 2 with specific details]",
            "- [Continue with all observable trends]",
            "",
            "Insights:",
            "1. [Specific insight with numbers]",
            "2. [Comparative insight]",
            "3. [Pattern observation]",
            "4. [Additional insights...]",
            "",
            "Source: [Exact source text]",
            "",
            "## QUALITY CHECKLIST",
            "",
            "Before finalizing, verify:",
            "- ✅ ALL captions copied EXACTLY (every dot, every space)",
            "- ✅ ALL visual elements described (none skipped)",
            "- ✅ ALL visible numbers extracted and included",
            "- ✅ ALL data series identified with values",
            "- ✅ ALL text from image reproduced verbatim",
            "- ✅ At least 3-7 insights provided per chart",
            "- ✅ Description is comprehensive enough for RAG retrieval",
            "- ✅ Someone could reconstruct key information from description alone",
            "",
            "## CRITICAL REMINDERS",
            "",
            "1. **Caption Precision**: '1.3.1' ≠ '1.31' - Read decimal points carefully",
            "2. **Completeness**: RAG systems need ALL details - be thorough",
            "3. **Every Element**: Describe EVERY chart/diagram separately",
            "4. **All Numbers**: Extract ALL visible data values",
            "5. **Exact Text**: Copy ALL visible text word-for-word",
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