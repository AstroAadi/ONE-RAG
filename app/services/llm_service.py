"""
Enhanced LLM Service with Vision Support
Detects image references in context and includes them in LLM calls
"""

from langchain_aws import ChatBedrock
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from app.core.config import settings
import logging
from typing import List, Dict, Optional, Tuple
import re
import json
import base64
import os
import boto3
from threading import Lock
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class ConversationHistory:
    """Manage conversation history with expiration"""
    
    def __init__(self, max_turns: int = 5, ttl_minutes: int = 5):
        self.max_turns = max_turns
        self.ttl_minutes = ttl_minutes
        self.history: List[Dict[str, str]] = []
        self.last_accessed = datetime.now()
    
    def add_turn(self, question: str, answer: str, context_used: str):
        # Store only summary of context, not full text
        context_summary = context_used[:500] + "..." if len(context_used) > 500 else context_used
        
        self.history.append({
            'question': question,
            'answer': answer,
            'context_summary': context_summary
        })
        if len(self.history) > self.max_turns:
            self.history = self.history[-self.max_turns:]
        
        self.last_accessed = datetime.now()
    
    def get_formatted_history(self) -> str:
        self.last_accessed = datetime.now()
        
        if not self.history:
            return ""
        formatted = "Previous conversation:\n"
        for i, turn in enumerate(self.history, 1):
            formatted += f"\nQ{i}: {turn['question']}\nA{i}: {turn['answer'][:200]}...\n"
        return formatted
    
    def is_expired(self) -> bool:
        """Check if conversation has expired"""
        return datetime.now() - self.last_accessed > timedelta(minutes=self.ttl_minutes)
    
    def clear(self):
        self.history = []
        self.last_accessed = datetime.now()


class LLMService:
    """
    LLM service that detects and includes images from context
    """
    
    def __init__(self):
        self.aws_config = {
            "region_name": settings.aws_region,
        }
        
        if settings.aws_access_key_id and settings.aws_secret_access_key:
            self.aws_config["aws_access_key_id"] = settings.aws_access_key_id
            self.aws_config["aws_secret_access_key"] = settings.aws_secret_access_key
        
        self.model_id = settings.bedrock_model_id
        self._llm = None
        self.conversations: Dict[str, ConversationHistory] = {}
        self._conversation_lock = Lock()  # FIXED: Thread-safe conversation access
        
        # Initialize Bedrock runtime for vision
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
                    region_name=settings.aws_region
                )
        except Exception as e:
            logger.warning(f"Could not initialize Bedrock runtime: {e}")
            self.bedrock_runtime = None
        
        logger.info(f"Enhanced LLM Service initialized with model: {self.model_id}")
    
    def get_llm(self) -> ChatBedrock:
        """Get or create LangChain ChatBedrock instance"""
        if self._llm is None:
            self._llm = ChatBedrock(
                model_id=self.model_id,
                model_kwargs={
                    "temperature": settings.temperature,
                    "max_tokens": settings.max_tokens
                },
                **self.aws_config
            )
            logger.info("Created new ChatBedrock instance")
        return self._llm
    
    def _cleanup_expired_conversations(self):
        """Remove expired conversations to prevent memory leak"""
        with self._conversation_lock:
            expired_ids = [
                conv_id for conv_id, conv in self.conversations.items()
                if conv.is_expired()
            ]
            for conv_id in expired_ids:
                del self.conversations[conv_id]
                logger.info(f"Cleaned up expired conversation: {conv_id}")
    
    def generate_answer(
        self,
        question: str,
        context: str,
        conversation_id: Optional[str] = None,
        include_citations: bool = True,
        temperature: Optional[float] = None
    ) -> str:
        """
        Generate answer with automatic image detection and inclusion
        FIXED: Thread-safe conversation handling with automatic cleanup
        
        Args:
            temperature: Optional temperature override (0.0-1.0)
        """
        try:
            # Cleanup expired conversations before processing
            self._cleanup_expired_conversations()
            
            # Extract image references from context
            image_refs = self._extract_image_references(context)
            
            # Get conversation history if available (thread-safe)
            history_text = ""
            if conversation_id:
                with self._conversation_lock:
                    if conversation_id in self.conversations:
                        history_text = self.conversations[conversation_id].get_formatted_history()
            
            # If images found, use vision-enabled generation
            if image_refs and self.bedrock_runtime:
                logger.info(f"Found {len(image_refs)} image(s) in context, using vision-enabled generation")
                answer = self._generate_with_vision(
                    question=question,
                    context=context,
                    image_refs=image_refs,
                    history=history_text,
                    temperature=temperature
                )
            else:
                # Standard text-only generation
                answer = self._generate_text_only(
                    question=question,
                    context=context,
                    history=history_text,
                    temperature=temperature
                )
            
            # Verify citations
            if include_citations:
                answer = self._verify_citations(answer, context)
            
            # Update conversation history (thread-safe)
            if conversation_id:
                with self._conversation_lock:
                    if conversation_id not in self.conversations:
                        self.conversations[conversation_id] = ConversationHistory()
                    self.conversations[conversation_id].add_turn(question, answer, context)
            
            return answer
            
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            raise
    
    def _extract_image_references(self, context: str) -> List[Dict[str, str]]:
        """
        Extract image references from context
        
        Format in markdown:
        IMAGE_PATH: path/to/image.png
        PAGE: 5
        DESCRIPTION: ...
        """
        image_refs = []
        
        # Find all image path references
        pattern = r'IMAGE_PATH:\s*([^\n]+)'
        matches = re.finditer(pattern, context)
        
        for match in matches:
            image_path = match.group(1).strip()
            
            # Get surrounding context (description, page, etc.)
            start_pos = max(0, match.start() - 500)
            end_pos = min(len(context), match.end() + 1000)
            surrounding_text = context[start_pos:end_pos]
            
            # Extract page number
            page_match = re.search(r'PAGE:\s*(\d+)', surrounding_text)
            page = page_match.group(1) if page_match else "unknown"
            
            # Extract description
            desc_match = re.search(r'DESCRIPTION:\s*\n(.+?)(?:\n---|$)', surrounding_text, re.DOTALL)
            description = desc_match.group(1).strip() if desc_match else ""
            
            # Resolve path (relative to uploads directory)
            full_path = os.path.join(settings.upload_dir, image_path)
            
            if os.path.exists(full_path):
                image_refs.append({
                    'path': full_path,
                    'relative_path': image_path,
                    'page': page,
                    'description': description
                })
                logger.info(f"Found image reference: {image_path} (page {page})")
            else:
                logger.warning(f"Image not found: {full_path}")
        
        return image_refs
    
    def _generate_with_vision(
        self,
        question: str,
        context: str,
        image_refs: List[Dict[str, str]],
        history: str = "",
        temperature: Optional[float] = None
    ) -> str:
        """
        Generate answer using Bedrock with vision (direct API call)
        Includes images in the request
        FIXED: Added context compression to prevent overflow
        """
        try:
            # Use provided temperature or default
            temp = temperature if temperature is not None else settings.temperature
            
            # FIXED: Compress context before building request to prevent overflow
            compressed_context = self._compress_context(context, question, max_length=6000)
            
            # Build content array with text and images
            content_parts = []
            
            # Add history if available
            if history:
                content_parts.append({
                    "type": "text",
                    "text": f"{history}\n\n"
                })
            
            # Add main context (remove image blocks to avoid duplication)
            clean_context = re.sub(
                r'---\n\*\*\[.*?\]\*\*.*?IMAGE_PATH:.*?---\n',
                '',
                compressed_context,  # Use compressed context
                flags=re.DOTALL
            )
            
            content_parts.append({
                "type": "text",
                "text": f"Context from documents:\n{clean_context}"
            })
            
            # Add images
            for img_ref in image_refs[:5]:  # Limit to 5 images
                try:
                    with open(img_ref['path'], 'rb') as f:
                        image_data = base64.standard_b64encode(f.read()).decode('utf-8')
                    
                    # Determine media type
                    ext = os.path.splitext(img_ref['path'])[1].lower()
                    media_types = {
                        '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
                        '.png': 'image/png', '.gif': 'image/gif', '.webp': 'image/webp'
                    }
                    media_type = media_types.get(ext, 'image/jpeg')
                    
                    # Add image with its description
                    content_parts.append({
                        "type": "text",
                        "text": f"\n[Image from page {img_ref['page']}]\n{img_ref['description']}\n"
                    })
                    
                    content_parts.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_data
                        }
                    })
                except Exception as e:
                    logger.error(f"Error loading image {img_ref['path']}: {e}")
            
            # Add question
            content_parts.append({
                "type": "text",
                "text": f"\nQuestion: {question}\n\nInstructions:\n" +
                        "- Answer using the text context AND the images provided\n" +
                        "- Reference specific images when relevant (e.g., 'as shown in the image from page 5')\n" +
                        "- For tables/charts in images, describe the data you see\n" +
                        "- Be specific about numbers, trends, and visual details\n" +
                        "- If answer not in context, say so clearly\n\nAnswer:"
            })
            
            # Call Bedrock directly
            request_body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": settings.max_tokens,
                "temperature": temp,  # Use provided or default temperature
                "messages": [{
                    "role": "user",
                    "content": content_parts
                }]
            }
            
            response = self.bedrock_runtime.invoke_model(
                modelId="us.anthropic.claude-3-5-sonnet-20241022-v2:0",
                body=json.dumps(request_body)
            )
            
            response_body = json.loads(response['body'].read())
            
            if 'content' in response_body and response_body['content']:
                return response_body['content'][0]['text']
            else:
                raise Exception("No content in vision response")
            
        except Exception as e:
            logger.error(f"Vision generation failed: {e}, falling back to text-only")
            return self._generate_text_only(question, context, history)
    
    def _generate_text_only(
        self,
        question: str,
        context: str,
        history: str = "",
        temperature: Optional[float] = None
    ) -> str:
        """Standard text-only generation using LangChain"""
        try:
            # Use provided temperature or default
            temp = temperature if temperature is not None else settings.temperature
            
            llm = ChatBedrock(
                model_id=self.model_id,
                model_kwargs={
                    "temperature": temp,
                    "max_tokens": settings.max_tokens
                },
                **self.aws_config
            )
            
            # Compress context if needed
            compressed_context = self._compress_context(context, question)
            
            # Build prompt
            prompt_template = """You are a helpful AI assistant answering questions based on provided documents.

"""
            if history:
                prompt_template += "{history}\n\n"
            
            prompt_template += """Context from documents:
{context}

Question: {question}

Instructions:
- Answer using ONLY information from the context
- If answer not found, say "I don't have enough information to answer this question"
- Be concise and accurate
- Reference sources when relevant (e.g., [Source 1])
- For visual content (tables, charts), use the descriptions provided
- Use proper markdown formatting (bullet points, **bold**, headings)

Answer:"""
            
            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question", "history"]
            )
            
            chain = prompt | llm | StrOutputParser()
            
            answer = chain.invoke({
                "context": compressed_context,
                "question": question,
                "history": history
            })
            
            return answer
            
        except Exception as e:
            logger.error(f"Text generation failed: {e}")
            raise
    
    def _compress_context(
        self,
        context: str,
        question: str,
        max_length: int = 8000 
    ) -> str:
        """Compress context by removing less relevant parts"""
        if len(context) <= max_length:
            return context
        
        # Split by sources
        sources = context.split('[Source')
        if len(sources) <= 1:
            return context[:max_length]
        
        # Rank sources by relevance to question
        question_lower = question.lower()
        question_keywords = set(re.findall(r'\w+', question_lower))
        
        ranked_sources = []
        for i, source in enumerate(sources):
            if i == 0 and not source.strip():
                continue
            
            source_text = f"[Source{source}"
            source_lower = source.lower()
            relevance = sum(1 for kw in question_keywords if kw in source_lower)
            
            ranked_sources.append({
                'text': source_text,
                'relevance': relevance,
                'length': len(source_text)
            })
        
        # Sort by relevance
        ranked_sources.sort(key=lambda x: x['relevance'], reverse=True)
        
        # Build compressed context
        final_context = ""
        current_length = 0
        
        for source in ranked_sources:
            if current_length + source['length'] <= max_length:
                final_context += source['text']
                current_length += source['length']
            else:
                break
        
        return final_context if final_context else context[:max_length]
    
    def _verify_citations(self, answer: str, context: str) -> str:
        """Verify citations exist in context"""
        citation_pattern = r'\[Source\s+(\d+)\]'
        cited_sources = set(re.findall(citation_pattern, answer))
        context_sources = set(re.findall(r'\[Source\s+(\d+)\]', context))
        
        invalid_sources = cited_sources - context_sources
        
        if invalid_sources:
            logger.warning(f"Found invalid citations: {invalid_sources}")
            for source_num in invalid_sources:
                answer = re.sub(
                    rf'\[Source\s+{source_num}\]',
                    '[Citation Unavailable]',
                    answer
                )
        
        return answer
    
    def detect_hallucination(self, answer: str, context: str) -> Tuple[bool, float]:
        """Simple hallucination detection"""
        try:
            # Extract claims with numbers
            claim_pattern = r'[^.!?]*\d+[^.!?]*[.!?]'
            claims = re.findall(claim_pattern, answer)
            
            if not claims:
                return False, 1.0
            
            # Check support in context
            supported_claims = 0
            context_lower = context.lower()
            
            for claim in claims:
                claim_terms = set(re.findall(r'\b\w{4,}\b', claim.lower()))
                matches = sum(1 for term in claim_terms if term in context_lower)
                
                if matches >= len(claim_terms) * 0.5:
                    supported_claims += 1
            
            support_ratio = supported_claims / len(claims) if claims else 1.0
            is_hallucination = support_ratio < 0.6
            
            if is_hallucination:
                logger.warning(f"Potential hallucination detected. Support ratio: {support_ratio}")
            
            return is_hallucination, support_ratio
            
        except Exception as e:
            logger.error(f"Error detecting hallucination: {str(e)}")
            return False, 1.0
    
    def clear_conversation(self, conversation_id: str):
        """Clear conversation history (thread-safe)"""
        with self._conversation_lock:
            if conversation_id in self.conversations:
                self.conversations[conversation_id].clear()
                logger.info(f"Cleared conversation {conversation_id}")
    
    def update_model(self, new_model_id: str) -> bool:
        """Update model ID"""
        try:
            self.model_id = new_model_id
            self._llm = None
            logger.info(f"Model updated to: {new_model_id}")
            return True
        except Exception as e:
            logger.error(f"Error updating model: {str(e)}")
            return False
    
    def get_current_model(self) -> str:
        """Get current model ID"""
        return self.model_id