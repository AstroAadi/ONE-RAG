"""
Enhanced LLM Service with Vision Support and DATABASE-BACKED CONVERSATION HISTORY
- Stores conversation history in database (last 4 turns per user)
- FIFO deletion when exceeding limit
- Proper user isolation
"""

from langchain_aws import ChatBedrock
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from app.core.config import settings
from app.db.models import ConversationHistory
from sqlalchemy.orm import Session
from sqlalchemy import desc
import logging
from typing import List, Dict, Optional, Tuple
import re
import json
import base64
import os
import boto3
from datetime import datetime

logger = logging.getLogger(__name__)


class LLMService:
    """
    LLM service with DATABASE-BACKED conversation history
    
    Key changes:
    - Stores conversation in database, not in-memory
    - FIFO deletion: keeps only last 4 Q&A pairs per conversation
    - User-isolated conversations
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
    
    def get_conversation_history(
        self, 
        conversation_id: str, 
        user_id: str,
        db: Session,
        max_turns: int = 4
    ) -> str:
        """
        Get conversation history from DATABASE (last N turns)
        
        Args:
            conversation_id: Conversation identifier
            user_id: User identifier (for isolation)
            db: Database session
            max_turns: Maximum number of Q&A pairs to retrieve (default: 4)
            
        Returns:
            Formatted conversation history string
        """
        try:
            # Get last N*2 messages (N questions + N answers)
            messages = db.query(ConversationHistory).filter(
                ConversationHistory.conversation_id == conversation_id,
                ConversationHistory.user_id == user_id
            ).order_by(
                desc(ConversationHistory.created_at)
            ).limit(max_turns * 2).all()
            
            if not messages:
                return ""
            
            # Reverse to get chronological order
            messages.reverse()
            
            # Format history
            formatted = "Previous conversation:\n"
            for i in range(0, len(messages), 2):
                if i + 1 < len(messages):
                    question_msg = messages[i]
                    answer_msg = messages[i + 1]
                    
                    formatted += f"\nQ: {question_msg.content}\n"
                    # Truncate long answers
                    answer_text = answer_msg.content[:200] + "..." if len(answer_msg.content) > 200 else answer_msg.content
                    formatted += f"A: {answer_text}\n"
            
            return formatted
            
        except Exception as e:
            logger.error(f"Error retrieving conversation history: {e}")
            return ""
    
    def save_conversation_turn(
        self,
        conversation_id: str,
        user_id: str,
        question: str,
        answer: str,
        db: Session,
        max_turns: int = 4
    ):
        """
        Save conversation turn to DATABASE with FIFO deletion
        
        Args:
            conversation_id: Conversation identifier
            user_id: User identifier
            question: User's question
            answer: Assistant's answer
            db: Database session
            max_turns: Maximum Q&A pairs to keep (default: 4)
        """
        try:
            # Count existing messages for this conversation
            message_count = db.query(ConversationHistory).filter(
                ConversationHistory.conversation_id == conversation_id,
                ConversationHistory.user_id == user_id
            ).count()
            
            # If we already have max_turns * 2 messages (Q&A pairs), delete oldest pair
            if message_count >= (max_turns * 2):
                # Delete oldest 2 messages (1 Q&A pair)
                oldest_messages = db.query(ConversationHistory).filter(
                    ConversationHistory.conversation_id == conversation_id,
                    ConversationHistory.user_id == user_id
                ).order_by(
                    ConversationHistory.created_at.asc()
                ).limit(2).all()
                
                for msg in oldest_messages:
                    db.delete(msg)
                
                logger.debug(f"Deleted oldest Q&A pair from conversation {conversation_id}")
            
            # Save new question
            question_msg = ConversationHistory(
                conversation_id=conversation_id,
                user_id=user_id,
                role="user",
                content=question,
                created_at=datetime.utcnow()
            )
            db.add(question_msg)
            
            # Save new answer
            answer_msg = ConversationHistory(
                conversation_id=conversation_id,
                user_id=user_id,
                role="assistant",
                content=answer,
                created_at=datetime.utcnow()
            )
            db.add(answer_msg)
            
            db.commit()
            
            logger.debug(f"Saved conversation turn for user {user_id}, conversation {conversation_id}")
            
        except Exception as e:
            logger.error(f"Error saving conversation: {e}")
            db.rollback()
    
    def generate_answer(
        self,
        question: str,
        context: str,
        user_id: str,
        db: Session,
        conversation_id: Optional[str] = None,
        include_citations: bool = True,
        temperature: Optional[float] = None
    ) -> str:
        """
        Generate answer with DATABASE-BACKED conversation history
        
        Args:
            question: User's question
            context: Retrieved document context
            user_id: User identifier (for conversation isolation)
            db: Database session
            conversation_id: Optional conversation ID for multi-turn
            include_citations: Whether to verify citations
            temperature: Optional temperature override (0.0-1.0)
        
        Returns:
            Generated answer string
        """
        try:
            # Extract image references from context
            image_refs = self._extract_image_references(context)
            
            # Get conversation history from DATABASE if available
            history_text = ""
            if conversation_id:
                history_text = self.get_conversation_history(
                    conversation_id, 
                    user_id, 
                    db,
                    max_turns=4  # Last 4 Q&A pairs
                )
            
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
            
            # Save to DATABASE
            if conversation_id:
                self.save_conversation_turn(
                    conversation_id=conversation_id,
                    user_id=user_id,
                    question=question,
                    answer=answer,
                    db=db,
                    max_turns=4
                )
            
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
            desc_match = re.search(r'DESCRIPTION:\s*([^\n]+)', surrounding_text)
            description = desc_match.group(1) if desc_match else ""
            
            image_refs.append({
                'path': image_path,
                'page': page,
                'description': description,
                'surrounding_context': surrounding_text[:500]
            })
        
        return image_refs
    
    def _load_image_as_base64(self, image_path: str) -> Optional[Tuple[str, str]]:
        """
        Load image file and convert to base64
        
        Returns:
            Tuple of (base64_data, media_type) or None if failed
        """
        try:
            # Handle relative paths
            if not os.path.isabs(image_path):
                image_path = os.path.join(settings.upload_dir, image_path)
            
            if not os.path.exists(image_path):
                logger.warning(f"Image not found: {image_path}")
                return None
            
            # Determine media type
            ext = os.path.splitext(image_path)[1].lower()
            media_type_map = {
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.png': 'image/png',
                '.gif': 'image/gif',
                '.webp': 'image/webp'
            }
            media_type = media_type_map.get(ext, 'image/jpeg')
            
            # Read and encode
            with open(image_path, 'rb') as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')
            
            return image_data, media_type
            
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            return None
    
    def _generate_with_vision(
        self,
        question: str,
        context: str,
        image_refs: List[Dict],
        history: str = "",
        temperature: Optional[float] = None
    ) -> str:
        """Generate answer with vision support using Bedrock API directly"""
        try:
            # Use provided temperature or default
            temp = temperature if temperature is not None else settings.vision_temperature
            
            # Build content parts (text + images)
            content_parts = []
            
            # Add text context
            context_summary = self._compress_context(context, question, max_length=6000)
            
            text_content = ""
            if history:
                text_content += f"{history}\n\n"
            
            text_content += f"Context from documents:\n{context_summary}\n\n"
            text_content += f"Question: {question}\n\n"
            
            # Add images
            for img_ref in image_refs[:5]:  # Limit to 5 images
                image_data = self._load_image_as_base64(img_ref['path'])
                if image_data:
                    base64_data, media_type = image_data
                    content_parts.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": base64_data
                        }
                    })
                    
                    # Add image context
                    text_content += f"\n[Image from page {img_ref['page']}]\n"
                    if img_ref['description']:
                        text_content += f"Description: {img_ref['description']}\n"
            
            # Add text as final content part
            content_parts.append({
                "type": "text",
                "text": text_content + 
                        "Instructions:\n" +
                        "- Answer using information from BOTH text context AND images\n" +
                        "- Reference specific images when relevant (e.g., 'As shown in the image...')\n" +
                        "- Be specific about numbers, trends, and visual details\n" +
                        "- If answer not in context, say so clearly\n\nAnswer:"
            })
            
            # Call Bedrock directly
            request_body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": settings.vision_max_tokens,
                "temperature": temp,
                "messages": [{
                    "role": "user",
                    "content": content_parts
                }]
            }
            
            response = self.bedrock_runtime.invoke_model(
                modelId=settings.bedrock_vision_model_id,  # Use config!
                body=json.dumps(request_body)
            )
            
            response_body = json.loads(response['body'].read())
            
            if 'content' in response_body and response_body['content']:
                return response_body['content'][0]['text']
            else:
                raise Exception("No content in vision response")
            
        except Exception as e:
            logger.error(f"Vision generation failed: {e}, falling back to text-only")
            return self._generate_text_only(question, context, history, temperature)
    
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
    
    def clear_conversation(self, conversation_id: str, user_id: str, db: Session):
        """Clear conversation history from DATABASE"""
        try:
            db.query(ConversationHistory).filter(
                ConversationHistory.conversation_id == conversation_id,
                ConversationHistory.user_id == user_id
            ).delete()
            db.commit()
            logger.info(f"Cleared conversation {conversation_id} for user {user_id}")
        except Exception as e:
            logger.error(f"Error clearing conversation: {e}")
            db.rollback()
    
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