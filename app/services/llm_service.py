"""
LLM Service - Text-Only Version (No Vision)
All image understanding happens at ingestion time via ImageDescriptionService
This service handles only text-based RAG queries
"""

from langchain_aws import ChatBedrock
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from app.core.config import settings
from app.db.models import ConversationHistory
from sqlalchemy.orm import Session
from sqlalchemy import desc
import logging
from typing import Optional, Tuple
import re
from datetime import datetime

logger = logging.getLogger(__name__)


class LLMService:
    """
    Text-only LLM service with database-backed conversation history
    
    Key Changes:
    - Removed all vision-related code
    - No image path detection or processing
    - Pure text-based RAG
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
        
        logger.info(f"LLM Service initialized (text-only) with model: {self.model_id}")
    
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
        Get conversation history from database (last N turns)
        
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
        Save conversation turn to database with FIFO deletion
        
        Args:
            conversation_id: Conversation identifier
            user_id: User identifier
            question: User's question
            answer: Assistant's answer
            db: Database session
            max_turns: Maximum Q&A pairs to keep (default: 4)
        """
        try:
            # Count existing messages
            message_count = db.query(ConversationHistory).filter(
                ConversationHistory.conversation_id == conversation_id,
                ConversationHistory.user_id == user_id
            ).count()
            
            # If at limit, delete oldest pair
            if message_count >= (max_turns * 2):
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
            
            logger.debug(f"Saved conversation turn for user {user_id}")
            
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
        Generate answer using text-only LLM (NO VISION)
        
        Args:
            question: User's question
            context: Retrieved document context (text only)
            user_id: User identifier
            db: Database session
            conversation_id: Optional conversation ID
            include_citations: Whether to verify citations
            temperature: Optional temperature override
        
        Returns:
            Generated answer
        """
        try:
            # Get conversation history if conversation_id provided
            history = ""
            if conversation_id:
                history = self.get_conversation_history(
                    conversation_id, user_id, db
                )
            
            # Generate answer (text-only)
            answer = self._generate_text_only(
                question, context, history, temperature
            )
            
            # Verify citations if requested
            if include_citations:
                answer = self._verify_citations(answer, context)
            
            # Save to conversation history if conversation_id provided
            if conversation_id:
                self.save_conversation_turn(
                    conversation_id, user_id, question, answer, db
                )
            
            return answer
            
        except Exception as e:
            logger.error(f"Answer generation error: {e}")
            import traceback
            traceback.print_exc()
            return f"I apologize, but I encountered an error: {str(e)}"
    
    def _generate_text_only(
        self,
        question: str,
        context: str,
        history: str = "",
        temperature: Optional[float] = None
    ) -> str:
        """
        Standard text-only generation
        
        This is now the ONLY generation method - no vision support
        """
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
- The context may include image descriptions marked as "[Image Description]" - use these descriptions as factual information
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
        """Clear conversation history from database"""
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