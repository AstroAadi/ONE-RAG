"""
LLM Service - Using LangChain with HuggingFace
Works with HuggingFace Router API for reliable model access
"""

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import logging
from typing import Optional, Tuple
import re

from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import desc

from app.core.config import settings
from app.db.models import ConversationHistory

logger = logging.getLogger(__name__)


class LLMGenerationError(Exception):
    """
    Raised when the LLM could not produce an answer.

    Carries the provider's HTTP status and message, because the exception text
    from the client library buries them behind a request id.
    """

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        provider_message: Optional[str] = None
    ):
        super().__init__(message)
        self.status_code = status_code
        self.provider_message = provider_message


def _describe_provider_error(error: Exception) -> Tuple[Optional[int], str]:
    """
    Pull the HTTP status and the provider's own message out of a client error.

    huggingface_hub raises HfHubHTTPError whose str() is mostly a request id,
    while the useful detail sits in the attached response body.
    """
    response = getattr(error, "response", None)
    status_code = getattr(response, "status_code", None)

    provider_message = ""

    if response is not None:
        try:
            payload = response.json()
            detail = payload.get("error", payload)
            if isinstance(detail, dict):
                provider_message = str(detail.get("message", "")) or str(detail)
            else:
                provider_message = str(detail)
        except Exception:
            provider_message = (getattr(response, "text", "") or "")[:500]

    if not provider_message:
        provider_message = str(error)

    return (int(status_code) if status_code else None), provider_message.strip()


class LLMService:
    """
    LLM service using LangChain with HuggingFace
    
    Supports seamless model switching via LangChain
    """
    
    def __init__(self):
        self.model_id = settings.llm_model_id
        self.api_token = settings.huggingface_api_token
        self.llm = None
        self._initialize_llm()
        
        logger.info(f"✅ LLM Service initialized: {self.model_id}")
    
    def _initialize_llm(self):
        """Initialize LangChain HuggingFace LLM"""
        try:
            # Use HuggingFaceEndpoint with LangChain
            endpoint = HuggingFaceEndpoint(
                repo_id=self.model_id,
                huggingfacehub_api_token=self.api_token,
                temperature=settings.temperature,
                max_new_tokens=settings.max_tokens,
                timeout=120
            )
            
            # Wrap in ChatHuggingFace for chat-like interface
            self.llm = ChatHuggingFace(llm=endpoint)
            
            logger.info(f"✅ LangChain LLM initialized: {self.model_id}")
            
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise RuntimeError(f"LLM initialization failed: {str(e)}")
    
    def get_conversation_history(
        self,
        conversation_id: str,
        user_id: str,
        db: Session,
        max_turns: int = 4
    ) -> str:
        """Get conversation history from database"""
        try:
            messages = db.query(ConversationHistory).filter(
                ConversationHistory.conversation_id == conversation_id,
                ConversationHistory.user_id == user_id
            ).order_by(
                desc(ConversationHistory.created_at)
            ).limit(max_turns * 2).all()
            
            if not messages:
                return ""
            
            messages.reverse()
            
            formatted = "Previous conversation:\n"
            for i in range(0, len(messages), 2):
                if i + 1 < len(messages):
                    question_msg = messages[i]
                    answer_msg = messages[i + 1]
                    
                    formatted += f"\nQ: {question_msg.content}\n"
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
        """Save conversation turn with FIFO deletion"""
        try:
            message_count = db.query(ConversationHistory).filter(
                ConversationHistory.conversation_id == conversation_id,
                ConversationHistory.user_id == user_id
            ).count()
            
            if message_count >= (max_turns * 2):
                oldest_messages = db.query(ConversationHistory).filter(
                    ConversationHistory.conversation_id == conversation_id,
                    ConversationHistory.user_id == user_id
                ).order_by(
                    ConversationHistory.created_at.asc()
                ).limit(2).all()
                
                for msg in oldest_messages:
                    db.delete(msg)
            
            question_msg = ConversationHistory(
                conversation_id=conversation_id,
                user_id=user_id,
                role="user",
                content=question,
                created_at=datetime.utcnow()
            )
            db.add(question_msg)
            
            answer_msg = ConversationHistory(
                conversation_id=conversation_id,
                user_id=user_id,
                role="assistant",
                content=answer,
                created_at=datetime.utcnow()
            )
            db.add(answer_msg)
            
            db.commit()
            
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
        """Generate answer using LangChain"""
        try:
            history = ""
            if conversation_id:
                history = self.get_conversation_history(
                    conversation_id, user_id, db
                )
            
            answer = self._generate_with_langchain(
                question, context, history, temperature
            )
            
            if include_citations:
                answer = self._verify_citations(answer, context)
            
            if conversation_id:
                self.save_conversation_turn(
                    conversation_id, user_id, question, answer, db
                )
            
            return answer
            
        except LLMGenerationError:
            raise
        except Exception as e:
            # Returning the error text as the answer would surface a failure as
            # a normal reply, and persist it into conversation history and the
            # query log as if the model had said it. Propagate instead so the
            # caller can respond with a real error.
            status_code, provider_message = _describe_provider_error(e)

            logger.error(
                f"Answer generation failed (model={self.model_id}, "
                f"status={status_code}): {provider_message}",
                exc_info=True
            )

            raise LLMGenerationError(
                provider_message or str(e),
                status_code=status_code,
                provider_message=provider_message
            ) from e
    
    def _generate_with_langchain(
        self,
        question: str,
        context: str,
        history: str = "",
        temperature: Optional[float] = None
    ) -> str:
        """Generate answer using LangChain pipeline"""
        try:
            # Update temperature if provided
            if temperature is not None and temperature != settings.temperature:
                # Reinitialize with new temperature
                endpoint = HuggingFaceEndpoint(
                    repo_id=self.model_id,
                    huggingfacehub_api_token=self.api_token,
                    temperature=temperature,
                    max_new_tokens=settings.max_tokens,
                    timeout=120
                )
                llm = ChatHuggingFace(llm=endpoint)
            else:
                llm = self.llm
            
            # Compress context if needed
            compressed_context = self._compress_context(context, question)
            
            # Build prompt template
            template_parts = [
                "You are a helpful AI assistant answering questions based on provided documents."
            ]
            
            if history:
                template_parts.append(f"\n{history}")
            
            template_parts.extend([
                "\nContext from documents:",
                "{context}",
                "\nQuestion: {question}",
                "\nInstructions:",
                "- Answer using ONLY information from the context",
                "- If answer not found, say 'I don't have enough information'",
                "- Be concise and accurate",
                "- Reference sources when relevant (e.g., [Source 1])",
                "- Use proper markdown formatting",
                "\nAnswer:"
            ])
            
            prompt = ChatPromptTemplate.from_template("\n".join(template_parts))
            
            # Create chain
            chain = prompt | llm | StrOutputParser()
            
            # Invoke chain
            answer = chain.invoke({
                "context": compressed_context,
                "question": question
            })
            
            return answer.strip()
            
        except Exception as e:
            logger.error(f"LangChain generation failed: {e}")
            raise
    
    def _compress_context(self, context: str, question: str, max_length: int = 4000) -> str:
        """Compress context to fit API limits"""
        if len(context) <= max_length:
            return context
        
        sources = context.split('[Source')
        if len(sources) <= 1:
            return context[:max_length]
        
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
        
        ranked_sources.sort(key=lambda x: x['relevance'], reverse=True)
        
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
            claim_pattern = r'[^.!?]*\d+[^.!?]*[.!?]'
            claims = re.findall(claim_pattern, answer)
            
            if not claims:
                return False, 1.0
            
            supported_claims = 0
            context_lower = context.lower()
            
            for claim in claims:
                claim_terms = set(re.findall(r'\b\w{4,}\b', claim.lower()))
                matches = sum(1 for term in claim_terms if term in context_lower)
                
                if matches >= len(claim_terms) * 0.5:
                    supported_claims += 1
            
            support_ratio = supported_claims / len(claims) if claims else 1.0
            is_hallucination = support_ratio < 0.6
            
            return is_hallucination, support_ratio
            
        except Exception as e:
            logger.error(f"Error detecting hallucination: {str(e)}")
            return False, 1.0
    
    def clear_conversation(self, conversation_id: str, user_id: str, db: Session):
        """Clear conversation history"""
        try:
            db.query(ConversationHistory).filter(
                ConversationHistory.conversation_id == conversation_id,
                ConversationHistory.user_id == user_id
            ).delete()
            db.commit()
        except Exception as e:
            logger.error(f"Error clearing conversation: {e}")
            db.rollback()
    
    def update_model(self, new_model_id: str) -> bool:
        """Update model ID and reinitialize"""
        try:
            if new_model_id not in settings.available_llm_models:
                logger.error(f"Invalid model ID: {new_model_id}")
                return False
            
            self.model_id = new_model_id
            self._initialize_llm()
            
            logger.info(f"✅ Model updated to: {new_model_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating model: {str(e)}")
            return False
    
    def get_current_model(self) -> str:
        """Get current model ID"""
        return self.model_id