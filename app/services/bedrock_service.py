import boto3
import json
from typing import List, Dict
from app.core.config import settings


class BedrockService:
    """Service for interacting with AWS Bedrock models"""
    
    def __init__(self):
        """Initialize Bedrock client"""
        # Try to use credentials from settings first, fallback to default AWS credentials
        client_kwargs = {
            'service_name': 'bedrock-runtime',
            'region_name': settings.aws_region
        }
        
        # Only add credentials if they're provided in settings
        if settings.aws_access_key_id and settings.aws_secret_access_key:
            client_kwargs['aws_access_key_id'] = settings.aws_access_key_id
            client_kwargs['aws_secret_access_key'] = settings.aws_secret_access_key
        
        self.bedrock_runtime = boto3.client(**client_kwargs)
        self.llm_model_id = settings.bedrock_llm_model_id
        self.embedding_model_id = settings.bedrock_embedding_model_id
    
    def generate_embeddings_with_cohere(self, texts: List[str], input_type: str = "search_document") -> List[List[float]]:
        """
        Generate embeddings using Cohere Embed English v3
        
        Args:
            texts: List of texts to embed
            input_type: Type of input ("search_document" or "search_query")
            
        Returns:
            List of embedding vectors
        """
        try:
            embeddings = []
            
            # Cohere has a 2048 character limit per text
            max_chars = 2048
            
            # Truncate texts that exceed the limit
            truncated_texts = []
            for text in texts:
                if len(text) > max_chars:
                    # Truncate to max_chars - 100 to be safe
                    truncated = text[:max_chars - 100] + "..."
                    truncated_texts.append(truncated)
                    print(f"Warning: Truncated text from {len(text)} to {len(truncated)} characters")
                else:
                    truncated_texts.append(text)
            
            # Process in batches (Cohere supports batch processing)
            batch_size = 96  # Cohere's max batch size
            
            for i in range(0, len(truncated_texts), batch_size):
                batch_texts = truncated_texts[i:i + batch_size]
                
                request_body = {
                    "texts": batch_texts,
                    "input_type": input_type,
                    "truncate": "END"
                }
                
                response = self.bedrock_runtime.invoke_model(
                    modelId=self.embedding_model_id,
                    contentType="application/json",
                    accept="application/json",
                    body=json.dumps(request_body)
                )
                
                response_body = json.loads(response['body'].read())
                batch_embeddings = response_body.get('embeddings', [])
                embeddings.extend(batch_embeddings)
            
            return embeddings
            
        except Exception as e:
            print(f"Error generating embeddings with Cohere: {str(e)}")
            raise
    
    def generate_response_with_claude(
        self, 
        question: str, 
        context: str,
        max_tokens: int = 2000,
        temperature: float = 0.7
    ) -> str:
        """
        Generate response using Claude via Bedrock
        
        Args:
            question: User's question
            context: Retrieved context from knowledge base
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            
        Returns:
            Generated response text
        """
        try:
            # Construct the prompt
            prompt = f"""You are a helpful AI assistant. Answer the user's question based on the provided context.

Context:
{context}

Question: {question}

Instructions:
- Answer the question based solely on the information provided in the context
- If the answer cannot be found in the context, say "I don't have enough information to answer this question"
- Be concise and accurate
- Cite specific parts of the context when relevant
- Use proper markdown formatting to enhance readability, like use bullet points, paragraphs, tables, and headings where appropriate
Answer:"""

            request_body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            }
            
            response = self.bedrock_runtime.invoke_model(
                modelId=self.llm_model_id,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(request_body)
            )
            
            response_body = json.loads(response['body'].read())
            
            # Extract text from Claude's response
            answer = response_body['content'][0]['text']
            
            return answer
            
        except Exception as e:
            print(f"Error generating response with Claude: {str(e)}")
            raise
    
    def generate_single_embedding(self, text: str, input_type: str = "search_query") -> List[float]:
        """
        Generate embedding for a single text
        
        Args:
            text: Text to embed
            input_type: Type of input
            
        Returns:
            Embedding vector
        """
        embeddings = self.generate_embeddings_with_cohere([text], input_type)
        return embeddings[0] if embeddings else []