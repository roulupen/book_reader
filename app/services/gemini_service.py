import google.generativeai as genai
from typing import List, Dict, Any, Optional
from app.config import settings
import logging

logger = logging.getLogger(__name__)

class GeminiService:
    def __init__(self):
        if not settings.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY environment variable is required")
        
        genai.configure(api_key=settings.GEMINI_API_KEY)
        
        # Try different model names in order of preference
        model_names = ['gemini-2.0-flash']
        self.model = None
        
        for model_name in model_names:
            try:
                self.model = genai.GenerativeModel(model_name)
                logger.info(f"Successfully initialized Gemini model: {model_name}")
                break
            except Exception as e:
                logger.warning(f"Failed to initialize model {model_name}: {str(e)}")
                continue
        
        if not self.model:
            raise ValueError("Failed to initialize any Gemini model. Please check your API key and model availability.")
    
    def test_connection(self) -> Dict[str, Any]:
        """Test the Gemini API connection and return status"""
        try:
            # Try a simple generation to test the connection
            response = self.model.generate_content("Hello")
            return {
                "status": "success",
                "message": "Gemini API connection successful",
                "model_name": self.model.model_name if hasattr(self.model, 'model_name') else "unknown"
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Gemini API connection failed: {str(e)}",
                "model_name": None
            }
    
    @staticmethod
    def list_available_models() -> List[str]:
        """List available Gemini models"""
        try:
            models = genai.list_models()
            model_names = []
            for model in models:
                if 'generateContent' in model.supported_generation_methods:
                    model_names.append(model.name)
            return model_names
        except Exception as e:
            logger.error(f"Error listing models: {str(e)}")
            return []
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for text chunks using Gemini"""
        try:
            embeddings = []
            
            for text in texts:
                # Use Gemini's embedding model
                result = genai.embed_content(
                    model="models/text-embedding-004",
                    content=text,
                    task_type="retrieval_document"
                )
                embeddings.append(result['embedding'])
                
            logger.info(f"Generated {len(embeddings)} embeddings using Gemini")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings with Gemini: {str(e)}")
            # Fallback to empty list (ChromaDB will use default embeddings)
            return []
    
    def generate_query_embedding(self, query: str) -> List[float]:
        """Generate embedding for a query using Gemini"""
        try:
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=query,
                task_type="retrieval_query"
            )
            logger.info("Generated query embedding using Gemini")
            return result['embedding']
            
        except Exception as e:
            logger.error(f"Error generating query embedding with Gemini: {str(e)}")
            return []
    
    def answer_question(self, question: str, context_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate an answer based on the question and context chunks"""
        try:
            # Combine context chunks
            context = "\n\n".join([chunk["content"] for chunk in context_chunks])
            
            # Create prompt
            prompt = f"""
Based on the following context from a book, please answer the question. 

Context:
{context}

Question: {question}

Instructions:
1. If you can find relevant information in the context, provide a detailed answer using that information.
2. If the context does not contain information relevant to the question, respond with: "I couldn't find information about this topic in the provided book content. The book may not cover this subject, or the relevant information might be in a different section."
3. Do NOT mention specific technical terms, clauses, or concepts that aren't in the question when saying information cannot be found.
4. Format your response using markdown for better readability:
   - Use **bold** for important points
   - Use *italics* for emphasis
   - Use bullet points or numbered lists when appropriate
   - Use code blocks for any technical content
   - Use headers (##, ###) to organize longer responses
   - Use > for quotes or important excerpts from the text

Please provide your answer based only on the information available in the context above.
"""
            
            # Generate response
            response = self.model.generate_content(prompt)
            
            return {
                "answer": response.text,
                "sources": [],  # No longer showing sources to user
                "confidence_score": self._calculate_confidence(context_chunks)
            }
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error generating answer: {error_msg}")
            
            # Provide more specific error messages
            if "404" in error_msg and "not found" in error_msg:
                answer = "I apologize, but the AI model is currently unavailable. This might be due to model updates or API changes. Please try again later."
            elif "quota" in error_msg.lower() or "limit" in error_msg.lower():
                answer = "I apologize, but the API quota has been exceeded. Please try again later or check your API usage."
            elif "authentication" in error_msg.lower() or "api key" in error_msg.lower():
                answer = "I apologize, but there's an authentication issue with the AI service. Please check your API key configuration."
            else:
                answer = f"I apologize, but I encountered an error while processing your question: {error_msg}"
            
            return {
                "answer": answer,
                "sources": [],
                "confidence_score": 0.0
            }
    
    def summarize_text(self, text: str, max_length: int = 500) -> str:
        """Summarize a text chunk"""
        try:
            prompt = f"""
Please provide a concise summary of the following text in no more than {max_length} characters:

{text}

Summary:
"""
            response = self.model.generate_content(prompt)
            return response.text
            
        except Exception as e:
            logger.error(f"Error summarizing text: {str(e)}")
            return text[:max_length] + "..." if len(text) > max_length else text
    
    def extract_key_topics(self, text: str) -> List[str]:
        """Extract key topics from text"""
        try:
            prompt = f"""
Please extract the main topics and themes from the following text. 
Return them as a simple comma-separated list of key terms and concepts.

Text:
{text}

Key topics (comma-separated):
"""
            response = self.model.generate_content(prompt)
            topics = [topic.strip() for topic in response.text.split(",")]
            return topics[:10]  # Limit to top 10 topics
            
        except Exception as e:
            logger.error(f"Error extracting topics: {str(e)}")
            return []
    
    def _calculate_confidence(self, context_chunks: List[Dict[str, Any]]) -> float:
        """Calculate confidence score based on context relevance"""
        if not context_chunks:
            return 0.0
        
        # Simple confidence calculation based on number of chunks and their distances
        total_distance = sum(chunk.get("distance", 1.0) for chunk in context_chunks)
        avg_distance = total_distance / len(context_chunks)
        
        # Convert distance to confidence (lower distance = higher confidence)
        confidence = max(0.0, min(1.0, 1.0 - avg_distance))
        return round(confidence, 2)
