"""
FastEmbed-based embedding service for fast, local embeddings
"""
import logging
from typing import List, Optional, Dict, Any
from fastembed import TextEmbedding
import numpy as np

logger = logging.getLogger(__name__)

class FastEmbeddingService:
    """Fast local embedding service using FastEmbed"""
    
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        """
        Initialize FastEmbed embedding service
        
        Args:
            model_name: Model to use for embeddings. Options:
                - "BAAI/bge-small-en-v1.5" (default, fast and good quality)
                - "BAAI/bge-base-en-v1.5" (better quality, slower)
                - "sentence-transformers/all-MiniLM-L6-v2" (very fast)
                - "sentence-transformers/all-mpnet-base-v2" (high quality)
        """
        self.model_name = model_name
        self.embedding_model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the FastEmbed model"""
        try:
            logger.info(f"Initializing FastEmbed model: {self.model_name}")
            self.embedding_model = TextEmbedding(model_name=self.model_name)
            logger.info(f"Successfully initialized FastEmbed model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize FastEmbed model {self.model_name}: {str(e)}")
            # Fallback to a more reliable model
            try:
                fallback_model = "sentence-transformers/all-MiniLM-L6-v2"
                logger.info(f"Trying fallback model: {fallback_model}")
                self.embedding_model = TextEmbedding(model_name=fallback_model)
                self.model_name = fallback_model
                logger.info(f"Successfully initialized fallback model: {fallback_model}")
            except Exception as fallback_error:
                logger.error(f"Failed to initialize fallback model: {str(fallback_error)}")
                raise ValueError("Could not initialize any FastEmbed model")
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors (each vector is a list of floats)
        """
        if not self.embedding_model:
            raise ValueError("Embedding model not initialized")
        
        if not texts:
            return []
        
        try:
            logger.info(f"Generating embeddings for {len(texts)} texts using FastEmbed")
            
            # FastEmbed returns a generator of numpy arrays
            embeddings_generator = self.embedding_model.embed(texts)
            
            # Convert to list of lists
            embeddings = []
            for embedding in embeddings_generator:
                # Convert numpy array to list of floats
                if isinstance(embedding, np.ndarray):
                    embeddings.append(embedding.tolist())
                else:
                    embeddings.append(list(embedding))
            
            logger.info(f"Generated {len(embeddings)} embeddings successfully")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings with FastEmbed: {str(e)}")
            return []
    
    def generate_query_embedding(self, query: str) -> List[float]:
        """
        Generate embedding for a single query
        
        Args:
            query: Query string to embed
            
        Returns:
            Embedding vector as list of floats
        """
        if not query.strip():
            return []
        
        try:
            embeddings = self.generate_embeddings([query])
            if embeddings:
                return embeddings[0]
            return []
            
        except Exception as e:
            logger.error(f"Error generating query embedding: {str(e)}")
            return []
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current embedding model"""
        return {
            "model_name": self.model_name,
            "provider": "FastEmbed",
            "local": True,
            "fast": True,
            "dimension": self._get_embedding_dimension()
        }
    
    def _get_embedding_dimension(self) -> Optional[int]:
        """Get the dimension of embeddings from this model"""
        try:
            # Generate a test embedding to get dimension
            test_embedding = self.generate_query_embedding("test")
            return len(test_embedding) if test_embedding else None
        except Exception:
            return None
    
    def test_connection(self) -> bool:
        """Test if the embedding service is working"""
        try:
            test_embedding = self.generate_query_embedding("test connection")
            return len(test_embedding) > 0
        except Exception as e:
            logger.error(f"FastEmbed test connection failed: {str(e)}")
            return False

# Gemini fallback service for comparison/backup
class GeminiEmbeddingService:
    """Gemini AI embedding service (slower but high quality)"""
    
    def __init__(self, gemini_service):
        self.gemini_service = gemini_service
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using Gemini AI"""
        return self.gemini_service.generate_embeddings(texts)
    
    def generate_query_embedding(self, query: str) -> List[float]:
        """Generate query embedding using Gemini AI"""
        return self.gemini_service.generate_query_embedding(query)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get Gemini model information"""
        return {
            "model_name": "text-embedding-004",
            "provider": "Gemini AI",
            "local": False,
            "fast": False,
            "dimension": 768  # Gemini embedding dimension
        }
    
    def test_connection(self) -> bool:
        """Test Gemini embedding connection"""
        try:
            return len(self.generate_query_embedding("test")) > 0
        except Exception:
            return False

# Factory function to create the best available embedding service
def create_embedding_service(prefer_local: bool = True, gemini_service=None) -> Any:
    """
    Create the best available embedding service
    
    Args:
        prefer_local: If True, prefer FastEmbed over Gemini
        gemini_service: Gemini service instance for fallback
        
    Returns:
        Embedding service instance
    """
    if prefer_local:
        try:
            # Try FastEmbed first (faster)
            service = FastEmbeddingService()
            if service.test_connection():
                logger.info("Using FastEmbed for embeddings (fast, local)")
                return service
        except Exception as e:
            logger.warning(f"FastEmbed initialization failed: {str(e)}")
    
    # Fallback to Gemini if available
    if gemini_service:
        try:
            gemini_embedding_service = GeminiEmbeddingService(gemini_service)
            if gemini_embedding_service.test_connection():
                logger.info("Using Gemini AI for embeddings (high quality, slower)")
                return gemini_embedding_service
        except Exception as e:
            logger.warning(f"Gemini embedding service failed: {str(e)}")
    
    # If we get here, try FastEmbed again as last resort
    try:
        service = FastEmbeddingService()
        logger.info("Using FastEmbed as last resort")
        return service
    except Exception as e:
        logger.error(f"All embedding services failed: {str(e)}")
        raise ValueError("No embedding service available")
