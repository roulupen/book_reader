"""
Ultra-lightweight embedding service using ChromaDB's built-in embedding functions
No heavy dependencies required - ChromaDB handles everything!
"""
import logging
from typing import List, Optional, Dict, Any
from chromadb.utils import embedding_functions
from app.config import settings

logger = logging.getLogger(__name__)

class ChromaEmbeddingService:
    """Ultra-lightweight embedding service using ChromaDB's built-in functions"""
    
    def __init__(self, embedding_function_name: str = "default"):
        """
        Initialize ChromaDB embedding service
        
        Args:
            embedding_function_name: Type of embedding function to use
                - "default": ChromaDB's built-in default (fastest, no dependencies)
                - "sentence_transformer": Minimal sentence transformer (small model)
        """
        self.embedding_function_name = embedding_function_name
        self.embedding_function = None
        self._initialize_embedding_function()
    
    def _initialize_embedding_function(self):
        """Initialize the ChromaDB embedding function"""
        try:
            if self.embedding_function_name == "default":
                logger.info("Initializing ChromaDB DefaultEmbeddingFunction (ultra-lightweight)")
                self.embedding_function = embedding_functions.DefaultEmbeddingFunction()
                logger.info("✅ Successfully initialized ChromaDB DefaultEmbeddingFunction")
                
            elif self.embedding_function_name == "sentence_transformer":
                logger.info("Initializing ChromaDB SentenceTransformerEmbeddingFunction with minimal model")
                # Use the smallest, fastest model
                model_name = getattr(settings, 'CHROMA_EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
                self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name=model_name
                )
                logger.info(f"✅ Successfully initialized SentenceTransformerEmbeddingFunction: {model_name}")
                
            else:
                # Fallback to default
                logger.warning(f"Unknown embedding function: {self.embedding_function_name}, using default")
                self.embedding_function = embedding_functions.DefaultEmbeddingFunction()
                self.embedding_function_name = "default"
                
        except Exception as e:
            logger.error(f"Failed to initialize embedding function {self.embedding_function_name}: {str(e)}")
            # Ultimate fallback to default
            try:
                logger.info("Falling back to ChromaDB DefaultEmbeddingFunction")
                self.embedding_function = embedding_functions.DefaultEmbeddingFunction()
                self.embedding_function_name = "default"
                logger.info("✅ Fallback to DefaultEmbeddingFunction successful")
            except Exception as fallback_error:
                logger.error(f"Even fallback failed: {str(fallback_error)}")
                raise ValueError("Could not initialize any ChromaDB embedding function")
    
    def generate_embeddings(self, texts: List[str], progress_callback=None) -> List[List[float]]:
        """
        Generate embeddings using ChromaDB's embedding functions (very fast and lightweight)
        
        Args:
            texts: List of text strings to embed
            progress_callback: Optional callback function(current, total, message)
            
        Returns:
            List of embedding vectors
        """
        if not self.embedding_function:
            raise ValueError("ChromaDB embedding function not initialized")
        
        if not texts:
            return []
        
        try:
            total_texts = len(texts)
            batch_size = getattr(settings, 'EMBEDDING_BATCH_SIZE', 100)
            
            logger.info(f"Generating embeddings for {total_texts} texts using ChromaDB {self.embedding_function_name} function")
            
            if progress_callback:
                progress_callback(0, total_texts, f"Starting ChromaDB {self.embedding_function_name} embedding generation...")
            
            embeddings = []
            
            # Process in batches for progress reporting and memory efficiency
            for i in range(0, total_texts, batch_size):
                batch_texts = texts[i:i + batch_size]
                current_batch_end = min(i + batch_size, total_texts)
                
                # Generate embeddings for this batch (very fast with ChromaDB)
                batch_embeddings = self.embedding_function(batch_texts)
                
                # Convert to proper Python lists (avoid numpy array issues)
                for embedding in batch_embeddings:
                    if hasattr(embedding, 'tolist'):
                        embeddings.append(embedding.tolist())
                    elif isinstance(embedding, (list, tuple)):
                        embeddings.append(list(embedding))
                    else:
                        embeddings.append(embedding)
                
                # Report progress
                if progress_callback:
                    progress_callback(
                        current_batch_end, 
                        total_texts, 
                        f"Generated embeddings {current_batch_end}/{total_texts} (ChromaDB batch {i//batch_size + 1})"
                    )
            
            if progress_callback:
                progress_callback(total_texts, total_texts, f"ChromaDB {self.embedding_function_name} embedding generation completed!")
            
            logger.info(f"Generated {len(embeddings)} embeddings successfully with ChromaDB {self.embedding_function_name}")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings with ChromaDB {self.embedding_function_name}: {str(e)}")
            if progress_callback:
                progress_callback(0, len(texts), f"Error: {str(e)}")
            return []
    
    def generate_query_embedding(self, query: str) -> List[float]:
        """Generate embedding for a single query"""
        if not query.strip():
            return []
        
        try:
            embeddings = self.generate_embeddings([query])
            if embeddings and len(embeddings) > 0:
                # Ensure we return a proper Python list, not numpy array
                embedding = embeddings[0]
                if hasattr(embedding, 'tolist'):
                    return embedding.tolist()
                elif isinstance(embedding, (list, tuple)):
                    return list(embedding)
                else:
                    return embedding
            return []
        except Exception as e:
            logger.error(f"Error generating query embedding: {str(e)}")
            return []
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current embedding model"""
        return {
            "model_name": self.embedding_function_name,
            "provider": "ChromaDB",
            "local": True,
            "fast": True,
            "lightweight": True,
            "dimension": self._get_embedding_dimension()
        }
    
    def _get_embedding_dimension(self) -> Optional[int]:
        """Get the dimension of embeddings from this model"""
        try:
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
            logger.error(f"ChromaDB embedding test connection failed: {str(e)}")
            return False

class GeminiEmbeddingService:
    """Gemini AI embedding service wrapper for fallback"""
    
    def __init__(self, gemini_service):
        """Initialize with a GeminiService instance"""
        self.gemini_service = gemini_service
    
    def generate_embeddings(self, texts: List[str], progress_callback=None) -> List[List[float]]:
        """Generate embeddings using Gemini AI"""
        if progress_callback:
            progress_callback(0, len(texts), "Starting Gemini embedding generation...")
        
        embeddings = self.gemini_service.generate_embeddings(texts)
        
        if progress_callback:
            if embeddings:
                progress_callback(len(texts), len(texts), "Gemini embedding generation completed!")
            else:
                progress_callback(0, len(texts), "Gemini embedding generation failed")
        
        return embeddings
    
    def generate_query_embedding(self, query: str) -> List[float]:
        """Generate embedding for a single query using Gemini"""
        return self.gemini_service.generate_query_embedding(query)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the Gemini embedding model"""
        return {
            "model_name": "text-embedding-004",
            "provider": "Gemini AI",
            "local": False,
            "fast": False,
            "lightweight": False,
            "dimension": 768
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
    Create the best available embedding service based on configuration
    
    Args:
        prefer_local: If True, prefer local embedding services
        gemini_service: Gemini service instance for fallback
        
    Returns:
        Embedding service instance
    """
    provider = getattr(settings, 'EMBEDDING_PROVIDER', 'chromadb').lower()
    
    # Try ChromaDB first (ultra-lightweight, no dependencies)
    if provider == "chromadb" or provider == "chroma":
        try:
            embedding_type = getattr(settings, 'CHROMA_EMBEDDING_TYPE', 'default')
            service = ChromaEmbeddingService(embedding_type)
            if service.test_connection():
                logger.info(f"Using ChromaDB {embedding_type} embedding function (ultra-lightweight, no dependencies)")
                return service
        except Exception as e:
            logger.warning(f"ChromaDB embedding service initialization failed: {str(e)}")
    
    # Try sentence transformer via ChromaDB (still lightweight)
    elif provider == "sentence_transformers" or provider == "sentence_transformer":
        try:
            service = ChromaEmbeddingService("sentence_transformer")
            if service.test_connection():
                logger.info("Using ChromaDB SentenceTransformer embedding function (lightweight)")
                return service
        except Exception as e:
            logger.warning(f"ChromaDB SentenceTransformer embedding service failed: {str(e)}")
    
    # Gemini fallback
    elif provider == "gemini" and gemini_service:
        try:
            gemini_embedding_service = GeminiEmbeddingService(gemini_service)
            if gemini_embedding_service.test_connection():
                logger.info("Using Gemini AI for embeddings (cloud-based)")
                return gemini_embedding_service
        except Exception as e:
            logger.warning(f"Gemini embedding service failed: {str(e)}")
    
    # Fallback priority: ChromaDB default -> ChromaDB sentence_transformer -> Gemini
    fallback_services = [
        ("ChromaDB Default", lambda: ChromaEmbeddingService("default")),
        ("ChromaDB SentenceTransformer", lambda: ChromaEmbeddingService("sentence_transformer")),
    ]
    
    if gemini_service:
        fallback_services.append(("Gemini", lambda: GeminiEmbeddingService(gemini_service)))
    
    for service_name, service_creator in fallback_services:
        try:
            service = service_creator()
            if service.test_connection():
                logger.info(f"Using {service_name} as fallback embedding service")
                return service
        except Exception as e:
            logger.warning(f"{service_name} fallback failed: {str(e)}")
    
    logger.error("All embedding services failed")
    raise ValueError("No embedding service available")