"""
Configurable embedding service supporting multiple providers for optimal speed
"""
import logging
from typing import List, Optional, Dict, Any
from fastembed import TextEmbedding
import numpy as np
from app.config import settings

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
    
    def generate_embeddings(self, texts: List[str], progress_callback=None) -> List[List[float]]:
        """
        Generate embeddings for multiple texts with optional progress callback
        
        Args:
            texts: List of text strings to embed
            progress_callback: Optional callback function(current, total, message)
            
        Returns:
            List of embedding vectors (each vector is a list of floats)
        """
        if not self.embedding_model:
            raise ValueError("Embedding model not initialized")
        
        if not texts:
            return []
        
        try:
            total_texts = len(texts)
            logger.info(f"Generating embeddings for {total_texts} texts using FastEmbed")
            
            if progress_callback:
                progress_callback(0, total_texts, "Starting embedding generation...")
            
            # Process in batches for better progress reporting
            batch_size = 50  # Process 50 texts at a time for progress updates
            embeddings = []
            
            for i in range(0, total_texts, batch_size):
                batch_texts = texts[i:i + batch_size]
                
                # Generate embeddings for this batch
                batch_embeddings_generator = self.embedding_model.embed(batch_texts)
                
                batch_embeddings = []
                for j, embedding in enumerate(batch_embeddings_generator):
                    # Convert numpy array to list of floats
                    if isinstance(embedding, np.ndarray):
                        batch_embeddings.append(embedding.tolist())
                    else:
                        batch_embeddings.append(list(embedding))
                    
                    # Report progress for each embedding in batch
                    current_total = i + j + 1
                    if progress_callback:
                        progress_callback(current_total, total_texts, f"Generated embedding {current_total}/{total_texts}")
                
                embeddings.extend(batch_embeddings)
            
            if progress_callback:
                progress_callback(total_texts, total_texts, "Embedding generation completed!")
            
            logger.info(f"Generated {len(embeddings)} embeddings successfully")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings with FastEmbed: {str(e)}")
            if progress_callback:
                progress_callback(0, len(texts), f"Error: {str(e)}")
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
    
    def generate_embeddings(self, texts: List[str], progress_callback=None) -> List[List[float]]:
        """Generate embeddings using Gemini AI with progress tracking"""
        if progress_callback:
            progress_callback(0, len(texts), "Starting Gemini embedding generation...")
        
        # Gemini processes all at once, so we simulate progress
        embeddings = self.gemini_service.generate_embeddings(texts)
        
        if progress_callback:
            if embeddings:
                progress_callback(len(texts), len(texts), "Gemini embedding generation completed!")
            else:
                progress_callback(0, len(texts), "Gemini embedding generation failed")
        
        return embeddings
    
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

class SentenceTransformersEmbeddingService:
    """Ultra-fast embedding service using sentence-transformers directly"""
    
    def __init__(self, model_name: str = None):
        """
        Initialize SentenceTransformers embedding service
        
        Args:
            model_name: Model to use. If None, uses config setting.
                Fast options:
                - "all-MiniLM-L6-v2" (fastest, 384 dimensions)
                - "all-MiniLM-L12-v2" (good balance)
                - "all-mpnet-base-v2" (best quality, slower)
        """
        try:
            from sentence_transformers import SentenceTransformer
            self.SentenceTransformer = SentenceTransformer
        except ImportError:
            raise ImportError("sentence-transformers not installed. Run: pip install sentence-transformers")
        
        self.model_name = model_name or settings.SENTENCE_TRANSFORMERS_MODEL
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the SentenceTransformers model"""
        try:
            logger.info(f"Initializing SentenceTransformers model: {self.model_name}")
            self.model = self.SentenceTransformer(self.model_name)
            logger.info(f"Successfully initialized SentenceTransformers model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize SentenceTransformers model {self.model_name}: {str(e)}")
            # Fallback to fastest model
            try:
                fallback_model = "all-MiniLM-L6-v2"
                logger.info(f"Trying fallback model: {fallback_model}")
                self.model = self.SentenceTransformer(fallback_model)
                self.model_name = fallback_model
                logger.info(f"Successfully initialized fallback model: {fallback_model}")
            except Exception as fallback_error:
                logger.error(f"Failed to initialize fallback model: {str(fallback_error)}")
                raise ValueError("Could not initialize any SentenceTransformers model")
    
    def generate_embeddings(self, texts: List[str], progress_callback=None) -> List[List[float]]:
        """
        Generate embeddings using SentenceTransformers (very fast)
        
        Args:
            texts: List of text strings to embed
            progress_callback: Optional callback function(current, total, message)
            
        Returns:
            List of embedding vectors
        """
        if not self.model:
            raise ValueError("SentenceTransformers model not initialized")
        
        if not texts:
            return []
        
        try:
            total_texts = len(texts)
            batch_size = settings.EMBEDDING_BATCH_SIZE
            
            logger.info(f"Generating embeddings for {total_texts} texts using SentenceTransformers (batch_size={batch_size})")
            
            if progress_callback:
                progress_callback(0, total_texts, "Starting SentenceTransformers embedding generation...")
            
            embeddings = []
            
            # Process in batches for progress reporting and memory efficiency
            for i in range(0, total_texts, batch_size):
                batch_texts = texts[i:i + batch_size]
                current_batch_end = min(i + batch_size, total_texts)
                
                # Generate embeddings for this batch (very fast)
                batch_embeddings = self.model.encode(batch_texts, convert_to_numpy=True, show_progress_bar=False)
                
                # Convert to list of lists
                for embedding in batch_embeddings:
                    embeddings.append(embedding.tolist())
                
                # Report progress
                if progress_callback:
                    progress_callback(
                        current_batch_end, 
                        total_texts, 
                        f"Generated embeddings {current_batch_end}/{total_texts} (batch {i//batch_size + 1})"
                    )
            
            if progress_callback:
                progress_callback(total_texts, total_texts, "SentenceTransformers embedding generation completed!")
            
            logger.info(f"Generated {len(embeddings)} embeddings successfully with SentenceTransformers")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings with SentenceTransformers: {str(e)}")
            if progress_callback:
                progress_callback(0, len(texts), f"Error: {str(e)}")
            return []
    
    def generate_query_embedding(self, query: str) -> List[float]:
        """Generate embedding for a single query"""
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
            "provider": "SentenceTransformers",
            "local": True,
            "fast": True,
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
            logger.error(f"SentenceTransformers test connection failed: {str(e)}")
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
    provider = settings.EMBEDDING_PROVIDER.lower()
    
    # Try the configured provider first
    if provider == "sentence_transformers":
        try:
            service = SentenceTransformersEmbeddingService()
            if service.test_connection():
                logger.info(f"Using SentenceTransformers for embeddings: {service.model_name} (ultra-fast, local)")
                return service
        except Exception as e:
            logger.warning(f"SentenceTransformers initialization failed: {str(e)}")
    
    elif provider == "fastembed":
        try:
            # Use the configured FastEmbed model
            service = FastEmbeddingService(settings.FASTEMBED_MODEL)
            if service.test_connection():
                logger.info(f"Using FastEmbed for embeddings: {service.model_name} (fast, local)")
                return service
        except Exception as e:
            logger.warning(f"FastEmbed initialization failed: {str(e)}")
    
    elif provider == "gemini" and gemini_service:
        try:
            gemini_embedding_service = GeminiEmbeddingService(gemini_service)
            if gemini_embedding_service.test_connection():
                logger.info("Using Gemini AI for embeddings (high quality, slower)")
                return gemini_embedding_service
        except Exception as e:
            logger.warning(f"Gemini embedding service failed: {str(e)}")
    
    # Fallback priority: SentenceTransformers (fastest) -> FastEmbed -> Gemini
    fallback_services = [
        ("SentenceTransformers", lambda: SentenceTransformersEmbeddingService()),
        ("FastEmbed", lambda: FastEmbeddingService("sentence-transformers/all-MiniLM-L6-v2")),  # Fastest FastEmbed model
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
