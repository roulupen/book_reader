import chromadb
import uuid
from typing import List, Dict, Any, Optional
from chromadb.config import Settings as ChromaSettings
from app.config import settings
import logging

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self, embedding_service=None):
        self.client = chromadb.PersistentClient(
            path=settings.CHROMA_PERSIST_DIRECTORY,
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            name=settings.CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
        self.embedding_service = embedding_service
    
    def add_documents(self, book_id: str, texts: List[str], metadatas: List[Dict[str, Any]], embeddings: List[List[float]] = None) -> bool:
        """Add document chunks to the vector store with optional custom embeddings"""
        try:
            # Generate unique IDs for each chunk
            ids = [f"{book_id}_{i}" for i in range(len(texts))]
            
            # Clean metadata and add book_id
            cleaned_metadatas = []
            for metadata in metadatas:
                cleaned_metadata = {"book_id": book_id}
                for key, value in metadata.items():
                    # ChromaDB only accepts specific types and no None values
                    if value is not None:
                        if isinstance(value, (str, int, float, bool)):
                            cleaned_metadata[key] = value
                        else:
                            # Convert other types to string
                            cleaned_metadata[key] = str(value)
                cleaned_metadatas.append(cleaned_metadata)
            
            # Use custom embeddings if provided, otherwise ChromaDB will generate them
            if embeddings and len(embeddings) == len(texts):
                self.collection.add(
                    documents=texts,
                    metadatas=cleaned_metadatas,
                    ids=ids,
                    embeddings=embeddings
                )
                logger.info(f"Added {len(texts)} documents with custom embeddings for book {book_id}")
            else:
                self.collection.add(
                    documents=texts,
                    metadatas=cleaned_metadatas,
                    ids=ids
                )
                logger.info(f"Added {len(texts)} documents with default embeddings for book {book_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding documents for book {book_id}: {str(e)}")
            return False
    
    def query_documents(self, book_id: str, query: str, n_results: int = 5, query_embedding: List[float] = None) -> List[Dict[str, Any]]:
        """Query documents for a specific book with optional custom query embedding"""
        try:
            # Use custom query embedding if provided (check for None and non-empty list)
            if query_embedding is not None and len(query_embedding) > 0:
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=n_results,
                    where={"book_id": book_id}
                )
                logger.info(f"Queried with custom embedding for book {book_id}")
            else:
                results = self.collection.query(
                    query_texts=[query],
                    n_results=n_results,
                    where={"book_id": book_id}
                )
                logger.info(f"Queried with text for book {book_id}")
            
            # Format results - safe checking for nested lists
            documents = []
            if (results.get("documents") and 
                len(results["documents"]) > 0 and 
                results["documents"][0] is not None):
                
                for i, doc in enumerate(results["documents"][0]):
                    documents.append({
                        "content": doc,
                        "metadata": results["metadatas"][0][i] if (results.get("metadatas") and 
                                                                   len(results["metadatas"]) > 0 and 
                                                                   len(results["metadatas"][0]) > i) else {},
                        "distance": results["distances"][0][i] if (results.get("distances") and 
                                                                   len(results["distances"]) > 0 and 
                                                                   len(results["distances"][0]) > i) else 0.0
                    })
            
            return documents
            
        except Exception as e:
            logger.error(f"Error querying documents for book {book_id}: {str(e)}")
            return []
    
    def delete_book_documents(self, book_id: str) -> bool:
        """Delete all documents for a specific book"""
        try:
            # Get all document IDs for this book
            results = self.collection.get(where={"book_id": book_id})
            
            if results["ids"]:
                self.collection.delete(ids=results["ids"])
                logger.info(f"Deleted {len(results['ids'])} documents for book {book_id}")
                return True
            else:
                logger.info(f"No documents found for book {book_id}")
                return True
                
        except Exception as e:
            logger.error(f"Error deleting documents for book {book_id}: {str(e)}")
            return False
    
    def get_book_chunk_count(self, book_id: str) -> int:
        """Get the number of chunks for a specific book"""
        try:
            results = self.collection.get(where={"book_id": book_id})
            return len(results["ids"]) if results["ids"] else 0
        except Exception as e:
            logger.error(f"Error getting chunk count for book {book_id}: {str(e)}")
            return 0
    
    def list_books(self) -> List[str]:
        """List all book IDs in the vector store"""
        try:
            results = self.collection.get()
            book_ids = set()
            
            if results["metadatas"]:
                for metadata in results["metadatas"]:
                    if "book_id" in metadata:
                        book_ids.add(metadata["book_id"])
            
            return list(book_ids)
            
        except Exception as e:
            logger.error(f"Error listing books: {str(e)}")
            return []
