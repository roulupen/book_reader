"""
Persistent metadata storage for book information
"""
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
from app.config import settings

logger = logging.getLogger(__name__)

class MetadataStore:
    """Persistent storage for book metadata"""
    
    def __init__(self):
        self.metadata_file = Path(settings.CHROMA_PERSIST_DIRECTORY) / "books_metadata.json"
        self._metadata: Dict[str, Dict[str, Any]] = {}
        self._load_metadata()
    
    def _load_metadata(self):
        """Load metadata from persistent storage"""
        try:
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Convert datetime strings back to datetime objects
                    for book_id, metadata in data.items():
                        if 'upload_date' in metadata and isinstance(metadata['upload_date'], str):
                            metadata['upload_date'] = datetime.fromisoformat(metadata['upload_date'])
                    self._metadata = data
                    logger.info(f"Loaded metadata for {len(self._metadata)} books from persistent storage")
            else:
                logger.info("No existing metadata file found, starting with empty metadata")
                self._metadata = {}
        except Exception as e:
            logger.error(f"Error loading metadata: {str(e)}")
            self._metadata = {}
    
    def _save_metadata(self):
        """Save metadata to persistent storage"""
        try:
            # Ensure directory exists
            self.metadata_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert datetime objects to strings for JSON serialization
            serializable_data = {}
            for book_id, metadata in self._metadata.items():
                serializable_metadata = metadata.copy()
                if 'upload_date' in serializable_metadata and isinstance(serializable_metadata['upload_date'], datetime):
                    serializable_metadata['upload_date'] = serializable_metadata['upload_date'].isoformat()
                serializable_data[book_id] = serializable_metadata
            
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_data, f, indent=2, ensure_ascii=False)
            
            logger.debug(f"Saved metadata for {len(self._metadata)} books to persistent storage")
        except Exception as e:
            logger.error(f"Error saving metadata: {str(e)}")
    
    def add_book(self, book_id: str, filename: str, chunk_count: int, upload_date: Optional[datetime] = None):
        """Add or update book metadata"""
        if upload_date is None:
            upload_date = datetime.now()
        
        self._metadata[book_id] = {
            "filename": filename,
            "upload_date": upload_date,
            "status": "indexed",
            "chunk_count": chunk_count
        }
        self._save_metadata()
        logger.info(f"Added metadata for book {book_id}: {filename}")
    
    def remove_book(self, book_id: str) -> bool:
        """Remove book metadata"""
        if book_id in self._metadata:
            del self._metadata[book_id]
            self._save_metadata()
            logger.info(f"Removed metadata for book {book_id}")
            return True
        return False
    
    def get_book(self, book_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific book"""
        return self._metadata.get(book_id)
    
    def list_books(self) -> Dict[str, Dict[str, Any]]:
        """Get all book metadata"""
        return self._metadata.copy()
    
    def has_book(self, book_id: str) -> bool:
        """Check if book exists in metadata"""
        return book_id in self._metadata
    
    def get_book_count(self) -> int:
        """Get total number of books"""
        return len(self._metadata)
    
    def update_book_status(self, book_id: str, status: str):
        """Update book status"""
        if book_id in self._metadata:
            self._metadata[book_id]["status"] = status
            self._save_metadata()
    
    def sync_with_vector_store(self, vector_store):
        """Synchronize metadata with vector store to recover from inconsistencies"""
        try:
            # Get all book IDs from vector store
            vector_book_ids = set(vector_store.list_books())
            metadata_book_ids = set(self._metadata.keys())
            
            # Find orphaned metadata (metadata without vector data)
            orphaned_metadata = metadata_book_ids - vector_book_ids
            if orphaned_metadata:
                logger.warning(f"Found {len(orphaned_metadata)} orphaned metadata entries, removing...")
                for book_id in orphaned_metadata:
                    self.remove_book(book_id)
            
            # Find orphaned vectors (vector data without metadata)
            orphaned_vectors = vector_book_ids - metadata_book_ids
            if orphaned_vectors:
                logger.warning(f"Found {len(orphaned_vectors)} orphaned vector entries")
                # Try to recover basic metadata for orphaned vectors
                for book_id in orphaned_vectors:
                    chunk_count = vector_store.get_book_chunk_count(book_id)
                    if chunk_count > 0:
                        # Create basic metadata for orphaned book
                        self.add_book(
                            book_id=book_id,
                            filename=f"recovered_book_{book_id[:8]}.unknown",
                            chunk_count=chunk_count,
                            upload_date=datetime.now()
                        )
                        logger.info(f"Recovered metadata for orphaned book {book_id}")
            
            logger.info(f"Metadata sync complete. {len(self._metadata)} books in metadata store")
            
        except Exception as e:
            logger.error(f"Error syncing metadata with vector store: {str(e)}")

# Global metadata store instance
_metadata_store = None

def get_metadata_store() -> MetadataStore:
    """Get the global metadata store instance"""
    global _metadata_store
    if _metadata_store is None:
        _metadata_store = MetadataStore()
    return _metadata_store
