import os
import uuid
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import PyPDF2
from docx import Document
import logging
from datetime import datetime
from app.config import settings

logger = logging.getLogger(__name__)

class FileProcessor:
    def __init__(self):
        self.data_folder = Path(settings.DATA_FOLDER)
        self.upload_folder = Path(settings.UPLOAD_FOLDER)
        
    def save_uploaded_file(self, file_content: bytes, filename: str) -> Tuple[str, str]:
        """Save uploaded file temporarily and return book_id and temp file path"""
        # Generate unique book ID
        book_id = str(uuid.uuid4())
        
        # Get file extension
        file_extension = Path(filename).suffix.lower()
        if file_extension not in settings.ALLOWED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {file_extension}")
        
        # Create unique filename for temporary upload folder (will be deleted after indexing)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_filename = f"{book_id}_{timestamp}_{filename}"
        temp_file_path = self.upload_folder / safe_filename
        
        # Save file to temporary upload folder
        with open(temp_file_path, "wb") as f:
            f.write(file_content)
        
        logger.info(f"Saved file {filename} temporarily as {safe_filename} with book_id {book_id}")
        return book_id, str(temp_file_path)
    
    def extract_text_from_file(self, file_path: str) -> str:
        """Extract text content from various file formats"""
        file_path = Path(file_path)
        file_extension = file_path.suffix.lower()
        
        try:
            if file_extension == ".pdf":
                return self._extract_text_from_pdf(file_path)
            elif file_extension == ".docx":
                return self._extract_text_from_docx(file_path)
            elif file_extension == ".txt":
                return self._extract_text_from_txt(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")
                
        except Exception as e:
            logger.error(f"Error extracting text from {file_path}: {str(e)}")
            raise
    
    def _extract_text_from_pdf(self, file_path: Path) -> str:
        """Extract text from PDF file"""
        text = ""
        try:
            with open(file_path, "rb") as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    if page_text.strip():
                        text += f"\n[Page {page_num + 1}]\n{page_text}\n"
        except Exception as e:
            logger.error(f"Error reading PDF {file_path}: {str(e)}")
            raise
        
        return text.strip()
    
    def _extract_text_from_docx(self, file_path: Path) -> str:
        """Extract text from DOCX file"""
        try:
            doc = Document(file_path)
            text = ""
            
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    text += paragraph.text + "\n"
            
            return text.strip()
            
        except Exception as e:
            logger.error(f"Error reading DOCX {file_path}: {str(e)}")
            raise
    
    def _extract_text_from_txt(self, file_path: Path) -> str:
        """Extract text from TXT file"""
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                return file.read().strip()
        except UnicodeDecodeError:
            # Try with different encoding
            try:
                with open(file_path, "r", encoding="latin-1") as file:
                    return file.read().strip()
            except Exception as e:
                logger.error(f"Error reading TXT {file_path}: {str(e)}")
                raise
    
    def chunk_text(self, text: str, chunk_size: int = None, overlap: int = None) -> List[Dict[str, Any]]:
        """Split text into semantically meaningful chunks with metadata"""
        chunk_size = chunk_size or settings.CHUNK_SIZE
        overlap = overlap or settings.CHUNK_OVERLAP
        
        if not text.strip():
            return []
        
        # Enhanced chunking strategy
        chunks = self._semantic_chunking(text, chunk_size, overlap)
        
        logger.info(f"Created {len(chunks)} semantic chunks from text of length {len(text)}")
        return chunks
    
    def _semantic_chunking(self, text: str, chunk_size: int, overlap: int) -> List[Dict[str, Any]]:
        """Advanced semantic chunking that preserves context"""
        chunks = []
        
        # First, split by major sections (double newlines, chapters, etc.)
        major_sections = self._split_into_sections(text)
        
        chunk_index = 0
        global_start = 0
        
        for section in major_sections:
            section_chunks = self._chunk_section(section, chunk_size, overlap, global_start, chunk_index)
            chunks.extend(section_chunks)
            chunk_index += len(section_chunks)
            global_start += len(section)
        
        return chunks
    
    def _split_into_sections(self, text: str) -> List[str]:
        """Split text into logical sections"""
        import re
        
        # Patterns for section breaks
        section_patterns = [
            r'\n\s*Chapter\s+\d+',  # Chapter breaks
            r'\n\s*CHAPTER\s+\d+',  # CHAPTER breaks
            r'\n\s*\d+\.\s+[A-Z]',  # Numbered sections
            r'\n\s*[A-Z][A-Z\s]+\n',  # All caps headers
            r'\n\s*#{1,3}\s+',  # Markdown headers
        ]
        
        # Try to find natural section breaks
        for pattern in section_patterns:
            matches = list(re.finditer(pattern, text))
            if matches and len(matches) > 1:
                sections = []
                start = 0
                for match in matches:
                    if start < match.start():
                        sections.append(text[start:match.start()])
                    start = match.start()
                sections.append(text[start:])
                return [s.strip() for s in sections if s.strip()]
        
        # Fallback: split by double newlines
        sections = text.split('\n\n')
        return [s.strip() for s in sections if s.strip()]
    
    def _chunk_section(self, section: str, chunk_size: int, overlap: int, global_start: int, start_index: int) -> List[Dict[str, Any]]:
        """Chunk a single section with smart boundary detection"""
        if len(section) <= chunk_size:
            # Section fits in one chunk
            page_match = self._extract_page_number(section)
            return [{
                "content": section,
                "metadata": {
                    "chunk_index": start_index,
                    "start_char": global_start,
                    "end_char": global_start + len(section),
                    "length": len(section),
                    "page": page_match,
                    "section_type": self._classify_section(section)
                }
            }]
        
        chunks = []
        start = 0
        chunk_index = start_index
        
        while start < len(section):
            end = start + chunk_size
            
            # Smart boundary detection
            if end < len(section):
                end = self._find_best_break_point(section, start, end, chunk_size)
            
            chunk_text = section[start:end].strip()
            
            if chunk_text:
                page_match = self._extract_page_number(chunk_text)
                
                chunks.append({
                    "content": chunk_text,
                    "metadata": {
                        "chunk_index": chunk_index,
                        "start_char": global_start + start,
                        "end_char": global_start + end,
                        "length": len(chunk_text),
                        "page": page_match,
                        "section_type": self._classify_section(chunk_text)
                    }
                })
                
                chunk_index += 1
            
            # Move start with overlap, but ensure progress
            start = max(start + 1, end - overlap)
            if start >= end:
                start = end
        
        return chunks
    
    def _find_best_break_point(self, text: str, start: int, end: int, chunk_size: int) -> int:
        """Find the best point to break a chunk while preserving meaning"""
        # Priority order for break points
        break_points = [
            # Paragraph breaks (highest priority)
            (r'\n\s*\n', 2),
            # Sentence endings
            (r'[.!?]\s+[A-Z]', 1),
            # Comma after long phrases
            (r',\s+', 0.5),
            # Semicolon
            (r';\s+', 0.7),
            # Colon
            (r':\s+', 0.6),
        ]
        
        import re
        
        # Look backwards from the end to find the best break point
        search_window = min(chunk_size // 3, end - start)
        search_start = max(start + chunk_size // 2, end - search_window)
        
        best_break = end
        best_priority = -1
        
        for pattern, priority in break_points:
            matches = list(re.finditer(pattern, text[search_start:end]))
            if matches:
                # Take the last match (closest to desired end)
                match = matches[-1]
                break_point = search_start + match.end()
                if priority > best_priority:
                    best_break = break_point
                    best_priority = priority
        
        return best_break
    
    def _classify_section(self, text: str) -> str:
        """Classify the type of text section for better indexing"""
        text_lower = text.lower()
        
        # Check for different content types
        if any(word in text_lower for word in ['chapter', 'section', 'part']):
            return 'header'
        elif text.count('.') > len(text) / 50:  # Lots of periods = prose
            return 'prose'
        elif text.count(':') > len(text) / 100:  # Lists or definitions
            return 'list'
        elif any(word in text_lower for word in ['figure', 'table', 'chart']):
            return 'figure'
        elif text.count('\n') / len(text) > 0.05:  # Lots of line breaks
            return 'structured'
        else:
            return 'general'
    
    def _extract_page_number(self, text: str) -> int:
        """Extract page number from text chunk, returns 0 if not found"""
        import re
        
        # Look for [Page X] pattern at the beginning of the chunk
        page_pattern = r"\[Page (\d+)\]"
        match = re.search(page_pattern, text[:100])
        
        if match:
            return int(match.group(1))
        
        # Return 0 instead of None to avoid ChromaDB issues
        return 0
    
    def delete_temp_file(self, file_path: str) -> bool:
        """Delete temporary file after processing"""
        try:
            temp_file = Path(file_path)
            if temp_file.exists():
                temp_file.unlink()
                logger.info(f"Deleted temporary file: {file_path}")
                return True
            else:
                logger.warning(f"Temporary file not found: {file_path}")
                return False
        except Exception as e:
            logger.error(f"Error deleting temporary file {file_path}: {str(e)}")
            return False

    def cleanup_temp_files(self, book_id: str) -> bool:
        """Clean up any remaining temporary files for a book"""
        try:
            deleted_count = 0
            # Check both upload and data folders for any remaining files
            for folder in [self.upload_folder, self.data_folder]:
                for file_path in folder.glob(f"{book_id}_*"):
                    if file_path.is_file():
                        file_path.unlink()
                        logger.info(f"Cleaned up file: {file_path}")
                        deleted_count += 1
            
            if deleted_count > 0:
                logger.info(f"Cleaned up {deleted_count} files for book {book_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error cleaning up files for book {book_id}: {str(e)}")
            return False
    
    def get_file_info(self, book_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a book file (Note: files are deleted after indexing)"""
        # Since files are deleted after indexing, we can't provide file info
        # This method is kept for compatibility but will return None
        logger.info(f"File info requested for book {book_id}, but files are deleted after indexing")
        return None
