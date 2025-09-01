from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import logging
from typing import List, Dict, Any
from datetime import datetime
import os
from pathlib import Path

from app.config import settings
from app.models import (
    BookUploadResponse, BookDeleteResponse, QueryRequest, QueryResponse,
    BookInfo, ErrorResponse
)
from app.services.vector_store import VectorStore
from app.services.gemini_service import GeminiService
from app.services.file_processor import FileProcessor
from app.services.embedding_service import create_embedding_service
from app.services.metadata_store import get_metadata_store
from app.utils import markdown_to_html

# Configure logging
logging.basicConfig(level=getattr(logging, settings.LOG_LEVEL.upper()))
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title=settings.API_TITLE,
    description=settings.API_DESCRIPTION,
    version=settings.API_VERSION
)

# Initialize services
gemini_service = GeminiService()
embedding_service = create_embedding_service(prefer_local=True, gemini_service=gemini_service)
vector_store = VectorStore(embedding_service)
file_processor = FileProcessor()
metadata_store = get_metadata_store()

# Setup templates
templates_dir = Path(__file__).parent / "templates"
templates_dir.mkdir(exist_ok=True)
templates = Jinja2Templates(directory=str(templates_dir))

# Sync metadata store with vector store on startup
try:
    metadata_store.sync_with_vector_store(vector_store)
    logger.info(f"Application started with {metadata_store.get_book_count()} books available")
except Exception as e:
    logger.error(f"Error during metadata sync: {str(e)}")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serve the main HTML interface"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload", response_model=BookUploadResponse)
async def upload_book(file: UploadFile = File(...)):
    """Upload and index a book"""
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        file_extension = Path(file.filename).suffix.lower()
        if file_extension not in settings.ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type. Allowed: {', '.join(settings.ALLOWED_EXTENSIONS)}"
            )
        
        # Check file size
        content = await file.read()
        if len(content) > settings.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400, 
                detail=f"File too large. Maximum size: {settings.MAX_FILE_SIZE // (1024*1024)}MB"
            )
        
        # Save file temporarily and get book_id
        book_id, temp_file_path = file_processor.save_uploaded_file(content, file.filename)
        
        try:
            # Extract text from file
            logger.info(f"Extracting text from {file.filename}")
            text_content = file_processor.extract_text_from_file(temp_file_path)
            
            if not text_content.strip():
                raise HTTPException(status_code=400, detail="No text content found in file")
            
            # Chunk the text
            chunks = file_processor.chunk_text(text_content)
            if not chunks:
                raise HTTPException(status_code=400, detail="Failed to create text chunks")
            
            # Generate embeddings using FastEmbed for faster indexing
            texts = [chunk["content"] for chunk in chunks]
            metadatas = [chunk["metadata"] for chunk in chunks]
            
            logger.info(f"Generating embeddings for {len(texts)} chunks using {embedding_service.get_model_info()['provider']}")
            embeddings = embedding_service.generate_embeddings(texts)
            
            # Add to vector store with custom embeddings
            success = vector_store.add_documents(book_id, texts, metadatas, embeddings)
            if not success:
                raise HTTPException(status_code=500, detail="Failed to index book")
            
            # Store metadata persistently
            metadata_store.add_book(
                book_id=book_id,
                filename=file.filename,
                chunk_count=len(chunks)
            )
            
        finally:
            # Always delete the temporary file after processing
            file_processor.delete_temp_file(temp_file_path)
            logger.info(f"Deleted temporary file after indexing: {temp_file_path}")
        
        logger.info(f"Successfully indexed book {book_id} with {len(chunks)} chunks")
        
        return BookUploadResponse(
            book_id=book_id,
            filename=file.filename,
            status="success",
            message=f"Book indexed successfully with {len(chunks)} chunks",
            indexed_at=datetime.now()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading book: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.delete("/books/{book_id}", response_model=BookDeleteResponse)
async def delete_book(book_id: str):
    """Delete a book and its index"""
    try:
        if not metadata_store.has_book(book_id):
            raise HTTPException(status_code=404, detail="Book not found")
        
        # Delete from vector store
        vector_success = vector_store.delete_book_documents(book_id)
        
        # Clean up any remaining temporary files (just in case)
        file_cleanup_success = file_processor.cleanup_temp_files(book_id)
        
        # Remove from metadata store
        metadata_store.remove_book(book_id)
        
        status = "success" if vector_success else "partial"
        message = "Book and all associated data deleted successfully"
        
        if file_cleanup_success:
            logger.info(f"Completed cleanup of all data for book {book_id}")
        else:
            logger.warning(f"Some cleanup operations may have failed for book {book_id}")
        
        logger.info(f"Deleted book {book_id}")
        
        return BookDeleteResponse(
            book_id=book_id,
            status=status,
            message=message
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting book {book_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/query", response_model=QueryResponse)
async def query_book(query: QueryRequest):
    """Ask a question about a book"""
    try:
        if not metadata_store.has_book(query.book_id):
            raise HTTPException(status_code=404, detail="Book not found")
        
        # Generate query embedding for better semantic search
        query_embedding = embedding_service.generate_query_embedding(query.question)
        
        # Query vector store for relevant chunks using custom embedding
        context_chunks = vector_store.query_documents(
            book_id=query.book_id,
            query=query.question,
            n_results=5,
            query_embedding=query_embedding if query_embedding else None
        )
        
        if not context_chunks:
            return QueryResponse(
                book_id=query.book_id,
                question=query.question,
                answer="I couldn't find information about this topic in the provided book content. The book may not cover this subject, or the relevant information might be in a different section.",
                sources=[],
                confidence_score=0.0
            )
        
        # Generate answer using Gemini
        result = gemini_service.answer_question(query.question, context_chunks)
        
        # Convert markdown to HTML for better formatting
        formatted_answer = markdown_to_html(result["answer"])
        
        return QueryResponse(
            book_id=query.book_id,
            question=query.question,
            answer=formatted_answer,
            sources=result["sources"],
            confidence_score=result["confidence_score"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error querying book {query.book_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/books", response_model=List[BookInfo])
async def list_books():
    """List all uploaded books"""
    try:
        book_list = []
        for book_id, metadata in metadata_store.list_books().items():
            book_list.append(BookInfo(
                book_id=book_id,
                filename=metadata["filename"],
                upload_date=metadata["upload_date"],
                status=metadata["status"],
                chunk_count=metadata["chunk_count"]
            ))
        
        return book_list
        
    except Exception as e:
        logger.error(f"Error listing books: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/books/{book_id}", response_model=BookInfo)
async def get_book_info(book_id: str):
    """Get information about a specific book"""
    try:
        metadata = metadata_store.get_book(book_id)
        if not metadata:
            raise HTTPException(status_code=404, detail="Book not found")
        return BookInfo(
            book_id=book_id,
            filename=metadata["filename"],
            upload_date=metadata["upload_date"],
            status=metadata["status"],
            chunk_count=metadata["chunk_count"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting book info {book_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.exception_handler(404)
async def not_found_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=404,
        content=ErrorResponse(
            error="Not Found",
            message="The requested resource was not found"
        ).dict()
    )

@app.get("/health")
async def health_check():
    """Health check endpoint with service status"""
    try:
        # Test services
        gemini_status = gemini_service.test_connection()
        embedding_status = embedding_service.test_connection()
        embedding_info = embedding_service.get_model_info()
        
        return {
            "status": "healthy",
            "services": {
                "vector_store": "operational",
                "file_processor": "operational",
                "gemini_ai": gemini_status,
                "embedding_service": {
                    "status": "connected" if embedding_status else "disconnected",
                    "provider": embedding_info.get("provider", "unknown"),
                    "model": embedding_info.get("model_name", "unknown"),
                    "local": embedding_info.get("local", False),
                    "fast": embedding_info.get("fast", False)
                }
            },
            "available_models": gemini_service.list_available_models()
        }
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "services": {
                    "vector_store": "unknown",
                    "file_processor": "unknown",
                    "gemini_ai": {"status": "error", "message": str(e)}
                }
            }
        )

@app.exception_handler(500)
async def internal_error_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal Server Error",
            message="An unexpected error occurred"
        ).dict()
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
