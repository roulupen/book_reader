from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

class BookUploadResponse(BaseModel):
    book_id: str
    filename: str
    status: str
    message: str
    indexed_at: datetime

class BookDeleteResponse(BaseModel):
    book_id: str
    status: str
    message: str

class QueryRequest(BaseModel):
    book_id: str
    question: str

class QueryResponse(BaseModel):
    book_id: str
    question: str
    answer: str
    sources: List[str]
    confidence_score: Optional[float] = None

class BookInfo(BaseModel):
    book_id: str
    filename: str
    upload_date: datetime
    status: str
    chunk_count: int

class ErrorResponse(BaseModel):
    error: str
    message: str
    details: Optional[str] = None
