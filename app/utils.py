"""
Utility functions for the Book Reader RAG application
"""
import os
import logging
import markdown
from typing import Optional
from pathlib import Path

def setup_logging(log_level: str = "INFO") -> None:
    """Setup application logging"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('app.log')
        ]
    )

def validate_file_extension(filename: str, allowed_extensions: set) -> bool:
    """Validate if file extension is allowed"""
    file_extension = Path(filename).suffix.lower()
    return file_extension in allowed_extensions

def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format"""
    if size_bytes == 0:
        return "0B"
    
    size_names = ["B", "KB", "MB", "GB"]
    i = 0
    
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f}{size_names[i]}"

def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe storage"""
    # Remove or replace unsafe characters
    unsafe_chars = ['<', '>', ':', '"', '/', '\\', '|', '?', '*']
    safe_filename = filename
    
    for char in unsafe_chars:
        safe_filename = safe_filename.replace(char, '_')
    
    # Limit filename length
    if len(safe_filename) > 255:
        name, ext = os.path.splitext(safe_filename)
        safe_filename = name[:255-len(ext)] + ext
    
    return safe_filename

def check_disk_space(path: str, required_bytes: int) -> bool:
    """Check if there's enough disk space"""
    try:
        stat = os.statvfs(path)
        available_bytes = stat.f_bavail * stat.f_frsize
        return available_bytes > required_bytes
    except (OSError, AttributeError):
        # Fallback for systems that don't support statvfs
        return True

def create_directories(*paths: str) -> None:
    """Create directories if they don't exist"""
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)

class FileValidator:
    """File validation utility class"""
    
    def __init__(self, max_size: int, allowed_extensions: set):
        self.max_size = max_size
        self.allowed_extensions = allowed_extensions
    
    def validate(self, filename: str, file_size: int) -> tuple[bool, Optional[str]]:
        """Validate file and return (is_valid, error_message)"""
        
        # Check file extension
        if not validate_file_extension(filename, self.allowed_extensions):
            return False, f"Unsupported file type. Allowed: {', '.join(self.allowed_extensions)}"
        
        # Check file size
        if file_size > self.max_size:
            max_size_mb = self.max_size / (1024 * 1024)
            return False, f"File too large. Maximum size: {max_size_mb:.1f}MB"
        
        # Check if file is empty
        if file_size == 0:
            return False, "File is empty"
        
        return True, None

def markdown_to_html(text: str) -> str:
    """Convert markdown text to HTML"""
    try:
        # Configure markdown with extensions for better formatting
        md = markdown.Markdown(extensions=[
            'markdown.extensions.fenced_code',
            'markdown.extensions.tables',
            'markdown.extensions.nl2br',
            'markdown.extensions.sane_lists'
        ])
        
        # Convert to HTML
        html = md.convert(text)
        
        # Clean up and ensure safe HTML
        return html
        
    except Exception as e:
        logging.error(f"Error converting markdown to HTML: {str(e)}")
        # Fallback: return text with basic line break conversion
        return text.replace('\n', '<br>')
