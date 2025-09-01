#!/usr/bin/env python3
"""
Startup script for the Book Reader RAG application
"""
import os
import sys
import uvicorn
from pathlib import Path

def check_environment():
    """Check if required environment variables are set"""
    required_vars = ["GEMINI_API_KEY"]
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print("❌ Missing required environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\n📝 Please set these variables or create a .env file")
        print("   Example: export GEMINI_API_KEY='your-api-key-here'")
        print("   Or copy .env.example to .env and fill in the values")
        return False
    
    return True

def check_dependencies():
    """Check if required directories exist"""
    from app.config import settings
    
    required_dirs = [
        settings.DATA_FOLDER,
        settings.UPLOAD_FOLDER,
        settings.CHROMA_PERSIST_DIRECTORY
    ]
    
    for dir_path in required_dirs:
        Path(dir_path).mkdir(exist_ok=True)
        print(f"✅ Directory ready: {dir_path}")

def main():
    """Main startup function"""
    print("🚀 Starting Book Reader RAG Application...")
    
    # Check environment
    if not check_environment():
        sys.exit(1)
    
    # Check and create directories
    check_dependencies()
    
    print("✅ All checks passed!")
    print("📚 Book Reader RAG is starting...")
    
    # Import settings after environment check
    from app.config import settings
    
    print(f"🌐 Open your browser to: http://localhost:{settings.API_PORT}")
    print(f"📖 API Documentation: http://localhost:{settings.API_PORT}/docs")
    print("\n⏹️  Press Ctrl+C to stop the server\n")
    
    # Start the server
    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True,
        log_level=settings.LOG_LEVEL.lower()
    )

if __name__ == "__main__":
    main()
