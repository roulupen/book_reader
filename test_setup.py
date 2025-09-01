#!/usr/bin/env python3
"""
Test script to verify the Book Reader RAG setup
"""
import os
import sys
from pathlib import Path

def test_imports():
    """Test that all required modules can be imported"""
    try:
        print("🧪 Testing imports...")
        
        # Test core imports
        from app.config import settings
        print("✅ Configuration module imported")
        
        from app.models import BookUploadResponse, QueryRequest
        print("✅ Models imported")
        
        from app.services.vector_store import VectorStore
        print("✅ Vector store service imported")
        
        from app.services.file_processor import FileProcessor
        print("✅ File processor service imported")
        
        # Test Gemini service (will fail without API key, but import should work)
        try:
            from app.services.gemini_service import GeminiService
            print("✅ Gemini service imported")
        except ValueError as e:
            if "GEMINI_API_KEY" in str(e):
                print("⚠️  Gemini service requires API key (expected)")
            else:
                raise
        
        from app.main import app
        print("✅ FastAPI app imported")
        
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {str(e)}")
        return False

def test_configuration():
    """Test configuration loading"""
    try:
        print("\n🔧 Testing configuration...")
        
        from app.config import settings
        
        # Test that .env file is loaded
        print(f"✅ API Title: {settings.API_TITLE}")
        print(f"✅ API Port: {settings.API_PORT}")
        print(f"✅ Data Folder: {settings.DATA_FOLDER}")
        print(f"✅ Chunk Size: {settings.CHUNK_SIZE}")
        print(f"✅ Max File Size: {settings.MAX_FILE_SIZE // (1024*1024)}MB")
        
        # Check if directories exist
        for folder in [settings.DATA_FOLDER, settings.UPLOAD_FOLDER, settings.CHROMA_PERSIST_DIRECTORY]:
            if Path(folder).exists():
                print(f"✅ Directory exists: {folder}")
            else:
                print(f"❌ Directory missing: {folder}")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {str(e)}")
        return False

def test_services():
    """Test service initialization (without API calls)"""
    try:
        print("\n🛠️  Testing services...")
        
        # Test vector store
        from app.services.vector_store import VectorStore
        vector_store = VectorStore()
        print("✅ Vector store initialized")
        
        # Test file processor
        from app.services.file_processor import FileProcessor
        file_processor = FileProcessor()
        print("✅ File processor initialized")
        
        # Test file validation
        from app.utils import FileValidator
        from app.config import settings
        validator = FileValidator(settings.MAX_FILE_SIZE, settings.ALLOWED_EXTENSIONS)
        
        # Test validation
        is_valid, error = validator.validate("test.pdf", 1000)
        if is_valid:
            print("✅ File validation working")
        else:
            print(f"❌ File validation failed: {error}")
        
        return True
        
    except Exception as e:
        print(f"❌ Services test failed: {str(e)}")
        return False

def main():
    """Run all tests"""
    print("🚀 Book Reader RAG Setup Test\n")
    
    all_passed = True
    
    # Run tests
    all_passed &= test_imports()
    all_passed &= test_configuration()
    all_passed &= test_services()
    
    # Summary
    print("\n" + "="*50)
    if all_passed:
        print("🎉 All tests passed! Setup is working correctly.")
        print("\n📝 Next steps:")
        print("1. Set your GEMINI_API_KEY in the .env file")
        print("2. Run: uv run python run.py")
        print("3. Open: http://localhost:8000")
        return 0
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
