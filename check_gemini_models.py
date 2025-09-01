#!/usr/bin/env python3
"""
Script to check available Gemini models and test the API connection
"""
import os
import sys
from pathlib import Path

# Add the app directory to the Python path
sys.path.insert(0, str(Path(__file__).parent))

def check_api_key():
    """Check if API key is available"""
    from app.config import settings
    
    if not settings.GEMINI_API_KEY or settings.GEMINI_API_KEY == "your-gemini-api-key-here":
        print("❌ GEMINI_API_KEY is not set or still has default value")
        print("📝 Please set your actual API key in the .env file")
        return False
    
    print(f"✅ API key is configured (starts with: {settings.GEMINI_API_KEY[:10]}...)")
    return True

def list_models():
    """List available Gemini models"""
    try:
        print("\n🔍 Checking available Gemini models...")
        
        import google.generativeai as genai
        from app.config import settings
        
        genai.configure(api_key=settings.GEMINI_API_KEY)
        
        models = genai.list_models()
        available_models = []
        
        print("\n📋 Available models:")
        for model in models:
            if 'generateContent' in model.supported_generation_methods:
                available_models.append(model.name)
                print(f"  ✅ {model.name}")
                print(f"     Description: {getattr(model, 'description', 'No description')}")
                print(f"     Supported methods: {model.supported_generation_methods}")
                print()
        
        return available_models
        
    except Exception as e:
        print(f"❌ Error listing models: {str(e)}")
        return []

def test_model(model_name):
    """Test a specific model"""
    try:
        print(f"\n🧪 Testing model: {model_name}")
        
        import google.generativeai as genai
        from app.config import settings
        
        genai.configure(api_key=settings.GEMINI_API_KEY)
        model = genai.GenerativeModel(model_name)
        
        # Test with a simple prompt
        response = model.generate_content("Hello! Please respond with 'Model is working correctly.'")
        print(f"✅ Model response: {response.text}")
        return True
        
    except Exception as e:
        print(f"❌ Model test failed: {str(e)}")
        return False

def test_gemini_service():
    """Test the Gemini service class"""
    try:
        print("\n🛠️  Testing GeminiService class...")
        
        from app.services.gemini_service import GeminiService
        
        service = GeminiService()
        status = service.test_connection()
        
        if status["status"] == "success":
            print(f"✅ GeminiService initialized successfully")
            print(f"   Model: {status.get('model_name', 'unknown')}")
            print(f"   Message: {status['message']}")
        else:
            print(f"❌ GeminiService test failed: {status['message']}")
            
        return status["status"] == "success"
        
    except Exception as e:
        print(f"❌ GeminiService test failed: {str(e)}")
        return False

def main():
    """Main function"""
    print("🔍 Gemini API Model Checker\n")
    
    # Check API key
    if not check_api_key():
        return 1
    
    # List available models
    available_models = list_models()
    
    if not available_models:
        print("❌ No models available or API error")
        return 1
    
    # Test recommended models
    recommended_models = ['gemini-1.5-flash', 'gemini-1.5-pro', 'gemini-pro']
    working_models = []
    
    for model_name in recommended_models:
        full_model_name = None
        # Find the full model name
        for available in available_models:
            if model_name in available:
                full_model_name = available
                break
        
        if full_model_name:
            if test_model(full_model_name):
                working_models.append(full_model_name)
        else:
            print(f"⚠️  Model {model_name} not found in available models")
    
    # Test the service
    service_working = test_gemini_service()
    
    # Summary
    print("\n" + "="*60)
    print("📊 SUMMARY")
    print("="*60)
    print(f"Available models: {len(available_models)}")
    print(f"Working models tested: {len(working_models)}")
    print(f"GeminiService status: {'✅ Working' if service_working else '❌ Failed'}")
    
    if working_models:
        print(f"\n🎉 Recommended model to use: {working_models[0]}")
        print("   Your application should work correctly now!")
    else:
        print("\n❌ No working models found. Please check your API key and try again.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
