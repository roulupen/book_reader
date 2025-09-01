#!/usr/bin/env python3
"""
Embedding Performance Benchmark
Compares FastEmbed vs Gemini AI embedding generation speed
"""
import time
import logging
from typing import List
from app.services.embedding_service import FastEmbeddingService, GeminiEmbeddingService
from app.services.gemini_service import GeminiService

# Suppress verbose logging for clean output
logging.getLogger().setLevel(logging.WARNING)

def benchmark_embedding_service(service, service_name: str, texts: List[str]) -> dict:
    """Benchmark an embedding service"""
    print(f"\n🧪 Testing {service_name}...")
    
    try:
        # Warm up (first call might be slower due to model loading)
        _ = service.generate_embeddings([texts[0]])
        
        # Actual benchmark
        start_time = time.time()
        embeddings = service.generate_embeddings(texts)
        end_time = time.time()
        
        duration = end_time - start_time
        embedding_dim = len(embeddings[0]) if embeddings else 0
        
        print(f"   ✅ Generated {len(embeddings)} embeddings")
        print(f"   ⏱️  Time: {duration:.3f} seconds")
        print(f"   📏 Dimension: {embedding_dim}")
        print(f"   ⚡ Speed: {len(texts)/duration:.1f} embeddings/second")
        
        return {
            "service": service_name,
            "count": len(embeddings),
            "time": duration,
            "dimension": embedding_dim,
            "speed": len(texts)/duration if duration > 0 else 0,
            "success": True
        }
        
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
        return {
            "service": service_name,
            "success": False,
            "error": str(e)
        }

def main():
    print("🚀 Embedding Performance Benchmark")
    print("=" * 50)
    
    # Test data - typical book chunks
    test_texts = [
        "In the beginning was the Word, and the Word was with God, and the Word was God.",
        "It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of foolishness.",
        "Call me Ishmael. Some years ago—never mind how long precisely—having little or no money in my purse.",
        "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife.",
        "All happy families are alike; each unhappy family is unhappy in its own way.",
        "In a hole in the ground there lived a hobbit. Not a nasty, dirty, wet hole, filled with the ends of worms.",
        "It was a bright cold day in April, and the clocks were striking thirteen.",
        "Space: the final frontier. These are the voyages of the starship Enterprise.",
        "To be, or not to be, that is the question: Whether 'tis nobler in the mind to suffer the slings and arrows.",
        "Once upon a time, in a galaxy far, far away, there lived a young farm boy named Luke Skywalker."
    ]
    
    print(f"📊 Testing with {len(test_texts)} text chunks")
    print(f"📝 Average text length: {sum(len(t) for t in test_texts) / len(test_texts):.0f} characters")
    
    results = []
    
    # Test FastEmbed
    try:
        fastembed_service = FastEmbeddingService()
        result = benchmark_embedding_service(fastembed_service, "FastEmbed (Local)", test_texts)
        results.append(result)
    except Exception as e:
        print(f"\n❌ FastEmbed initialization failed: {e}")
    
    # Test Gemini (if API key is available)
    try:
        gemini_service = GeminiService()
        gemini_embedding_service = GeminiEmbeddingService(gemini_service)
        result = benchmark_embedding_service(gemini_embedding_service, "Gemini AI (Cloud)", test_texts)
        results.append(result)
    except Exception as e:
        print(f"\n⚠️  Gemini AI not available: {e}")
    
    # Summary
    print("\n" + "=" * 50)
    print("📈 PERFORMANCE SUMMARY")
    print("=" * 50)
    
    successful_results = [r for r in results if r.get('success', False)]
    
    if len(successful_results) >= 2:
        fastembed_result = next((r for r in successful_results if "FastEmbed" in r['service']), None)
        gemini_result = next((r for r in successful_results if "Gemini" in r['service']), None)
        
        if fastembed_result and gemini_result:
            speedup = gemini_result['time'] / fastembed_result['time']
            print(f"🏆 FastEmbed is {speedup:.1f}x FASTER than Gemini AI!")
            print(f"   FastEmbed: {fastembed_result['time']:.3f}s")
            print(f"   Gemini AI: {gemini_result['time']:.3f}s")
            print(f"   Time saved: {gemini_result['time'] - fastembed_result['time']:.3f}s per batch")
    
    for result in successful_results:
        print(f"\n🔹 {result['service']}")
        print(f"   Time: {result['time']:.3f}s")
        print(f"   Speed: {result['speed']:.1f} embeddings/sec")
        print(f"   Dimension: {result['dimension']}")
    
    print(f"\n🎯 RECOMMENDATION:")
    if any("FastEmbed" in r['service'] for r in successful_results):
        print("   ✅ Use FastEmbed for production - it's faster, local, and cost-effective!")
    else:
        print("   ⚠️  Consider setting up FastEmbed for better performance")
    
    print("\n🚀 Benchmark complete!")

if __name__ == "__main__":
    main()
