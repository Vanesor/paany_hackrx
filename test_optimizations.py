#!/usr/bin/env python3
"""
Test script for the optimized RAG system components.
"""

import asyncio
import sys
import os

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(__file__))

async def test_optimizations():
    print("🧪 Testing RAG System Optimizations")
    print("=" * 50)
    
    # Test 1: Document processing optimizations
    print("\n1. Testing semantic chunking...")
    try:
        from document_processor import split_text_semantic, enhance_chunks_with_context
        
        sample_text = """
        This is a sample document. It contains multiple sentences and paragraphs.
        
        This is a new paragraph. It should be handled intelligently by the semantic chunker.
        The chunker should respect natural boundaries like sentences and paragraphs.
        
        Another paragraph here. This helps test the semantic awareness of the chunking algorithm.
        """
        
        chunks = split_text_semantic(sample_text)
        print(f"   ✅ Generated {len(chunks)} semantic chunks")
        
        enhanced = enhance_chunks_with_context(chunks)
        print(f"   ✅ Enhanced {len(enhanced)} chunks with context")
        
    except Exception as e:
        print(f"   ❌ Document processing test failed: {e}")
    
    # Test 2: FAISS vector store
    print("\n2. Testing FAISS vector store...")
    try:
        import faiss
        import numpy as np
        from vector_store import OptimizedVectorStore
        
        print(f"   ✅ FAISS version: {faiss.__version__}")
        
        # Create a simple vector store
        vs = OptimizedVectorStore(dimension=768)
        print(f"   ✅ OptimizedVectorStore created with dimension {vs.dimension}")
        
    except Exception as e:
        print(f"   ❌ FAISS vector store test failed: {e}")
    
    # Test 3: Semantic cache
    print("\n3. Testing semantic cache...")
    try:
        from cache import SemanticCache, redis_client
        
        # Test redis connection
        redis_client.ping()
        print("   ✅ Redis connection working")
        
        cache = SemanticCache(redis_client, threshold=0.92)
        print(f"   ✅ SemanticCache created with threshold {cache.threshold}")
        
    except Exception as e:
        print(f"   ❌ Semantic cache test failed: {e}")
    
    # Test 4: Main application
    print("\n4. Testing main application...")
    try:
        from main import OptimizedRAGSystem, app
        
        rag_system = OptimizedRAGSystem()
        print("   ✅ OptimizedRAGSystem created successfully")
        print(f"   ✅ FastAPI app: {app.title}")
        
    except Exception as e:
        print(f"   ❌ Main application test failed: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 Optimization testing complete!")
    print("\nNext steps:")
    print("1. Start the server: python main.py")
    print("2. Test with real documents using the /api/v1/hackrx/run endpoint")
    print("3. Monitor performance improvements in the logs")

if __name__ == "__main__":
    asyncio.run(test_optimizations())
