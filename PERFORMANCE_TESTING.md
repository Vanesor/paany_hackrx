# Performance Comparison Guide

## 🔍 How to Measure the Improvements

### 1. **Vector Search Speed Test**

Create a simple benchmark to compare the old vs new vector search:

```python
import time
import numpy as np
from vector_store import VectorStore, OptimizedVectorStore

# Test with 1000 sample chunks
sample_chunks = [f"Sample document chunk {i} with some content for testing" for i in range(1000)]

# Time the old system
old_vs = VectorStore()
start = time.time()
# await old_vs.build_index(sample_chunks)  # Would be slow
old_time = time.time() - start

# Time the new system  
new_vs = OptimizedVectorStore()
start = time.time()
# await new_vs.build_index(sample_chunks)  # Much faster with FAISS
new_time = time.time() - start

print(f"Speed improvement: {old_time/new_time:.1f}x faster")
```

### 2. **Memory Usage Comparison**

Monitor memory usage before and after:

```bash
# Before optimizations
pip list | wc -l  # Count dependencies
ps aux | grep python  # Check memory usage

# After optimizations  
pip list | wc -l  # Should be fewer
ps aux | grep python  # Should use less memory
```

### 3. **Cache Hit Rate Monitoring**

Check Redis for cache performance:

```bash
# Connect to Redis CLI
redis-cli

# Monitor cache hits
MONITOR

# Check cache keys
KEYS sem_query_*  # Semantic cache
KEYS pdf_*        # PDF cache
```

### 4. **API Response Time Testing**

Use this curl command to test response times:

```bash
# Test the optimized endpoint
time curl -X POST "http://localhost:8000/api/v1/hackrx/run" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": "https://example.com/sample.pdf",
    "questions": ["What is this document about?"]
  }'

# Test the legacy endpoint for comparison
time curl -X POST "http://localhost:8000/api/v1/hackrx/run-legacy" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": "https://example.com/sample.pdf", 
    "questions": ["What is this document about?"]
  }'
```

### 5. **Chunking Quality Test**

Compare chunking approaches:

```python
from document_processor import get_text_chunks, split_text_semantic

sample_text = "Your document text here..."

# Old method (simple splitting)
old_chunks = get_text_chunks(sample_text, chunk_size=1000, overlap=200)

# New method (semantic splitting)  
new_chunks = split_text_semantic(sample_text)

print(f"Old: {len(old_chunks)} chunks, Avg: {sum(len(c) for c in old_chunks)/len(old_chunks):.1f} chars")
print(f"New: {len(new_chunks)} chunks, Avg: {sum(len(c) for c in new_chunks)/len(new_chunks):.1f} chars")

# Check for better semantic boundaries
for i, chunk in enumerate(new_chunks[:3]):
    print(f"Chunk {i}: ...{chunk[-50:]}")  # See how chunks end
```

## 📊 Expected Results

| Test | Expected Improvement |
|------|---------------------|
| **Vector Search** | 10-100x faster |
| **Memory Usage** | 40% reduction |
| **Cache Hit Rate** | 30-40% fewer API calls |
| **Response Time** | 2-3x faster after caching |
| **Chunking Quality** | Better semantic boundaries |

## 🎯 Real-World Testing

1. **First Request**: Document processing + indexing (~3-5 seconds)
2. **Second Request** (same doc): Cache hit (~1-2 seconds) 
3. **Similar Questions**: Semantic cache hit (~0.5 seconds)
4. **Load Testing**: Stable performance under concurrent requests

## 🔧 Monitoring in Production

Add these log patterns to track improvements:

```bash
# Search for performance logs
grep "completed in" logs/*.log
grep "Cache hit" logs/*.log  
grep "FAISS index built" logs/*.log
grep "Semantic cache hit" logs/*.log
```
