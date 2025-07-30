# Gemini 2.5 Flash RAG System - Action Plan

## Project Overview
Build an LLM-powered intelligent query-retrieval system using Gemini 2.5 Flash and Gemini Embedding models for insurance, legal, HR, and compliance document analysis.

## Phase 1: Environment Setup & Project Initialization (Days 1-2)

### 1.1 Development Environment
```bash
# Create project structure
mkdir gemini-rag-system
cd gemini-rag-system
mkdir -p {src,tests,docs,data,config}

# Python environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# Install core dependencies
pip install fastapi uvicorn pydantic google-generativeai
pip install pinecone-client sentence-transformers
pip install PyMuPDF python-docx python-email-validator
pip install asyncio aiofiles python-multipart
```

### 1.2 API Keys & Configuration
```bash
# Create .env file
touch .env

# Add required API keys
GOOGLE_API_KEY=your_gemini_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENVIRONMENT=your_environment
HACKRX_API_TOKEN=6e8b43cca9d29b261843a3b1c53382bdaa5b2c9e96db92da679278c6dc0042ca
```

### 1.3 Project Structure Setup
```
src/
├── core/
│   ├── __init__.py
│   ├── config.py
│   └── database.py
├── models/
│   ├── __init__.py
│   ├── schemas.py
│   └── embeddings.py
├── services/
│   ├── __init__.py
│   ├── document_processor.py
│   ├── embedding_service.py
│   ├── retrieval_service.py
│   └── llm_service.py
├── api/
│   ├── __init__.py
│   ├── routes.py
│   └── dependencies.py
└── main.py
```

## Phase 2: Core Components Development (Days 3-5)

### 2.1 Configuration Management
```python
# src/core/config.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    google_api_key: str
    pinecone_api_key: str
    pinecone_environment: str
    hackrx_api_token: str
    
    class Config:
        env_file = ".env"

settings = Settings()
```

### 2.2 Document Processing Service
```python
# src/services/document_processor.py
import PyMuPDF
from docx import Document
import aiohttp
from typing import List, Dict

class DocumentProcessor:
    def __init__(self):
        self.chunk_size = 1000
        self.chunk_overlap = 200
    
    async def process_document_from_url(self, url: str) -> List[Dict]:
        # Download document
        # Parse based on file type
        # Chunk into segments
        # Return structured chunks with metadata
        pass
```

### 2.3 Embedding Service with Gemini
```python
# src/services/embedding_service.py
import google.generativeai as genai
from typing import List
import numpy as np

class GeminiEmbeddingService:
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = 'models/text-embedding-004'
    
    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            result = genai.embed_content(
                model=self.model,
                content=text,
                task_type="retrieval_document"
            )
            embeddings.append(result['embedding'])
        return embeddings
    
    async def embed_query(self, query: str) -> List[float]:
        result = genai.embed_content(
            model=self.model,
            content=query,
            task_type="retrieval_query"
        )
        return result['embedding']
```

### 2.4 Vector Database Setup
```python
# src/services/retrieval_service.py
import pinecone
from typing import List, Dict

class PineconeRetrieval:
    def __init__(self, api_key: str, environment: str):
        pinecone.init(api_key=api_key, environment=environment)
        self.index_name = "gemini-rag-index"
        self.dimension = 768  # Gemini embedding dimension
        
    def create_index(self):
        if self.index_name not in pinecone.list_indexes():
            pinecone.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric="cosine"
            )
        self.index = pinecone.Index(self.index_name)
    
    async def upsert_documents(self, documents: List[Dict]):
        # Batch upsert documents with embeddings
        pass
    
    async def search(self, query_embedding: List[float], top_k: int = 10):
        # Search similar documents
        pass
```

### 2.5 LLM Service with Gemini 2.5 Flash
```python
# src/services/llm_service.py
import google.generativeai as genai
import json
from typing import List, Dict

class GeminiLLMService:
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
    async def generate_answer(self, query: str, context: List[str]) -> Dict:
        prompt = self._create_structured_prompt(query, context)
        
        response = self.model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.1,
                max_output_tokens=1000,
                response_mime_type="application/json"
            )
        )
        
        return json.loads(response.text)
    
    def _create_structured_prompt(self, query: str, context: List[str]) -> str:
        context_text = "\n\n".join([f"Context {i+1}: {ctx}" for i, ctx in enumerate(context)])
        
        return f"""
        You are an expert insurance and legal document analyzer. Based on the provided context, answer the query with precision and cite your sources.

        Query: {query}

        Context:
        {context_text}

        Provide your response in the following JSON format:
        {{
            "answer": "Direct, comprehensive answer to the query",
            "confidence": 0.95,
            "supporting_evidence": ["specific clause 1", "specific clause 2"],
            "document_sources": ["source1", "source2"],
            "reasoning": "Step-by-step explanation of how you arrived at the answer"
        }}
        """
```

## Phase 3: API Development (Days 6-7)

### 3.1 Pydantic Models
```python
# src/models/schemas.py
from pydantic import BaseModel
from typing import List, Optional

class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

class DetailedResponse(BaseModel):
    answer: str
    confidence: float
    supporting_evidence: List[str]
    document_sources: List[str]
    reasoning: str
```

### 3.2 FastAPI Routes
```python
# src/api/routes.py
from fastapi import APIRouter, Depends, HTTPException
from ..models.schemas import QueryRequest, QueryResponse
from ..services import DocumentProcessor, GeminiEmbeddingService, PineconeRetrieval, GeminiLLMService

router = APIRouter(prefix="/hackrx", tags=["hackrx"])

@router.post("/run", response_model=QueryResponse)
async def process_queries(request: QueryRequest):
    try:
        # 1. Process documents from URL
        doc_processor = DocumentProcessor()
        chunks = await doc_processor.process_document_from_url(request.documents)
        
        # 2. Generate embeddings and store in Pinecone
        embedding_service = GeminiEmbeddingService(settings.google_api_key)
        retrieval_service = PineconeRetrieval(settings.pinecone_api_key, settings.pinecone_environment)
        
        # Index documents if not already done
        await index_documents(chunks, embedding_service, retrieval_service)
        
        # 3. Process each query
        llm_service = GeminiLLMService(settings.google_api_key)
        answers = []
        
        for question in request.questions:
            # Get query embedding
            query_embedding = await embedding_service.embed_query(question)
            
            # Retrieve relevant context
            similar_docs = await retrieval_service.search(query_embedding, top_k=5)
            context = [doc['metadata']['text'] for doc in similar_docs['matches']]
            
            # Generate answer using Gemini 2.5 Flash
            response = await llm_service.generate_answer(question, context)
            answers.append(response['answer'])
        
        return QueryResponse(answers=answers)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### 3.3 Main Application
```python
# src/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .api.routes import router
from .core.config import settings

app = FastAPI(
    title="Gemini RAG System",
    description="LLM-powered intelligent query-retrieval system",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "gemini-rag-system"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

## Phase 4: Optimization & Testing (Days 8-9)

### 4.1 Performance Optimization
- Implement async batch processing for embeddings
- Add caching layer with Redis for frequent queries
- Optimize chunk size and overlap parameters
- Implement connection pooling for Pinecone

### 4.2 Testing Framework
```python
# tests/test_api.py
import pytest
from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_hackrx_run():
    payload = {
        "documents": "test_document_url",
        "questions": ["Test question?"]
    }
    response = client.post("/api/v1/hackrx/run", json=payload)
    assert response.status_code == 200
    assert "answers" in response.json()
```

### 4.3 Error Handling & Monitoring
- Add comprehensive error handling
- Implement logging with structured logs
- Add metrics collection for performance monitoring
- Create health checks for all dependencies

## Phase 5: Deployment & Production (Days 10-11)

### 5.1 Docker Configuration
```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY .env ./

EXPOSE 8000

CMD ["python", "-m", "src.main"]
```

### 5.2 Docker Compose
```yaml
# docker-compose.yml
version: '3.8'
services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - GOOGLE_API_KEY=${GOOGLE_API_KEY}
      - PINECONE_API_KEY=${PINECONE_API_KEY}
    depends_on:
      - redis
  
  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
```

### 5.3 Production Deployment
- Set up CI/CD pipeline
- Configure load balancing
- Implement auto-scaling
- Add monitoring and alerting

## Phase 6: Testing & Evaluation (Days 12-14)

### 6.1 Unit Testing
- Test individual components
- Mock external API calls
- Validate embedding generation
- Test document processing

### 6.2 Integration Testing
- End-to-end API testing
- Performance benchmarking
- Load testing with concurrent requests
- Error scenario testing

### 6.3 Accuracy Evaluation
- Test with known documents
- Validate against expected answers
- Measure precision and recall
- Optimize retrieval parameters

## Key Implementation Notes

### Token Efficiency Strategies
- Implement intelligent context truncation
- Use hierarchical summarization for long documents
- Optimize prompt templates for Gemini 2.5 Flash
- Implement caching for repeated queries

### Accuracy Improvements
- Fine-tune chunk sizes for document types
- Implement query expansion techniques
- Add metadata filtering for better retrieval
- Use confidence thresholds for answer quality

### Latency Optimization
- Implement async processing throughout
- Use connection pooling
- Cache embeddings for repeated documents
- Optimize Pinecone index configuration

## Success Metrics

- **Accuracy**: >90% correct answers on test dataset
- **Latency**: <3 seconds average response time
- **Token Efficiency**: <2000 tokens per query on average
- **Throughput**: Handle 100+ concurrent requests
- **Reliability**: 99.9% uptime with proper error handling

## Troubleshooting Guide

### Common Issues
1. **Embedding API Rate Limits**: Implement exponential backoff
2. **Pinecone Connection Errors**: Add retry logic with circuit breaker
3. **Memory Issues**: Optimize batch processing sizes
4. **Timeout Errors**: Adjust request timeout configurations

### Monitoring
- Track API response times
- Monitor embedding generation time
- Log retrieval accuracy metrics
- Alert on error rates exceeding thresholds

This action plan provides a comprehensive roadmap for implementing the Gemini 2.5 Flash-powered RAG system, with specific focus on the insurance/legal document analysis requirements and evaluation criteria.