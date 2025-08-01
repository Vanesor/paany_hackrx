# main.py
from fastapi import FastAPI, Header, HTTPException, status
from pydantic import BaseModel, HttpUrl, Field
import uvicorn
import time
import asyncio
import hashlib 
import redis
import logging
import uuid

from vector_store import VectorStore
from document_processor import get_document_from_url, get_text_chunks
from llm_handler import generate_answer
from config import TOP_K_RESULTS, HYBRID_SEARCH_ALPHA, TOKEN, PORT
from cache import get_cached_data, set_cached_data, redis_client

# Initialize logger
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Intelligent Query-Retrieval System",
    description="A high-accuracy, asynchronous API for document Q&A."
)

class QueryRequest(BaseModel):
    documents: HttpUrl
    questions: list[str]

class QueryResponse(BaseModel):
    answers: list[str]

class HealthCheckResponse(BaseModel):
    status: str = Field(..., example="ok")
    message: str = Field(..., example="Service is running")

@app.get("/api/health", response_model=HealthCheckResponse, tags=["Monitoring"])
async def health_check():
    try:
        redis_client.ping()
        return {"status": "ok", "message": "Service is running and Redis is connected"}
    except redis.exceptions.ConnectionError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service is running, but Redis is unavailable: {e}"
        )

async def enhance_query(original_query: str) -> str:
    """Uses the LLM to rephrase a query for better search results."""
    enhancement_prompt = f'Rephrase and expand the following user query to be more specific and comprehensive for searching through a technical document. Focus on key terms and concepts. Original query: "{original_query}"'
    enhanced = await generate_answer(context="No context available.", question=enhancement_prompt)
    enhanced = enhanced.strip().replace('"', '')
    print(f"Original Query: '{original_query}' | Enhanced Query: '{enhanced}'")
    return enhanced

@app.post("/api/v1/hackrx/run", response_model=QueryResponse, tags=["Query System"])
async def run_submission(request: QueryRequest, authorization: str = Header(None)):
    request_id = str(uuid.uuid4())[:8]  # Short request ID for tracking
    start_time = time.time()
    
    logger.info(f"[Request {request_id}] 📥 Starting new request with {len(request.questions)} questions")
    
    if not authorization or authorization != f"Bearer {TOKEN}":
        logger.warning(f"[Request {request_id}] ❌ Unauthorized access attempt")
        raise HTTPException(status_code=401, detail="Unauthorized")

    doc_url = str(request.documents)
    logger.info(f"[Request {request_id}] 📄 Processing document: {doc_url[:50]}...")
    
    vector_store = VectorStore()
    cached_data = get_cached_data(doc_url)

    if cached_data:
        logger.info(f"[Request {request_id}] ⚡ Using cached vector store")
        vector_store.index, vector_store.chunks, vector_store.bm25 = cached_data
    else:
        doc_start = time.time()
        logger.info(f"[Request {request_id}] 🔄 Downloading and processing document...")
        
        document_text = await get_document_from_url(doc_url)
        if not document_text:
            logger.error(f"[Request {request_id}] ❌ Failed to retrieve document")
            raise HTTPException(status_code=400, detail="Failed to retrieve or process the document.")
        
        doc_time = time.time() - doc_start
        logger.info(f"[Request {request_id}] ✅ Document processed in {doc_time:.2f}s")
        
        chunks = get_text_chunks(document_text)
        logger.info(f"[Request {request_id}] 📝 Split into {len(chunks)} text chunks")
        
        index_start = time.time()
        await vector_store.build_index(chunks)
        index_time = time.time() - index_start
        logger.info(f"[Request {request_id}] 🧠 Vector index built in {index_time:.2f}s")
        
        if vector_store.index is not None:
            data_to_cache = (vector_store.index, vector_store.chunks, vector_store.bm25)
            set_cached_data(doc_url, data_to_cache)
            logger.info(f"[Request {request_id}] 💾 Data cached for future requests")

    async def get_answer_for_question(question: str):
        q_start = time.time()
        cache_key = hashlib.md5((doc_url + question).encode()).hexdigest()
        cached_answer = redis_client.get(cache_key)
        if cached_answer:
            q_time = time.time() - q_start
            logger.info(f"[Request {request_id}] ⚡ Cached answer retrieved in {q_time:.3f}s")
            return cached_answer.decode('utf-8')

        # Core RAG Pipeline: Retrieve -> Generate
        search_start = time.time()
        retrieved_chunks = await vector_store.search(question, k=TOP_K_RESULTS, alpha=HYBRID_SEARCH_ALPHA)
        search_time = time.time() - search_start
        
        context_str = "\n---\n".join(retrieved_chunks)
        
        llm_start = time.time()
        answer = await generate_answer(context=context_str, question=question)
        llm_time = time.time() - llm_start
        
        q_time = time.time() - q_start
        logger.info(f"[Request {request_id}] 🔍 Question answered in {q_time:.2f}s (search: {search_time:.2f}s, LLM: {llm_time:.2f}s)")
        
        redis_client.setex(cache_key, 3600, answer)
        return answer

    logger.info(f"[Request {request_id}] 🚀 Processing {len(request.questions)} questions in parallel...")
    tasks = [get_answer_for_question(q) for q in request.questions]
    final_answers = await asyncio.gather(*tasks)
    
    total_time = time.time() - start_time
    logger.info(f"[Request {request_id}] ✅ Request completed in {total_time:.2f}s total")
    
    return QueryResponse(answers=final_answers)

if __name__ == "__main__":
    logger.info("🚀 Starting FastAPI server...")
    logger.info(f"📍 Server will run on http://0.0.0.0:{PORT}")
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=True)