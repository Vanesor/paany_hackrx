# main.py
from fastapi import FastAPI, Header, HTTPException, status, Request
from pydantic import BaseModel, HttpUrl, Field
import uvicorn
import time
import asyncio
import hashlib 
import redis
import logging

from vector_store import VectorStore
from document_processor import get_document_from_url, get_text_chunks
from llm_handler import generate_answer
from reranker import reranker_instance
from config import TOP_K_RESULTS, HYBRID_SEARCH_ALPHA, TOKEN, PORT
from cache import get_cached_data, set_cached_data, redis_client

# Get logger for this module
logger = logging.getLogger("main")

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

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Middleware for request logging and timing."""
    start_time = time.time()
    request_id = hashlib.md5(f"{time.time()}{request.client.host}".encode()).hexdigest()[:8]
    
    logger.info(f"[{request_id}] 🚀 {request.method} {request.url.path} - Request started from {request.client.host}")
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    logger.info(f"[{request_id}] ✅ {request.method} {request.url.path} - Completed in {process_time:.3f}s - Status: {response.status_code}")
    
    return response

@app.get("/api/health", response_model=HealthCheckResponse, tags=["Monitoring"])
async def health_check():
    start_time = time.time()
    logger.info("🏥 Health check requested")
    
    try:
        redis_client.ping()
        redis_time = time.time() - start_time
        logger.info(f"✅ Redis connection healthy - Response time: {redis_time:.3f}s")
        return {"status": "ok", "message": "Service is running and Redis is connected"}
    except redis.exceptions.ConnectionError as e:
        error_time = time.time() - start_time
        logger.error(f"❌ Redis connection failed after {error_time:.3f}s: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service is running, but Redis is unavailable: {e}"
        )

async def enhance_query(original_query: str) -> str:
    """Uses the LLM to rephrase a query for better search results."""
    start_time = time.time()
    logger.debug(f"🔍 Enhancing query: {original_query[:50]}...")
    
    enhancement_prompt = f'Rephrase and expand the following user query to be more specific and comprehensive for searching through a technical document. Focus on key terms and concepts. Original query: "{original_query}"'
    enhanced = await generate_answer(context="No context available.", question=enhancement_prompt)
    enhanced = enhanced.strip().replace('"', '')
    
    enhancement_time = time.time() - start_time
    logger.info(f"✨ Query enhanced in {enhancement_time:.3f}s: '{original_query}' → '{enhanced[:50]}...'")
    return enhanced

@app.post("/api/v1/hackrx/run", response_model=QueryResponse, tags=["Query System"])
async def run_submission(request: QueryRequest, authorization: str = Header(None)):
    overall_start_time = time.time()
    request_id = hashlib.md5(f"{time.time()}{str(request.documents)}".encode()).hexdigest()[:8]
    
    logger.info(f"[{request_id}] 📝 New request received with {len(request.questions)} questions")
    
    if not authorization or authorization != f"Bearer {TOKEN}":
        logger.warning(f"[{request_id}] 🚫 Unauthorized access attempt")
        raise HTTPException(status_code=401, detail="Unauthorized")

    doc_url = str(request.documents)
    logger.info(f"[{request_id}] 📄 Processing document URL: {doc_url}")
    
    # Initialize vector store
    vector_store_start = time.time()
    vector_store = VectorStore()
    logger.debug(f"[{request_id}] 🏗️ Vector store initialized in {time.time() - vector_store_start:.3f}s")
    
    # Check cache
    cache_start = time.time()
    cached_data = get_cached_data(doc_url)
    cache_time = time.time() - cache_start
    
    if cached_data:
        logger.info(f"[{request_id}] 💾 Cache HIT - Data loaded in {cache_time:.3f}s")
        load_start = time.time()
        vector_store.index, vector_store.chunks, vector_store.bm25 = cached_data
        load_time = time.time() - load_start
        logger.debug(f"[{request_id}] 📊 Vector store loaded from cache in {load_time:.3f}s - {len(vector_store.chunks)} chunks")
    else:
        logger.info(f"[{request_id}] 💾 Cache MISS - Will process document from scratch")
        
        # Download document
        doc_start = time.time()
        document_text = await get_document_from_url(doc_url)
        doc_time = time.time() - doc_start
        
        if not document_text:
            logger.error(f"[{request_id}] ❌ Failed to retrieve document after {doc_time:.3f}s")
            raise HTTPException(status_code=400, detail="Failed to retrieve or process the document.")
        
        logger.info(f"[{request_id}] 📥 Document downloaded in {doc_time:.3f}s - {len(document_text)} characters")
        
        # Create chunks
        chunk_start = time.time()
        chunks = get_text_chunks(document_text)
        chunk_time = time.time() - chunk_start
        logger.info(f"[{request_id}] ✂️ Text chunked in {chunk_time:.3f}s - Created {len(chunks)} chunks")
        
        # Build index
        index_start = time.time()
        await vector_store.build_index(chunks)
        index_time = time.time() - index_start
        logger.info(f"[{request_id}] 🔍 Index built in {index_time:.3f}s")
        
        # Cache results
        if vector_store.index is not None:
            cache_save_start = time.time()
            data_to_cache = (vector_store.index, vector_store.chunks, vector_store.bm25)
            set_cached_data(doc_url, data_to_cache)
            cache_save_time = time.time() - cache_save_start
            logger.info(f"[{request_id}] 💾 Data cached in {cache_save_time:.3f}s")

    # Process questions
    questions_start = time.time()
    logger.info(f"[{request_id}] 🤔 Processing {len(request.questions)} questions")

    async def get_answer_for_question(question: str, question_index: int):
        q_start = time.time()
        logger.debug(f"[{request_id}] Question {question_index + 1}: {question[:100]}...")
        
        cache_key = hashlib.md5((doc_url + question).encode()).hexdigest()
        
        # Check for cached answer
        cached_answer = redis_client.get(cache_key)
        if cached_answer:
            q_time = time.time() - q_start
            logger.info(f"[{request_id}] 💾 Question {question_index + 1} - Cache HIT in {q_time:.3f}s")
            return cached_answer.decode('utf-8')

        logger.debug(f"[{request_id}] 💾 Question {question_index + 1} - Cache MISS, processing...")
        
        # Enhance query
        enhance_start = time.time()
        enhanced_question = await enhance_query(question)
        enhance_time = time.time() - enhance_start
        logger.debug(f"[{request_id}] Question {question_index + 1} - Enhanced in {enhance_time:.3f}s")
        
        # Search for relevant chunks
        search_start = time.time()
        retrieved_chunks = await vector_store.search(enhanced_question, k=TOP_K_RESULTS, alpha=HYBRID_SEARCH_ALPHA)
        search_time = time.time() - search_start
        logger.debug(f"[{request_id}] Question {question_index + 1} - Search completed in {search_time:.3f}s - Found {len(retrieved_chunks)} chunks")
        
        # Rerank chunks
        rerank_start = time.time()
        reranked_chunks = reranker_instance.rerank(enhanced_question, retrieved_chunks, top_k=3)
        rerank_time = time.time() - rerank_start
        logger.debug(f"[{request_id}] Question {question_index + 1} - Reranking completed in {rerank_time:.3f}s - Top {len(reranked_chunks)} chunks selected")
        
        context_str = "\n---\n".join(reranked_chunks)
        
        # Generate answer
        llm_start = time.time()
        answer = await generate_answer(context=context_str, question=question)
        llm_time = time.time() - llm_start
        logger.debug(f"[{request_id}] Question {question_index + 1} - Answer generated in {llm_time:.3f}s")
        
        # Cache answer
        cache_answer_start = time.time()
        redis_client.setex(cache_key, 3600, answer)
        cache_answer_time = time.time() - cache_answer_start
        logger.debug(f"[{request_id}] Question {question_index + 1} - Answer cached in {cache_answer_time:.3f}s")
        
        q_total_time = time.time() - q_start
        logger.info(f"[{request_id}] ✅ Question {question_index + 1} completed in {q_total_time:.3f}s (enhance: {enhance_time:.3f}s, search: {search_time:.3f}s, rerank: {rerank_time:.3f}s, LLM: {llm_time:.3f}s)")
        
        return answer

    # Process all questions in parallel
    parallel_start = time.time()
    tasks = [get_answer_for_question(q, i) for i, q in enumerate(request.questions)]
    final_answers = await asyncio.gather(*tasks)
    parallel_time = time.time() - parallel_start
    
    questions_total_time = time.time() - questions_start
    logger.info(f"[{request_id}] 🎯 All {len(request.questions)} questions processed in {questions_total_time:.3f}s (parallel execution: {parallel_time:.3f}s)")
    
    # Final timing
    overall_end_time = time.time()
    total_execution_time = overall_end_time - overall_start_time
    logger.info(f"[{request_id}] 🏁 REQUEST COMPLETED in {total_execution_time:.2f}s")
    logger.info(f"[{request_id}] 📊 Performance breakdown - Vector setup: {cache_time:.3f}s, Questions: {questions_total_time:.3f}s")
    
    return QueryResponse(answers=final_answers)


if __name__ == "__main__":
    logger.info(f"🚀 Starting Intelligent Query-Retrieval System on port {PORT}")
    logger.info("🔧 System features: Hybrid search, query enhancement, smart caching, reranking")
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=True)