from fastapi import FastAPI, Header, HTTPException, status
from pydantic import BaseModel, HttpUrl, Field
import redis
import uvicorn
import time
import asyncio
import hashlib
import logging
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from vector_store import VectorStore
from document_processor import get_document_from_url, get_text_chunks
from llm_handler import generate_answer
from config import (
    TOP_K_RESULTS, TOKEN, PORT, REDIS_DISABLED, REDIS_URL,
    DENSE_WEIGHT, CHUNK_SIZE, CHUNK_OVERLAP
)
from cache import get_cached_data, set_cached_data, redis_client

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("main")

app = FastAPI(
    title="Intelligent Query-Retrieval System",
    description="An API to answer questions about documents using RAG, Gemini, and Redis."
)

class QueryRequest(BaseModel):
    documents: HttpUrl
    questions: list[str]

class QueryResponse(BaseModel):
    answers: list[str]

class HealthCheckResponse(BaseModel):
    status: str = "ok"
    message: str = "Service is running"

@app.get("/api/health", response_model=HealthCheckResponse, tags=["Monitoring"])
async def health_check():
    try:
        if not REDIS_DISABLED:
            redis_client.ping()
            return {"status": "ok", "message": "Service is running and Redis is connected"}
        return {"status": "ok", "message": "Service is running (Redis disabled)"}
    except redis.exceptions.ConnectionError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service is running, but Redis is unavailable: {e}"
        )

@app.post("/api/v1/hackrx/run", response_model=QueryResponse, tags=["Query System"])
async def run_submission(request: QueryRequest, authorization: str = Header(None)):
    start_time = time.time()
    logger.info(f"New request received with {len(request.questions)} questions")
    
    if not authorization or authorization != f"Bearer {TOKEN}":
        logger.warning("Unauthorized access attempt")
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    doc_url = str(request.documents)
    logger.info(f"Processing document URL: {doc_url}")
    
    vector_store = VectorStore()
    cached_data = None if REDIS_DISABLED else get_cached_data(doc_url)

    if cached_data:
        logger.info(f"Cache hit: Vector store found in cache")
        # Handle both old-style and new-style cache data
        if isinstance(cached_data, tuple):
            if len(cached_data) == 2:
                # Old cache format with just dense index and chunks
                vector_store.dense_index, vector_store.chunks = cached_data
                # Need to build sparse indices
                logger.info("Building sparse indices for cached data...")
                vector_store.tokenized_chunks = [vector_store._preprocess_text(chunk) for chunk in vector_store.chunks]
                vector_store.sparse_index = BM25Okapi(vector_store.tokenized_chunks)
                vector_store.tfidf_vectorizer = TfidfVectorizer(min_df=2, max_df=0.95, ngram_range=(1, 2))
                vector_store.tfidf_matrix = vector_store.tfidf_vectorizer.fit_transform(vector_store.chunks)
                vector_store.chunk_metadata = [vector_store._extract_metadata(chunk) for chunk in vector_store.chunks]
            elif len(cached_data) == 5:
                # Old hybrid format (without metadata)
                vector_store.dense_index, vector_store.chunks, vector_store.sparse_index, vector_store.tokenized_chunks, vector_store.tfidf_vectorizer = cached_data
                vector_store.tfidf_matrix = vector_store.tfidf_vectorizer.transform(vector_store.chunks)
                vector_store.chunk_metadata = [vector_store._extract_metadata(chunk) for chunk in vector_store.chunks]
            elif len(cached_data) == 6:
                # New format with all indices and metadata
                vector_store.dense_index, vector_store.chunks, vector_store.sparse_index, vector_store.tokenized_chunks, vector_store.tfidf_vectorizer, vector_store.tfidf_matrix = cached_data
                vector_store.chunk_metadata = [vector_store._extract_metadata(chunk) for chunk in vector_store.chunks]
            elif len(cached_data) == 7:
                # Complete format with all data
                vector_store.dense_index, vector_store.chunks, vector_store.sparse_index, vector_store.tokenized_chunks, vector_store.tfidf_vectorizer, vector_store.tfidf_matrix, vector_store.chunk_metadata = cached_data
        
        logger.debug(f"Loaded {len(vector_store.chunks)} chunks from cache")
    else:
        logger.info(f"Cache miss: Vector store not in cache. Processing document...")
        document_text = await get_document_from_url(doc_url)
        if not document_text:
            logger.error(f"Failed to retrieve or process document: {doc_url}")
            raise HTTPException(status_code=400, detail="Failed to retrieve or process the document.")
        
        logger.info("Creating text chunks from document")
        chunks = get_text_chunks(document_text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
        logger.info(f"Generated {len(chunks)} chunks, building vector index...")
        
        await vector_store.build_index(chunks)
        logger.debug("Vector indices built successfully")
        
        # Cache all indices for future use
        if not REDIS_DISABLED and vector_store.dense_index is not None and vector_store.sparse_index is not None:
            logger.info("Caching vector store for future use")
            data_to_cache = (
                vector_store.dense_index, 
                vector_store.chunks,
                vector_store.sparse_index,
                vector_store.tokenized_chunks,
                vector_store.tfidf_vectorizer,
                vector_store.tfidf_matrix,
                vector_store.chunk_metadata
            )
            set_cached_data(doc_url, data_to_cache)
            logger.debug("Vector store cached successfully")

    tasks = []
    final_answers = {} 
    logger.info(f"Processing {len(request.questions)} questions")

    for i, question in enumerate(request.questions):
        logger.debug(f"Question {i+1}: {question[:100]}...")
        cache_key = hashlib.md5((doc_url + question).encode()).hexdigest()
        
        cached_answer = None if REDIS_DISABLED else redis_client.get(cache_key)

        if cached_answer:
            logger.info(f"Found answer in cache for question {i+1}")
            final_answers[i] = cached_answer.decode('utf-8')
        else:
            logger.info(f"No cached answer found for question {i+1}, processing...")
            async def get_answer_task(q, key, index):
                logger.debug(f"Starting hybrid search for question {index+1}")
                # Use hybrid search with custom weight based on question type
                # For factual questions, give more weight to sparse retrieval
                if any(term in q.lower() for term in ['what', 'when', 'where', 'who']):
                    hybrid_weight = 0.5  # Balance between keywords and semantics
                # For conceptual questions, give more weight to dense retrieval
                elif any(term in q.lower() for term in ['why', 'how', 'explain', 'describe']):
                    hybrid_weight = 0.8  # Favor semantic understanding
                else:
                    hybrid_weight = DENSE_WEIGHT  # Default from config
                
                logger.debug(f"Using hybrid_weight={hybrid_weight} for question type")
                context_chunks = await vector_store.search(q, k=TOP_K_RESULTS, hybrid_weight=hybrid_weight)
                logger.debug(f"Found {len(context_chunks)} relevant chunks for question {index+1}")
                
                context_str = "\n---\n".join(context_chunks)
                logger.debug(f"Context length for question {index+1}: {len(context_str)} chars")
                
                logger.info(f"Generating answer for question {index+1}")
                answer = await generate_answer(context=context_str, question=q)
                logger.debug(f"Answer generated for question {index+1}: {answer[:100]}...")
                
                if not REDIS_DISABLED:
                    redis_client.setex(key, 3600, answer)
                    logger.debug(f"Answer for question {index+1} cached")
                
                final_answers[index] = answer
            tasks.append(get_answer_task(question, cache_key, i))

    if tasks:
        logger.info(f"Running {len(tasks)} parallel answer generation tasks")
        await asyncio.gather(*tasks)
        logger.info(f"All answer generation tasks completed")

    ordered_answers = [final_answers[i] for i in sorted(final_answers.keys())]
    logger.info(f"Returning {len(ordered_answers)} answers to the client")

    end_time = time.time()
    execution_time = end_time - start_time
    logger.info(f"Request completed in {execution_time:.2f} seconds")
    
    return QueryResponse(answers=ordered_answers)

if __name__ == "__main__":
    logger.info(f"Starting server on port {PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT)