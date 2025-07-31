from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, HttpUrl, Field
import uvicorn
import time
import asyncio
import hashlib
import logging
from vector_store import VectorStore
from document_processor import get_document_from_url, get_text_chunks
from llm_handler import generate_answer
from config import TOP_K_RESULTS, TOKEN
from cache import get_cached_data, set_cached_data, redis_client

# Configure logging
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
    status: str = Field(..., example="ok")
    message: str = Field(..., example="Service is running")

@app.get("/api/health", response_model=HealthCheckResponse, tags=["Monitoring"])
async def health_check():
    return {"status": "ok", "message": "Service is running"}

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
    cached_data = get_cached_data(doc_url)

    if cached_data:
        # Unpack the cached index and chunks
        logger.info(f"Cache hit: Vector store found in cache")
        vector_store.index, vector_store.chunks = cached_data
        logger.debug(f"Loaded {len(vector_store.chunks)} chunks from cache")
    else:
        logger.info(f"Cache miss: Vector store not in cache. Processing document...")
        document_text = await get_document_from_url(doc_url)
        if not document_text:
            logger.error(f"Failed to retrieve or process document: {doc_url}")
            raise HTTPException(status_code=400, detail="Failed to retrieve or process the document.")
        
        logger.info("Creating text chunks from document")
        chunks = get_text_chunks(document_text)
        logger.info(f"Generated {len(chunks)} chunks, building vector index...")
        
        await vector_store.build_index(chunks)
        logger.debug("Vector index built successfully")
        
        # Cache the NumPy index and the list of chunks together
        if vector_store.index is not None:
            logger.info("Caching vector store for future use")
            data_to_cache = (vector_store.index, vector_store.chunks)
            set_cached_data(doc_url, data_to_cache)
            logger.debug("Vector store cached successfully")

    tasks = []
    final_answers = {} 
    logger.info(f"Processing {len(request.questions)} questions")

    for i, question in enumerate(request.questions):
        logger.debug(f"Question {i+1}: {question[:100]}...")
        cache_key = hashlib.md5((doc_url + question).encode()).hexdigest()
        
        cached_answer = redis_client.get(cache_key)

        if cached_answer:
            logger.info(f"Found answer in cache for question {i+1}")
            final_answers[i] = cached_answer.decode('utf-8')
        else:
            logger.info(f"No cached answer found for question {i+1}, processing...")
            async def get_answer_task(q, key, index):
                logger.debug(f"Starting search for question {index+1}")
                context_chunks = vector_store.search(q, k=TOP_K_RESULTS)
                logger.debug(f"Found {len(context_chunks)} relevant chunks for question {index+1}")
                
                context_str = "\n---\n".join(context_chunks)
                logger.debug(f"Context length for question {index+1}: {len(context_str)} chars")
                
                logger.info(f"Generating answer for question {index+1}")
                answer = await generate_answer(context=context_str, question=q)
                logger.debug(f"Answer generated for question {index+1}: {answer[:100]}...")
                
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
    uvicorn.run(app, host="0.0.0.0", port=10000)