from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, HttpUrl, Field
import uvicorn
import time
import asyncio
import hashlib 
from vector_store import VectorStore
from document_processor import get_document_from_url, get_text_chunks
from llm_handler import generate_answer
from config import TOP_K_RESULTS, TOKEN
from cache import get_cached_data, set_cached_data, redis_client

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
    if not authorization or authorization != f"Bearer {TOKEN}":
        raise HTTPException(status_code=401, detail="Unauthorized")
    doc_url = str(request.documents)
    
    vector_store = VectorStore()
    cached_data = get_cached_data(doc_url)

    if cached_data:
        # Unpack the cached index and chunks
        vector_store.index, vector_store.chunks = cached_data
    else:
        print(f"Vector store not in cache. Processing: {doc_url}")
        document_text = await get_document_from_url(doc_url)
        if not document_text:
            raise HTTPException(status_code=400, detail="Failed to retrieve or process the document.")
        
        chunks = get_text_chunks(document_text)
        await vector_store.build_index(chunks)
        
        # Cache the NumPy index and the list of chunks together
        if vector_store.index is not None:
            data_to_cache = (vector_store.index, vector_store.chunks)
            set_cached_data(doc_url, data_to_cache)

    tasks = []
    final_answers = {} 

    for i, question in enumerate(request.questions):
        cache_key = hashlib.md5((doc_url + question).encode()).hexdigest()
        
        cached_answer = redis_client.get(cache_key)

        if cached_answer:
            print(f"Found final answer in cache for: '{question}'")
            final_answers[i] = cached_answer.decode('utf-8')
        else:
            async def get_answer_task(q, key, index):
                context_chunks = vector_store.search(q, k=TOP_K_RESULTS)
                context_str = "\n---\n".join(context_chunks)
                answer = await generate_answer(context=context_str, question=q)
                redis_client.setex(key, 3600, answer)
                final_answers[index] = answer
            tasks.append(get_answer_task(question, cache_key, i))

    if tasks:
        await asyncio.gather(*tasks)

    ordered_answers = [final_answers[i] for i in sorted(final_answers.keys())]

    end_time = time.time()
    print(f"Time of execution: {(end_time - start_time):.2f} seconds")
    return QueryResponse(answers=ordered_answers)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)