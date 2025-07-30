from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, HttpUrl
import uvicorn
import time
import asyncio 
from vector_store import VectorStore
from document_processor import get_document_from_url, get_text_chunks
from llm_handler import generate_answer
from config import TOP_K_RESULTS, TOKEN
from cache import get_vector_store, set_vector_store

app = FastAPI(
    title="Intelligent Query-Retrieval System",
    description="An API to answer questions about documents using RAG, Gemini, and Redis."
)

class QueryRequest(BaseModel):
    documents: HttpUrl
    questions: list[str]

class QueryResponse(BaseModel):
    answers: list[str]

@app.post("/api/v1/hackrx/run", response_model=QueryResponse, tags=["Query System"])
async def run_submission(request: QueryRequest, authorization: str = Header(None)):
    start_time = time.time()
    print("\n[API] Starting new request processing")
    print(f"[API] Received questions: {len(request.questions)}")
    
    if not authorization or authorization != f"Bearer {TOKEN}":
        print("[API] Authorization failed")
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    doc_url = str(request.documents)
    print(f"[API] Processing document URL: {doc_url}")
    
    vector_store = get_vector_store(doc_url)
    print(f"[API] Cache hit: {vector_store is not None}")

    if not vector_store:
        print(f"[API] Cache miss. Processing document: {doc_url}")
        document_text = get_document_from_url(doc_url)
        if not document_text:
            print("[API] Failed to retrieve document")
            raise HTTPException(status_code=400, detail="Failed to retrieve or process the document.")
        
        chunks = get_text_chunks(document_text)
        
        vector_store = VectorStore()
        vector_store.build_index(chunks)
        
        set_vector_store(doc_url, vector_store)

    async def get_answer_for_question(question: str):
        """A helper coroutine to process a single question."""
        print(f"\n[API] Processing question: {question[:100]}...")
        context_chunks = vector_store.search(question, k=TOP_K_RESULTS)
        print(f"[API] Retrieved {len(context_chunks)} context chunks")
        context_str = "\n---\n".join(context_chunks)
        print(f"[API] Total context length: {len(context_str)} characters")
        answer = await generate_answer(context=context_str, question=question)
        print(f"[API] Generated answer length: {len(answer)} characters")
        return answer

    print(f"[API] Creating {len(request.questions)} parallel tasks")
    tasks = [get_answer_for_question(q) for q in request.questions]
    
    print("[API] Gathering answers from all tasks")
    final_answers = await asyncio.gather(*tasks)

    end_time = time.time()
    print(f"Time of execution: {(end_time - start_time):.2f} seconds")
    return QueryResponse(answers=final_answers)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)