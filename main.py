from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, HttpUrl
import uvicorn
import dotenv
from vector_store import VectorStore
from document_processor import get_document_from_url, get_text_chunks
from llm_handler import generate_answer
from config import TOP_K_RESULTS, TOKEN

app = FastAPI(
    title="Intelligent Query-Retrieval System",
    description="An API to answer questions about documents using RAG and Gemini."
)

class QueryRequest(BaseModel):
    documents: HttpUrl
    questions: list[str]

class QueryResponse(BaseModel):
    answers: list[str]

# redis?
cache = {}

@app.post("/api/v1/hackrx/run", response_model=QueryResponse, tags=["Query System"])
async def run_submission(request: QueryRequest, authorization: str = Header(None)):
    """
    Processes a document and answers a list of questions about it.
    This endpoint implements a "process once, query many" strategy for speed.
    """
    if not authorization or authorization != f"Bearer {TOKEN}":
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    doc_url = str(request.documents)

    if doc_url not in cache:
        print(f"Document not in cache. Processing: {doc_url}")
        document_text = get_document_from_url(doc_url)
        if not document_text:
            raise HTTPException(status_code=400, detail="Failed to retrieve or process the document.")
        
        chunks = get_text_chunks(document_text)
        
        vector_store = VectorStore()
        vector_store.build_index(chunks)
        cache[doc_url] = vector_store
    else:
        print(f"Found document in cache. Using existing vector store for: {doc_url}")
        vector_store = cache[doc_url]

    final_answers = []
    for question in request.questions:
        context_chunks = vector_store.search(question, k=TOP_K_RESULTS)
        context_str = "\n---\n".join(context_chunks)
        
        answer = generate_answer(context=context_str, question=question)
        final_answers.append(answer)

    return QueryResponse(answers=final_answers)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)