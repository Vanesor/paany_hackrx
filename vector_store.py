import numpy as np
import faiss
import asyncio
import google.generativeai as genai
from config import PRIMARY_API_KEY, BACKUP_API_KEY, EMBEDDING_MODEL_NAME

class VectorStore:
    def __init__(self):
        self.model_name = EMBEDDING_MODEL_NAME
        self.index = None
        self.chunks = []
        print(f"VectorStore configured for Google AI model: {self.model_name}")

    async def _embed_content_with_fallback(self, content, task_type):
        """Internal method to handle embedding with primary and backup keys."""
        try:
            genai.configure(api_key=PRIMARY_API_KEY)
            return await asyncio.to_thread(
                genai.embed_content,
                model=self.model_name,
                content=content,
                task_type=task_type
            )
        except Exception as e:
            print(f"Primary API key failed for embedding: {e}. Trying backup key.")
            if BACKUP_API_KEY:
                try:
                    genai.configure(api_key=BACKUP_API_KEY)
                    return await asyncio.to_thread(
                        genai.embed_content,
                        model=self.model_name,
                        content=content,
                        task_type=task_type
                    )
                except Exception as e2:
                    print(f"Backup API key also failed for embedding: {e2}")
                    raise e2 
            raise e

    async def build_index(self, chunks: list[str]):
        """Asynchronously creates embeddings and builds the NumPy index."""
        self.chunks = chunks
        print(f"Generating embeddings for {len(chunks)} chunks...")
        
        result = await self._embed_content_with_fallback(self.chunks, "retrieval_document")
        embeddings = np.array(result['embedding'], dtype=np.float32)
        
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        self.index = embeddings / norms
        print("NumPy index built successfully.")

    def search(self, query: str, k: int) -> list[str]:
        """Searches the index for the most relevant chunks using NumPy."""
        result = genai.embed_content(
            model=self.model_name,
            content=query,
            task_type="retrieval_query"
        ) 
        
        query_embedding = np.array(result['embedding'], dtype=np.float32)
        query_norm = np.linalg.norm(query_embedding)
        normalized_query = query_embedding / query_norm
        
        similarities = np.dot(self.index, normalized_query)
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        
        return [self.chunks[i] for i in top_k_indices]