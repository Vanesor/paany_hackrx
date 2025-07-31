import numpy as np
import asyncio
import google.generativeai as genai
from config import GOOGLE_API_KEY, EMBEDDING_MODEL_NAME

genai.configure(api_key=GOOGLE_API_KEY)

class VectorStore:
    def __init__(self):
        self.model_name = EMBEDDING_MODEL_NAME
        self.index = None  # This will now be a NumPy array
        self.chunks = []
        print(f"VectorStore configured for Google AI model: {self.model_name}")

    async def build_index(self, chunks: list[str]):
        """Asynchronously creates embeddings and builds the NumPy index."""
        self.chunks = chunks
        print(f"Generating embeddings for {len(chunks)} chunks...")
        
        try:
            result = await asyncio.to_thread(
                genai.embed_content,
                model=self.model_name,
                content=self.chunks,
                task_type="retrieval_document"
            )
            embeddings = np.array(result['embedding'], dtype=np.float32)
            
            # Normalize embeddings for efficient cosine similarity
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            self.index = embeddings / norms
            print("NumPy index built successfully.")
            
        except Exception as e:
            print(f"Error getting Gemini embeddings: {e}")

    def search(self, query: str, k: int) -> list[str]:
        """Searches the index for the most relevant chunks using NumPy."""
        if self.index is None:
            return []
            
        # Embed the query
        result = genai.embed_content(
            model=self.model_name,
            content=query,
            task_type="retrieval_query"
        )
        query_embedding = np.array(result['embedding'], dtype=np.float32)
        
        # Normalize the query embedding
        query_norm = np.linalg.norm(query_embedding)
        normalized_query = query_embedding / query_norm
        
        # Compute cosine similarity (dot product of normalized vectors)
        similarities = np.dot(self.index, normalized_query)
        
        # Get the indices of the top k most similar chunks
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        
        return [self.chunks[i] for i in top_k_indices]