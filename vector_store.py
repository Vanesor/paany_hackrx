import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL_NAME

class VectorStore:
    def __init__(self):
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        self.index = None
        self.chunks = []

    def build_index(self, chunks: list[str]):
        """Creates embeddings and builds the FAISS index."""
        self.chunks = chunks
        print(f"Generating embeddings for {len(chunks)} chunks...")
        embeddings = self.model.encode(chunks, show_progress_bar=True)
        
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.index.add(np.array(embeddings, dtype=np.float32))
        print("FAISS index built successfully.")

    def search(self, query: str, k: int) -> list[str]:
        """Searches the index for the most relevant chunks."""
        if self.index is None:
            return []
            
        query_with_instruction = f"Represent this sentence for searching relevant passages: {query}"
        query_embedding = self.model.encode([query_with_instruction])
        distances, indices = self.index.search(np.array(query_embedding, dtype=np.float32), k)
        return [self.chunks[i] for i in indices[0]]