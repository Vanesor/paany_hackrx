import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from config import EMBEDDING_MODEL_NAME

if "embedding-001" in EMBEDDING_MODEL_NAME:
    from config import GOOGLE_API_KEY
    genai.configure(api_key=GOOGLE_API_KEY)


class VectorStore:
    def __init__(self):
        self.model_name = EMBEDDING_MODEL_NAME
        self.model = None
        self.index = None
        self.chunks = []
        self.embedding_dim = None

        if "embedding-001" in self.model_name:
            self.embedding_dim = 768
            print(f"Using Google AI embedding model: {self.model_name}")
        else:
            print(f"Loading local sentence-transformer model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()

    def _get_gemini_embeddings(self, texts: list[str]) -> np.ndarray:
        """Helper function to get embeddings from the Gemini API."""
        try:
            result = genai.embed_content(model=self.model_name,
                                         content=texts,
                                         task_type="retrieval_document")
            return np.array(result['embedding'])
        except Exception as e:
            print(f"Error getting Gemini embeddings: {e}")
            return None

    def build_index(self, chunks: list[str]):
        """Creates embeddings and builds the FAISS index."""
        print("\n[VectorStore] Starting index building process...")
        print(f"[VectorStore] Number of chunks to process: {len(chunks)}")
        print(f"[VectorStore] Average chunk length: {sum(len(c) for c in chunks)/len(chunks):.2f} characters")
        self.chunks = chunks
        print(f"[VectorStore] Generating embeddings using model: {self.model_name}")
        
        if "embedding-001" in self.model_name:
            embeddings = self._get_gemini_embeddings(self.chunks)
            if embeddings is None:
                print("Failed to generate embeddings.")
                return
        else:
            embeddings = self.model.encode(chunks, show_progress_bar=True)
        
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.index.add(np.array(embeddings, dtype=np.float32))
        print("FAISS index built successfully.")

    def search(self, query: str, k: int) -> list[str]:
        """Searches the index for the most relevant chunks."""
        if self.index is None:
            return []
            
        if "embedding-001" in self.model_name:
            result = genai.embed_content(model=self.model_name,
                                         content=query,
                                         task_type="retrieval_query")
            query_embedding = np.array([result['embedding']])
        else:
            query_with_instruction = f"Represent this sentence for searching relevant passages: {query}"
            query_embedding = self.model.encode([query_with_instruction])
        
        distances, indices = self.index.search(np.array(query_embedding, dtype=np.float32), k)
        return [self.chunks[i] for i in indices[0]]