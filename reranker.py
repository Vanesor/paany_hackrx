# reranker.py
from sentence_transformers import CrossEncoder
from config import RERANKER_MODEL_NAME

class Reranker:
    def __init__(self):
        self.model = CrossEncoder(RERANKER_MODEL_NAME)
        print("Cross-encoder reranker model loaded.")

    def rerank(self, query: str, documents: list[str], top_k: int) -> list[str]:
        """Reranks documents based on their relevance to the query."""
        if not documents:
            return []
            
        pairs = [[query, doc] for doc in documents]
        scores = self.model.predict(pairs, show_progress_bar=False)
        
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        return [doc for doc, score in scored_docs[:top_k]]

reranker_instance = Reranker()