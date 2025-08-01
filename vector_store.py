# vector_store.py
import numpy as np
import time
import logging
from rank_bm25 import BM25Okapi
from llm_handler import embed_content_with_fallback

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self):
        self.index = None
        self.chunks = []
        self.bm25 = None
        logger.info("🏗️  VectorStore initialized for hybrid search.")

    async def build_index(self, chunks: list[str]):
        """Builds both NumPy (vector) and BM25 (keyword) indices."""
        start_time = time.time()
        self.chunks = chunks
        logger.info(f"📊 Building index for {len(chunks)} text chunks...")
        
        # Vector embeddings
        embed_start = time.time()
        result = await embed_content_with_fallback(self.chunks, "retrieval_document")
        embeddings = np.array(result['embedding'], dtype=np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        self.index = embeddings / norms
        embed_time = time.time() - embed_start
        logger.info(f"🧠 Vector embeddings created in {embed_time:.2f}s")
        
        # BM25 index
        bm25_start = time.time()
        tokenized_chunks = [chunk.split() for chunk in chunks]
        self.bm25 = BM25Okapi(tokenized_chunks)
        bm25_time = time.time() - bm25_start
        logger.info(f"🔍 BM25 index built in {bm25_time:.2f}s")
        
        total_time = time.time() - start_time
        logger.info(f"✅ Complete index built in {total_time:.2f}s total")

    async def search(self, query: str, k: int, alpha: float) -> list[str]:
        """Performs a hybrid search combining vector and keyword results."""
        if self.index is None or self.bm25 is None: 
            logger.warning("⚠️ Search attempted on uninitialized index")
            return []

        start_time = time.time()
        logger.debug(f"🔍 Hybrid search for: '{query[:50]}...' (k={k}, α={alpha})")

        # Vector search
        vector_start = time.time()
        result = await embed_content_with_fallback(query, "retrieval_query")
        query_embedding = np.array(result['embedding'], dtype=np.float32)
        query_norm = np.linalg.norm(query_embedding)
        normalized_query = query_embedding / query_norm
        vector_scores = np.dot(self.index, normalized_query)
        vector_time = time.time() - vector_start

        # Keyword search
        keyword_start = time.time()
        tokenized_query = query.split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_scores = bm25_scores / (bm25_scores.max() or 1.0) # Normalize
        keyword_time = time.time() - keyword_start

        # Hybrid combination
        min_len = min(len(vector_scores), len(bm25_scores))
        hybrid_scores = (alpha * vector_scores[:min_len]) + ((1 - alpha) * bm25_scores[:min_len])
        
        top_k_indices = np.argsort(hybrid_scores)[-k:][::-1]
        results = [self.chunks[i] for i in top_k_indices]
        
        total_time = time.time() - start_time
        logger.debug(f"🎯 Search completed in {total_time:.3f}s (vector: {vector_time:.3f}s, keyword: {keyword_time:.3f}s)")
        
        return results