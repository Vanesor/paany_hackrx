# vector_store.py
import numpy as np
import time
import logging
from rank_bm25 import BM25Okapi
from llm_handler import embed_content_with_fallback

# Get logger for this module
logger = logging.getLogger("vector_store")

class VectorStore:
    def __init__(self):
        self.index = None
        self.chunks = []
        self.bm25 = None
        logger.info("🏗️ VectorStore initialized for hybrid search")

    async def build_index(self, chunks: list[str]):
        """Builds both NumPy (vector) and BM25 (keyword) indices."""
        start_time = time.time()
        logger.info(f"🔨 Building indices for {len(chunks)} chunks")
        
        self.chunks = chunks
        
        # Build vector index
        vector_start = time.time()
        logger.debug("📊 Generating embeddings for vector index...")
        result = await embed_content_with_fallback(self.chunks, "retrieval_document")
        embeddings = np.array(result['embedding'], dtype=np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        self.index = embeddings / norms
        vector_time = time.time() - vector_start
        logger.info(f"✅ Vector index built in {vector_time:.3f}s - Shape: {self.index.shape}")
        
        # Build BM25 index
        bm25_start = time.time()
        logger.debug("🔤 Building BM25 keyword index...")
        tokenized_chunks = [chunk.split() for chunk in chunks]
        self.bm25 = BM25Okapi(tokenized_chunks)
        bm25_time = time.time() - bm25_start
        logger.info(f"✅ BM25 index built in {bm25_time:.3f}s")
        
        total_time = time.time() - start_time
        logger.info(f"🎯 Index building completed in {total_time:.3f}s (Vector: {vector_time:.3f}s, BM25: {bm25_time:.3f}s)")

    async def search(self, query: str, k: int, alpha: float) -> list[str]:
        """Performs a hybrid search combining vector and keyword results."""
        start_time = time.time()
        logger.debug(f"🔍 Hybrid search for: {query[:50]}... (k={k}, alpha={alpha})")
        
        if self.index is None or self.bm25 is None: 
            logger.error("❌ Cannot search: indices not built")
            return []

        # Vector search
        vector_start = time.time()
        result = await embed_content_with_fallback(query, "retrieval_query")
        query_embedding = np.array(result['embedding'], dtype=np.float32)
        query_norm = np.linalg.norm(query_embedding)
        normalized_query = query_embedding / query_norm
        vector_scores = np.dot(self.index, normalized_query)
        vector_time = time.time() - vector_start
        logger.debug(f"🎯 Vector search completed in {vector_time:.3f}s")

        # BM25 search
        bm25_start = time.time()
        tokenized_query = query.split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_scores = bm25_scores / (bm25_scores.max() or 1.0)
        bm25_time = time.time() - bm25_start
        logger.debug(f"🔤 BM25 search completed in {bm25_time:.3f}s")

        # Combine scores
        combine_start = time.time()
        min_len = min(len(vector_scores), len(bm25_scores))
        hybrid_scores = (alpha * vector_scores[:min_len]) + ((1 - alpha) * bm25_scores[:min_len])
        
        top_k_indices = np.argsort(hybrid_scores)[-k:][::-1]
        results = [self.chunks[i] for i in top_k_indices]
        combine_time = time.time() - combine_start
        
        total_search_time = time.time() - start_time
        logger.info(f"✅ Hybrid search completed in {total_search_time:.3f}s (Vector: {vector_time:.3f}s, BM25: {bm25_time:.3f}s, Combine: {combine_time:.3f}s) - Found {len(results)} results")
        
        # Log top scores for debugging
        top_scores = sorted(hybrid_scores, reverse=True)[:3]
        logger.debug(f"📊 Top 3 hybrid scores: {[f'{score:.4f}' for score in top_scores]}")
        
        return results