import numpy as np
import asyncio
import time
import logging
import re
import string
import os
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Tuple, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from config import (
    EMBEDDING_MODEL_NAME, SENTENCE_TRANSFORMER_MODEL, 
    DENSE_WEIGHT, BM25_WEIGHT, TFIDF_WEIGHT, HYBRID_SEARCH_ENABLED
)

# Get logger for this module
logger = logging.getLogger("vector_store")

class VectorStore:
    def __init__(self):
        self.model_name = EMBEDDING_MODEL_NAME
        self.sentence_transformer_model_name = SENTENCE_TRANSFORMER_MODEL
        self.sentence_transformer = None
        self.dense_index: Optional[np.ndarray] = None  # Dense embeddings
        self.sparse_index = None  # Sparse embeddings (BM25)
        self.tfidf_vectorizer = None  # TF-IDF for additional sparse features
        self.tfidf_matrix = None  # Precomputed TF-IDF features
        self.chunks = []
        self.tokenized_chunks = []  # For BM25
        self.chunk_metadata = []  # Store metadata like page number for reranking
        self._initialize_embedding_model()
        logger.info(f"HybridVectorStore initialized with model: {self.model_name} - {self.sentence_transformer_model_name}")
        
    def _initialize_embedding_model(self):
        """Initialize the appropriate embedding model based on configuration."""
        if self.model_name == "sentence-transformers":
            try:
                logger.info(f"Loading SentenceTransformer model: {self.sentence_transformer_model_name}")
                self.sentence_transformer = SentenceTransformer(self.sentence_transformer_model_name)
                logger.info(f"SentenceTransformer model loaded successfully")
            except Exception as e:
                logger.error(f"Error loading SentenceTransformer model: {str(e)}")
                raise
                
    def _preprocess_text(self, text: str) -> List[str]:
        """Preprocess text for BM25 tokenization."""
        # Convert to lowercase and remove punctuation
        text = text.lower()
        # Keep numbers but remove punctuation
        text = re.sub(r'[^\w\s]', ' ', text)
        # Remove stop words for better matching (a basic implementation)
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'of'}
        tokens = text.split()
        tokens = [t for t in tokens if t not in stop_words and len(t) > 1]
        return tokens
        
    def _extract_metadata(self, chunk: str) -> Dict[str, Any]:
        """Extract metadata from a chunk for improved relevance ranking."""
        metadata = {}
        
        # Extract page numbers
        page_match = re.search(r'\[Page (\d+)\]', chunk)
        if page_match:
            metadata['page'] = int(page_match.group(1))
            
        # Detect if chunk contains figures/tables
        if re.search(r'(Figure|Table|Fig\.)\s+\d+', chunk, re.IGNORECASE):
            metadata['has_figure'] = True
            
        # Detect mathematical content
        if re.search(r'[=+\-*/^(){}[\]]+', chunk):
            metadata['has_math'] = True
            
        # Count numbers in the text (often important for scientific text)
        metadata['number_count'] = len(re.findall(r'\b\d+\b', chunk))
        
        return metadata
        
    async def _encode_with_sentence_transformers(self, texts, batch_size=64):
        """Generate embeddings using SentenceTransformer model with optimized parallel processing."""
        if self.sentence_transformer is None:
            logger.error("SentenceTransformer model not initialized")
            return None
            
        logger.info(f"Generating embeddings for {len(texts) if isinstance(texts, list) else 1} texts")
        
        try:
            # Convert single text to list if needed
            if not isinstance(texts, list):
                texts = [texts]
                return await asyncio.to_thread(self.sentence_transformer.encode, texts, convert_to_numpy=True)
            
            start_time = time.time()
            
            # Function to encode a batch of texts
            def encode_batch(batch):
                return self.sentence_transformer.encode(batch, show_progress_bar=False, convert_to_numpy=True)
            
            # Determine optimal number of workers based on CPU cores
            max_workers = min(os.cpu_count() or 4, 8)  # Limit to 8 workers max
            logger.info(f"Using ThreadPoolExecutor with {max_workers} workers and batch size {batch_size}")
            
            # Split texts into batches
            batches = [texts[i:i+batch_size] for i in range(0, len(texts), batch_size)]
            logger.info(f"Split {len(texts)} texts into {len(batches)} batches")
            
            # Process batches in parallel using ThreadPoolExecutor
            all_embeddings = []
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_results = list(executor.map(encode_batch, batches))
                
                for i, batch_result in enumerate(future_results):
                    all_embeddings.append(batch_result)
                    logger.info(f"Completed batch {i+1}/{len(batches)} ({(i+1)/len(batches)*100:.1f}%)")
            
            # Combine all embeddings
            if all_embeddings:
                embeddings = np.vstack(all_embeddings)
                end_time = time.time()
                logger.info(f"Generated {len(texts)} embeddings in {end_time - start_time:.2f} seconds " 
                           f"({len(texts) / (end_time - start_time):.1f} embeddings/second)")
                logger.info(f"Embedding shape: {embeddings.shape}")
                return embeddings
            else:
                logger.error("No embeddings generated")
                return None
                
        except Exception as e:
            logger.error(f"Error generating embeddings with SentenceTransformer: {str(e)}")
            return None
            
    async def build_index(self, chunks: list[str]):
        """Asynchronously creates both dense and sparse embeddings and builds indices."""
        logger.info(f"Building hybrid index for {len(chunks)} chunks")
        
        if not chunks:
            logger.error("No chunks provided, cannot build index")
            return
            
        self.chunks = chunks
        
        # Extract metadata from chunks
        self.chunk_metadata = [self._extract_metadata(chunk) for chunk in chunks]
        
        # Log chunk information
        avg_chunk_len = sum(len(c) for c in chunks) / len(chunks)
        logger.debug(f"Average chunk length: {avg_chunk_len:.1f} characters")
        
        # Build sparse representations (BM25) - more efficient tokenization
        logger.info("Building sparse (BM25) index...")
        start_time = time.time()
        
        # Use ThreadPoolExecutor for tokenization
        with ThreadPoolExecutor() as executor:
            self.tokenized_chunks = list(executor.map(self._preprocess_text, chunks))
            
        self.sparse_index = BM25Okapi(self.tokenized_chunks)
        logger.info(f"BM25 index built in {time.time() - start_time:.2f} seconds")
        
        # Build TF-IDF representations for additional sparse features
        logger.info("Building TF-IDF index...")
        start_time = time.time()
        self.tfidf_vectorizer = TfidfVectorizer(
            min_df=2, 
            max_df=0.95,
            ngram_range=(1, 2)  # Include bigrams
        )
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(chunks)
        logger.debug(f"TF-IDF index built in {time.time() - start_time:.2f} seconds")
        
        try:
        # Build dense representations
            logger.info(f"Building dense ({self.sentence_transformer_model_name}) index...")
            start_time = time.time()
            
            # Generate embeddings using our optimized parallel method
            embeddings = await self._encode_with_sentence_transformers(self.chunks)
            
            if embeddings is None:
                logger.error("Failed to generate embeddings")
                return
                
            # Normalize embeddings for cosine similarity
            logger.info("Normalizing embeddings...")
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            self.dense_index = embeddings / norms
            
            end_time = time.time()
            logger.info(f"Hybrid index built successfully in {end_time - start_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Error building dense index: {str(e)}")
            raise

    async def search(self, query: str, k: int, hybrid_weight: float = DENSE_WEIGHT) -> list[str]:
        """
        Searches using a hybrid approach combining dense and sparse retrieval.
        
        Args:
            query: The search query
            k: Number of results to return
            hybrid_weight: Weight for dense embeddings (0.0-1.0). Sparse = 1-hybrid_weight
        
        Returns:
            List of most relevant chunks
        """
        if not HYBRID_SEARCH_ENABLED:
            # Fallback to dense search only if hybrid search is disabled
            logger.info("Hybrid search disabled, using dense search only")
            dense_results = await self._dense_search(query, k=k)
            return [self.chunks[idx] for idx, _ in dense_results]
        
        if self.dense_index is None or self.sparse_index is None or len(self.chunks) == 0:
            logger.error("Cannot search: Indices or chunks not initialized")
            return []
            
        logger.info(f"Hybrid search for query: {query[:50]}...")
        logger.debug(f"Search parameters: top_k={k}, hybrid_weight={hybrid_weight}")
        
        try:
            start_time = time.time()
            
            # 1. Dense retrieval (semantic search)
            logger.debug("Generating query embedding for dense search")
            dense_results = await self._dense_search(query, k=k*2)  # Get more results for reranking
            
            # 2. Sparse retrieval (keyword match with BM25)
            logger.debug("Performing sparse BM25 search")
            tokenized_query = self._preprocess_text(query)
            bm25_scores = self.sparse_index.get_scores(tokenized_query)
            
            # 3. TF-IDF sparse retrieval for additional signal
            logger.debug("Performing TF-IDF search")
            query_tfidf = self.tfidf_vectorizer.transform([query])
            tfidf_scores = (self.tfidf_matrix @ query_tfidf.T).toarray().flatten()
            
            # 4. Get top sparse candidates
            # Combine BM25 and TF-IDF scores with their respective weights
            sparse_score = BM25_WEIGHT * bm25_scores
            if max(sparse_score) > 0:
                sparse_score = sparse_score / max(sparse_score)  # Normalize
                
            tfidf_score = TFIDF_WEIGHT * tfidf_scores
            if max(tfidf_score) > 0:
                tfidf_score = tfidf_score / max(tfidf_score)  # Normalize
                
            combined_sparse_scores = sparse_score + tfidf_score
            sparse_indices = np.argsort(combined_sparse_scores)[-k*2:][::-1]  # Get more for reranking
            
            # 5. Combine dense and sparse results with hybrid scoring
            logger.debug("Combining dense and sparse results for hybrid ranking")
            combined_scores = {}
            
            # Add dense scores (already in 0-1 range)
            for idx, score in dense_results:
                combined_scores[idx] = hybrid_weight * score
            
            # Add sparse scores (normalized)
            sparse_weight = 1.0 - hybrid_weight
            for idx in sparse_indices:
                norm_score = combined_sparse_scores[idx]
                if max(combined_sparse_scores) > 0:
                    norm_score = norm_score / max(combined_sparse_scores)
                    
                if idx in combined_scores:
                    combined_scores[idx] += sparse_weight * norm_score
                else:
                    combined_scores[idx] = sparse_weight * norm_score
            
            # 6. Apply metadata boosts (for advanced relevance)
            self._apply_metadata_boosts(query, combined_scores)
            
            # 7. Sort by combined score and get top k
            top_indices = sorted(combined_scores.keys(), 
                                key=lambda idx: combined_scores[idx], 
                                reverse=True)[:k]
            
            # Get matching chunks
            results = [self.chunks[i] for i in top_indices]
            
            end_time = time.time()
            logger.info(f"Found {len(results)} chunks in {end_time - start_time:.4f} seconds")
            
            # Detailed logging for debugging
            for i, idx in enumerate(top_indices[:min(3, len(top_indices))]):
                logger.debug(f"Top result #{i+1}: score={combined_scores[idx]:.4f}, chunk preview: {self.chunks[idx][:100]}...")
            
            return results
            
        except Exception as e:
            logger.error(f"Error during hybrid search: {str(e)}")
            return []
    
    def _apply_metadata_boosts(self, query: str, scores: Dict[int, float]):
        """Apply boosts based on chunk metadata and query characteristics."""
        # Check if query contains specific patterns to match with metadata
        has_math_query = bool(re.search(r'[=+\-*/^(){}[\]]+', query))
        has_figure_query = bool(re.search(r'(figure|table|graph|chart|plot|diagram)', query, re.IGNORECASE))
        has_number_query = bool(re.findall(r'\b\d+\b', query))
        
        # Apply boosts based on metadata matches
        for idx in list(scores.keys()):
            metadata = self.chunk_metadata[idx]
            
            # Boost chunks with figures/tables when query mentions them
            if has_figure_query and metadata.get('has_figure', False):
                scores[idx] *= 1.2  # 20% boost
                
            # Boost chunks with mathematical content for math queries
            if has_math_query and metadata.get('has_math', False):
                scores[idx] *= 1.1  # 10% boost
                
            # Boost chunks with numbers for queries with numbers
            if has_number_query and metadata.get('number_count', 0) > 0:
                scores[idx] *= 1.0 + min(0.3, metadata['number_count'] * 0.05)  # Up to 30% boost
    
    async def _dense_search(self, query: str, k: int) -> List[Tuple[int, float]]:
        """Performs dense vector search using the embedding model."""
        try:
            # Generate embedding for the query
            logger.debug("Generating query embedding")
            
            result = await self._encode_with_sentence_transformers(query)
            
            if result is None:
                logger.error("Failed to generate query embedding")
                return []
                
            query_embedding = np.array(result, dtype=np.float32)
            logger.debug(f"Query embedding shape: {query_embedding.shape}")
            
            # Normalize the query embedding
            query_norm = np.linalg.norm(query_embedding)
            normalized_query = query_embedding / query_norm
            
            # Make sure the query embedding is properly shaped for matrix multiplication
            if normalized_query.ndim == 1:
                normalized_query = normalized_query.reshape(-1, 1)  # Convert to column vector
            
            # Calculate similarities
            logger.debug(f"Calculating similarities against {len(self.chunks)} chunks")
            if self.dense_index is not None:
                # Make sure shapes are compatible for dot product
                if normalized_query.shape[1] == 1:  # If column vector
                    similarities = np.dot(self.dense_index, normalized_query).flatten()
                else:  # If row vector
                    similarities = np.dot(self.dense_index, normalized_query.T).flatten()
                
                # Get top-k indices and scores
                top_k_indices = np.argsort(similarities)[-k:][::-1]
                top_similarities = [similarities[i] for i in top_k_indices]
                
                # Return indices and scores
                return list(zip(top_k_indices, top_similarities))
            else:
                logger.error("Dense index is None, cannot perform search")
                return []
            
        except Exception as e:
            logger.error(f"Error during dense search: {str(e)}")
            return []