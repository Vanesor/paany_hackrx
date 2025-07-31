import numpy as np
import asyncio
import google.generativeai as genai
import logging
from config import PRIMARY_API_KEY, BACKUP_API_KEY, EMBEDDING_MODEL_NAME

# Get logger for this module
logger = logging.getLogger("vector_store")

class VectorStore:
    def __init__(self):
        self.model_name = EMBEDDING_MODEL_NAME
        self.index = None
        self.chunks = []
        logger.info(f"VectorStore initialized with model: {self.model_name}")

    async def _embed_content_with_fallback(self, content, task_type):
        """Internal method to handle embedding with primary and backup keys."""
        logger.debug(f"Generating embeddings with {self.model_name}, task_type={task_type}")
        
        content_info = f"{len(content)} items" if isinstance(content, list) else "single item"
        logger.debug(f"Content to embed: {content_info}")
        
        try:
            logger.debug("Using PRIMARY API key for embeddings")
            genai.configure(api_key=PRIMARY_API_KEY)
            
            result = await asyncio.to_thread(
                genai.embed_content,
                model=self.model_name,
                content=content,
                task_type=task_type
            )
            
            logger.debug(f"Embedding successful with PRIMARY key")
            return result
            
        except Exception as e:
            logger.warning(f"PRIMARY API key failed for embedding: {str(e)}")
            
            if BACKUP_API_KEY:
                logger.debug("Attempting with BACKUP API key")
                try:
                    genai.configure(api_key=BACKUP_API_KEY)
                    
                    result = await asyncio.to_thread(
                        genai.embed_content,
                        model=self.model_name,
                        content=content,
                        task_type=task_type
                    )
                    
                    logger.debug("Embedding successful with BACKUP key")
                    return result
                    
                except Exception as e2:
                    logger.error(f"BACKUP API key also failed: {str(e2)}")
                    raise e2
                    
            logger.error("No backup key available, embedding failed")
            raise e

    async def build_index(self, chunks: list[str]):
        """Asynchronously creates embeddings and builds the NumPy index."""
        logger.info(f"Building index for {len(chunks)} chunks")
        
        if not chunks:
            logger.error("No chunks provided, cannot build index")
            return
            
        self.chunks = chunks
        
        # Log chunk information
        avg_chunk_len = sum(len(c) for c in chunks) / len(chunks)
        logger.debug(f"Average chunk length: {avg_chunk_len:.1f} characters")
        
        try:
            logger.info("Generating embeddings...")
            start_time = __import__('time').time()
            
            result = await self._embed_content_with_fallback(self.chunks, "retrieval_document")
            
            if not result or 'embedding' not in result:
                logger.error("No embedding results returned")
                return
                
            embeddings = np.array(result['embedding'], dtype=np.float32)
            logger.debug(f"Generated embeddings shape: {embeddings.shape}")
            
            # Normalize embeddings for cosine similarity
            logger.debug("Normalizing embeddings...")
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            self.index = embeddings / norms
            
            end_time = __import__('time').time()
            logger.info(f"Index built successfully in {end_time - start_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Error building index: {str(e)}")
            raise

    def search(self, query: str, k: int) -> list[str]:
        """Searches the index for the most relevant chunks using NumPy."""
        if self.index is None or len(self.chunks) == 0:
            logger.error("Cannot search: Index or chunks not initialized")
            return []
            
        logger.info(f"Searching for query: {query[:50]}...")
        logger.debug(f"Search parameters: top_k={k}")
        
        try:
            # Generate embedding for the query
            logger.debug("Generating query embedding")
            start_time = __import__('time').time()
            
            result = genai.embed_content(
                model=self.model_name,
                content=query,
                task_type="retrieval_query"
            )
            
            query_embedding = np.array(result['embedding'], dtype=np.float32)
            logger.debug(f"Query embedding shape: {query_embedding.shape}")
            
            # Normalize the query embedding
            query_norm = np.linalg.norm(query_embedding)
            normalized_query = query_embedding / query_norm
            
            # Calculate similarities and get top results
            logger.debug(f"Calculating similarities against {len(self.chunks)} chunks")
            similarities = np.dot(self.index, normalized_query)
            
            # Get top-k indices
            top_k_indices = np.argsort(similarities)[-k:][::-1]
            top_similarities = [similarities[i] for i in top_k_indices]
            
            # Get matching chunks
            results = [self.chunks[i] for i in top_k_indices]
            
            end_time = __import__('time').time()
            logger.info(f"Found {len(results)} chunks in {end_time - start_time:.4f} seconds")
            
            # Log similarity scores
            for i, (chunk, sim) in enumerate(zip(results, top_similarities)):
                logger.debug(f"Match #{i+1}: similarity={sim:.4f}, chunk_len={len(chunk)}")
                
            return results
            
        except Exception as e:
            logger.error(f"Error during search: {str(e)}")
            return []