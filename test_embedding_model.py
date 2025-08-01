from sentence_transformers import SentenceTransformer
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_embedding_model():
    try:
        model_name = "BAAI/bge-large-en-v1.5"
        logger.info(f"Loading model: {model_name}")
        start_time = time.time()
        
        model = SentenceTransformer(model_name)
        
        end_time = time.time()
        logger.info(f"Model loaded successfully in {end_time - start_time:.2f} seconds")
        
        # Test embedding generation
        test_text = "This is a test sentence to verify the embedding model works correctly."
        logger.info(f"Generating embedding for test text: {test_text}")
        start_time = time.time()
        
        embedding = model.encode(test_text)
        
        end_time = time.time()
        logger.info(f"Embedding generated successfully in {end_time - start_time:.2f} seconds")
        logger.info(f"Embedding shape: {embedding.shape}")
        
        return True
    except Exception as e:
        logger.error(f"Error testing embedding model: {str(e)}")
        return False

if __name__ == "__main__":
    logger.info("Starting embedding model test")
    success = test_embedding_model()
    logger.info(f"Test completed. Success: {success}")
