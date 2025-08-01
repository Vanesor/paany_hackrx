import os
import logging
from dotenv import load_dotenv

load_dotenv()

LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG")
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

PRIMARY_API_KEY = os.getenv("GOOGLE_API_KEY")
BACKUP_API_KEY = os.getenv("GOOGLE_API_KEY_BACKUP")

PORT = int(os.getenv("PORT", 10000))
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
REDIS_DISABLED = os.getenv("REDIS_DISABLED", "false").lower() == "true"

# Hybrid search configuration
HYBRID_SEARCH_ENABLED = True
# Weight for dense embeddings (0.0-1.0). Sparse = 1-DENSE_WEIGHT
DENSE_WEIGHT = float(os.getenv("DENSE_WEIGHT", "0.75"))
# Weight for BM25 in sparse retrieval
BM25_WEIGHT = float(os.getenv("BM25_WEIGHT", "0.7"))
# Weight for TF-IDF in sparse retrieval
TFIDF_WEIGHT = float(os.getenv("TFIDF_WEIGHT", "0.3"))

# Embedding model configuration
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "models/embedding-001")
# Sentence Transformers configuration
SENTENCE_TRANSFORMER_MODEL = os.getenv("SENTENCE_TRANSFORMER_MODEL", "BAAI/bge-large-en-v1.5")
# OpenAI configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002")

# LLM Configuration
GENERATIVE_MODEL_NAME = "gemini-2.5-flash"
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "7"))

# Document processing configuration
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

# FAISS Configuration
FAISS_DIMENSION = int(os.getenv("FAISS_DIMENSION", "768"))
FAISS_M = int(os.getenv("FAISS_M", "32")) 
FAISS_EF_CONSTRUCTION = int(os.getenv("FAISS_EF_CONSTRUCTION", "200"))
FAISS_EF_SEARCH = int(os.getenv("FAISS_EF_SEARCH", "100"))

TOKEN = os.getenv("TOKEN", "6e8b43cca9d29b261843a3b1c53382bdaa5b2c9e96db92da679278c6dc0042ca")

PROMPT_TEMPLATE = """
You are an expert document analyst. Answer the question using ONLY the provided context. Be precise and factual.

STRICT FORMAT REQUIREMENTS:
- Provide a direct, concise answer (maximum 50 words)
- Include specific numbers, timeframes, percentages, and conditions mentioned in the context
- Use clear, professional language
- Do NOT add explanations, justifications, or additional commentary
- If the answer is not found in the context, respond: "The answer could not be found in the provided document."

EXAMPLES OF CORRECT ANSWERS:
- "A grace period of thirty days is provided for premium payment after the due date to renew or continue coverage from the first year onwards."
- "There is a waiting period of thirty-six (36) months of continuous coverage from the first policy commencement for pre-existing diseases (PED) to be covered."
- "The policy has a two-year waiting period for cataract surgery."
- "Yes, the policy indemnifies expenses for health check-ups up to the limit specified in the Policy Schedule for Plan A."

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""