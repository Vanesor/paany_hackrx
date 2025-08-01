# config.py
import os
import logging
from dotenv import load_dotenv

load_dotenv()

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# --- API Keys & Tokens ---
API_KEYS_STRING = os.getenv("GOOGLE_API_KEYS", "")
API_KEYS = [key.strip() for key in API_KEYS_STRING.split(',') if key.strip()]
TOKEN = os.getenv("TOKEN")

# --- Model Configuration ---
EMBEDDING_MODEL_NAME = "models/embedding-001"
GENERATIVE_MODEL_NAME = "gemini-2.5-flash"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# --- RAG Pipeline Configuration ---
TOP_K_RETRIEVAL = 10
TOP_K_RESULTS = 5 # Retrieve the top 5 results from hybrid search
HYBRID_SEARCH_ALPHA = 0.5 # Balance between vector and keyword search

# --- Caching & Server ---
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
PORT = int(os.getenv("PORT", 10000))

# --- Prompt Template for Final Answer Generation ---
PROMPT_TEMPLATE = """
Based *only* on the context provided below, provide a concise and direct answer to the user's question.
Synthesize all relevant details into a single, easy-to-read paragraph.
Do not add any preamble like "The answer is..." or "Justification:".

If the answer cannot be found in the context, respond with only this exact phrase: "The answer could not be found in the provided document."

CONTEXT:
---
{context}
---

QUESTION: {question}
"""