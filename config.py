# config.py
import os
from dotenv import load_dotenv

load_dotenv()

# --- Logging Configuration ---
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# --- API Keys & Tokens ---
API_KEYS_STRING = os.getenv("GOOGLE_API_KEYS", "")
print(API_KEYS_STRING)
API_KEYS = [key.strip() for key in API_KEYS_STRING.split(',') if key.strip()]
TOKEN = os.getenv("TOKEN")

# --- Model Configuration ---
EMBEDDING_MODEL_NAME = "models/embedding-001"
GENERATIVE_MODEL_NAME = "gemini-1.5-flash-latest"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# --- RAG Pipeline Configuration ---
TOP_K_RESULTS = 10
HYBRID_SEARCH_ALPHA = 0.5 # 50% vector score, 50% keyword score

# --- Caching & Server ---
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
print(REDIS_URL)
PORT = int(os.getenv("PORT", 8000))

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