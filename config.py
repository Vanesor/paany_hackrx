import os
from dotenv import load_dotenv

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5" 
EMBEDDING_MODEL_NAME= "models/embedding-001"
GENERATIVE_MODEL_NAME = "gemini-2.5-flash"
TOP_K_RESULTS = 5

TOKEN = "6e8b43cca9d29b261843a3b1c53382bdaa5b2c9e96db92da679278c6dc0042ca" # 

PROMPT_TEMPLATE = """
Using only the context below, answer the question in a single, concise paragraph. Focus on key details, avoiding unnecessary words or repetition. Do not include preamble or explanations.

If the question cannot be answered from the context, respond only with: 'The answer could not be found in the provided document.'

CONTEXT:
---
{context}
---

QUESTION: {question}
"""