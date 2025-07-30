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