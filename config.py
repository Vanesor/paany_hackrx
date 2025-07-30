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
Answer the question using only the provided context. Be direct and factual.

Format requirements:
- Single sentence answer (maximum 30 words)
- Include key numbers, timeframes, and conditions
- Use simple, clear language
- No explanatory phrases or justifications
- If not found: "The answer could not be found in the provided document."

Examples of good answers:
- "The grace period is 30 days after the premium due date."
- "Pre-existing diseases are covered after 36 months of continuous coverage."
- "Room rent is capped at 1% of Sum Insured per day for Plan A."

CONTEXT:
{context}

QUESTION: {question}

DIRECT ANSWER:"""