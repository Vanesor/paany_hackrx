import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TOKEN = os.getenv("TOKEN")

EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5"

GENERATIVE_MODEL_NAME = "gemini-1.5-flash-latest"

TOP_K_RESULTS = 5

PROMPT_TEMPLATE = """
SYSTEM INSTRUCTION:
You are a highly intelligent AI assistant tasked with analyzing legal and policy documents.
Your task is to answer the user's question based *ONLY* on the provided context.
Do not use any external knowledge.

Your response MUST be in two parts:
1.  **Direct Answer:** A clear and concise answer to the user's question.
2.  **Justification:** The detailed reasoning for your answer, including direct quotes from the provided context to support your conclusion.

If the context does not contain the answer, you must state: "The answer could not be found in the provided context."

CONTEXT:
---
{context}
---

QUESTION:
{question}
"""