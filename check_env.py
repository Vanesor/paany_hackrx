import os
from dotenv import load_dotenv

load_dotenv()

print("ENVIRONMENT VARIABLES:")
print(f"EMBEDDING_MODEL_NAME: {os.getenv('EMBEDDING_MODEL_NAME', 'Not set')}")
print(f"SENTENCE_TRANSFORMER_MODEL: {os.getenv('SENTENCE_TRANSFORMER_MODEL', 'Not set')}")
print(f"DENSE_WEIGHT: {os.getenv('DENSE_WEIGHT', 'Not set')}")
print(f"BM25_WEIGHT: {os.getenv('BM25_WEIGHT', 'Not set')}")
print(f"TFIDF_WEIGHT: {os.getenv('TFIDF_WEIGHT', 'Not set')}")
print(f"REDIS_DISABLED: {os.getenv('REDIS_DISABLED', 'Not set')}")
