# llm_handler.py
import google.generativeai as genai
import asyncio
import time
import logging
from config import API_KEYS, GENERATIVE_MODEL_NAME, PROMPT_TEMPLATE

# Get logger for this module
logger = logging.getLogger("llm_handler")

class ApiKeyManager:
    def __init__(self, keys):
        if not keys:
            raise ValueError("No API keys provided in GOOGLE_API_KEYS .env variable.")
        self.keys = keys
        self.current_key_index = 0
        logger.info(f"🔑 API Key Manager initialized with {len(keys)} keys")

    def get_key(self):
        """Rotates to the next key for load distribution."""
        key = self.keys[self.current_key_index]
        self.current_key_index = (self.current_key_index + 1) % len(self.keys)
        logger.debug(f"🔄 Using API key ending in ...{key[-4:]}")
        return key

api_key_manager = ApiKeyManager(API_KEYS)

async def _call_gemini_with_fallback(api_function, **kwargs):
    """A generic wrapper for Gemini API calls with key rotation and fallback."""
    start_time = time.time()
    initial_key_index = api_key_manager.current_key_index
    
    for i in range(len(API_KEYS)):
        # Start with the current key and loop through all keys once
        key_index = (initial_key_index + i) % len(API_KEYS)
        key = API_KEYS[key_index]
        attempt_start = time.time()
        
        try:
            genai.configure(api_key=key)
            result = await asyncio.to_thread(api_function, **kwargs)
            attempt_time = time.time() - attempt_start
            total_time = time.time() - start_time
            logger.debug(f"✅ API call successful with key ...{key[-4:]} in {attempt_time:.3f}s (total: {total_time:.3f}s)")
            return result
        except Exception as e:
            attempt_time = time.time() - attempt_start
            logger.warning(f"❌ API Key ending in ...{key[-4:]} failed after {attempt_time:.3f}s: {e}")
    
    total_time = time.time() - start_time
    logger.error(f"💀 All {len(API_KEYS)} API keys failed after {total_time:.3f}s")
    raise Exception("All API keys failed.")

async def embed_content_with_fallback(content, task_type):
    """Embeds content using the managed fallback logic."""
    start_time = time.time()
    content_info = f"{len(content)} items" if isinstance(content, list) else f"1 item ({len(content)} chars)"
    logger.debug(f"🧠 Embedding {content_info} with task_type: {task_type}")
    
    result = await _call_gemini_with_fallback(
        genai.embed_content,
        model="models/embedding-001",
        content=content,
        task_type=task_type
    )
    
    embed_time = time.time() - start_time
    logger.info(f"✅ Embedding completed in {embed_time:.3f}s for {content_info}")
    return result

async def generate_answer(context: str, question: str) -> str:
    """Generates an answer using the managed fallback logic."""
    start_time = time.time()
    logger.debug(f"🧠 Generating answer for question: {question[:50]}...")
    
    if not context: 
        logger.warning("⚠️ No context provided for answer generation")
        return "Could not retrieve relevant context to answer the question."
    
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    prompt_length = len(prompt)
    logger.debug(f"📝 Prompt prepared - Length: {prompt_length} characters")
    
    key_for_request = api_key_manager.get_key()
    llm_start = time.time()
    
    try:
        genai.configure(api_key=key_for_request)
        model = genai.GenerativeModel(GENERATIVE_MODEL_NAME)
        response = await model.generate_content_async(prompt)
        answer = response.text
        
        llm_time = time.time() - llm_start
        total_time = time.time() - start_time
        answer_length = len(answer)
        
        logger.info(f"✅ Answer generated in {llm_time:.3f}s (total: {total_time:.3f}s) - Output: {answer_length} chars")
        logger.debug(f"📤 Answer preview: {answer[:100]}...")
        
        return answer
    except Exception as e:
        error_time = time.time() - start_time
        logger.error(f"❌ Answer generation failed with key ...{key_for_request[-4:]} after {error_time:.3f}s: {e}")
        return "An error occurred while communicating with the generative model."