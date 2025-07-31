import google.generativeai as genai
import logging
from config import PRIMARY_API_KEY, BACKUP_API_KEY, GENERATIVE_MODEL_NAME, PROMPT_TEMPLATE

# Get logger for this module
logger = logging.getLogger("llm_handler")

async def generate_answer(context: str, question: str) -> str:
    """
    Asynchronously generates an answer using the Gemini API, with a fallback key.
    """
    logger.info(f"Generating answer for question: {question[:100]}...")
    
    if not context:
        logger.warning("No context provided for question")
        return "Could not retrieve relevant context to answer the question."

    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    logger.debug(f"Prompt length: {len(prompt)} characters")
    
    start_time = __import__('time').time()
    
    # Try with primary API key
    try:
        logger.debug(f"Using PRIMARY key with model {GENERATIVE_MODEL_NAME}")
        genai.configure(api_key=PRIMARY_API_KEY)
        
        model = genai.GenerativeModel(GENERATIVE_MODEL_NAME)
        logger.debug("Sending request to Gemini API...")
        
        response = await model.generate_content_async(prompt)
        answer = response.text
        
        end_time = __import__('time').time()
        logger.info(f"Answer generated successfully in {end_time - start_time:.2f} seconds")
        logger.debug(f"Answer length: {len(answer)} characters")
        
        return answer
        
    except Exception as primary_error:
        logger.warning(f"PRIMARY API key failed: {str(primary_error)}")
        
        # Try with backup API key if available
        if BACKUP_API_KEY:
            logger.debug("Attempting with BACKUP API key")
            try:
                genai.configure(api_key=BACKUP_API_KEY)
                model = genai.GenerativeModel(GENERATIVE_MODEL_NAME)
                
                response = await model.generate_content_async(prompt)
                answer = response.text
                
                end_time = __import__('time').time()
                logger.info(f"Answer generated with BACKUP key in {end_time - start_time:.2f} seconds")
                
                return answer
                
            except Exception as backup_error:
                logger.error(f"BACKUP API key also failed: {str(backup_error)}")
                return "An error occurred with both API keys."
        
        logger.error("No backup key available, answer generation failed")
        return "An error occurred with the primary API key and no backup was available."