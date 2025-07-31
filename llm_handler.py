import google.generativeai as genai
from config import PRIMARY_API_KEY, BACKUP_API_KEY, GENERATIVE_MODEL_NAME, PROMPT_TEMPLATE

async def generate_answer(context: str, question: str) -> str:
    """
    Asynchronously generates an answer using the Gemini API, with a fallback key.
    """
    if not context:
        return "Could not retrieve relevant context to answer the question."

    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    
    try:
        genai.configure(api_key=PRIMARY_API_KEY)
        model = genai.GenerativeModel(GENERATIVE_MODEL_NAME)
        response = await model.generate_content_async(prompt)
        return response.text
    except Exception as e:
        print(f"Primary API key failed: {e}. Trying backup key.")
        if BACKUP_API_KEY:
            try:
                genai.configure(api_key=BACKUP_API_KEY)
                model = genai.GenerativeModel(GENERATIVE_MODEL_NAME)
                response = await model.generate_content_async(prompt)
                return response.text
            except Exception as e2:
                print(f"Backup API key also failed: {e2}")
                return "An error occurred with both API keys."
        
        return "An error occurred with the primary API key and no backup was available."