import google.generativeai as genai
from config import GOOGLE_API_KEY, GENERATIVE_MODEL_NAME, PROMPT_TEMPLATE

genai.configure(api_key=GOOGLE_API_KEY)

def generate_answer(context: str, question: str) -> str:
    """
    Generates an answer using the Gemini API based on the provided context and question.
    """
    if not context:
        return "Could not retrieve relevant context to answer the question."

    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    
    try:
        model = genai.GenerativeModel(GENERATIVE_MODEL_NAME)
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Error generating answer with Gemini: {e}")
        return "An error occurred while communicating with the generative model."