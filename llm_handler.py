import google.generativeai as genai
from config import GOOGLE_API_KEY, GENERATIVE_MODEL_NAME, PROMPT_TEMPLATE

genai.configure(api_key=GOOGLE_API_KEY)

def generate_answer(context: str, question: str) -> str:
    """
    Generates an answer using the Gemini API based on the provided context and question.
    """
    print(f"\n[LLM] Processing question: {question[:100]}...")
    
    if not context:
        print("[LLM] Warning: No context provided")
        return "Could not retrieve relevant context to answer the question."

    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    print(f"[LLM] Generated prompt length: {len(prompt)} characters")
    
    try:
        print(f"[LLM] Initializing {GENERATIVE_MODEL_NAME}...")
        model = genai.GenerativeModel(GENERATIVE_MODEL_NAME)
        print("[LLM] Generating response...")
        response = model.generate_content(prompt)
        print("[LLM] Response generated successfully")
        return response.text
    except Exception as e:
        print(f"Error generating answer with Gemini: {e}")
        return "An error occurred while communicating with the generative model."