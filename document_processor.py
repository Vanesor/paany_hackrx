import requests
import fitz  # PyMuPDF?
from langchain.text_splitter import RecursiveCharacterTextSplitter

def get_document_from_url(url: str) -> str:
    """
    Downloads a PDF from a URL and extracts its text content.
    Uses PyMuPDF for fast and accurate text extraction.
    """
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()  
        
        with fitz.open(stream=response.content, filetype="pdf") as doc:
            text = "".join(page.get_text() for page in doc)
        return text
    except requests.RequestException as e:
        print(f"Error downloading or processing the PDF: {e}")
        return None

def get_text_chunks(text: str) -> list[str]:
    """
    Splits a long text into smaller, semantically coherent chunks.
    This is crucial for both embedding accuracy and managing token limits.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks