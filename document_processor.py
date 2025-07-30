import requests
import fitz  # PyMuPDF?

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

def get_text_chunks(text: str, chunk_size: int = 1000, overlap: int = 200):
    print(f"[DocProcessor] Starting text chunking process...")
    print(f"[DocProcessor] Input text length: {len(text)} characters")
    print(f"[DocProcessor] Chunk size: {chunk_size}, Overlap: {overlap}")
    
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    
    print(f"[DocProcessor] Generated {len(chunks)} chunks")
    return chunks