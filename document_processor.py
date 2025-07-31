import httpx
import fitz

async def get_document_from_url(url: str) -> str:
    """
    Asynchronously downloads a PDF from a URL and extracts its text content.
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=30)
            response.raise_for_status()
        
        with fitz.open(stream=response.content, filetype="pdf") as doc:
            text = "".join(page.get_text() for page in doc)
        return text
    except httpx.RequestError as e:
        print(f"Error downloading the PDF: {e}")
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