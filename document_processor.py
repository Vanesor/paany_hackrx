import httpx
import fitz
import logging

# Get logger for this module
logger = logging.getLogger("doc_processor")

async def get_document_from_url(url: str) -> str:
    """
    Asynchronously downloads a PDF from a URL and extracts its text content.
    """
    logger.info(f"Fetching document from URL: {url}")
    start_time = __import__('time').time()
    
    try:
        logger.debug("Initiating HTTP request...")
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=30)
            response.raise_for_status()
            
        content_size = len(response.content)
        logger.info(f"Downloaded {content_size/1024:.1f} KB")
        
        logger.debug("Processing PDF with PyMuPDF...")
        with fitz.open(stream=response.content, filetype="pdf") as doc:
            page_count = len(doc)
            logger.debug(f"PDF has {page_count} pages")
            
            # Extract text from each page
            text = ""
            for page_num, page in enumerate(doc):
                page_text = page.get_text()
                text += page_text
                logger.debug(f"Page {page_num+1}: extracted {len(page_text)} chars")
            
        end_time = __import__('time').time()
        logger.info(f"Document processed in {end_time - start_time:.2f} seconds")
        logger.info(f"Total text extracted: {len(text)} characters")
        
        return text
        
    except httpx.RequestError as e:
        logger.error(f"Error downloading PDF: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        return None

def get_text_chunks(text: str, chunk_size: int = 1000, overlap: int = 200):
    """Split text into overlapping chunks for embedding."""
    logger.info("Starting text chunking process...")
    
    if not text:
        logger.warning("Empty text provided for chunking")
        return []
        
    logger.debug(f"Input text length: {len(text)} characters")
    logger.debug(f"Configuration: chunk_size={chunk_size}, overlap={overlap}")
    
    start_time = __import__('time').time()
    
    chunks = []
    start = 0
    while start < len(text):
        # Find chunk end
        end = min(start + chunk_size, len(text))
        
        # Try to find a better boundary (sentence or paragraph end)
        if end < len(text):
            for i in range(min(100, end - start)):
                if text[end - i] in '.!?\n' and text[end - i - 1] not in '.!?\n':
                    end = end - i
                    logger.debug(f"Adjusted chunk boundary to natural break at position {end}")
                    break
        
        # Extract the chunk
        chunk = text[start:end]
        chunks.append(chunk)
        
        # Log every 10th chunk for diagnostics
        if len(chunks) % 10 == 0:
            logger.debug(f"Created {len(chunks)} chunks so far...")
            
        # Move to next chunk with overlap
        start += (chunk_size - overlap)
    
    end_time = __import__('time').time()
    
    # Summarize chunking results
    avg_chunk_len = sum(len(c) for c in chunks) / max(len(chunks), 1)
    
    logger.info(f"Chunking completed in {end_time - start_time:.2f} seconds")
    logger.info(f"Generated {len(chunks)} chunks")
    logger.debug(f"Average chunk size: {avg_chunk_len:.1f} characters")
    
    return chunks