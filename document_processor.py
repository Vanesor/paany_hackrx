import httpx
import fitz  # PyMuPDF
import logging
import re
import time
from config import CHUNK_SIZE, CHUNK_OVERLAP

# Get logger for this module
logger = logging.getLogger("doc_processor")

async def get_document_from_url(url: str) -> str:
    """
    Asynchronously downloads a PDF from a URL and extracts its text content.
    """
    logger.info(f"Fetching document from URL: {url}")
    start_time = time.time()
    
    try:
        logger.debug("Initiating HTTP request...")
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=60)  # Increased timeout
            response.raise_for_status()
            
        content_size = len(response.content)
        logger.info(f"Downloaded {content_size/1024:.1f} KB")
        
        logger.debug("Processing PDF with PyMuPDF...")
        doc = fitz.open(stream=response.content, filetype="pdf")
        page_count = len(doc)
        logger.debug(f"PDF has {page_count} pages")
        
        # Extract text from each page
        text = ""
        for page_num in range(page_count):
            page = doc[page_num]
            page_text = page.get_text()
            # Add metadata about page number to help with context
            logger.debug(f"Page {page_num+1}: extracted {len(page_text)} chars")
            # Append page number metadata and the page text
            text += f"\n[Page {page_num+1}]\n{page_text}"
        
        doc.close()
        
        end_time = __import__('time').time()
        logger.info(f"Document processed in {end_time - start_time:.2f} seconds")
        logger.info(f"Total text extracted: {len(text)} characters")
        
        return text
        
    except httpx.RequestError as e:
        logger.error(f"Error downloading PDF: {str(e)}")
        return ""
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        return ""

def get_text_chunks(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    """
    Split text into semantically meaningful chunks for embedding.
    Uses a more intelligent chunking strategy with natural breaks.
    """
    logger.info("Starting text chunking process...")
    
    if not text:
        logger.warning("Empty text provided for chunking")
        return []
        
    logger.debug(f"Input text length: {len(text)} characters")
    logger.debug(f"Configuration: chunk_size={chunk_size}, overlap={overlap}")
    
    start_time = __import__('time').time()
    
    # Pre-process: Clean up text, removing excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    
    # Find natural chunk boundaries (paragraphs, sections)
    # Look for patterns like "[Page X]", section headers, etc.
    section_patterns = [
        r'\[Page \d+\]',                  # Page markers
        r'\n\s*(?:CHAPTER|Section)\s+\d+', # Chapter/section headers
        r'\n\s*[A-Z][A-Z\s]+(?:\n|\:)',    # ALL CAPS headers
        r'\n\s*(?:Figure|Table)\s+\d+',    # Figure/table captions
        r'\n\s*\d+\.\s+[A-Z]',             # Numbered sections
    ]
    
    # Compile the patterns for faster matching
    compiled_patterns = [re.compile(pattern) for pattern in section_patterns]
    
    # Find all potential break points
    break_points = set()
    for pattern in compiled_patterns:
        for match in pattern.finditer(text):
            break_points.add(match.start())
    
    # Sort break points
    break_points = sorted(list(break_points))
    
    # Create chunks with respect to natural breaks
    chunks = []
    start = 0
    
    while start < len(text):
        # Determine end of current chunk
        end = min(start + chunk_size, len(text))
        
        # If we're not at the end of text, look for a natural break point
        if end < len(text):
            # Find nearest break point within reasonable range
            best_break = end
            
            # Look for a sentence break near chunk_size
            for i in range(max(0, end - 150), min(end + 150, len(text))):
                if i in break_points or (i > 0 and text[i-1] in '.!?' and text[i] == ' '):
                    best_break = i
                    break
            
            end = best_break
        
        # Extract chunk
        chunk = text[start:end].strip()
        if chunk:  # Only add non-empty chunks
            chunks.append(chunk)
        
        # Move to next chunk with overlap
        start = max(start + 1, end - overlap)
    
    # Post-process: Clean up chunks
    for i in range(len(chunks)):
        # Remove redundant page markers in the middle of chunks
        chunks[i] = re.sub(r'\[Page \d+\]\s*', ' ', chunks[i])
        # Ensure chunks don't start with partial sentences
        if i > 0 and not chunks[i][0].isupper() and not chunks[i][0].isdigit():
            # Look for the first sentence break
            match = re.search(r'[.!?]\s+[A-Z]', chunks[i])
            if match:
                # Move the beginning of this chunk to the start of a complete sentence
                new_start = match.end() - 1
                chunks[i] = chunks[i][new_start:]
    
    end_time = __import__('time').time()
    
    # Summarize chunking results
    avg_chunk_len = sum(len(c) for c in chunks) / max(len(chunks), 1)
    
    logger.info(f"Chunking completed in {end_time - start_time:.2f} seconds")
    logger.info(f"Generated {len(chunks)} chunks")
    logger.debug(f"Average chunk size: {avg_chunk_len:.1f} characters")
    
    return chunks