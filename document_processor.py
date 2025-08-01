# document_processor.py
import httpx
import fitz  # PyMuPDF
import docx  # python-docx
import io
import email
import time
import logging
from email import policy
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

# --- Text Extraction Helpers ---

def _extract_text_from_pdf(content: bytes) -> str:
    """Extracts text from PDF content."""
    with fitz.open(stream=content, filetype="pdf") as doc:
        return "".join(page.get_text() for page in doc)

def _extract_text_from_docx(content: bytes) -> str:
    """Extracts text from DOCX content."""
    doc = docx.Document(io.BytesIO(content))
    return "\n".join([para.text for para in doc.paragraphs])

def _extract_text_from_email(content: bytes) -> str:
    """Extracts and cleans text from email (.eml or .msg) content."""
    msg = email.message_from_bytes(content, policy=policy.default)
    
    text_parts = []
    
    # Extract headers
    for header in ['From', 'To', 'Subject', 'Date']:
        if msg[header]:
            text_parts.append(f"{header}: {msg[header]}")
            
    text_parts.append("\n--- Body ---\n")

    # Extract body
    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            content_disposition = str(part.get("Content-Disposition"))
            if "attachment" not in content_disposition:
                if content_type == "text/plain":
                    text_parts.append(part.get_payload(decode=True).decode())
                elif content_type == "text/html":
                    html_content = part.get_payload(decode=True).decode()
                    soup = BeautifulSoup(html_content, "html.parser")
                    text_parts.append(soup.get_text())
    else:
        # Not multipart, just get the payload
        if msg.get_content_type() == "text/plain":
            text_parts.append(msg.get_payload(decode=True).decode())
        elif msg.get_content_type() == "text/html":
            html_content = msg.get_payload(decode=True).decode()
            soup = BeautifulSoup(html_content, "html.parser")
            text_parts.append(soup.get_text())
            
    return "\n".join(text_parts).strip()


# --- Main Document Processing Function ---

async def get_document_from_url(url: str) -> str | None:
    """
    Asynchronously downloads a document from a URL, detects its type,
    and extracts its text content.
    """
    start_time = time.time()
    logger.info(f"📥 Downloading document from: {url[:50]}...")
    
    try:
        download_start = time.time()
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=30, follow_redirects=True)
            response.raise_for_status()
        download_time = time.time() - download_start

        content = response.content
        content_type = response.headers.get("content-type", "").lower()
        file_size = len(content)

        logger.info(f"✅ Downloaded {file_size} bytes in {download_time:.2f}s - Content-Type: {content_type}")

        extract_start = time.time()
        if "pdf" in content_type:
            logger.debug("📄 Extracting text from PDF...")
            text = _extract_text_from_pdf(content)
        elif "vnd.openxmlformats-officedocument.wordprocessingml.document" in content_type or url.endswith(".docx"):
            logger.debug("📝 Extracting text from DOCX...")
            text = _extract_text_from_docx(content)
        elif "message/rfc822" in content_type or url.endswith(".eml") or url.endswith(".msg"):
            logger.debug("📧 Extracting text from email...")
            text = _extract_text_from_email(content)
        else:
            logger.error(f"❌ Unsupported document type: {content_type}")
            return None

        extract_time = time.time() - extract_start
        total_time = time.time() - start_time
        text_length = len(text)
        
        logger.info(f"✅ Extracted {text_length} characters in {extract_time:.2f}s (total: {total_time:.2f}s)")
        return text

    except httpx.RequestError as e:
        total_time = time.time() - start_time
        logger.error(f"❌ Download failed after {total_time:.2f}s: {e}")
        return None
    except Exception as e:
        total_time = time.time() - start_time
        logger.error(f"❌ Processing failed after {total_time:.2f}s: {e}")
        return None

def get_text_chunks(text: str) -> list[str]:
    """Splits text into semantically coherent chunks."""
    start_time = time.time()
    logger.debug(f"✂️  Splitting {len(text)} characters into chunks...")
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_text(text)
    
    split_time = time.time() - start_time
    logger.info(f"✅ Split into {len(chunks)} chunks in {split_time:.3f}s")
    
    return chunks