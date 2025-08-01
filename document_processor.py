# document_processor.py
import httpx
import fitz  # PyMuPDF
import docx  # python-docx
import io
import email
from email import policy
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter

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
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=30, follow_redirects=True)
            response.raise_for_status()

        content = response.content
        content_type = response.headers.get("content-type", "").lower()

        print(f"Downloaded document with Content-Type: {content_type}")

        if "pdf" in content_type:
            return _extract_text_from_pdf(content)
        elif "vnd.openxmlformats-officedocument.wordprocessingml.document" in content_type or url.endswith(".docx"):
            return _extract_text_from_docx(content)
        elif "message/rfc822" in content_type or url.endswith(".eml") or url.endswith(".msg"):
            return _extract_text_from_email(content)
        else:
            print(f"Unsupported document type: {content_type}. Please provide a PDF, DOCX, or EML file.")
            return None

    except httpx.RequestError as e:
        print(f"Error downloading the document: {e}")
        return None
    except Exception as e:
        print(f"Error processing document: {e}")
        return None

def get_text_chunks(text: str) -> list[str]:
    """Splits text into semantically coherent chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
        length_function=len,
    )
    return splitter.split_text(text)