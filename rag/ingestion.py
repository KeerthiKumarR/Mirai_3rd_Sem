import os
from pathlib import Path
import pypdf

def load_pdf(pdf_path: str) -> list[dict]:
    """
    Ingests and processes a PDF file.
    
    Args:
        pdf_path: Path to the target PDF file.
        
    Returns:
        List of dictionaries containing 'page_number' and 'text'.
    """
    path = Path(pdf_path)
    if not path.is_file():
        raise FileNotFoundError(f"PDF file not found at: {pdf_path}")
        
    reader = pypdf.PdfReader(str(path))
    total_pages = len(reader.pages)
    documents = []
    
    for idx, page in enumerate(reader.pages):
        page_num = idx + 1
        raw_text = page.extract_text() or ""
        # Clean up text while preserving structure
        cleaned_text = raw_text.strip()
        documents.append({
            "page_number": page_num,
            "text": cleaned_text,
            "total_pages": total_pages
        })
        
    return documents
