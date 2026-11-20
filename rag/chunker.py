def chunk_pages(pages: list[dict], chunk_size: int = 600, chunk_overlap: int = 150) -> list[dict]:
    """
    Divides PDF pages into meaningful text chunks with appropriate overlap.
    Preserves page numbers and generates deterministic chunk IDs.
    
    Args:
        pages: List of dictionaries with 'page_number' and 'text'.
        chunk_size: Target character size for each chunk.
        chunk_overlap: Overlap size between consecutive chunks.
        
    Returns:
        List of chunk dictionaries with 'chunk_id', 'text', 'page', and 'metadata'.
    """
    chunks = []
    
    for page_info in pages:
        page_num = page_info["page_number"]
        text = page_info["text"]
        
        if not text:
            continue
            
        start = 0
        chunk_idx = 0
        text_len = len(text)
        
        while start < text_len:
            end = min(start + chunk_size, text_len)
            
            # If not at the very end of the text, try to find a natural break point (newline or period)
            if end < text_len:
                break_point = max(text.rfind("\n", start, end), text.rfind(". ", start, end))
                if break_point > start + (chunk_size // 2):
                    end = break_point + 1
                    
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                chunk_id = f"page_{page_num}_chunk_{chunk_idx}"
                chunks.append({
                    "chunk_id": chunk_id,
                    "text": chunk_text,
                    "page": page_num,
                    "metadata": {
                        "page": page_num,
                        "chunk_id": chunk_id,
                        "source": f"Page {page_num}"
                    }
                })
                chunk_idx += 1
                
            if end >= text_len:
                break
                
            # Advance start index accounting for overlap
            start = max(end - chunk_overlap, start + 1)
            
    return chunks
