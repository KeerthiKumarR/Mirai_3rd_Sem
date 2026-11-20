import re
from rag.config import SIMILARITY_DISTANCE_THRESHOLD, TOP_K_RESULTS
from rag.vector_store import VectorStore

class Retriever:
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store

    def retrieve(self, query: str, top_k: int = TOP_K_RESULTS) -> dict:
        """
        Retrieves relevant chunks for a question, applying distance thresholding
        and semantic relevance filtering.
        
        Returns:
            {
                "is_relevant": bool,
                "chunks": list of chunk dicts,
                "sources": list of page/chunk IDs (e.g., ["Page 10"]),
                "combined_context": str
            }
        """
        res = self.vector_store.query(query_text=query, n_results=top_k)
        
        documents = res.get("documents", [[]])[0]
        distances = res.get("distances", [[]])[0]
        metadatas = res.get("metadatas", [[]])[0]
        ids = res.get("ids", [[]])[0]
        
        if not documents:
            return {
                "is_relevant": False,
                "chunks": [],
                "sources": [],
                "combined_context": ""
            }
            
        top_distance = distances[0] if distances else 999.0
        
        # Relevance threshold check:
        # In cosine distance space, relevant matches for this manual are typically <= 0.45.
        # Queries outside the manual (like Eiffel Tower or Noma) score > 0.65.
        is_relevant = top_distance < 0.50
        
        # Check entity presence: if query mentions specific named entities, check if they appear
        # in the retrieved context
        named_entities = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", query)
        stop_words = {"What", "Which", "At", "According", "Under", "Where", "When", "How", "The"}
        key_entities = [e for e in named_entities if e not in stop_words]
        
        combined_text = "\n\n".join(documents).lower()
        if key_entities:
            entity_match = False
            for ent in key_entities:
                for word in ent.lower().split():
                    if len(word) > 3 and word in combined_text:
                        entity_match = True
                        break
                if entity_match:
                    break
            if not entity_match:
                is_relevant = False

        if not is_relevant:
            return {
                "is_relevant": False,
                "chunks": [],
                "sources": [],
                "combined_context": ""
            }
            
        # Select chunks tightly clustered around the top match (within +0.20 of min distance and <= 0.50)
        selected_chunks = []
        sources = []
        seen_pages = set()
        
        for doc, dist, meta, cid in zip(documents, distances, metadatas, ids):
            if dist <= min(top_distance + 0.15, 0.50):
                page_num = meta.get("page")
                page_str = f"Page {page_num}" if page_num else meta.get("source", cid)
                if page_str not in seen_pages:
                    seen_pages.add(page_str)
                    sources.append(page_str)
                selected_chunks.append({
                    "id": cid,
                    "text": doc,
                    "distance": dist,
                    "page": page_num,
                    "metadata": meta
                })
                
        if not selected_chunks:
            return {
                "is_relevant": False,
                "chunks": [],
                "sources": [],
                "combined_context": ""
            }
            
        return {
            "is_relevant": True,
            "chunks": selected_chunks,
            "sources": sources,
            "combined_context": "\n\n---\n\n".join([c["text"] for c in selected_chunks])
        }
