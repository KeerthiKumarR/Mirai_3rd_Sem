import os
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions

from rag.config import (
    CHROMA_PERSIST_DIR,
    COLLECTION_NAME,
    DEFAULT_PDF_PATH,
    CHUNK_SIZE,
    CHUNK_OVERLAP
)
from rag.ingestion import load_pdf
from rag.chunker import chunk_pages

class VectorStore:
    def __init__(self, persist_dir: str = CHROMA_PERSIST_DIR, collection_name: str = COLLECTION_NAME):
        """
        Initializes the ChromaDB persistent client and embedding function.
        """
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        
        # Ensure persist directory exists
        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize persistent client
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        
        # Standardized embedding function using ONNX all-MiniLM-L6-v2 (local & offline)
        self.embedding_function = embedding_functions.DefaultEmbeddingFunction()
        
        # Get or create the ChromaDB collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"}
        )

    def initialize(self, pdf_path: str = DEFAULT_PDF_PATH, force_reindex: bool = False):
        """
        Loads and indexes the PDF document if not already indexed in the persistent store.
        If already indexed, reloads the database without generating embeddings again.
        """
        current_count = self.collection.count()
        if current_count > 0 and not force_reindex:
            print(f"[*] ChromaDB collection '{self.collection_name}' already populated with {current_count} chunks. Reloading from disk...")
            return current_count
            
        print(f"[*] Ingesting PDF from: {pdf_path}")
        pages = load_pdf(pdf_path)
        print(f"[*] Loaded {len(pages)} pages. Generating text chunks...")
        
        chunks = chunk_pages(pages, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        print(f"[*] Created {len(chunks)} chunks. Generating embeddings and storing in ChromaDB...")
        
        if force_reindex and current_count > 0:
            print("[*] Clearing existing collection for reindexing...")
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function,
                metadata={"hnsw:space": "cosine"}
            )
            
        # Add documents in batches
        batch_size = 50
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            self.collection.add(
                documents=[c["text"] for c in batch],
                ids=[c["chunk_id"] for c in batch],
                metadatas=[c["metadata"] for c in batch]
            )
            
        total_indexed = self.collection.count()
        print(f"[+] Successfully indexed and persisted {total_indexed} chunks in ChromaDB at '{self.persist_dir}'.")
        return total_indexed

    def query(self, query_text: str, n_results: int = 4) -> dict:
        """
        Queries the persistent ChromaDB collection.
        
        Returns dict with keys: 'documents', 'distances', 'metadatas', 'ids'
        """
        return self.collection.query(
            query_texts=[query_text],
            n_results=n_results
        )
