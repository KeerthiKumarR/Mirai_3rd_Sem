import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

# Default PDF Path
DEFAULT_PDF_PATH = os.getenv("PDF_PATH", str(BASE_DIR / "grand_bistro_culinary_recipe_manual_50p.pdf"))

# ChromaDB persistence directory
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", str(BASE_DIR / "chroma_db"))

# Chunking configuration
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "600"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

# Collection name in ChromaDB
COLLECTION_NAME = "grand_bistro_manual"

# LLM API keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# Retrieval thresholds
SIMILARITY_DISTANCE_THRESHOLD = float(os.getenv("SIMILARITY_DISTANCE_THRESHOLD", "1.10"))
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "4"))
