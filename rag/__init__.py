from rag.vector_store import VectorStore
from rag.retriever import Retriever
from rag.generator import AnswerGenerator
from rag.config import DEFAULT_PDF_PATH

class CulinaryRAGSystem:
    _instance = None

    def __init__(self, pdf_path: str = DEFAULT_PDF_PATH, force_reindex: bool = False):
        self.vector_store = VectorStore()
        self.vector_store.initialize(pdf_path=pdf_path, force_reindex=force_reindex)
        self.retriever = Retriever(self.vector_store)
        self.generator = AnswerGenerator()

    @classmethod
    def get_instance(cls, pdf_path: str = DEFAULT_PDF_PATH):
        if cls._instance is None:
            cls._instance = cls(pdf_path=pdf_path)
        return cls._instance

    def answer_question(self, question: str) -> dict:
        """
        Executes end-to-end RAG pipeline for a given question.
        Returns:
            {"answer": <str>, "sources": [<page_or_chunk_id>, ...]}
        """
        retrieval_res = self.retriever.retrieve(question)
        return self.generator.generate(question, retrieval_res)

def answer(question: str) -> dict:
    """
    Required assignment function.
    Returns {"answer": <str>, "sources": [<page_or_chunk_id>, ...]}
    """
    rag = CulinaryRAGSystem.get_instance()
    return rag.answer_question(question)

__all__ = ["CulinaryRAGSystem", "answer"]
