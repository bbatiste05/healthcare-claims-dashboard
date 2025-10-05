# copilot/rag.py
import pandas as pd
from pathlib import Path

class SimpleRAG:
    """
    Simple Retrieval-Augmented Generation (RAG) utility.
    Loads optional local reference data (ICD, CPT, provider metadata)
    and returns lightweight 'snippets' to provide external context to GPT.
    """

    def __init__(self, base_dir: str | Path = "data/"):
        # Ensure the path works whether a string or Path object is passed
        self.base = Path(base_dir)
        self.icd = None
        self.cpt = None
        self.nppes = None

        # Try loading any local reference files if they exist
        self._load_references()

    def _load_references(self):
        """Load optional ICD, CPT, and NPPES reference CSVs."""
        try:
            if (self.base / "icd.csv").exists():
                self.icd = pd.read_csv(self.base / "icd.csv")
            if (self.base / "cpt.csv").exists():
                self.cpt = pd.read_csv(self.base / "cpt.csv")
            if (self.base / "nppes.csv").exists():
                self.nppes = pd.read_csv(self.base / "nppes.csv")
        except Exception as e:
            print(f"[SimpleRAG] Warning: failed to load references: {e}")

    def search(self, query: str, k: int = 5):
        """
        Return lightweight text snippets that GPT can use as external context.
        This is intentionally simple — you could later expand with embeddings or BM25.
        """
        snippets = []

        # Match keywords to ICD/CPT if present
        if self.icd is not None and "icd" in query.lower():
            results = self.icd.head(k).to_dict(orient="records")
            snippets.extend(results)

        if self.cpt is not None and "cpt" in query.lower():
            results = self.cpt.head(k).to_dict(orient="records")
            snippets.extend(results)

        if self.nppes is not None and "provider" in query.lower():
            results = self.nppes.head(k).to_dict(orient="records")
            snippets.extend(results)

        return snippets
