import pandas as pd
from pathlib import Path
import faiss
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

CSV_PATH = os.getenv('CSV_PATH')
MODEL_NAME = os.getenv('EMBEDDING_MODEL')


class FaissRetriever:
    def __init__(self, csv_path: Path = CSV_PATH, model_name: str = MODEL_NAME):
        self.csv_path = csv_path
        self.model_name = model_name

        self.model: SentenceTransformer | None = None
        self.messages = None
        self.index = None
        self.dim = None

    def load(self):
        # Load model
        self.model = SentenceTransformer(self.model_name)

        # Load messages
        df = pd.read_csv(self.csv_path, encoding="latin-1")
        df = df.rename(columns={"v1": "label", "v2": "text"})
        self.messages = df["text"].tolist()

        # Build embeddings
        embeddings = self.model.encode(self.messages)
        X = embeddings.astype("float32")

        # Build FAISS index
        self.dim = X.shape[1]
        self.index = faiss.IndexFlatL2(self.dim)
        self.index.add(X)

    def retrieve_similar(self, query_text: str, k: int = 5):
        if self.index is None:
            raise RuntimeError("Call .load() first")

        query_vec = self.model.encode([query_text]).astype("float32")
        distances, indices = self.index.search(query_vec, k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            results.append({
                "index": int(idx),
                "distance": float(dist),
                "text": self.messages[idx],
            })
        return results