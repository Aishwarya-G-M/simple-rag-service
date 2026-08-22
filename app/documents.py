import pandas as pd
from pathlib import Path
from typing import List, Dict, Any

def load_sms_spam_csv(path: str) -> List[Dict[str, Any]]:
    """
    Load the SMS spam CSV and return a list of document dicts.
    Each document has:
      - id: string
      - text: SMS message
      - label: 'ham' or 'spam'
      - source: 'sms_spam_csv'
    """
    df = pd.read_csv(path, encoding="latin-1", header=0)

    # Normalize column names
    # Expected: v1 -> label, v2 -> text
    df = df.rename(columns={"v1": "label", "v2": "text"})

    docs = []
    for idx, row in df.iterrows():
        docs.append(
            {
                "id": str(idx),
                "text": str(row["text"]),
                "label": str(row["label"]),
                "source": "sms_spam_csv",
            }
        )
    return docs

def load_documents(data_dir: str = "data") -> List[Dict[str, Any]]:
    """
    Load all documents for the RAG service.
    For now, we just load the SMS spam CSV.
    """
    data_path = Path(data_dir) / "sms_spam.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Expected CSV at {data_path}")

    return load_sms_spam_csv(str(data_path))