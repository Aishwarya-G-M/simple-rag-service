import pandas as pd
from pathlib import Path
import os
from dotenv import load_dotenv

# The purpose of this file was to verify all CSV messages were embedded and inspect the matrix shape.
# This logic is now moved to faiss_retriever.py
load_dotenv()

model = os.getenv('EMBEDDING_MODEL')

csv_path = os.getenv('CSV_PATH')

# Read with the correct encoding
df = pd.read_csv(csv_path, encoding="latin-1")
df = df.rename(columns={"v1": "label", "v2": "text"})

messages = df["text"].tolist()

embeddings = model.encode(messages, show_progress_bar=True)

#print(f"Number of messages: {len(messages)}")
#print(f"Embedding shape: {embeddings.shape}")
#print(f"First message: {messages[0]}")
#print(f"First embedding (first 5 dims): {embeddings[0][:5]}")