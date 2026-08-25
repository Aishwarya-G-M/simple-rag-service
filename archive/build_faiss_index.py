import pandas as pd
import faiss
import os
from dotenv import load_dotenv

load_dotenv()

model = os.getenv('EMBEDDING_MODEL')

csv_path = os.getenv('CSV_PATH')

df = pd.read_csv(csv_path, encoding="latin-1")
df = df.rename(columns={"v1": "label", "v2": "text"})

messages = df["text"].tolist()

# Create embeddings
embeddings = model.encode(messages)

# Ensure float32 and 2D array for FAISS
X = embeddings.astype("float32")

dim = X.shape[1]

# Create a flat L2 index
index = faiss.IndexFlatL2(dim)

# Add vectors to the index
index.add(X)

#print(f"FAISS index size: {index.ntotal}")
#print(f"Index dimension: {dim}")

# Example query
query_text = "Free entry to win a cash prize. Text WIN now."
query_vec = model.encode([query_text]).astype("float32")

k = 5  # number of nearest neighbors
distances, indices = index.search(query_vec, k)

#print("Query:", query_text)
#print("Indices of nearest neighbors:", indices[0])
#print("Distances:", distances[0])
#print("Nearest messages:")
for i in indices[0]:
    print("-", messages[i])