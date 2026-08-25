import os
from dotenv import load_dotenv

load_dotenv()

# This was the first try to verify that the embedding model works.
model = os.getenv('EMBEDDING_MODEL')

text = "Free entry to win a cash prize. Text WIN now."
embedding = model.encode(text)

#print(f"Embedding dimensions: {len(embedding)}")
#print(f"First five values: {embedding[:5]}")