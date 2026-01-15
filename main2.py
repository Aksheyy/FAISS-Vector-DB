
import json
import numpy as np
import faiss
from openai import AzureOpenAI
from config import AZURE_API_KEY, AZURE_ENDPOINT, AZURE_DEPLOYMENT

# Initialize Azure OpenAI client
client = AzureOpenAI(
    api_key=AZURE_API_KEY,
    azure_endpoint=AZURE_ENDPOINT,
    api_version="2024-02-15-preview"
)

#  Connection Test
try:
    test_response = client.embeddings.create(model=AZURE_DEPLOYMENT, input="Connection test")
    print("Azure OpenAI connection successful!")
    print(f"Embedding length: {len(test_response.data[0].embedding)}")
except Exception as e:
    print("Connection failed:", e)
    exit()

#  Load requirements
with open("requirements.json", "r") as f:
    data = json.load(f)

requirements = data["requirements"]

#  Use raw_text for better context
texts = [req["raw_text"] for req in requirements]
ids = [req["id"] for req in requirements]

#  Generate embeddings
embeddings = []
for text in texts:
    response = client.embeddings.create(model=AZURE_DEPLOYMENT, input=text)
    embeddings.append(response.data[0].embedding)

#  Convert to NumPy and normalize for cosine similarity
embeddings = np.array(embeddings).astype("float32")
faiss.normalize_L2(embeddings)

#  Create FAISS index using cosine similarity
dimension = len(embeddings[0])
index = faiss.IndexFlatIP(dimension)  # Inner Product for cosine similarity
index.add(embeddings)

#  Save FAISS index
faiss.write_index(index, "faiss_index.bin")
print("FAISS index saved successfully!")

#  Save metadata mapping
with open("metadata.json", "w") as f:
    json.dump({"ids": ids, "texts": texts}, f)

#  Query example
query = "The proposed POS solution must minimally meet the capabilities of critical functions and features that"
query_embedding = client.embeddings.create(model=AZURE_DEPLOYMENT, input=query).data[0].embedding
query_vector = np.array([query_embedding]).astype("float32")
faiss.normalize_L2(query_vector)  # Normalize query for cosine similarity

#  Search top 3 matches
distances, indices = index.search(query_vector, k=3)

# Load metadata
with open("metadata.json", "r") as f:
    meta = json.load(f)

print("\nTop matches:")
for rank, i in enumerate(indices[0]):
    print(f"{rank+1}. {meta['ids'][i]}: {meta['texts'][i]} (Score: {distances[0][rank]:.4f})")
