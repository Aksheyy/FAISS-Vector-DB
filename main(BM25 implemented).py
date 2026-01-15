
import json
import numpy as np
import faiss
from openai import AzureOpenAI
from config import AZURE_API_KEY, AZURE_ENDPOINT, AZURE_DEPLOYMENT
from rank_bm25 import BM25Okapi

# Initialize Azure OpenAI client
client = AzureOpenAI(
    api_key=AZURE_API_KEY,
    azure_endpoint=AZURE_ENDPOINT,
    api_version="2024-02-15-preview"
)

# ✅ Connection Test
try:
    test_response = client.embeddings.create(model=AZURE_DEPLOYMENT, input="Connection test")
    print("✅ Azure OpenAI connection successful!")
except Exception as e:
    print("❌ Connection failed:", e)
    exit()

# ✅ Load requirements
with open("requirements.json", "r") as f:
    data = json.load(f)

requirements = data["requirements"]

# Use raw_text for better context
texts = [req["raw_text"] for req in requirements]
ids = [req["id"] for req in requirements]

# ✅ Generate embeddings
embeddings = []
for text in texts:
    response = client.embeddings.create(model=AZURE_DEPLOYMENT, input=text)
    embeddings.append(response.data[0].embedding)

# ✅ Convert to NumPy and normalize for cosine similarity
embeddings = np.array(embeddings).astype("float32")
faiss.normalize_L2(embeddings)

# ✅ Create FAISS index
dimension = len(embeddings[0])
index = faiss.IndexFlatIP(dimension)
index.add(embeddings)

# ✅ Save FAISS index
faiss.write_index(index, "faiss_index.bin")

# ✅ Save metadata
with open("metadata.json", "w") as f:
    json.dump({"ids": ids, "texts": texts}, f)

# ✅ Prepare BM25
tokenized_texts = [text.lower().split() for text in texts]
bm25 = BM25Okapi(tokenized_texts)

# ✅ Query
query = "POS should allow offline transactions"
query_embedding = client.embeddings.create(model=AZURE_DEPLOYMENT, input=query).data[0].embedding
query_vector = np.array([query_embedding]).astype("float32")
faiss.normalize_L2(query_vector)

# ✅ FAISS Search
faiss_distances, faiss_indices = index.search(query_vector, k=10)

# ✅ BM25 Search
tokenized_query = query.lower().split()
bm25_scores = bm25.get_scores(tokenized_query)

# ✅ Normalize BM25 scores (Min-Max)
bm25_min = min(bm25_scores)
bm25_max = max(bm25_scores)
bm25_normalized = [(score - bm25_min) / (bm25_max - bm25_min) if bm25_max != bm25_min else 0 for score in bm25_scores]

# ✅ Combine Scores (Hybrid)
alpha = 0.6  # weight for FAISS
beta = 0.4   # weight for BM25
combined_scores = {}

for i in range(len(texts)):
    semantic_score = 0
    if i in faiss_indices[0]:
        semantic_score = faiss_distances[0][list(faiss_indices[0]).index(i)]
    combined_scores[i] = alpha * semantic_score + beta * bm25_normalized[i]

# ✅ Sort by combined score
sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:3]

# ✅ Display Results
print("\nTop Hybrid Matches:")
for rank, (idx, score) in enumerate(sorted_results):
    print(f"{rank+1}. {ids[idx]}: {texts[idx]} (Score: {score:.4f})")
