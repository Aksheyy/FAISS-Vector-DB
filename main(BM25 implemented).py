
import json
import re
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from openai import AzureOpenAI
from config import AZURE_API_KEY, AZURE_ENDPOINT, AZURE_DEPLOYMENT

# =========================================================
# Azure OpenAI Client
# =========================================================
client = AzureOpenAI(
    api_key=AZURE_API_KEY,
    azure_endpoint=AZURE_ENDPOINT,
    api_version="2024-02-15-preview"
)

# =========================================================
# Load Requirements
# =========================================================
with open("requirements.json", "r", encoding="utf-8") as f:
    data = json.load(f)

requirements = [
    r for r in data.get("requirements", [])
    if r.get("raw_text") and r.get("normalized_text")
]

# =========================================================
# Helper Functions
# =========================================================
def clean(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def tokenize(text: str):
    return clean(text).split()

def build_embedding_text(req: dict) -> str:
    return f"""
    Requirement: {req.get('normalized_text', '')}
    Description: {req.get('raw_text', '')}
    Action: {req.get('action', {}).get('verb', '')}
            ({req.get('action', {}).get('modality', '')})
    Object: {req.get('object', {}).get('primary', '')}
    Category: {req.get('constraint', {}).get('type', 'unknown')}
    Context: Airport Retail POS
    """

def build_bm25_text(req: dict) -> str:
    return " ".join([
        req.get("normalized_text", ""),
        req.get("object", {}).get("primary", ""),
        req.get("constraint", {}).get("type", "")
    ])

def infer_constraint_boost(query: str):
    q = query.lower()
    if any(w in q for w in ["offline", "outage", "network", "disconnect"]):
        return "resilience"
    if any(w in q for w in ["gst", "tax", "regulation", "compliance"]):
        return "compliance"
    if any(w in q for w in ["security", "access", "encrypt", "audit"]):
        return "security"
    if any(w in q for w in ["report", "analytics"]):
        return "reporting"
    if any(w in q for w in ["payment", "refund", "wallet", "currency"]):
        return "functional"
    return None

# =========================================================
# Build Documents
# =========================================================
embedding_docs = [build_embedding_text(r) for r in requirements]
bm25_docs = [build_bm25_text(r) for r in requirements]
ids = [r.get("id") for r in requirements]

# =========================================================
# Create Embeddings (Batch)
# =========================================================
response = client.embeddings.create(
    model=AZURE_DEPLOYMENT,
    input=embedding_docs
)

embeddings = np.array(
    [e.embedding for e in response.data],
    dtype="float32"
)

faiss.normalize_L2(embeddings)

# =========================================================
# FAISS Index
# =========================================================
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(embeddings)

# =========================================================
# BM25 Index
# =========================================================
bm25 = BM25Okapi([tokenize(t) for t in bm25_docs])

# =========================================================
# Hybrid Search
# =========================================================
def hybrid_search(query, top_k=5, alpha=0.7):
    beta = 1 - alpha

    # ---- Vector Search ----
    q_embedding = client.embeddings.create(
        model=AZURE_DEPLOYMENT,
        input=query
    ).data[0].embedding

    q_vector = np.array([q_embedding], dtype="float32")
    faiss.normalize_L2(q_vector)

    faiss_scores, faiss_indices = index.search(q_vector, top_k)

    # ---- BM25 Search ----
    bm25_scores = np.array(bm25.get_scores(tokenize(query)))
    bm25_scores = (bm25_scores - bm25_scores.min()) / (
        bm25_scores.max() - bm25_scores.min() + 1e-9
    )

    # ---- Intent Boost ----
    constraint_hint = infer_constraint_boost(query)

    results = []
    for rank, idx in enumerate(faiss_indices[0]):
        score = (
            alpha * faiss_scores[0][rank]
            + beta * bm25_scores[idx]
        )

        # Boost by constraint type
        if constraint_hint and requirements[idx].get("constraint", {}).get("type") == constraint_hint:
            score *= 1.15

        # Boost MUST requirements
        if requirements[idx].get("action", {}).get("modality") == "must":
            score *= 1.10

        # Confidence weighting
        score *= requirements[idx].get("confidence_score", 1.0)

        results.append((idx, score))

    return sorted(results, key=lambda x: x[1], reverse=True)

# =========================================================
# Run Example Query
# =========================================================
query = "offline transactions"
results = hybrid_search(query, top_k=4)

print("\n✅ Top Hybrid Search Results:\n")
for rank, (idx, score) in enumerate(results, 1):
    r = requirements[idx]
    print(f"{rank}. {r['id']} — {r['normalized_text']}")
    print(f"   Category: {r.get('constraint', {}).get('type')}")
    print("")
