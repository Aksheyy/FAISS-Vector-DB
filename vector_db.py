import json
import os
import numpy as np
import faiss
from openai import AzureOpenAI
from rank_bm25 import BM25Okapi
from config import AZURE_API_KEY, AZURE_ENDPOINT, AZURE_DEPLOYMENT
import pickle  # For saving embeddings
from dotenv import load_dotenv

load_dotenv()  # Ensure environment variables are loaded

class RequirementsVectorDB:
    def __init__(self, data_file="requirements.json", index_file="faiss_index.bin", metadata_file="metadata.json", embeddings_file="embeddings.pkl"):
        self.data_file = data_file
        self.index_file = index_file
        self.metadata_file = metadata_file
        self.embeddings_file = embeddings_file
        self.client = AzureOpenAI(
            api_key=AZURE_API_KEY,
            azure_endpoint=AZURE_ENDPOINT,
            api_version="2024-02-15-preview"
        )
        self.index = None
        self.bm25 = None
        self.ids = []
        self.texts = []  # Enriched texts for embedding and search
        self.raw_texts = []  # Raw texts for display
        self.embeddings = None

    def test_connection(self):
        try:
            response = self.client.embeddings.create(model=AZURE_DEPLOYMENT, input="Connection test")
            print("✅ Azure OpenAI connection successful!")
            return True
        except Exception as e:
            print("❌ Connection failed:", e)
            return False

    def load_data(self):
        with open(self.data_file, "r") as f:
            data = json.load(f)
        requirements = data["requirements"]
        self.texts = []
        self.raw_texts = []
        for req in requirements:
            # Use normalized_text as base and add context from object and constraint for better retrieval
            context_parts = [req["normalized_text"]]
            if "object" in req and "primary" in req["object"]:
                context_parts.append(req["object"]["primary"])
            if "constraint" in req and "type" in req["constraint"]:
                context_parts.append(req["constraint"]["type"])
            # Optionally add secondary objects
            if "object" in req and "secondary" in req["object"]:
                context_parts.extend(req["object"]["secondary"])
            enriched_text = " ".join(context_parts)
            self.texts.append(enriched_text)
            self.raw_texts.append(req["raw_text"])  # For display
        self.ids = [req["id"] for req in requirements]

    def generate_embeddings(self, force_regenerate=False):
        if os.path.exists(self.embeddings_file) and not force_regenerate:
            print("Loading existing embeddings...")
            with open(self.embeddings_file, "rb") as f:
                self.embeddings = pickle.load(f)
        else:
            print("Generating embeddings...")
            embeddings = []
            for text in self.texts:
                response = self.client.embeddings.create(model=AZURE_DEPLOYMENT, input=text)
                embeddings.append(response.data[0].embedding)
            self.embeddings = np.array(embeddings).astype("float32")
            faiss.normalize_L2(self.embeddings)  # Normalize for cosine similarity
            with open(self.embeddings_file, "wb") as f:
                pickle.dump(self.embeddings, f)
            print("Embeddings saved.")

    def build_index(self, index_type="flat_ip"):
        if self.embeddings is None:
            raise ValueError("Embeddings not generated. Call generate_embeddings first.")
        dimension = self.embeddings.shape[1]
        if index_type == "flat_ip":
            self.index = faiss.IndexFlatIP(dimension)
        elif index_type == "ivf":  # For larger datasets
            nlist = min(100, len(self.embeddings) // 39)  # Rule of thumb
            quantizer = faiss.IndexFlatIP(dimension)
            self.index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
            self.index.train(self.embeddings)
        else:
            raise ValueError("Unsupported index type")
        self.index.add(self.embeddings)
        faiss.write_index(self.index, self.index_file)
        print("Index built and saved.")

    def load_index(self):
        if os.path.exists(self.index_file):
            self.index = faiss.read_index(self.index_file)
            print("Index loaded.")
        else:
            raise FileNotFoundError("Index file not found. Build index first.")

    def prepare_bm25(self):
        tokenized_texts = [text.lower().split() for text in self.texts]  # Simple tokenization; improve with NLTK
        self.bm25 = BM25Okapi(tokenized_texts)

    def save_metadata(self):
        with open(self.metadata_file, "w") as f:
            json.dump({"ids": self.ids, "texts": self.texts, "raw_texts": self.raw_texts}, f)

    def load_metadata(self):
        with open(self.metadata_file, "r") as f:
            data = json.load(f)
        self.ids = data["ids"]
        self.texts = data["texts"]
        self.raw_texts = data.get("raw_texts", self.texts)  # Fallback if not present

    def query(self, query_text, k=10, alpha=0.6, beta=0.4):
        # Generate query embedding
        response = self.client.embeddings.create(model=AZURE_DEPLOYMENT, input=query_text)
        query_embedding = np.array([response.data[0].embedding]).astype("float32")
        faiss.normalize_L2(query_embedding)

        # FAISS search
        distances, indices = self.index.search(query_embedding, k)

        # BM25 search
        tokenized_query = query_text.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)

        # Normalize BM25 scores (using softmax for better distribution)
        exp_scores = np.exp(bm25_scores - np.max(bm25_scores))
        bm25_normalized = exp_scores / np.sum(exp_scores)

        # Combine scores
        combined_scores = {}
        for i in range(len(self.texts)):
            semantic_score = 0
            if i in indices[0]:
                pos = list(indices[0]).index(i)
                semantic_score = distances[0][pos]
            combined_scores[i] = alpha * semantic_score + beta * bm25_normalized[i]

        # Sort and return top results
        sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:k]
        return [(self.ids[idx], self.raw_texts[idx], score) for idx, score in sorted_results]

    def find_duplicates(self, threshold=0.8, k=5):
        """
        Find potential duplicate requirements within the dataset.
        For each requirement, find the top-k most similar others above the threshold.
        Returns a list of tuples: (id1, id2, similarity_score, text1, text2)
        """
        duplicates = []
        for i, text in enumerate(self.texts):
            # Query for similar requirements, excluding itself
            results = self.query(text, k=k+1)  # +1 to account for self
            for id_, sim_raw_text, score in results:
                if id_ != self.ids[i] and score >= threshold:
                    duplicates.append((self.ids[i], id_, score, self.raw_texts[i], sim_raw_text))
        return duplicates

# Usage example
if __name__ == "__main__":
    db = RequirementsVectorDB()
    if not db.test_connection():
        exit()

    db.load_data()
    db.generate_embeddings()
    db.build_index()  # Use "ivf" for larger datasets
    db.prepare_bm25()
    db.save_metadata()

    # Query example for similar requirements
    query_req = "POS should allow offline transactions"
    results = db.query(query_req, k=3)
    print(f"\nSimilar requirements to: '{query_req}'")
    for rank, (id_, text, score) in enumerate(results, 1):
        print(f"{rank}. {id_}: {text} (Score: {score:.4f})")

    # Find duplicates within the dataset
    print("\nPotential duplicates in the dataset:")
    duplicates = db.find_duplicates(threshold=0.7, k=2)
    for id1, id2, score, text1, text2 in duplicates[:5]:  # Show top 5
        print(f"Duplicate: {id1} <-> {id2} (Score: {score:.4f})")
        print(f"  Text1: {text1}")
        print(f"  Text2: {text2}\n")