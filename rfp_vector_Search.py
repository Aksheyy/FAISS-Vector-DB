# rfp_intelligent_assistant.py
"""
Ultra-Intelligent RFP Requirements Assistant

This system acts like an intelligent assistant (like me!) that:
- Understands ANY query naturally
- Extracts information dynamically
- Formats responses adaptively based on what's needed
- Provides summaries, counts, lists, comparisons automatically

NO rigid structure - pure conversational intelligence.
STRICTLY GROUNDED - Only uses requirements from the database.
"""

import json
import os
import re
import sys
import pickle
from pathlib import Path
from collections import defaultdict, Counter
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional

try:
    import numpy as np
    import faiss
    from openai import AzureOpenAI
    from rank_bm25 import BM25Okapi
    from dotenv import load_dotenv
except ImportError as e:
    print(f"Missing dependency: {e}")
    sys.exit(1)

# =========================
# Configuration
# =========================
SCRIPT_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = SCRIPT_DIR
DEFAULT_DATA_FILE = os.getenv("REQS_JSON") or str(SCRIPT_DIR / "requirements.json")

INDEX_FILE = str(ARTIFACT_DIR / "faiss_index.index")
EMBS_FILE = str(ARTIFACT_DIR / "embeddings.npy")
METADATA_FILE = str(ARTIFACT_DIR / "metadata.pkl")
TEXTS_FILE = str(ARTIFACT_DIR / "texts.pkl")
BM25_FILE = str(ARTIFACT_DIR / "bm25.pkl")
VOCAB_FILE = str(ARTIFACT_DIR / "bm25_vocab.pkl")

SIMILARITY_THRESHOLD = 0.30
RRF_K = 60

SHOW_DEBUG = os.getenv("DEBUG", "").lower() == "true"

# =========================
# Azure OpenAI Setup
# =========================
load_dotenv()
client = AzureOpenAI(
    api_key=os.getenv('OPENAI_API_KEY'),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2023-05-15"),
    azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT')
)
EMBED_MODEL = os.getenv('AZURE_OPENAI_DEPLOYMENT')
CHAT_MODEL = os.getenv('AZURE_OPENAI_CHAT_DEPLOYMENT')

if not all([os.getenv('AZURE_OPENAI_ENDPOINT'), os.getenv('OPENAI_API_KEY'), EMBED_MODEL, CHAT_MODEL]):
    raise ValueError("Missing Azure OpenAI credentials")

# =========================
# Utilities
# =========================
def word_tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9]+", (text or "").lower())

def normalize_vecs(vecs: np.ndarray) -> np.ndarray:
    faiss.normalize_L2(vecs)
    return vecs

def get_embedding(text: str) -> np.ndarray:
    resp = client.embeddings.create(input=text, model=EMBED_MODEL)
    emb = np.array(resp.data[0].embedding, dtype='float32')[None, :]
    return normalize_vecs(emb)

def call_llm(messages: List[Dict[str, str]], max_tokens: int = 800, temperature: float = 0.0) -> str:
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature
    )
    return resp.choices[0].message.content.strip()

# =========================
# Data Loading
# =========================
def build_searchable_text(req: Dict[str, Any]) -> str:
    """Build comprehensive searchable text"""
    parts = []
    parts.append(req.get("normalized_text", ""))
    parts.append(req.get("raw_text", ""))
    
    cls = req.get("classification", {})
    entities = req.get("entities", {})
    action = req.get("action", {})
    constraint = req.get("constraint", {})
    
    if action.get("verb"):
        parts.append(f"action: {action.get('verb')} {action.get('modality', '')}")
    if cls.get('requirement_type'):
        parts.append(f"type: {cls.get('requirement_type')}")
    if cls.get('criticality'):
        parts.append(f"criticality: {cls.get('criticality')}")
    if constraint.get("description"):
        parts.append(f"constraint: {constraint.get('description')}")
    
    for k in ("systems", "standards", "regulations"):
        vals = entities.get(k, [])
        if isinstance(vals, list) and vals:
            parts.append(f"{k}: " + ", ".join(str(v) for v in vals))
    
    return " | ".join([p for p in parts if p]).lower()

def load_requirements(path: str) -> List[Dict[str, Any]]:
    """Load requirements from JSON"""
    with Path(path).open("r") as f:
        data = json.load(f)
    
    reqs = []
    for r in data.get("requirements", []):
        if r.get("client_reference_id") and r.get("normalized_text"):
            if not r.get("relationships", {}).get("duplicate_of"):
                reqs.append(r)
    return reqs

def build_or_load_indexes(data_file: str):
    """Build or load indexes"""
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    
    if all(Path(p).exists() for p in [INDEX_FILE, EMBS_FILE, METADATA_FILE, TEXTS_FILE, BM25_FILE, VOCAB_FILE]):
        print("Loading indexes...")
        index = faiss.read_index(INDEX_FILE)
        embeddings = np.load(EMBS_FILE)
        with open(METADATA_FILE, "rb") as f:
            metadata = pickle.load(f)
        with open(TEXTS_FILE, "rb") as f:
            texts = pickle.load(f)
        with open(BM25_FILE, "rb") as f:
            bm25 = pickle.load(f)
        with open(VOCAB_FILE, "rb") as f:
            tokenized = pickle.load(f)
        return index, embeddings, metadata, texts, bm25, tokenized
    
    print("Building indexes...")
    reqs = load_requirements(data_file)
    texts = [build_searchable_text(r) for r in reqs]
    
    tokenized = [word_tokenize(t) for t in texts]
    bm25 = BM25Okapi(tokenized)
    
    print(f"Generating embeddings for {len(texts)} requirements...")
    embs = []
    for i, t in enumerate(texts):
        if i % 50 == 0:
            print(f"  {i+1}/{len(texts)}")
        embs.append(get_embedding(t)[0])
    embeddings = np.array(embs, dtype="float32")
    normalize_vecs(embeddings)
    
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    
    metadata = {r["client_reference_id"]: r for r in reqs}
    
    faiss.write_index(index, INDEX_FILE)
    np.save(EMBS_FILE, embeddings)
    with open(METADATA_FILE, "wb") as f:
        pickle.dump(metadata, f)
    with open(TEXTS_FILE, "wb") as f:
        pickle.dump(texts, f)
    with open(BM25_FILE, "wb") as f:
        pickle.dump(bm25, f)
    with open(VOCAB_FILE, "wb") as f:
        pickle.dump(tokenized, f)
    
    print("Indexes built.")
    return index, embeddings, metadata, texts, bm25, tokenized

# =========================
# Intelligent Assistant
# =========================
class IntelligentRfpAssistant:
    """
    Acts like an intelligent assistant that understands queries naturally
    and provides adaptive responses like a human would.
    
    STRICTLY GROUNDED: Only uses requirements from the database.
    """
    
    def __init__(self, data_file: str = DEFAULT_DATA_FILE):
        self.index, self.embeddings, self.metadata, self.texts, self.bm25, self.tokenized = build_or_load_indexes(data_file)
        self.ids = list(self.metadata.keys())
        self.id_to_idx = {rid: i for i, rid in enumerate(self.ids)}
        print(f"Loaded {len(self.ids)} requirements\n")
    
    def answer_query(self, query: str) -> str:
        """
        Main method - understands query and provides intelligent response.
        The LLM decides everything: what to search, how to analyze, how to respond.
        """
        
        # First, validate that this is a requirements query
        if not self._is_requirements_query(query):
            return self._handle_non_requirements_query(query)
        
        # Let LLM understand what data operations are needed
        analysis_plan = self._get_analysis_plan(query)
        
        if SHOW_DEBUG:
            print("\n=== Analysis Plan ===")
            print(json.dumps(analysis_plan, indent=2))
        
        # Execute the data operations
        data_context = self._execute_data_operations(analysis_plan)
        
        if SHOW_DEBUG:
            print("\n=== Data Context ===")
            print(json.dumps(data_context, indent=2)[:500] + "...")
        
        # Validate that we found relevant requirements
        if not data_context.get("requirements") and not data_context.get("groups"):
            return self._handle_no_results(query)
        
        # Let LLM generate the final response using ONLY the retrieved data
        response = self._generate_intelligent_response(query, analysis_plan, data_context)
        
        return response
    
    def _is_requirements_query(self, query: str) -> bool:
        """
        Check if the query is about requirements in the database.
        Returns False for general questions not related to requirements.
        """
        # Keywords that indicate a requirements query
        req_indicators = [
            'requirement', 'req', 'specification', 'spec',
            'list', 'show', 'find', 'get', 'search',
            'how many', 'count', 'total',
            'what', 'which', 'where',
            'mandatory', 'optional', 'critical',
            'security', 'functional', 'compliance',
            'similar', 'compare', 'related',
            'pos-', 'fr-', 'sec-', 'nfr-',  # Common ID patterns
        ]
        
        query_lower = query.lower()
        
        # Check for requirement indicators
        has_indicator = any(indicator in query_lower for indicator in req_indicators)
        
        # If no indicators, it's likely not a requirements query
        if not has_indicator:
            return False
        
        # Additional check: general knowledge questions
        general_patterns = [
            r'^what is ',
            r'^who is ',
            r'^when was ',
            r'^how does .+ work',
            r'^explain .+ (concept|theory|principle)',
            r'^define ',
        ]
        
        for pattern in general_patterns:
            if re.match(pattern, query_lower):
                # Check if it's asking about requirements specifically
                if 'requirement' not in query_lower:
                    return False
        
        return True
    
    def _handle_non_requirements_query(self, query: str) -> str:
        """
        Handle queries that are not about requirements.
        """
        return """I'm a specialized assistant for RFP requirements analysis. I can only answer questions about the requirements in this database.

I can help you with:
• Listing requirements by type, criticality, or system
• Counting and grouping requirements
• Finding similar requirements
• Comparing requirements
• Searching for specific features or capabilities

Please ask a question about the requirements in the database."""
    
    def _handle_no_results(self, query: str) -> str:
        """
        Handle cases where no requirements match the query.
        """
        return f"""I couldn't find any requirements matching your query: "{query}"

This could mean:
• No requirements in the database match these criteria
• The search terms might need adjustment
• The feature/concept you're looking for might not be covered in these requirements

Try:
• Using different search terms
• Broadening your search criteria
• Asking about a different aspect of the requirements

Available requirement types: functional, security, compliance, integration, non-functional
Available criticality levels: mandatory, high, medium, low"""
    
    def _get_analysis_plan(self, query: str) -> Dict[str, Any]:
        """
        LLM analyzes the query and decides what data operations are needed.
        """
        
        system_prompt = """You are an intelligent data analyst for an RFP requirements database.

Given a user query, you need to determine what data operations are required to answer it.

CRITICAL: You can ONLY work with requirements that exist in the database. Do NOT make up or invent requirements.

AVAILABLE OPERATIONS:
1. retrieve_by_id - Get specific requirements by ID
2. search_semantic - Search by meaning/concepts
3. filter_by_attributes - Filter by type, criticality, systems, etc.
4. count - Count requirements matching criteria
5. group_and_aggregate - Group by field and count
6. find_similar - Find similar requirements
7. analyze_text - Analyze text content across requirements
8. summarize - Summarize matching requirements

REQUIREMENT SCHEMA:
- client_reference_id: e.g., "POS-FR-001"
- normalized_text, raw_text: Requirement descriptions
- classification.requirement_type: functional, non_functional, security, compliance, integration
- classification.criticality: mandatory, high, medium, low
- entities.systems: List of systems
- entities.standards, regulations, regions
- action.verb, action.modality
- constraint.type, constraint.description

Analyze the query and return a JSON plan."""

        user_prompt = f"""Query: "{query}"

Determine what operations are needed to answer this query using ONLY the requirements database.

Return JSON with this structure:
{{
  "query_type": "list|count|summary|comparison|analysis",
  "operations": [
    {{
      "operation": "retrieve_by_id|search_semantic|filter_by_attributes|count|group_and_aggregate|find_similar|analyze_text|summarize",
      "parameters": {{
        "ids": [],  // for retrieve_by_id
        "concepts": [],  // for search_semantic
        "filters": {{  // for filter_by_attributes
          "requirement_type": [],
          "criticality": [],
          "systems": [],
          "text_contains": [],
          "text_excludes": []
        }},
        "group_by": [],  // for group_and_aggregate
        "similarity_anchor": null,  // for find_similar
        "analysis_focus": ""  // for analyze_text/summarize
      }}
    }}
  ],
  "response_format": "list|table|summary|grouped|count",
  "explanation": "What the user is asking for and how to answer it"
}}

Be thorough - extract all relevant information from the query."""

        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            response = call_llm(messages, max_tokens=700)
            
            # Extract JSON
            match = re.search(r'\{.*\}', response, re.DOTALL)
            if match:
                return json.loads(match.group(0))
        except Exception as e:
            if SHOW_DEBUG:
                print(f"Plan generation failed: {e}")
        
        # Fallback
        return {
            "query_type": "list",
            "operations": [{"operation": "search_semantic", "parameters": {"concepts": [query]}}],
            "response_format": "list",
            "explanation": "Basic search"
        }
    
    def _execute_data_operations(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the data operations specified in the plan.
        """
        results = {
            "requirements": [],
            "count": 0,
            "groups": {},
            "summary_data": []
        }
        
        for op in plan.get("operations", []):
            operation = op.get("operation")
            params = op.get("parameters", {})
            
            if operation == "retrieve_by_id":
                ids = params.get("ids", [])
                for rid in ids:
                    if rid in self.metadata:
                        results["requirements"].append(self.metadata[rid])
            
            elif operation == "search_semantic":
                concepts = params.get("concepts", [])
                reqs = self._semantic_search(" ".join(concepts), top_k=100)
                results["requirements"].extend(reqs)
            
            elif operation == "filter_by_attributes":
                filters = params.get("filters", {})
                if not results["requirements"]:
                    # Filter from all requirements
                    candidates = list(self.metadata.values())
                else:
                    candidates = results["requirements"]
                
                filtered = self._apply_filters(candidates, filters)
                results["requirements"] = filtered
            
            elif operation == "count":
                if results["requirements"]:
                    results["count"] = len(results["requirements"])
                else:
                    results["count"] = len(self.metadata)
            
            elif operation == "group_and_aggregate":
                group_by = params.get("group_by", [])
                if group_by:
                    groups = defaultdict(list)
                    reqs_to_group = results["requirements"] if results["requirements"] else list(self.metadata.values())
                    
                    for req in reqs_to_group:
                        key_parts = []
                        for field in group_by:
                            if field in ["requirement_type", "criticality"]:
                                val = req.get("classification", {}).get(field, "unknown")
                            else:
                                val = "unknown"
                            key_parts.append(val)
                        key = " | ".join(key_parts)
                        groups[key].append(req)
                    
                    results["groups"] = {k: len(v) for k, v in groups.items()}
            
            elif operation == "find_similar":
                anchor = params.get("similarity_anchor")
                if anchor:
                    similar = self._find_similar(anchor, top_k=20)
                    results["requirements"] = similar
            
            elif operation == "analyze_text":
                focus = params.get("analysis_focus", "")
                reqs_to_analyze = results["requirements"] if results["requirements"] else list(self.metadata.values())
                
                # Find requirements mentioning the focus
                relevant = []
                for req in reqs_to_analyze:
                    text = (req.get("normalized_text", "") + " " + req.get("raw_text", "")).lower()
                    if focus.lower() in text:
                        relevant.append(req)
                results["requirements"] = relevant
            
            elif operation == "summarize":
                # Just mark that we need to summarize
                results["summary_data"] = results["requirements"] if results["requirements"] else list(self.metadata.values())[:10]
        
        # Deduplicate requirements
        if results["requirements"]:
            seen = set()
            unique = []
            for req in results["requirements"]:
                rid = req["client_reference_id"]
                if rid not in seen:
                    seen.add(rid)
                    unique.append(req)
            results["requirements"] = unique
            results["count"] = len(unique)
        
        return results
    
    def _semantic_search(self, query: str, top_k: int = 100) -> List[Dict[str, Any]]:
        """Perform semantic search"""
        q_emb = get_embedding(query)
        
        # Vector search
        D, I = self.index.search(q_emb, min(top_k, len(self.ids)))
        vec_results = [(self.ids[idx], float(score)) for idx, score in zip(I[0], D[0]) 
                       if score >= SIMILARITY_THRESHOLD]
        
        # BM25 search
        tokens = word_tokenize(query)
        bm25_scores = self.bm25.get_scores(tokens)
        bm25_ranked = sorted(enumerate(bm25_scores), key=lambda x: x[1], reverse=True)[:top_k]
        bm25_results = [(self.ids[idx], score) for idx, score in bm25_ranked if score > 0]
        
        # RRF fusion
        rrf_scores = defaultdict(float)
        for rank, (rid, _) in enumerate(vec_results):
            rrf_scores[rid] += 1.0 / (RRF_K + rank + 1)
        for rank, (rid, _) in enumerate(bm25_results):
            rrf_scores[rid] += 1.0 / (RRF_K + rank + 1)
        
        # Get top results
        top_ids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [self.metadata[rid] for rid, _ in top_ids]
    
    def _apply_filters(self, requirements: List[Dict[str, Any]], filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply filters to requirements"""
        filtered = []
        
        for req in requirements:
            cls = req.get("classification", {})
            entities = req.get("entities", {})
            text = (req.get("normalized_text", "") + " " + req.get("raw_text", "")).lower()
            
            match = True
            
            # Type filter
            if filters.get("requirement_type"):
                req_type = cls.get("requirement_type", "").lower()
                allowed = [t.lower() for t in filters["requirement_type"]]
                if req_type not in allowed:
                    match = False
            
            # Criticality filter
            if filters.get("criticality"):
                crit = cls.get("criticality", "").lower()
                allowed = [c.lower() for c in filters["criticality"]]
                if crit not in allowed:
                    match = False
            
            # Systems filter
            if filters.get("systems"):
                sys_list = [s.lower() for s in entities.get("systems", [])]
                required = [s.lower() for s in filters["systems"]]
                if not any(s in sys_list for s in required):
                    match = False
            
            # Text contains
            if filters.get("text_contains"):
                for phrase in filters["text_contains"]:
                    if phrase.lower() not in text:
                        match = False
            
            # Text excludes
            if filters.get("text_excludes"):
                for phrase in filters["text_excludes"]:
                    if phrase.lower() in text:
                        match = False
            
            if match:
                filtered.append(req)
        
        return filtered
    
    def _find_similar(self, anchor: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """Find similar requirements"""
        anchor_id = None
        if anchor in self.id_to_idx:
            anchor_id = anchor
        else:
            # Search for it
            results = self._semantic_search(anchor, top_k=5)
            if results:
                anchor_id = results[0]["client_reference_id"]
        
        if not anchor_id:
            return []
        
        idx = self.id_to_idx[anchor_id]
        anchor_emb = self.embeddings[idx:idx+1]
        D, I = self.index.search(anchor_emb, min(top_k + 1, len(self.ids)))
        
        similar = []
        for idx, score in zip(I[0], D[0]):
            rid = self.ids[idx]
            if rid != anchor_id and score >= SIMILARITY_THRESHOLD:
                similar.append(self.metadata[rid])
        
        return similar
    
    def _generate_intelligent_response(self, query: str, plan: Dict[str, Any], data: Dict[str, Any]) -> str:
        """
        LLM generates the final response based on the query and retrieved data.
        The response is adaptive - formatted based on what makes sense.
        
        STRICTLY GROUNDED: Only uses requirements from the database.
        """
        
        system_prompt = """You are an intelligent RFP requirements assistant providing answers to users.

CRITICAL INSTRUCTIONS:
1. You can ONLY use information from the requirements provided in the context
2. DO NOT use your general knowledge or make up requirements
3. DO NOT invent requirement IDs or content
4. If the data doesn't contain what the user is asking for, say so clearly
5. ONLY cite requirement IDs that are actually in the provided data

You have access to a requirements database and have retrieved relevant data based on the user's query.

Your job is to provide a clear, comprehensive, and well-formatted answer that:
1. Directly addresses what the user asked
2. ONLY uses requirements from the provided context
3. Presents information in the most useful format (list, table, summary, count, etc.)
4. Highlights key points and insights FROM THE DATA
5. Uses proper formatting (bullet points, numbering, sections as needed)

Be conversational but professional. Adapt your response format to the query type:
- List queries → Provide organized lists with key details
- Count queries → Give count with breakdown if relevant
- Summary queries → Provide concise summary with key examples
- Comparison queries → Use clear groupings or tables
- Analysis queries → Provide insights with supporting evidence

Always include requirement IDs when listing specific requirements.
Be concise but comprehensive.

IF THE DATA DOESN'T ANSWER THE QUESTION: Clearly state what information is missing or not available in the requirements database."""

        # Build context from data
        context_parts = []
        context_parts.append(f"User Query: {query}")
        context_parts.append(f"Query Type: {plan.get('query_type', 'unknown')}")
        context_parts.append("")
        
        if data.get("requirements"):
            reqs = data["requirements"]
            context_parts.append(f"Retrieved {len(reqs)} requirement(s) from database:")
            for i, req in enumerate(reqs[:50], 1):  # Limit to first 50 for context
                rid = req["client_reference_id"]
                text = req.get("normalized_text", "")
                cls = req.get("classification", {})
                req_type = cls.get("requirement_type", "")
                crit = cls.get("criticality", "")
                context_parts.append(f"{i}. {rid} [{req_type} | {crit}]: {text}")
        
        if data.get("groups"):
            context_parts.append("\nGrouped data from database:")
            for group, count in sorted(data["groups"].items(), key=lambda x: x[1], reverse=True):
                context_parts.append(f"- {group}: {count}")
        
        if data.get("count"):
            context_parts.append(f"\nTotal count in database: {data['count']}")
        
        context = "\n".join(context_parts)
        
        # Truncate if too long
        if len(context) > 6000:
            context = context[:6000] + "\n... (data truncated - showing first 50 requirements)"
        
        user_prompt = f"""{context}

Provide a comprehensive answer to the user's query using ONLY the requirements shown above.

IMPORTANT: 
- Do NOT add requirements not shown in the data above
- Do NOT use your general knowledge
- Only cite requirement IDs that appear in the data above
- If the data doesn't fully answer the question, say so"""

        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            response = call_llm(messages, max_tokens=1500, temperature=0.3)
            
            # Post-process: Validate that mentioned IDs actually exist
            response = self._validate_response(response, data)
            
            return response
        except Exception as e:
            if SHOW_DEBUG:
                print(f"Response generation failed: {e}")
            return f"Found {data.get('count', len(data.get('requirements', [])))} matching requirement(s)."
    
    def _validate_response(self, response: str, data: Dict[str, Any]) -> str:
        """
        Validate that the response only references requirements that actually exist in the data.
        """
        # Extract all requirement IDs mentioned in the response
        mentioned_ids = re.findall(r'\b[A-Z]+-[A-Z]+-\d+\b', response)
        
        # Get valid IDs from the data
        valid_ids = set()
        if data.get("requirements"):
            valid_ids = {req["client_reference_id"] for req in data["requirements"]}
        
        # Check for invalid IDs
        invalid_ids = set(mentioned_ids) - valid_ids - set(self.metadata.keys())
        
        if invalid_ids:
            if SHOW_DEBUG:
                print(f"Warning: Response mentioned non-existent IDs: {invalid_ids}")
            # Remove lines mentioning invalid IDs
            lines = response.split('\n')
            filtered_lines = []
            for line in lines:
                has_invalid = any(inv_id in line for inv_id in invalid_ids)
                if not has_invalid:
                    filtered_lines.append(line)
            response = '\n'.join(filtered_lines)
        
        return response

# =========================
# CLI
# =========================
def main():
    data_file = DEFAULT_DATA_FILE if len(sys.argv) < 2 else sys.argv[1]
    print(f"[Info] Using: {Path(data_file).resolve()}\n")
    
    try:
        assistant = IntelligentRfpAssistant(data_file=data_file)
    except Exception as e:
        print(f"Failed to initialize: {e}")
        sys.exit(1)
    
    print("=" * 70)
    print("Intelligent RFP Requirements Assistant")
    print("Ask me anything about requirements - I'll understand and respond naturally")
    print("=" * 70)
    print("\nExample queries:")
    print("  • List all mandatory security requirements")
    print("  • How many requirements enforce audit logging?")
    print("  • Summarize fraud prevention requirements")
    print("  • Group requirements by criticality")
    print("  • Find requirements similar to POS-FR-001")
    print("  • What requirements ensure PCI-DSS compliance?")
    print("\nType 'quit' to exit.\n")
    
    while True:
        try:
            query = input("→ Ask: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        
        if not query:
            continue
        if query.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        
        try:
            # Get intelligent response
            response = assistant.answer_query(query)
            
            # Display
            print("\n" + "=" * 70)
            print(response)
            print("=" * 70 + "\n")
            
        except Exception as e:
            print(f"\n❌ Error: {e}\n")
            if SHOW_DEBUG:
                import traceback
                traceback.print_exc()

if __name__ == "__main__":
    main()