import os
import json
import chromadb
from chromadb.utils import embedding_functions
import google.generativeai as genai
import re
import sys
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger("RAG_Engine")

# Path Management
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PERSIST_DIRECTORY = os.path.join(BASE_DIR, "chroma_db")
FINE_TUNED_MODEL_PATH = os.path.join(BASE_DIR, "fine_tuned_model")
DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"

def validate_model(model_path: str, base_model: str) -> bool:
    """
    Validation Rigor: Verifies if the fine-tuned model is functional and 
    presents expected structure before loading.
    """
    if not os.path.exists(model_path):
        return False
    
    
    required_files = ["model.safetensors", "config.json", "tokenizer.json"]
    for f in required_files:
        if not os.path.exists(os.path.join(model_path, f)):
            logger.warning(f"Missing essential model file: {f}. Falling back to base model.")
            return False
            
    
    return True

embedding_func = None
collection = None

def get_engine():
    global embedding_func, collection
    if collection is not None:
        return collection
    
    try:
        if validate_model(FINE_TUNED_MODEL_PATH, DEFAULT_MODEL_NAME):
            model_name = FINE_TUNED_MODEL_PATH
            logger.info(f"Loading Fine-Tuned Model: {model_name}")
        else:
            model_name = DEFAULT_MODEL_NAME
            logger.info(f"Loading Base Model: {model_name}")

        embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)
        client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
        collection = client.get_or_create_collection(
            name="shl_assessments_v2",
            embedding_function=embedding_func
        )
        return collection
    except Exception as e:
        logger.error(f"Failed to initialize Vector Store: {e}")
        raise e

def ingest_data(json_path: str):
    """
    Ingests product data into the vector store with robustness checks.
    """
    if not os.path.isabs(json_path):
        json_path = os.path.join(BASE_DIR, json_path)

    if not os.path.exists(json_path):
        logger.error(f"Data file not found for ingestion: {json_path}")
        return

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            products = json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in data file: {e}")
        return

    ids = []
    documents = []
    metadatas = []

    curr_collection = get_engine()
    logger.info(f"Ingesting {len(products)} products into {curr_collection.name}...")

    for i, product in enumerate(products):
        
        content = f"{product.get('name', 'Unknown')} - {product.get('description', '')}"
        
        
        test_types = product.get("test_type", [])
        if isinstance(test_types, str):
            test_types = [test_types]
            
        metadata = {
            "name": product.get("name", "Unknown"),
            "url": product.get("url", ""),
            "remote_support": str(product.get("remote_support", "No")),
            "adaptive_support": str(product.get("adaptive_support", "No")),
            "description": product.get("description", "")[:400], 
            "duration": str(product.get("duration", 0)),
            "test_type": str(test_types) 
        }
        
        ids.append(f"prod_{i}")
        documents.append(content)
        metadatas.append(metadata)

    if ids:
        curr_collection = get_engine()
        batch_size = 100
        for idx in range(0, len(ids), batch_size):
            curr_collection.upsert(
                ids=ids[idx:idx+batch_size], 
                documents=documents[idx:idx+batch_size], 
                metadatas=metadatas[idx:idx+batch_size]
            )
        logger.info("Ingestion complete.")



EXPANSION_MAP = {
    "backend": "backend server database api microservices system design",
    "frontend": "frontend ui ux javascript react angular vue html css",
    "fullstack": "fullstack full-stack web developer backend frontend",
    "data science": "data science machine learning statistics python pandas numpy analysis",
    "sales": "sales selling negotiation business development b2b account manager",
    "python": "python programming scripting language django flask",
    "java": "java j2ee spring hibernate jvm",
    "c++": "c++ cpp systems programming memory management",
    "manager": "manager leadership team lead management supervision",
    "graduate": "graduate entry level junior",
}

def expand_query(query: str) -> str:
    query_lower = query.lower()
    expanded_terms = []
    for key, synonyms in EXPANSION_MAP.items():
        if key in query_lower:
            expanded_terms.append(synonyms)
    
    if expanded_terms:
        return f"{query} {' '.join(expanded_terms)}"
    return query

def search(query: str, n_results: int = 50) -> list:
    """
    Retrieves initial candidates using expanded vector search.
    """
    expanded_query = expand_query(query)
    
    try:
        curr_collection = get_engine()
        results = curr_collection.query(
            query_texts=[expanded_query],
            n_results=n_results
        )
        
        candidates = []
        if results["ids"] and results["metadatas"]:
            for i in range(len(results["ids"][0])):
                candidates.append(results["metadatas"][0][i])
        return candidates
    except Exception as e:
        logger.error(f"Search failed: {e}")
        return []

def detect_query_intents(query: str) -> set:
    INTENT_MAP = {
        "DATA_SCIENCE": ["data science", "data scientist", "machine learning", "pandas", "numpy", "statistics", "tableau", "powerbi", "analysis"],
        "SOFTWARE_ENG": ["developer", "engineer", "backend", "frontend", "fullstack", "java", "c++", ".net", "api", "algorithms", "data structures", "system design", "coding", "software"],
        "SALES": ["sales", "business development", "account manager", "selling", "negotiation"],
        "FINANCE": ["finance", "accounting", "audit", "tax", "financial"],
        "HR": ["human resources", "recruitment", "personnel", "hr"],
        "PYTHON": ["python"], 
    }
    query_lower = query.lower()
    detected = set()
    for intent, keywords in INTENT_MAP.items():
        if any(k in query_lower for k in keywords):
            detected.add(intent)
    
    if "PYTHON" in detected:
        if "DATA_SCIENCE" not in detected and "SOFTWARE_ENG" not in detected:
            if any(term in query_lower for term in ["analysis", "model", "regression"]):
                detected.add("DATA_SCIENCE")
            else:
                detected.add("SOFTWARE_ENG")
    return detected

def get_candidate_intents(candidate: dict) -> set:
    text = (candidate.get('name', '') + " " + candidate.get('description', '')).lower()
    detected = set()
    
    for intent, keywords in {
        "DATA_SCIENCE": ["data science", "machine learning", "statistics"],
        "SOFTWARE_ENG": ["developer", "engineer", "coding", "software", "java", "python"],
        "SALES": ["sales", "selling"],
        "FINANCE": ["finance", "accounting"]
    }.items():
        if any(k in text for k in keywords):
            detected.add(intent)
    return detected

def heuristic_rerank(query: str, candidates: list, max_k: int = 10) -> list:
    """
    Deterministic fallback reranker with intent alignment.
    """
    query_lower = query.lower()
    query_intents = detect_query_intents(query)
    
    scored_candidates = []
    q_keywords = [k for k in query_lower.split() if len(k) > 2]
    
    for c in candidates:
        score = 0
        name_lower = c.get('name', '').lower()
        desc_lower = c.get('description', '').lower()
        cand_intents = get_candidate_intents(c)
        
        # Exact matching
        for k in q_keywords:
            if k in name_lower: score += 5
            elif k in desc_lower: score += 1
        
        # Intent matching
        if query_intents & cand_intents:
            score += 10
            
        scored_candidates.append((score, c))
        
    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    return [c for s, c in scored_candidates[:max_k]]

def rerank_with_llm(query: str, candidates: list, api_key: str = None) -> list:
    """
    Primary reranker using Gemini Pro with Balancing Logic.
    """
    if not api_key:
        logger.warning("No Gemini API Key provided. Falling back to Heuristic Reranker.")
        return heuristic_rerank(query, candidates)

    logger.info("Engaging Gemini Reranker...")
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-pro')

        candidates_str = ""
        for idx, c in enumerate(candidates):
            candidates_str += f"ID {idx}: {c.get('name')} | {c.get('description')[:100]}... | Type: {c.get('test_type')}\n"

        prompt = f"""
        Role: Expert Recruiter.
        Task: Recommend 5 to 10 assessments for the query: "{query}".
        
        Candidates:
        {candidates_str}
        
        Rules:
        1. INTENT MATCHING: Detect the core role and skills. Discard irrelevant assessments.
        2. BALANCING: If the query requests both technical (Knowledge/Skills) and soft skills (Personality/Behavior), you MUST include at least 2 results from EACH category.
        3. KEYWORD PREFERENCE: Prioritize specialized tests (e.g. 'Automata') over generic ones if specific skills are mentioned.
        4. QUANTITY: Return between 5 and 10 results.
        5. OUTPUT: Strictly JSON list of objects from Candidates (copy all fields exactly).
        """

        response = model.generate_content(prompt)
        text = response.text
        
        text = re.sub(r'```json|```', '', text).strip()
        selected = json.loads(text)
        
        
        if len(selected) < 5:
            logger.info("LLM returned minimal results. Supplementing with heuristic.")
            h_list = heuristic_rerank(query, candidates)
            seen_urls = {s.get('url') for s in selected}
            for h in h_list:
                if h.get('url') not in seen_urls and len(selected) < 10:
                    selected.append(h)
                    seen_urls.add(h.get('url'))
                    
        return selected

    except Exception as e:
        logger.error(f"LLM Reranking failed: {e}")
        return heuristic_rerank(query, candidates)
