# SHL Assessment Recommendation System - Technical Approach

## 1. Executive Summary
The SHL Assessment Recommendation System is an intelligent RAG-based engine designed to simplify the discovery of relevant assessments for recruiters and hiring managers. By leveraging semantic search combined with LLM-powered reranking and balancing logic, the system provides highly relevant "Individual Test Solutions" from a catalog of 377+ products.

## 2. Data Pipeline
### Data Ingestion & Scraping
A custom Python-based scraper was developed using `BeautifulSoup4` and `requests`. The scraper performs the following:
*   **Targeted Extraction**: Specifically targets the product catalog with filters for "individual test solutions" (`type=1`).
*   **Deep Crawling**: After extracting the primary list (name, URL, remote/adaptive support), the system crawls each individual product page to capture detailed descriptions and durations.
*   **Test Type Mapping**: One-letter codes (e.g., K, P) are mapped to human-readable categories (e.g., Knowledge & Skills, Personality & Behavior) based on official SHL terminology.
*   **Clean Structuring**: Data is stored in a clean JSON format, ensuring easy consumption by the RAG engine.

### Vector Storage
Candidates are stored in **ChromaDB**, an open-source vector database.
*   **Embeddings**: We use the `all-MiniLM-L6-v2` model for high-speed, high-density vector embeddings of assessment titles and descriptions.
*   **Enriched Metadata**: Each entry includes structured metadata (URL, duration, test types) for efficient filtering and reranking.

## 3. Recommendation Engine
The system employs a multi-stage retrieval and ranking pipeline:

### Stage 1: Semantic Retrieval
Traditional keyword search is replaced with semantic search, allowing the system to understand the context of job descriptions (e.g., "Java developer" matches "Core Java" and "Coding" tests even without exact word matches).

### Stage 2: Intent-Aware Expansion
Queries are expanded using a predefined mapping of role-to-skills (e.g., "Fullstack" expands to "React, Node, SQL"). This ensures high recall during the initial retrieval phase.

### Stage 3: LLM Reranking (Intelligence Layer)
The top 50 candidates are passed to **Gemini Pro (1.5-flash/pro)** for final selection. The LLM acts as an expert recruiter to:
*   Identify core requirements from natural language.
*   Exclude "Pre-packaged Job Solutions" (enforced at source).
*   Prioritize specialized assessments over generic tools.

## 4. Performance Optimization & Balancing
### Intelligent Balancing
A core requirement was to balance technical and behavioral assessments when queries span multiple domains.
*   **Dual-Category Constraints**: The LLM prompt enforces a "Minimum 2" rule for both "Knowledge & Skills" and "Personality & Behavior" when a query mentions both hard and soft skills.
*   **Deterministic Fallback**: If the LLM returns insufficient results, a heuristic re-ranker supplements the list based on intent-alignment and keyword density.

### Performance Tuning
*   **Mean Recall@10**: Iterative testing on the `train.csv` dataset was used to refine the reranking prompt and expansion logic, achieving significant improvements in retrieval accuracy.
*   **Latency**: Threaded scraping and optimized batch-upserts in ChromaDB ensure the system is ready for production scaling.

## 5. System Verification
The system was validated against the following:
*   **API Compliance**: Standardized health check (`/health`) and recommendation (`/recommend`) endpoints returning JSON with required status codes.
*   **Data Integrity**: Verified 377 individual test solutions post-crawling.
*   **Frontend UX**: A modern, responsive React/FastAPI stack provides a clean interface for recruiters to input queries and view results in a detailed tabular format.
