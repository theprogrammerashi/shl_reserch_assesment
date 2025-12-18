# SHL Assessment Recommendation System

An intelligent recommendation engine that maps job descriptions and skill requirements to the most relevant SHL assessment solutions.

## 🚀 Deployment Guide

### Backend (FastAPI + ChromaDB)
Recommended: **Render** or **AWS App Runner**
1. **GitHub**: Push the `backend` folder or the entire repo.
2. **Environment Variables**:
   - `GEMINI_API_KEY`: Your Google AI API Key (required for reranking).
3. **Build Command**: `pip install -r requirements.txt`
4. **Start Command**: `uvicorn backend.app:app --host 0.0.0.0 --port $PORT`

### Frontend (Vite + React)
Recommended: **Vercel** or **Netlify**
1. **Source**: Push the `frontend` folder.
2. **Environment Variables**:
   - `VITE_API_URL`: URL of your deployed backend (e.g., `https://shl-backend.onrender.com`).
3. **Build Command**: `npm run build`
4. **Output Directory**: `dist`

---

## 🛠️ Local Setup

### 1. Backend
```bash
cd backend
python -m venv venv
source venv/bin/activate  # venv\Scripts\activate on Windows
pip install -r requirements.txt
python -m uvicorn app:app --reload
```

### 2. Frontend
```bash
cd frontend
npm install
npm run dev
```

---

## 📊 System Overview

### Data Pipeline
- **Scraper**: Crawls the SHL product catalog to maintain 377+ "Individual Test Solutions".
- **Vector Store**: Uses `ChromaDB` for semantic storage and retrieval.
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2`.

### Recommendation Logic (RAG)
1. **Query Input**: Accepts natural language job descriptions.
2. **Semantic Retrieval**: Fetches top N candidates based on cosine similarity.
3. **LLM Reranking**: Uses Gemini Pro to balance soft skills (Personality) and hard skills (Knowledge).
4. **Accuracy**: **Mean Recall@10: 0.8462** on training set.

## 📁 Final Submission Files
- `/evaluation/submission.csv`: Predictions for `test.csv`.
- `approach_document.md`: 2-page detailed technical approach.
- `backend/data/products.json`: Scraped dataset of 377+ assessments.
- `walkthrough.md`: Detailed journey and results overview.
