from fastapi import FastAPI, HTTPException, Response, Request
from pydantic import BaseModel, Field
from typing import List, Optional
import uvicorn
import os
import json
import contextlib
import logging
import sys
import asyncio
from dotenv import load_dotenv


load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger("API_Gateway")


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

try:
    import rag_engine
except ImportError as e:
    logger.critical(f"Critical dependency failure: rag_engine.py not found. {e}")
    sys.exit(1)

@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles application state transitions.
    """
    logger.info("Initializing Application Bootstrap...")
    data_path = os.path.join(BASE_DIR, "data", "products.json")
    
    async def background_ingestion():
        try:
            # Small delay to allow port binding to finalize
            await asyncio.sleep(1)
            count = rag_engine.get_engine().count()
            if count == 0:
                logger.info("Vector Store empty. Starting cold-start ingestion...")
                rag_engine.ingest_data(data_path)
            else:
                logger.info(f"Vector Store Active. Entities: {count}")
        except Exception as e:
            logger.error(f"Background Ingestion failure: {e}")

    # Fire and forget - lets the server start listening immediately
    asyncio.create_task(background_ingestion())
        
    yield
    logger.info("Graceful shutdown sequence complete.")

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="SHL Assessment Recommender",
    description="Intelligent RAG-based assessment suggestion engine.",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Schemas

class QueryRequest(BaseModel):
    query: str = Field(..., example="Java Developer with leadership skills")

class AssessmentResponse(BaseModel):
    url: str
    name: str
    adaptive_support: str
    description: str
    duration: int
    remote_support: str
    test_type: List[str]

class RecommendResponse(BaseModel):
    recommended_assessments: List[AssessmentResponse]

# Endpoints

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/recommend", response_model=RecommendResponse, status_code=200)
async def recommend_assessments(request: QueryRequest, response: Response):
    """
    Recommendation handler. Returns custom 288 status code per assessment requirements.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    logger.info(f"Processing Recommendation for: {request.query[:50]}...")

    try:
        # 1. Retrieval
        raw_candidates = rag_engine.search(request.query)
        
        # 2. Reranking
        api_key = os.environ.get("GEMINI_API_KEY")
        final_list = rag_engine.rerank_with_llm(request.query, raw_candidates, api_key)

        # 3. Defensive Serialization & Transformation
        formatted_results = []
        for item in final_list:
            # Duration parse
            try:
                dur = int(float(item.get("duration", 0)))
            except (ValueError, TypeError):
                dur = 0
                
            
            tt = item.get("test_type", [])
            if isinstance(tt, str):
                try:
                    import ast
                    tt = ast.literal_eval(tt)
                except Exception:
                    tt = [tt] if tt else []
                    
            formatted_results.append(AssessmentResponse(
                url=item.get("url", ""),
                name=item.get("name", "Unknown Assessment"),
                adaptive_support=str(item.get("adaptive_support", "No")),
                description=item.get("description", ""),
                duration=dur,
                remote_support=str(item.get("remote_support", "No")),
                test_type=tt if isinstance(tt, list) else [str(tt)]
            ))

        
        response.status_code = 200
        return RecommendResponse(recommended_assessments=formatted_results)

    except Exception as e:
        logger.exception("Internal Recommendation Pipeline Failure")
        raise HTTPException(status_code=500, detail="An internal processing error occurred.")

@app.post("/feedback")
def submit_feedback(data: dict):
    """
    Captures user engagement data for future Reinforcement Learning.
    """
    feedback_file = os.path.join(BASE_DIR, "data", "user_feedback.json")
    
    try:
        store = []
        if os.path.exists(feedback_file):
            with open(feedback_file, "r") as f:
                store = json.load(f)
        
        store.append(data)
        with open(feedback_file, "w") as f:
            json.dump(store, f, indent=4)
            
        return {"status": "recorded"}
    except Exception as e:
        logger.error(f"Feedback capture failed: {e}")
        return {"status": "bypass"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
