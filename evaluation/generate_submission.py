import requests
import pandas as pd
import os
import time

# Configuration
API_URL = "http://localhost:8000/recommend"
DATA_DIR = os.path.dirname(__file__)
TEST_FILE = os.path.join(DATA_DIR, "test.csv")
SUBMISSION_FILE = os.path.join(DATA_DIR, "submission.csv")

def get_recommendations(query, k=10):
    """
    Calls the API to get recommendations for a given query.
    """
    try:
        response = requests.post(API_URL, json={"query": query})
        if response.status_code == 200:
            data = response.json()
            recs = data.get("recommended_assessments", [])
            return [rec["url"] for rec in recs[:k]]
        else:
            print(f"API Error ({response.status_code}) for '{query}': {response.text}")
    except Exception as e:
        print(f"Connection Error for '{query}': {e}")
    return []

def generate_submission():
    if not os.path.exists(TEST_FILE):
        print(f"Test file not found at {TEST_FILE}. Please ensure it exists.")
        return

    print(f"Loading test data from {TEST_FILE}...")
    try:
        # Load test set. Expecting 'Query' column.
        df = pd.read_csv(TEST_FILE)
    except Exception as e:
        print(f"Error reading test CSV: {e}")
        return

    if "Query" not in df.columns:
        print("Test CSV must have a 'Query' column.")
        return

    results = []
    queries = df["Query"].unique()
    
    print(f"Generating predictions for {len(queries)} unique queries...")
    
    for i, query in enumerate(queries):
        print(f"[{i+1}/{len(queries)}] Processing: {query[:50]}...")
        preds = get_recommendations(query, k=10)
        
        if not preds:
            print(f"  WARNING: No recommendations found for '{query[:50]}'")
        
        # Format requirements: Long format (Query, Assessment_url)
        # We need minimum 5, maximum 10.
        for url in preds:
            results.append({"Query": query, "Assessment_url": url})
            
        # Give API a moment if needed
        time.sleep(0.5)

    submission_df = pd.DataFrame(results)
    submission_df.to_csv(SUBMISSION_FILE, index=False)
    print(f"Successfully saved {len(submission_df)} prediction rows to {SUBMISSION_FILE}")

if __name__ == "__main__":
    generate_submission()
