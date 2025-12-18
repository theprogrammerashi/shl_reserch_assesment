import requests
import pandas as pd
import os
import json
import time

# Configuration
API_URL = "http://localhost:8000/recommend"
DATA_DIR = os.path.dirname(__file__)
TRAIN_FILE = os.path.join(DATA_DIR, "train.csv")
TEST_FILE = os.path.join(DATA_DIR, "test.csv")
SUBMISSION_FILE = os.path.join(DATA_DIR, "submission.csv")

def get_recommendations(query, k=10):
    try:
        response = requests.post(API_URL, json={"query": query})
        if response.status_code == 200:
            data = response.json()
            
            recs = data.get("recommended_assessments", [])
            return [rec["url"] for rec in recs[:k]]
    except Exception as e:
        print(f"Error fetching recommendations for '{query}': {e}")
    return []

def evaluate_recall_at_10():
    if not os.path.exists(TRAIN_FILE):
        print(f"Train file not found at {TRAIN_FILE}. Skipping evaluation.")
        return

    print("Loading train data...")
    
    try:
        df = pd.read_csv(TRAIN_FILE)
    except Exception as e:
        print(f"Error reading train CSV: {e}")
        return

    # Check columns
    if "Query" not in df.columns or "Assessment_url" not in df.columns:
        print("Train CSV must have 'Query' and 'Assessment_url' columns.")
        return

    print(f"Evaluating on {len(df)} queries...")
    
    hits = 0
    total = 0
    failures = []
    
    
    grouped = df.groupby("Query")['Assessment_url'].apply(list)
    
    for query, ground_truths in grouped.items():
        print(f"Processing: {query}...")
        preds = get_recommendations(query, k=10)
        
        # Calculate Recall@10
        # Recall = (Relevant items in Top 10) / (Total Relevant items)
        
        # Normalize URLs for comparison 
        preds_norm = [u.strip().rstrip('/') for u in preds]
        gts_norm = [str(u).strip().rstrip('/') for u in ground_truths]
        
        # Intersection
        intersection = set(preds_norm) & set(gts_norm)
        recall = len(intersection) / len(gts_norm) if len(gts_norm) > 0 else 0
        
        print(f"  Found {len(intersection)}/{len(gts_norm)} relevant items. Recall: {recall:.2f}")
        
        if recall < 1.0:
            failures.append({
                "Query": query,
                "Expected": gts_norm,
                "Got_Top3": preds_norm[:3],
                "Recall": recall
            })
        
        total += 1
        hits += recall

    if failures:
        fail_df = pd.DataFrame(failures)
        fail_path = os.path.join(DATA_DIR, "failure_analysis.csv")
        fail_df.to_csv(fail_path, index=False)
        print(f"\nSaved {len(failures)} failures to {fail_path}")

    if total > 0:
        mean_recall = hits / total
        print(f"\nMean Recall@10: {mean_recall:.4f}")
    else:
        print("No queries evaluated.")


def generate_predictions():
    if not os.path.exists(TEST_FILE):
        print(f"Test file not found at {TEST_FILE}. Skipping predictions.")
        return

    print("\nGenerating predictions for test set...")
    try:
        df = pd.read_csv(TEST_FILE)
    except:
        print("Error reading test CSV.")
        return

    if "Query" not in df.columns:
        print("Test CSV must have a 'Query' column.")
        return

    results = []
    
    for query in df["Query"].unique():
        print(f"Predicting for: {query}")
        preds = get_recommendations(query, k=10)
        
        
        
        for url in preds[:5]: # Top 5
            results.append({"Query": query, "Assessment_url": url})
            
    submission_df = pd.DataFrame(results)
    submission_df.to_csv(SUBMISSION_FILE, index=False)
    print(f"Saved predictions to {SUBMISSION_FILE}")

if __name__ == "__main__":
    evaluate_recall_at_10()
    generate_predictions()
