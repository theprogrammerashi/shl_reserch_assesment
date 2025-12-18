import os
import sys
from backend.rag_engine import ingest_data

def main():
    json_path = os.path.join("backend", "data", "products.json")
    if not os.path.exists(json_path):
        
        json_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "data", "products.json"))
        
    print(f"Starting ingestion from {json_path}...")
    ingest_data(json_path)
    print("Ingestion complete.")

if __name__ == "__main__":
    main()
