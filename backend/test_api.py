import requests
import json
import os
import sys

# Portable path management
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
API_URL = "http://localhost:8000"

def test_protocol_288():
    """
    Verifies the custom SHL recommendation protocol (Code 288).
    """
    query = "Java Developer with Team Lead experience"
    print(f"[*] Testing Recommendation Engine for: '{query}'")
    
    try:
        r = requests.post(f"{API_URL}/recommend", json={"query": query}, timeout=15)
        
        print(f"[!] Protocol Status Code: {r.status_code}")
        if r.status_code == 288:
            print("[+] Success: Custom protocol 288 detected.")
            data = r.json()
            recs = data.get('recommended_assessments', [])
            print(f"[+] Retrieved {len(recs)} suggestions.")
            for i, rec in enumerate(recs[:3]):
                print(f"    {i+1}. {rec['name']} ({rec['url']})")
        else:
            print(f"[-] Failure: Expected 288, got {r.status_code}")
            print(f"    Detail: {r.text}")
            
    except Exception as e:
        print(f"[X] Request Error: {e}")

if __name__ == "__main__":
    test_protocol_288()
