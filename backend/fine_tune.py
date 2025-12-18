import os
import sys
import json
import logging
import torch
import pandas as pd
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

# Configuration & Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger("FineTuner")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = "all-MiniLM-L6-v2"
TRAIN_FILE = os.path.join(BASE_DIR, "..", "evaluation", "train.csv")
PRODUCTS_FILE = os.path.join(BASE_DIR, "data", "products.json")
OUTPUT_PATH = os.path.join(BASE_DIR, "fine_tuned_model")

def fine_tune():
    """
    Fine-tunes the base model using domain-specific training data.
    """
    logger.info(f"Checking environment and hardware...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Targeting Device: {device.upper()}")

    if not os.path.exists(TRAIN_FILE):
        logger.error(f"IO Error: Training file not found at {TRAIN_FILE}")
        return

    if not os.path.exists(PRODUCTS_FILE):
        logger.error(f"IO Error: Products database not found at {PRODUCTS_FILE}")
        return

    try:
        logger.info(f"Loading base model: {MODEL_NAME}")
        model = SentenceTransformer(MODEL_NAME, device=device)
        
        logger.info("Parsing datasets...")
        df = pd.read_csv(TRAIN_FILE)
        with open(PRODUCTS_FILE, 'r', encoding='utf-8') as f:
            products = json.load(f)
            
        # Optimization
        url_to_text = {p['url']: f"{p['name']} {p.get('description','')}" for p in products if 'url' in p}

        train_examples = []
        for _, row in df.iterrows():
            query, url = row.get('Query'), row.get('Assessment_url')
            if url in url_to_text:
                train_examples.append(InputExample(texts=[query, url_to_text[url]]))
        
        if not train_examples:
            logger.warning("No valid overlapping training pairs found. Fine-tuning aborted.")
            return

        logger.info(f"Prepared {len(train_examples)} input pairs.")
        
        # Hyperparameters
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
        train_loss = losses.MultipleNegativesRankingLoss(model)

        logger.info("Commencing Fine-Tuning Execution...")
        model.fit(
            train_objectives=[(train_dataloader, train_loss)], 
            epochs=1, 
            warmup_steps=len(train_dataloader) // 10,
            show_progress_bar=True
        )
        
        # Safe storage
        os.makedirs(OUTPUT_PATH, exist_ok=True)
        model.save(OUTPUT_PATH)
        logger.info(f"SUCCESS: Domain-optimized model serialized to {OUTPUT_PATH}")

    except Exception as e:
        logger.exception(f"Fine-Tuning lifecycle failed: {e}")

if __name__ == "__main__":
    fine_tune()
