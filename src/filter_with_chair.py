import os
import sys
import json
import pandas as pd
from tqdm import tqdm

# --- Add the CHAIR utility to our Python path ---
# This allows us to import the 'chair.py' script from the cloned repository
HALLUCINATION_DIR = os.path.join(os.getcwd(), 'utils/Hallucination')
sys.path.append(HALLUCINATION_DIR)
from utils.chair import CHAIR

# --- 1. DEFINE FILE PATHS ---
PROJECT_DIR = os.getcwd()
# Input file from the previous step
CAPTIONS_FILE = os.path.join(PROJECT_DIR, 'data/contrastive_set/generated_captions.json')
# We need the instance annotations which contain object data
INSTANCE_ANNOTATION_FILE = os.path.join(PROJECT_DIR, 'data/mscoco/annotations/instances_train2017.json')
# The final output file for our contrastive pairs
OUTPUT_PAIRS_FILE = os.path.join(PROJECT_DIR, 'data/contrastive_set/contrastive_pairs.json')

# --- 2. MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    print("--- Starting Step 2B: Filtering with CHAIR metric ---")

    # --- Load the generated captions data ---
    print(f"Loading generated captions from {CAPTIONS_FILE}...")
    with open(CAPTIONS_FILE, 'r') as f:
        generated_data = json.load(f)
    print(f"Loaded {len(generated_data)} generated captions.")

    # --- Initialize the CHAIR evaluator ---
    # This object loads all the necessary MSCOCO ground-truth data
    print("Initializing CHAIR evaluator...")
    evaluator = CHAIR(INSTANCE_ANNOTATION_FILE)
    print("CHAIR evaluator ready.")

    # --- Process captions and create contrastive pairs ---
    contrastive_pairs = []
    print("Evaluating captions for hallucinations...")
    for item in tqdm(generated_data, desc="Filtering Captions"):
        image_id = item['image_id']
        generated_caption = item['generated_caption']
        
        # The core of our filtering: calculate the CHAIR score
        # We pass the caption as a list containing a single dictionary
        caption_data = [{'image_id': image_id, 'caption': generated_caption}]
        evaluator.get_metrics(caption_data)
        
        # CHAIRi is the fraction of hallucinated objects. A score > 0 means at least one hallucination.
        chair_i_score = evaluator.chair_i_per_sent
        
        # If the caption has at least one hallucinated object, it's a negative example
        if chair_i_score > 0:
            # We pair the hallucinated caption (negative) with a ground-truth caption (positive)
            positive_caption = item['ground_truth_captions']
            
            pair = {
                "image_id": image_id,
                "positive": positive_caption,
                "negative": generated_caption,
                "chair_i_score": chair_i_score
            }
            contrastive_pairs.append(pair)

    print(f"\nFound {len(contrastive_pairs)} pairs with hallucinations.")

    # --- Save the final contrastive pairs to a file ---
    with open(OUTPUT_PAIRS_FILE, 'w') as f:
        json.dump(contrastive_pairs, f, indent=4)

    print(f"Successfully saved contrastive pairs to {OUTPUT_PAIRS_FILE}")

    # --- Display a sample for verification ---
    if contrastive_pairs:
        df = pd.DataFrame(contrastive_pairs)
        print("\nSample of final contrastive pairs:")
        # Use pd.option_context to prevent long captions from being truncated
        with pd.option_context('display.max_colwidth', None):
            print(df.head())