import torch
import os
import sys
import json
import numpy as np
from tqdm import tqdm
import pandas as pd
import nltk

# --- 1. ADD UTILS TO PATH AND IMPORT CHAIR ---
UTILS_DIR = os.path.join(os.getcwd(), 'utils')
sys.path.append(UTILS_DIR)
import chair # Import the module so we can patch it

# --- 2. DEFINE FILE PATHS ---
PROJECT_DIR = os.getcwd()
# Directory where all annotations are stored
ANNOTATIONS_DIR = os.path.join(PROJECT_DIR, 'data/mscoco/annotations')
# The specific instance file needed by CHAIR
VAL_INSTANCES_FILE = os.path.join(ANNOTATIONS_DIR, 'instances_val2017.json')

# The two result files you just created
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')
BASELINE_FILE = os.path.join(RESULTS_DIR, 'baseline_eval_results.json')
STEERED_FILE = os.path.join(RESULTS_DIR, 'steered_eval_results.json')

# --- 3. MONKEY-PATCH for MSCOCO 2017 DATASET ---
# This is critical. We patch the chair module to read our 2017 files
# instead of the 2014 files it's hardcoded for.

def combine_coco_captions_2017(annotation_path):
    """Loads and combines train and val 2017 caption annotations."""
    val_path = os.path.join(annotation_path, 'captions_val2017.json')
    train_path = os.path.join(annotation_path, 'captions_train2017.json')
    
    if not os.path.exists(val_path) or not os.path.exists(train_path):
        raise FileNotFoundError(f"MSCOCO 2017 train/val caption annotations not found in {annotation_path}")

    val_caps = json.load(open(val_path))
    train_caps = json.load(open(train_path))
    
    # We only need the 'annotations' part for CHAIR's ground truth
    all_annotations = val_caps['annotations'] + train_caps['annotations']
    return {'annotations': all_annotations}

def combine_coco_instances_2017(annotation_path):
    """Loads and combines train and val 2017 instance annotations."""
    val_path = os.path.join(annotation_path, 'instances_val2017.json')
    train_path = os.path.join(PROJECT_DIR, 'data/mscoco/annotations/instances_train2017.json')

    if not os.path.exists(val_path) or not os.path.exists(train_path):
        raise FileNotFoundError(f"MSCOCO 2017 train/val instance annotations not found.")

    val_instances = json.load(open(val_path))
    train_instances = json.load(open(train_path))
    
    # CHAIR needs 'categories' from train and all annotations
    all_instances = {
        'categories': train_instances['categories'],
        'annotations': val_instances['annotations'] + train_instances['annotations']
    }
    return all_instances

# Replace the original functions in the imported 'chair' module with our new ones
chair.combine_coco_captions = combine_coco_captions_2017
chair.combine_coco_instances = combine_coco_instances_2017

# Now we can safely import the CHAIR class
from chair import CHAIR

# --- 4. EVALUATION FUNCTION ---
def evaluate_captions(caption_data, evaluator):
    """
    Takes a list of caption data and a pre-initialized CHAIR evaluator.
    Returns the CHAIRi and CHAIRs scores (as percentages).
    """
    total_chair_i = 0.0
    total_chair_s = 0.0
    num_captions = 0
    
    for item in tqdm(caption_data, desc="Scoring captions", leave=False):
        image_id = item['image_id']
        caption = item['caption']
        
        # Use the CHAIR object's methods to process the caption
        # This gets the standardized list of objects mentioned
        _, node_words, _, _ = evaluator.caption_to_words(caption)
        
        # Get the ground truth objects for this image from the evaluator's database
        gt_objects = evaluator.imid_to_objects.get(image_id, set())
        
        # Find all words in our caption that are NOT in the ground truth
        hallucinated_words = [word for word in node_words if word not in gt_objects]
        
        chair_i_score = 0.0
        if len(node_words) > 0:
            # CHAIRi = fraction of mentioned objects that were hallucinated
            chair_i_score = len(hallucinated_words) / float(len(node_words))
        
        total_chair_i += chair_i_score
        
        # CHAIRs = 1 if the caption has any hallucinations, 0 otherwise
        if chair_i_score > 0:
            total_chair_s += 1
        
        num_captions += 1
        
    if num_captions == 0:
        return 0.0, 0.0
        
    # Return the average scores as percentages
    avg_chair_i = (total_chair_i / num_captions) * 100
    avg_chair_s = (total_chair_s / num_captions) * 100
    return avg_chair_i, avg_chair_s

# --- 5. MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    print("--- Starting Step 4b: Final Quantitative Evaluation ---")

    # Download NLTK data for CHAIR's text processing if not present
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("NLTK 'punkt' tokenizer not found. Downloading...")
        nltk.download('punkt')
    
    # --- Load all data ---
    print(f"Loading baseline results from {BASELINE_FILE}...")
    with open(BASELINE_FILE, 'r') as f:
        baseline_data = json.load(f)
    print(f"Loaded {len(baseline_data)} baseline captions.")
        
    print(f"Loading steered results from {STEERED_FILE}...")
    with open(STEERED_FILE, 'r') as f:
        steered_data = json.load(f)
    print(f"Loaded {len(steered_data)} steered captions.")

    # Initialize the CHAIR evaluator ONCE. This is a heavy operation.
    # We pass the directory containing all our annotation files.
    print(f"Initializing CHAIR evaluator with annotations from {ANNOTATIONS_DIR}...")
    evaluator = CHAIR(ANNOTATIONS_DIR)
    print("CHAIR evaluator ready.")

    # --- Run evaluations ---
    print("\nScoring baseline captions...")
    baseline_chair_i, baseline_chair_s = evaluate_captions(baseline_data, evaluator)
    
    print("\nScoring steered captions...")
    steered_chair_i, steered_chair_s = evaluate_captions(steered_data, evaluator)

    # --- 4. Print the final results ---
    print("\n\n--- FINAL EXPERIMENT RESULTS ---")
    print("Lower is better for both metrics.\n")
    
    # For reference, published baselines for LLaVA-1.5 7B are ~15.4 (CHAIRi) and ~50.0 (CHAIRs)
    # Your baseline will differ based on prompt/test set, but should be in this ballpark.
    
    data = {
        "Baseline": {
            "CHAIRi (%)": baseline_chair_i,
            "CHAIRs (%)": baseline_chair_s
        },
        "Steered (Ours)": {
            "CHAIRi (%)": steered_chair_i,
            "CHAIRs (%)": steered_chair_s
        },
        "Improvement": {
            "CHAIRi (%)": baseline_chair_i - steered_chair_i,
            "CHAIRs (%)": baseline_chair_s - steered_chair_s
        }
    }
    
    df = pd.DataFrame(data).T
    print(df.to_string(float_format="%.2f"))
    
    # You must access the specific numeric values *inside* the dictionary
    print(f"\nImprovement (CHAIRi): {data['Improvement']['CHAIRi (%)']:.2f} points")
    print(f"Improvement (CHAIRs): {data['Improvement']['CHAIRs (%)']:.2f} points")

    print("\n--- Experiment Finished ---")