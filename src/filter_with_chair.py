import os
import sys
import json
import pandas as pd
from tqdm import tqdm
import nltk

# --- Add the utils directory to our Python path ---
# This allows us to import the 'chair.py' module to patch it
UTILS_DIR = os.path.join(os.getcwd(), 'utils')
sys.path.append(UTILS_DIR)
import chair # Import the module itself

# --- 1. DEFINE FILE PATHS ---
PROJECT_DIR = os.getcwd()
ANNOTATIONS_DIR = os.path.join(PROJECT_DIR, 'data/mscoco/annotations')
CAPTIONS_FILE = os.path.join(PROJECT_DIR, 'data/contrastive_set/generated_captions.json')
OUTPUT_PAIRS_FILE = os.path.join(PROJECT_DIR, 'data/contrastive_set/contrastive_pairs.json')

# --- 2. MONKEY-PATCH for MSCOCO 2017 DATASET ---
# The standalone chair.py script is hardcoded for the 2014 dataset.
# We dynamically replace its data loading functions to support our 2017 dataset.

def combine_coco_captions_2017(annotation_path):
    """Loads and combines train and val 2017 caption annotations."""
    val_path = os.path.join(annotation_path, 'captions_val2017.json')
    train_path = os.path.join(annotation_path, 'captions_train2017.json')
    
    if not os.path.exists(val_path) or not os.path.exists(train_path):
        raise FileNotFoundError("MSCOCO 2017 train/val caption annotations not found.")

    val_caps = json.load(open(val_path))
    train_caps = json.load(open(train_path))
    
    all_annotations = val_caps['annotations'] + train_caps['annotations']
    return {'annotations': all_annotations}

def combine_coco_instances_2017(annotation_path):
    """Loads and combines train and val 2017 instance annotations."""
    val_path = os.path.join(annotation_path, 'instances_val2017.json')
    train_path = os.path.join(annotation_path, 'instances_train2017.json')

    if not os.path.exists(val_path) or not os.path.exists(train_path):
        raise FileNotFoundError("MSCOCO 2017 train/val instance annotations not found.")

    val_instances = json.load(open(val_path))
    train_instances = json.load(open(train_path))
    
    all_instances = {
        'categories': train_instances['categories'],
        'annotations': val_instances['annotations'] + train_instances['annotations']
    }
    return all_instances

# Replace the original functions in the imported 'chair' module with our new ones
chair.combine_coco_captions = combine_coco_captions_2017
chair.combine_coco_instances = combine_coco_instances_2017

# Now we can import the CHAIR class, which will use our patched functions
from chair import CHAIR

# --- 3. MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    print("--- Starting Step 2B: Filtering with CHAIR metric (patched for MSCOCO 2017) ---")

    # Download NLTK data required by the CHAIR script if not present
    try:
        nltk.data.find('tokenizers/punkt')
    except nltk.downloader.DownloadError:
        print("NLTK 'punkt' tokenizer not found. Downloading...")
        nltk.download('punkt')

    # --- Load the generated captions data ---
    print(f"Loading generated captions from {CAPTIONS_FILE}...")
    with open(CAPTIONS_FILE, 'r') as f:
        generated_data = json.load(f)
    print(f"Loaded {len(generated_data)} generated captions.")

    # --- Initialize the CHAIR evaluator ---
    # This builds the ground-truth object database in memory using our patched functions.
    print("Initializing CHAIR evaluator (this may take a moment)...")
    evaluator = CHAIR(ANNOTATIONS_DIR)
    print("CHAIR evaluator ready.")

    # --- Process captions and create contrastive pairs ---
    contrastive_pairs = []
    print("Evaluating captions for hallucinations...")
    for item in tqdm(generated_data, desc="Filtering Captions"):
        image_id = item['image_id']
        generated_caption = item['generated_caption']
        
        # Use the CHAIR object's methods to process the caption
        _, node_words, _, _ = evaluator.caption_to_words(generated_caption)
        
        # Get the ground truth objects for this image
        gt_objects = evaluator.imid_to_objects.get(image_id, set())
        
        # Find hallucinated words
        hallucinated_words = [word for word in node_words if word not in gt_objects]
        
        chair_i_score = 0.0
        if len(node_words) > 0:
            chair_i_score = len(hallucinated_words) / float(len(node_words))
        
        # If the caption has at least one hallucinated object, it's a negative example
        if chair_i_score > 0:
            # Pair the hallucinated caption (negative) with a ground-truth caption (positive)
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
        with pd.option_context('display.max_colwidth', None):
            print(df.head())