import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from pycocotools.coco import COCO
from PIL import Image
import os
import json
import pandas as pd
from tqdm import tqdm

# --- 1. DEFINE FILE PATHS ---
# This assumes you run the script from the root of your project directory
PROJECT_DIR = os.getcwd()
ANNOTATION_FILE = os.path.join(PROJECT_DIR, 'data/mscoco/annotations/captions_train2017.json')
IMAGE_DIR = os.path.join(PROJECT_DIR, 'data/mscoco/train2017')
OUTPUT_FILE = os.path.join(PROJECT_DIR, 'data/contrastive_set/generated_captions.json')
MODEL_CACHE_DIR = os.path.join(PROJECT_DIR, 'model_cache')

# --- 2. MODEL LOADING FUNCTION (Full Precision for Local A100) ---
def load_llava_model(model_id="llava-hf/llava-1.5-7b-hf", cache_dir=MODEL_CACHE_DIR):
    """
    Loads the LLaVA model and its associated processor in full float16 precision.
    """
    print(f"Loading model in full float16 precision: {model_id}...")
    os.makedirs(cache_dir, exist_ok=True)

    model = LlavaForConditionalGeneration.from_pretrained(
        model_id, 
        torch_dtype=torch.float16, 
        low_cpu_mem_usage=True,
        cache_dir=cache_dir
    ).to("cuda")

    processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir)
    
    print("Model and processor loaded successfully onto the GPU.")
    return model, processor

# --- 3. HELPER FUNCTIONS ---
def get_image_path(coco_obj, image_id):
    """Constructs the full path for an image given its ID."""
    img_info = coco_obj.loadImgs(image_id)
    return os.path.join(IMAGE_DIR, img_info[0]['file_name'])

def get_ground_truth_captions(coco_obj, image_id):
    """Retrieves all ground-truth captions for an image."""
    ann_ids = coco_obj.getAnnIds(imgIds=image_id)
    anns = coco_obj.loadAnns(ann_ids)
    return [ann['caption'] for ann in anns]

def generate_caption(model, processor, image, prompt_text):
    """Generates a caption for a given image using the LLaVA model."""
    prompt = f"USER: <image>\n{prompt_text}\nASSISTANT:"
    
    inputs = processor(text=prompt, images=image, return_tensors='pt').to("cuda", torch.float16)

    output = model.generate(**inputs, max_new_tokens=100, do_sample=False)

    full_response = processor.batch_decode(output, skip_special_tokens=True)
    
    try:
        assistant_response = full_response[0].split("ASSISTANT:")[1].strip()
        return assistant_response
    except IndexError:
        print(f"Warning: Could not parse ASSISTANT response from: {full_response}")
        return "" # Return empty string if parsing fails

# --- 4. MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    # Load the model and processor
    model, processor = load_llava_model()

    # Load COCO annotations
    print("Loading MSCOCO annotations...")
    coco = COCO(ANNOTATION_FILE)
    print("Annotations loaded.")
    image_ids = coco.getImgIds()

    # --- Configuration ---
    # Start with a small number, then increase to ~200-300 for the final run
    NUM_IMAGES_TO_PROCESS = 5000  
    PROMPT_FOR_GENERATION = "Describe this image in detail."

    results = []
    image_ids_subset = image_ids[:NUM_IMAGES_TO_PROCESS]

    print(f"Starting caption generation for {len(image_ids_subset)} images...")

    if os.path.exists(OUTPUT_FILE):
        print(f"Output file found. Loading existing results from {OUTPUT_FILE}")
        with open(OUTPUT_FILE, 'r') as f:
            results = json.load(f)
        
        processed_image_ids = {res['image_id'] for res in results}
        print(f"{len(processed_image_ids)} images already processed.")
        
        image_ids_subset = [img_id for img_id in image_ids_subset if img_id not in processed_image_ids]
        print(f"Resuming generation for the remaining {len(image_ids_subset)} images.")

    for image_id in tqdm(image_ids_subset, desc="Generating Captions"):
        try:
            image_path = get_image_path(coco, image_id)
            if not os.path.exists(image_path):
                print(f"Warning: Image file not found, skipping: {image_path}")
                continue
                
            image = Image.open(image_path).convert("RGB")
            ground_truths = get_ground_truth_captions(coco, image_id)
            generated_caption = generate_caption(model, processor, image, PROMPT_FOR_GENERATION)
            
            if generated_caption: # Only append if caption generation was successful
                results.append({
                    "image_id": image_id,
                    "image_file": os.path.basename(image_path),
                    "ground_truth_captions": ground_truths,
                    "generated_caption": generated_caption
                })

        except Exception as e:
            print(f"Error processing image {image_id}: {e}")
            continue

    # --- Save final results ---
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"\nCaption generation complete. Results saved to {OUTPUT_FILE}")

    # Display the first few results for a quick check
    df = pd.DataFrame(results)
    print("\nSample of generated data:")
    print(df.head())