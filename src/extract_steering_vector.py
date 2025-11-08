import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from pycocotools.coco import COCO
from PIL import Image
import os
import json
import numpy as np
from tqdm import tqdm

# --- 1. DEFINE FILE PATHS ---
PROJECT_DIR = os.getcwd()
CONTRASTIVE_PAIRS_FILE = os.path.join(PROJECT_DIR, 'data/contrastive_set/contrastive_pairs.json')
IMAGE_DIR = os.path.join(PROJECT_DIR, 'data/mscoco/train2017')
# --- FIX #1: Added path to the annotation file ---
ANNOTATION_FILE = os.path.join(PROJECT_DIR, 'data/mscoco/annotations/captions_train2017.json')
STEERING_VECTORS_DIR = os.path.join(PROJECT_DIR, 'steering_vectors')
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

# --- 3. ACTIVATION EXTRACTION LOGIC ---
activations = {}

def get_activation(name):
    """A hook function to store the output of a module."""
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook

def get_kv_activations(model, processor, image, caption):
    """
    Performs a forward pass and extracts the Key and Value activations 
    from the final token of the input.
    """
    global activations
    activations = {}
    
    prompt = f"USER: <image>\n{caption}"
    inputs = processor(text=prompt, images=image, return_tensors='pt').to("cuda", torch.float16)
    
    hooks = []
    for i, layer in enumerate(model.language_model.layers):
        hooks.append(layer.self_attn.k_proj.register_forward_hook(get_activation(f'k_{i}')))
        hooks.append(layer.self_attn.v_proj.register_forward_hook(get_activation(f'v_{i}')))
        
    with torch.no_grad():
        model(**inputs)
        
    for hook in hooks:
        hook.remove()
        
    num_layers = len(model.language_model.layers)
    key_activations = []
    value_activations = []

    for i in range(num_layers):
        key_activations.append(activations[f'k_{i}'][0, -1, :])
        value_activations.append(activations[f'v_{i}'][0, -1, :])
        
    return key_activations, value_activations

# --- 4. MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    print("--- Starting Step 3: Extracting Steering Vector ---")
    
    model, processor = load_llava_model()
    num_layers = len(model.language_model.layers)

    # --- FIX #2: Load COCO annotations to map image IDs to filenames ---
    print("Loading COCO annotations...")
    coco = COCO(ANNOTATION_FILE)
    print("Annotations loaded.")

    with open(CONTRASTIVE_PAIRS_FILE, 'r') as f:
        contrastive_pairs = json.load(f)
    print(f"Loaded {len(contrastive_pairs)} contrastive pairs.")

    key_diffs = [[] for _ in range(num_layers)]
    value_diffs = [[] for _ in range(num_layers)]

    for pair in tqdm(contrastive_pairs, desc="Processing pairs"):
        image_id = pair['image_id']
        
        img_info = coco.loadImgs(image_id)
        if not img_info:
            print(f"Warning: Image ID {image_id} not found in annotations. Skipping.")
            continue
        image_path = os.path.join(IMAGE_DIR, img_info[0]['file_name'])
        
        if not os.path.exists(image_path):
            print(f"Warning: Image file {image_path} not found. Skipping.")
            continue
            
        image = Image.open(image_path).convert("RGB")
        
        # --- FIX #3: Select the first caption from the 'positive' list ---
        positive_caption = pair['positive']
        negative_caption = pair['negative']
        
        pos_keys, pos_values = get_kv_activations(model, processor, image, positive_caption)
        neg_keys, neg_values = get_kv_activations(model, processor, image, negative_caption)
        
        for i in range(num_layers):
            key_diffs[i].append(pos_keys[i] - neg_keys[i])
            value_diffs[i].append(pos_values[i] - neg_values[i])

    steering_vectors_k = []
    steering_vectors_v = []

    print("\nAveraging differences to create steering vectors...")
    for i in range(num_layers):
        layer_key_diffs = torch.stack(key_diffs[i])
        mean_key_diff = layer_key_diffs.mean(dim=0)
        steering_vectors_k.append(mean_key_diff)
        
        layer_value_diffs = torch.stack(value_diffs[i])
        mean_value_diff = layer_value_diffs.mean(dim=0)
        steering_vectors_v.append(mean_value_diff)

    os.makedirs(STEERING_VECTORS_DIR, exist_ok=True)
    torch.save(steering_vectors_k, os.path.join(STEERING_VECTORS_DIR, 'steering_vectors_k.pt'))
    torch.save(steering_vectors_v, os.path.join(STEERING_VECTORS_DIR, 'steering_vectors_v.pt'))

    print(f"\nSteering vectors saved successfully to {STEERING_VECTORS_DIR}")
    print(f"Generated {len(steering_vectors_k)} key vectors and {len(steering_vectors_v)} value vectors.")
    print("--- Step 3 Complete ---")