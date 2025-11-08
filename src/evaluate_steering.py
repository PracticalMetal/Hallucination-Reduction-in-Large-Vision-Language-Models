import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration, DynamicCache
from pycocotools.coco import COCO
from PIL import Image
import os
import json
import numpy as np
from tqdm import tqdm

# --- 1. DEFINE FILE PATHS ---
PROJECT_DIR = os.getcwd()
# ---!! IMPORTANT: We are now using the VALIDATION set ---
IMAGE_DIR = os.path.join(PROJECT_DIR, 'data/mscoco/val2017') 
ANNOTATION_FILE = os.path.join(PROJECT_DIR, 'data/mscoco/annotations/captions_val2017.json') 
STEERING_VECTORS_DIR = os.path.join(PROJECT_DIR, 'steering_vectors')
MODEL_CACHE_DIR = os.path.join(PROJECT_DIR, 'model_cache')
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results') # Output directory

# --- 2. MODEL LOADING FUNCTION ---
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
    model.eval() # Set model to evaluation mode
    return model, processor

# --- 3. GENERATION FUNCTIONS (from verify_steering_vector.py) ---

def generate_baseline(model, processor, image, prompt_text):
    """Generates a caption without any steering."""
    prompt = f"USER: <image>\n{prompt_text}\nASSISTANT:"
    inputs = processor(text=prompt, images=image, return_tensors='pt').to("cuda", torch.float16)
    
    output = model.generate(**inputs, max_new_tokens=100, do_sample=False)
    
    # Decode and clean up
    full_response_list = processor.batch_decode(output, skip_special_tokens=True)
    try:
        return full_response_list[0].split("ASSISTANT:")[1].strip()
    except IndexError:
        return full_response_list[0].strip() # Fallback if parsing fails

@torch.inference_mode()
def generate_with_steering(model, processor, image, prompt_text,
                           steering_k_list, steering_v_list, coeff_k, coeff_v):
    # 0) Build the multimodal prompt with LLaVA
    prompt = f"USER: <image>\n{prompt_text}\nASSISTANT"
    inputs = processor(text=prompt, images=image, return_tensors='pt').to(model.device, torch.float16)

    prompt2 = f"USER: <image>\n{prompt_text}\nASSISTANT:"
    inputs2 = processor(text=prompt2, images=image, return_tensors='pt').to(model.device, torch.float16)

    # 1) Prefill to build cache
    out = model(**inputs, use_cache=True, return_dict=True)
    cache = DynamicCache.from_legacy_cache(out.past_key_values)  # convert tuple -> Cache

    # 2) Edit last-token KV per layer
    legacy = list(cache.to_legacy_cache())  # [(k,v), ...] with shapes [B, H, T, D]
    for i, (k, v) in enumerate(legacy):
        nh, hd = k.shape[1], k.shape[3]
        k2 = k.clone()
        v2 = v.clone()
        k2[0, :, -1, :] += coeff_k * steering_k_list[i].reshape(nh, hd).to(k2.dtype).to(k2.device)
        v2[0, :, -1, :] += coeff_v * steering_v_list[i].reshape(nh, hd).to(v2.dtype).to(v2.device)
        legacy[i] = (k2, v2)
    cache = DynamicCache.from_legacy_cache(tuple(legacy))  # rewrap edits

    # 3) Seed generation with the last text token
    seed_ids = inputs2["input_ids"][:, -1:]                  # K = 1
    past_len = cache.get_seq_length()                       # N
    cache_pos = torch.arange(past_len, past_len + seed_ids.shape[1],
                             device=seed_ids.device)        # [N]
    # attention_mask must represent past + new tokens
    attn = torch.cat(
        [inputs2["attention_mask"], inputs2["attention_mask"].new_ones((inputs2["attention_mask"].size(0), seed_ids.size(1)))],
        dim=-1
    )

    # 4) Resume decoding

    out_ids = model.generate(
        input_ids=seed_ids,
        past_key_values=cache,      # pass Cache object
        cache_position=cache_pos,   # explicit, avoids empty cache_position bug
        attention_mask= attn,     # pass full attention mask
        max_new_tokens=100,
        do_sample=False,
    )
    response = processor.batch_decode(out_ids, skip_special_tokens=True)[0].strip()
    return response.lstrip(": ").strip()

# --- 4. MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    
    # ---!! SET YOUR OPTIMAL VALUES HERE!! ---
    # Use the best-performing values you found during verification
    YOUR_COEFF_K = 0.1 
    YOUR_COEFF_V = 2.0
    # -----------------------------------------

    NUM_EVAL_IMAGES = 100 # Set this to a number you are comfortable with. 500 is a good start.
    PROMPT_TEXT = "Describe this image in detail."

    # Define output file paths
    BASELINE_OUTPUT_FILE = os.path.join(RESULTS_DIR, 'baseline_eval_results.json')
    STEERED_OUTPUT_FILE = os.path.join(RESULTS_DIR, 'steered_eval_results.json')
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Load model and data
    model, processor = load_llava_model()
    coco = COCO(ANNOTATION_FILE)
    image_ids = coco.getImgIds()[:NUM_EVAL_IMAGES] # Limit to NUM_EVAL_IMAGES
    
    # Load steering vectors
    steering_vectors_k_cpu = torch.load(os.path.join(STEERING_VECTORS_DIR, 'steering_vectors_k.pt'))
    steering_vectors_v_cpu = torch.load(os.path.join(STEERING_VECTORS_DIR, 'steering_vectors_v.pt'))
    steering_vectors_k = [v.to("cuda") for v in steering_vectors_k_cpu]
    steering_vectors_v = [v.to("cuda") for v in steering_vectors_v_cpu]
    
    print(f"\nStarting evaluation on {len(image_ids)} images from the validation set...")
    
    # --- 1. Generate BASELINE results ---
    print(f"\n--- Generating BASELINE captions (saving to {BASELINE_OUTPUT_FILE}) ---")
    baseline_results = []
    for image_id in tqdm(image_ids, desc="Baseline Generation"):
        img_info = coco.loadImgs(image_id)
        image_path = os.path.join(IMAGE_DIR, img_info[0]['file_name'])
        image = Image.open(image_path).convert("RGB")
        
        caption = generate_baseline(model, processor, image, PROMPT_TEXT)
        # We save in the format that our utils/chair.py script expects
        baseline_results.append({"image_id": image_id, "caption": caption})
    
    with open(BASELINE_OUTPUT_FILE, 'w') as f:
        json.dump(baseline_results, f, indent=4)
    print("Baseline results saved.")

    # --- 2. Generate STEERED results ---
    print(f"\n--- Generating STEERED captions (saving to {STEERED_OUTPUT_FILE}) ---")
    steered_results = []
    for image_id in tqdm(image_ids, desc="Steered Generation"):
        img_info = coco.loadImgs(image_id)
        image_path = os.path.join(IMAGE_DIR, img_info['file_name'])
        image = Image.open(image_path).convert("RGB")
        
        caption = generate_with_steering(
            model, processor, image, PROMPT_TEXT,
            steering_vectors_k, steering_vectors_v,
            YOUR_COEFF_K, YOUR_COEFF_V
        )
        # We save in the format that our utils/chair.py script expects
        steered_results.append({"image_id": image_id, "caption": caption})

    with open(STEERED_OUTPUT_FILE, 'w') as f:
        json.dump(steered_results, f, indent=4)
    print("Steered results saved.")
    
    print("\n--- Step 4a (Generation) Complete ---")
    print(f"Next step: Run the CHAIR metric on both {BASELINE_OUTPUT_FILE} and {STEERED_OUTPUT_FILE}.")