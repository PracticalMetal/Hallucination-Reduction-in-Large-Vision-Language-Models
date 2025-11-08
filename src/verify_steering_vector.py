import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration, DynamicCache
from pycocotools.coco import COCO
from PIL import Image
import os
import json
import numpy as np

# --- 1. DEFINE FILE PATHS ---
PROJECT_DIR = os.getcwd()
CONTRASTIVE_PAIRS_FILE = os.path.join(PROJECT_DIR, 'data/contrastive_set/contrastive_pairs.json')
IMAGE_DIR = os.path.join(PROJECT_DIR, 'data/mscoco/train2017')
ANNOTATION_FILE = os.path.join(PROJECT_DIR, 'data/mscoco/annotations/captions_train2017.json')
STEERING_VECTORS_DIR = os.path.join(PROJECT_DIR, 'steering_vectors')
MODEL_CACHE_DIR = os.path.join(PROJECT_DIR, 'model_cache')

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
    return model, processor

# --- 3. GENERATION FUNCTIONS ---
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

# # Requires: transformers >= 4.47, torch >= 2.2
# # Docs:
# #   https://huggingface.co/docs/transformers/en/cache_explanation
# #   https://huggingface.co/docs/transformers/en/kv_cache

# import torch
# from transformers import DynamicCache

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
    model, processor = load_llava_model()
    coco = COCO(ANNOTATION_FILE)
    with open(CONTRASTIVE_PAIRS_FILE, 'r') as f:
        contrastive_pairs = json.load(f)
        
    # Load steering vectors
    steering_vectors_k_cpu = torch.load(os.path.join(STEERING_VECTORS_DIR, 'steering_vectors_k.pt'))
    steering_vectors_v_cpu = torch.load(os.path.join(STEERING_VECTORS_DIR, 'steering_vectors_v.pt'))
    
    # Move steering vectors to the same device as the model (GPU)
    steering_vectors_k = [v.to("cuda") for v in steering_vectors_k_cpu]
    steering_vectors_v = [v.to("cuda") for v in steering_vectors_v_cpu]
    
    # --- Select a test case ---
    test_pair = max(contrastive_pairs, key=lambda x: x['chair_i_score'])
    
    image_id = test_pair['image_id']
    img_info = coco.loadImgs(image_id)
    image_path = os.path.join(IMAGE_DIR, img_info[0]['file_name'])
    image = Image.open(image_path).convert("RGB")
    
    prompt_text = "Describe this image in detail."
    
    print("\n--- VERIFICATION TEST ---")
    print(f"Using Image ID: {image_id}")
    print(f"Original Factual Caption (Positive): {test_pair['positive']}") # Print first factual caption
    print(f"Original Hallucinated Caption (Negative): {test_pair['negative']}")
    print(f"CHAIR Score: {test_pair['chair_i_score']}\n")
    
    # --- Run baseline generation ---
    print("--- 1. Generating with BASELINE model (no steering) ---")
    baseline_caption = generate_baseline(model, processor, image, prompt_text)
    print(f"Output: {baseline_caption}\n")
    
    # --- Run steered generation ---
    COEFF_K = 0
    COEFF_V = 0
    
    print(f"--- 2. Generating WITH STEERING (k_coeff={COEFF_K}, v_coeff={COEFF_V}) ---")
    steered_caption = generate_with_steering(
        model, processor, image, prompt_text, 
        steering_vectors_k, steering_vectors_v, 
        COEFF_K, COEFF_V
    )
    print(f"Output: {steered_caption}\n")
    
    print("--- Verification Complete ---")