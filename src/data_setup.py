import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
import os
import requests
import zipfile
from tqdm import tqdm


### For GPU tasks, we will load the model in full float16 precision.
# def load_llava_model(model_id="llava-hf/llava-1.5-7b-hf", cache_dir="./model_cache"):
#     """
#     Loads the LLaVA model and its associated processor in full precision (float16).
    
#     Args:
#         model_id (str): The Hugging Face model identifier.
#         cache_dir (str): The directory to cache the downloaded model.

#     Returns:
#         tuple: A tuple containing the loaded model and processor.
#     """
#     print(f"Loading model in full float16 precision: {model_id}...")
    
#     # Ensure the cache directory exists
#     os.makedirs(cache_dir, exist_ok=True)

#     # Load the model in float16, without 4-bit quantization.
#     model = LlavaForConditionalGeneration.from_pretrained(
#         model_id, 
#         torch_dtype=torch.float16, 
#         low_cpu_mem_usage=True,
#         cache_dir=cache_dir
#     ).to("cuda") # Move the model to the GPU

#     # The processor handles both image preprocessing and text tokenization
#     processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir)
    
#     print("Model and processor loaded successfully onto the GPU.")
#     return model, processor


### Currenty using 4bit quantization to reduce memory usage.
def load_llava_model(model_id="llava-hf/llava-1.5-7b-hf", cache_dir="./model_cache"):
    """
    Loads the LLaVA model and its associated processor in full precision (float16).
    
    Args:
        model_id (str): The Hugging Face model identifier.
        cache_dir (str): The directory to cache the downloaded model.

    Returns:
        tuple: A tuple containing the loaded model and processor.
    """
    print(f"Loading model in full float16 precision: {model_id}...")
    
    # Ensure the cache directory exists
    os.makedirs(cache_dir, exist_ok=True)

    # Load the model in float16, without 4-bit quantization.
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id, 
        torch_dtype=torch.float16, 
        low_cpu_mem_usage=True,
        cache_dir=cache_dir
    ).to("cuda") # Move the model to the GPU

    # The processor handles both image preprocessing and text tokenization
    processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir)
    
    print("Model and processor loaded successfully onto the GPU.")
    return model, processor


def download_and_unzip(url, target_dir):
    """
    Downloads a file from a URL, shows a progress bar, and unzips it.
    """
    filename = url.split('/')[-1]
    filepath = os.path.join(target_dir, filename)
    
    # Check if the unzipped directory already exists to avoid re-downloading
    unzipped_dir_name = filename.replace('.zip', '')
    if os.path.exists(os.path.join(target_dir, unzipped_dir_name)) or \
       (filename == 'annotations_trainval2017.zip' and os.path.exists(os.path.join(target_dir, 'annotations'))):
        print(f"'{unzipped_dir_name}' already exists. Skipping download.")
        return

    print(f"Downloading {filename}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for bad status codes
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filepath, 'wb') as f, tqdm(
            desc=filename,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=1024):
                size = f.write(data)
                bar.update(size)
        
        print(f"Unzipping {filename}...")
        with zipfile.ZipFile(filepath, 'r') as zip_ref:
            zip_ref.extractall(target_dir)
        os.remove(filepath) # Clean up the zip file after extraction
        print(f"Finished processing {filename}.")

    except requests.exceptions.RequestException as e:
        print(f"Error downloading {filename}: {e}")
        if os.path.exists(filepath):
            os.remove(filepath)

def setup_mscoco_dataset(base_dir="data/mscoco"):
    """
    Downloads and sets up the MSCOCO 2017 dataset (images and annotations).
    """
    print("--- Setting up MSCOCO Dataset ---")
    os.makedirs(base_dir, exist_ok=True)

    # URLs for MSCOCO 2017 dataset components
    urls = {
        "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
        "train_images": "http://images.cocodataset.org/zips/train2017.zip",
        "val_images": "http://images.cocodataset.org/zips/val2017.zip"
    }

    # Download and process each component
    download_and_unzip(urls["annotations"], base_dir)
    download_and_unzip(urls["train_images"], base_dir)
    download_and_unzip(urls["val_images"], base_dir)
    
    print("\nMSCOCO dataset setup complete.")
    print(f"Data is located in: {os.path.abspath(base_dir)}")



# --- You can run this function to download the data ---
model, processor = load_llava_model() 
setup_mscoco_dataset()