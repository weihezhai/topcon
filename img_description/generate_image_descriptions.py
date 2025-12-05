import os
import sys
import argparse
import json
import torch
from PIL import Image
from tqdm import tqdm
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# Parse GPU IDs early before importing torch (consistent with your training script)
parser = argparse.ArgumentParser(description="Generate image descriptions for scientific papers using Qwen2-VL")
parser.add_argument("--gpu_ids", type=int, nargs='+', default=None, help="GPU IDs to use (e.g., --gpu_ids 0)")
args, unknown = parser.parse_known_args()

if args.gpu_ids:
    gpu_ids_str = ','.join(map(str, args.gpu_ids))
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids_str
    print(f"Set CUDA_VISIBLE_DEVICES to: {gpu_ids_str}")

class TeeOutput:
    """Class to duplicate stdout to both console and log file"""
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, 'w', buffering=1)
        
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        
    def flush(self):
        self.terminal.flush()
        self.log.flush()
        
    def close(self):
        self.log.close()

def get_image_files(directory):
    """Recursively find all image files in a directory"""
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    image_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if os.path.splitext(file)[1].lower() in valid_extensions:
                image_files.append(os.path.join(root, file))
    return image_files

def main():
    # Re-parse arguments with all options
    parser = argparse.ArgumentParser(description="Generate image descriptions for scientific papers using Qwen2-VL")
    parser.add_argument("--data_root", type=str, default="/ceph/hpc/home/euweihez/topcon/d2025d08-005-users/data_src/balanced/balanced_llm", help="Root directory containing paper folders")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2-VL-7B-Instruct", help="Path to Qwen2-VL model (or Qwen2.5-VL-7B-Instruct)")
    parser.add_argument("--output_file", type=str, default="image_descriptions.json", help="Output JSON file path")
    parser.add_argument("--gpu_ids", type=int, nargs='+', default=None, help="GPU IDs to use")
    parser.add_argument("--batch_size", type=int, default=1, help="Inference batch size (keep 1 for variable image sizes)")
    args = parser.parse_args()

    # Setup logging
    log_file = "image_generation.log"
    sys.stdout = TeeOutput(log_file)
    
    print(f"Using Data Root: {args.data_root}")
    print(f"Loading model: {args.model_path}")
    
    # Load model and processor
    # Note: Qwen2-VL requires 'qwen_vl_utils' and transformers>=4.45.0
    try:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            args.model_path,
            torch_dtype="auto",
            device_map="auto",
            attn_implementation="flash_attention_2" if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else "sdpa"
        )
        processor = AutoProcessor.from_pretrained(args.model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Ensure you have installed: pip install git+https://github.com/huggingface/transformers qwen-vl-utils")
        return

    print("Model loaded successfully.")
    if hasattr(model, 'hf_device_map'):
        print(f"Device map: {model.hf_device_map}")

    results = {}
    
    # Check if output file exists to resume
    if os.path.exists(args.output_file):
        print(f"Found existing output file {args.output_file}, loading to resume...")
        try:
            with open(args.output_file, 'r') as f:
                results = json.load(f)
        except json.JSONDecodeError:
            print("Could not decode existing JSON, starting fresh.")

    # Iterate through paper directories
    if not os.path.exists(args.data_root):
        print(f"Error: Data root {args.data_root} does not exist.")
        return

    # Get list of paper directories
    paper_dirs = [d for d in os.listdir(args.data_root) if os.path.isdir(os.path.join(args.data_root, d))]
    print(f"Found {len(paper_dirs)} paper directories.")

    for paper_id in tqdm(paper_dirs, desc="Processing papers"):
        # Skip if already fully processed (optional logic, here we just update)
        if paper_id in results and len(results[paper_id]) > 0:
            # You might want to skip or check if all images are done
            pass

        paper_path = os.path.join(args.data_root, paper_id)
        figures_path = os.path.join(paper_path, "figures")
        
        # Check if figures directory exists
        if not os.path.exists(figures_path):
            continue

        image_paths = get_image_files(figures_path)
        if not image_paths:
            continue

        if paper_id not in results:
            results[paper_id] = {}

        for img_path in image_paths:
            # Create a relative path key for the JSON
            rel_path = os.path.relpath(img_path, args.data_root)
            
            # Skip if already described
            if rel_path in results[paper_id]:
                continue

            try:
                # Prepare conversation for Qwen2-VL
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": img_path},
                            {"type": "text", "text": "Describe this image from a scientific paper in detail. Identify the type of figure (e.g., plot, diagram, architecture), describe the axes, legends, data trends, and any structural elements shown."},
                        ],
                    }
                ]

                # Prepare inputs
                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                
                image_inputs, video_inputs = process_vision_info(messages)
                
                inputs = processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )
                
                # Move inputs to model device
                inputs = inputs.to(model.device)

                # Generate description
                with torch.no_grad():
                    generated_ids = model.generate(**inputs, max_new_tokens=512)
                
                # Decode output
                generated_ids_trimmed = [
                    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]

                # Store result
                results[paper_id][rel_path] = output_text
                
                # Periodic save (every image to be safe)
                with open(args.output_file, 'w') as f:
                    json.dump(results, f, indent=2)
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                torch.cuda.empty_cache()

    print(f"Processing complete. Results saved to {args.output_file}")

if __name__ == "__main__":
    main()
