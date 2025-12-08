import os
import sys
import argparse
import json
import torch
import traceback
# from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor
from huggingface_hub import snapshot_download
# from qwen_vl_utils import process_vision_info

# Parse GPU IDs early before importing torch (consistent with your training script)
parser = argparse.ArgumentParser(description="Generate image descriptions for scientific papers using Qwen3-VL")
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
    # print(f"Found {len(image_files)} images in {directory}")
    # print(image_files)
    return image_files

def main():
    # Re-parse arguments with all options
    parser = argparse.ArgumentParser(description="Generate image descriptions for scientific papers using Qwen3-VL")
    parser.add_argument("--data_root", type=str, default="/ceph/hpc/home/euweihez/topcon/d2025d08-005-users/data_src/balanced/balanced_llm", help="Root directory containing paper folders")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-VL-8B-Instruct", help="Path to Qwen3-VL model")
    parser.add_argument("--output_file", type=str, default="/ceph/hpc/home/euweihez/topcon/d2025d08-005-users/img_description/llm/image_descriptions.json", help="Output JSON file path")
    parser.add_argument("--gpu_ids", type=int, nargs='+', default=None, help="GPU IDs to use")
    parser.add_argument("--batch_size", type=int, default=1, help="Inference batch size (keep 1 for variable image sizes)")
    args = parser.parse_args()

    # Setup logging
    log_file = "image_generation.log"
    sys.stdout = TeeOutput(log_file)
    
    # Check if model exists locally, if not download it
    local_model_dir = "/ceph/hpc/home/euweihez/qwen3-vl/8b"
    if args.model_path == "Qwen/Qwen3-VL-8B-Instruct":
        if not os.path.exists(local_model_dir) or not os.listdir(local_model_dir):
            print(f"Model not found at {local_model_dir}. Downloading {args.model_path}...")
            try:
                snapshot_download(repo_id=args.model_path, local_dir=local_model_dir)
                print(f"Model downloaded successfully to {local_model_dir}")
            except Exception as e:
                print(f"Error downloading model: {e}")
                print("Proceeding with default remote loading...")
        
        if os.path.exists(local_model_dir) and os.listdir(local_model_dir):
            print(f"Using local model from {local_model_dir}")
            args.model_path = local_model_dir

    print(f"Using Data Root: {args.data_root}")
    print(f"Loading model: {args.model_path}")
    
    # Load model and processor
    # Note: Qwen3-VL requires 'qwen_vl_utils' and transformers>=4.47.1
    try:
        model = AutoModelForImageTextToText.from_pretrained(
            args.model_path,
            dtype="auto",                # Qwen docs use `dtype`, not `torch_dtype`
            device_map="auto",
            attn_implementation="sdpa"
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

    prompts = {
    "Architecture Diagram": """
        If it is an Architecture Diagram, analyze the image based on the following:
        1. Clarity: Are distinct elements easily distinguishable and is the resolution high enough?
        2. Logic: Is the flow easy to follow with clearly defined inputs and outputs (e.g., tensor shapes)?
        3. Aesthetics: Evaluate the professionalism regarding color palettes, alignment, and whitespace usage.
        4. Self-Containedness: Can the main methodological idea be understood purely from the diagram without reading the full paper text?
    """,

    "Teaser Image": """
        If it is a Teaser Image, evaluate its ability to 'hook' the reader by checking:
        1. Impact: Is the visual contrast between the baseline and the proposed method immediately obvious?
        2. Storytelling: Does the image successfully convey the paper's core contribution in under 5 seconds?
        3. Layout: Is the space utilized effectively without feeling cluttered?
    """,

    "Plots / Charts": """
        If it is a Plot or Chart, critique the quantitative presentation by checking:
        1. Legibility: Are the axis labels, legends, and tick marks readable without zooming in?
        2. Vectorization: Do lines and markers appear crisp (vector-based) rather than pixelated?
        3. Distinction: Are the colors and line styles (solid/dashed) distinct enough to differentiate methods, even for color-blind readers?
    """,

    "Qualitative Examples": """
        If it is a Qualitative Example (visual results), assess the fairness and fidelity by checking:
        1. Detail: Do the zoomed-in patches effectively highlight the specific improvements claimed?
        2. Artifacts: Are there any visible generative artifacts (e.g., checkerboard patterns, blurring) that undermine the quality?
        3. Alignment: Are the comparisons strictly side-by-side with identical cropping/conditions?
    """
    }

    BASE_DESCRIPTION_PROMPT = (
        "1. Identify the type of figure (e.g., teaser image, Plots / Charts, Method / Architecture Diagram, Qualitative Examples).\n"
        "2. Describe the overall layout (number of subplots or panels, their arrangement).\n"
        "3. List all visual elements present (e.g., lines, bars, points, images, annotations, text blocks).\n"
        "4. Describe colors and shapes used for different categories or methods.\n"
        "5. If some text or details are too small or unreadable, explicitly say 'unreadable' instead of guessing.\n"
        "6. Assess the readability and informative quality of this image.\n"
)
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
                image_uri = os.path.abspath(img_path)

                # ===== 1) FIRST PASS: classify figure type =====
                classify_messages = [
                    {
                        "role": "system",
                        "content": [ 
                            # CHANGE: Wrap string in a list with type="text"
                            {
                                "type": "text",
                                "text": (
                                    "You are an expert in scientific figure design for top-tier AI conferences. "
                                    "Your job is to classify the type of the given figure from an AI paper.\n\n"
                                    "Respond with EXACTLY ONE of the following options (no extra words):\n"
                                    " - Teaser Image\n"
                                    " - Architecture Diagram\n"
                                    " - Plots / Charts\n"
                                    " - Qualitative Examples\n"
                                    " - Other"
                                )
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image_uri},
                            {
                                "type": "text",
                                "text": (
                                    "Look at this figure and decide its type. "
                                    "Reply with exactly one label from the list."
                                ),
                            },
                        ],
                    },
                ]

                class_inputs = processor.apply_chat_template(
                    classify_messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                ).to(model.device)

                with torch.no_grad():
                    class_ids = model.generate(**class_inputs, max_new_tokens=16)

                class_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(class_inputs.input_ids, class_ids)
                ]
                raw_type = processor.batch_decode(
                    class_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0].strip()

                # Normalise to one of our keys
                raw_type_lower = raw_type.lower()
                figure_type = "Other"
                if "teaser" in raw_type_lower:
                    figure_type = "Teaser Image"
                elif "architecture" in raw_type_lower or "diagram" in raw_type_lower or "method" in raw_type_lower:
                    figure_type = "Architecture Diagram"
                elif "plot" in raw_type_lower or "chart" in raw_type_lower or "curve" in raw_type_lower or "graph" in raw_type_lower:
                    figure_type = "Plots / Charts"
                elif "qualitative" in raw_type_lower or "example" in raw_type_lower or "visual result" in raw_type_lower or "comparison" in raw_type_lower:
                    figure_type = "Qualitative Examples"

                print(f"[{paper_id}] {rel_path} -> classified as: {figure_type} (raw: {raw_type})")

                # ===== 2) SECOND PASS: generate full evaluation using conditional prompt =====
                type_specific_prompt = prompts.get(figure_type, "")

                eval_messages = [
                    {
                        "role": "system",
                        "content": [
                            # CHANGE: Wrap string in a list with type="text"
                            {
                                "type": "text", 
                                "text": (
                                    "You are a senior Reviewer for top-tier AI conferences (CVPR/ICLR/NeurIPS). "
                                    "You are an expert in Scientific Visualization and Information Design. "
                                    "You receive one figure from an AI conference paper. "
                                    "Your task is to OBJECTIVELY DESCRIBE what is visible in the figure, "
                                    "without guessing the paper’s claims or performance."
                                )
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image_uri},
                            {
                                "type": "text",
                                "text": (
                                    f"The figure has been classified as: {figure_type}.\n\n"
                                    "First, briefly restate the figure type in your own words.\n\n"
                                    "Then follow these generic instructions:\n"
                                    f"{BASE_DESCRIPTION_PROMPT}\n\n"
                                    "Finally, apply the following additional criteria that depend on this figure type "
                                    "(skip this part if it seems irrelevant):\n"
                                    f"{type_specific_prompt}"
                                ),
                            },
                        ],
                    },
                ]

                eval_inputs = processor.apply_chat_template(
                    eval_messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                ).to(model.device)

                with torch.no_grad():
                    generated_ids = model.generate(**eval_inputs, max_new_tokens=512)

                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(eval_inputs.input_ids, generated_ids)
                ]

                output_text_list = processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                output_text = output_text_list[0]

                # Store both type and description (you can change this back to a plain string if you prefer)
                results[paper_id][rel_path] = {
                    "figure_type": figure_type,
                    "raw_type_response": raw_type,
                    "description": output_text,
                }

                # Periodic save (every image)
                with open(args.output_file, "w") as f:
                    json.dump(results, f, indent=2)

            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                traceback.print_exc()
                torch.cuda.empty_cache()

    print(f"Processing complete. Results saved to {args.output_file}")

if __name__ == "__main__":
    main()
