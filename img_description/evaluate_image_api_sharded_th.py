import os
import sys
import argparse
import json
import base64
import time
import mimetypes
import math
from openai import OpenAI
from tqdm import tqdm
import random


# --- CONFIGURATION ---
# RPM = 60. With 4 parallel jobs, each job gets 15 RPM.
# 60 sec / 15 req = 4.0 sec delay.
# We use 5.0s to be safe.
MIN_INTERVAL = 4.0 
LAST_REQ_TIME = 0

# --- Configuration ---
# You can set this via environment variable or hardcode it (not recommended for sharing)
os.environ["DASHSCOPE_API_KEY"] = "sk-5e081662e553472788958c33a522dde2" 

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

def encode_image(image_path):
    """
    Encodes a local image file to a base64 string.
    Necessary for sending local images to the API.
    """
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type is None:
        mime_type = "image/jpeg" # Default fallback

    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
    
    return f"data:{mime_type};base64,{base64_image}"

def call_qwen_api(client, model_name, messages, max_tokens=512):
    """Wrapper to handle API calls with Pacing and Exponential Backoff"""
    global LAST_REQ_TIME
    
    # --- 1. PROACTIVE PACING ---
    # Calculate how long since the last request
    elapsed = time.time() - LAST_REQ_TIME
    
    # If we are too fast, sleep the difference
    if elapsed < MIN_INTERVAL:
        sleep_needed = MIN_INTERVAL - elapsed
        # Add tiny random jitter (0-0.5s) so parallel jobs don't sync up perfectly
        time.sleep(sleep_needed + random.uniform(0, 0.5))

    retries = 6
    backoff = 10  # Start with a higher backoff (10s) since limits are tight
    
    for attempt in range(retries):
        try:
            LAST_REQ_TIME = time.time() # Mark the time of request
            
            completion = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.1, 
                stream=False
            )
            return completion.choices[0].message.content

        except Exception as e:
            error_str = str(e).lower()
            is_rate_limit = "429" in error_str or "limit" in error_str
            
            if attempt < retries - 1:
                # If it's a rate limit, wait longer. If it's a server error, wait standard time.
                wait_base = backoff if is_rate_limit else 2
                
                # Exponential Backoff: 10s -> 20s -> 40s...
                sleep_time = (wait_base * (2 ** attempt)) + random.uniform(1.0, 3.0)
                
                print(f"API Error (Attempt {attempt+1}/{retries}): {e}")
                print(f"Waiting {sleep_time:.2f}s before retry...")
                time.sleep(sleep_time)
            else:
                print(f"API Failed after {retries} attempts: {e}")
                raise e

def main():
    parser = argparse.ArgumentParser(description="Generate image descriptions for scientific papers using Qwen API")
    parser.add_argument("--data_root", type=str, default="/ceph/hpc/home/euweihez/topcon/d2025d08-005-users/data_src/balanced/balanced_llm", help="Root directory containing paper folders")
    parser.add_argument("--model_name", type=str, default="qwen3-vl-8b-instruct", help="Qwen API model name (e.g., qwen-vl-max, qwen-vl-plus)")
    parser.add_argument("--output_file", type=str, default="/ceph/hpc/home/euweihez/topcon/d2025d08-005-users/img_description/llm/image_descriptions.json", help="Output JSON file path")
    
    # --- NEW ARGUMENTS FOR PARALLELISM ---
    parser.add_argument("--shard_id", type=int, default=0, help="The index of the current job (0 to num_shards-1)")
    parser.add_argument("--num_shards", type=int, default=1, help="Total number of parallel jobs")
    
    args = parser.parse_args()

    # Setup logging with unique name per shard
    log_file = f"image_generation_shard_{args.shard_id}_th.log"
    sys.stdout = TeeOutput(log_file)

    # Modify output filename to be unique per shard
    # e.g., image_descriptions_0.json
    base, ext = os.path.splitext(args.output_file)
    shard_output_file = f"{base}_{args.shard_id}{ext}"
    
    # Check API Key
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("Error: DASHSCOPE_API_KEY environment variable is not set.")
        print("Please export your API key: export DASHSCOPE_API_KEY='sk-...'")
        return

    # Initialize OpenAI Client for Qwen
    print(f"Initializing Qwen Client with model: {args.model_name}")
    client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    )

    print(f"Using Data Root: {args.data_root}")

    results = {}
    
    # Resume logic for this specific shard file
    if os.path.exists(shard_output_file):
        print(f"Found existing output file {shard_output_file}, loading to resume...")
        try:
            with open(shard_output_file, 'r') as f:
                results = json.load(f)
        except json.JSONDecodeError:
            print("Could not decode existing JSON, starting fresh.")

    if not os.path.exists(args.data_root):
        print(f"Error: Data root {args.data_root} does not exist.")
        return

    # Get ALL directories first
    all_paper_dirs = sorted([d for d in os.listdir(args.data_root) if os.path.isdir(os.path.join(args.data_root, d))])
    total_papers = len(all_paper_dirs)

    # --- SHARDING LOGIC ---
    # Calculate which chunk of papers this specific job should handle
    chunk_size = math.ceil(total_papers / args.num_shards)
    start_idx = args.shard_id * chunk_size
    end_idx = min(start_idx + chunk_size, total_papers)
    
    # Slice the list
    my_paper_dirs = all_paper_dirs[start_idx:end_idx]
    
    print(f"--- SHARD INFO ---")
    print(f"Job ID: {args.shard_id} / {args.num_shards}")
    print(f"Processing indices: {start_idx} to {end_idx} (Total: {len(my_paper_dirs)} papers)")
    print(f"Outputting to: {shard_output_file}")
    print(f"------------------")

    # --- Prompts Configuration ---
    prompts = {
        "Architecture Diagram": """
            If it is an Architecture Diagram, analyze the image based on the following:
            1. Clarity: Are distinct elements easily distinguishable?
            2. Logic: Is the flow easy to follow (inputs/outputs)?
            3. Aesthetics: Evaluate professionalism, color, whitespace.
            4. Self-Containedness: Can it be understood without text?
        """,
        "Teaser Image": """
            If it is a Teaser Image, evaluate:
            1. Impact: Is the contrast between baseline and proposed method obvious?
            2. Storytelling: Does it convey the core contribution quickly?
            3. Layout: Is space utilized effectively?
        """,
        "Plots / Charts": """
            If it is a Plot or Chart, critique:
            1. Legibility: Axis labels, legends, tick marks.
            2. Vectorization: Crispness of lines.
            3. Distinction: Colors and line styles.
        """,
        "Qualitative Examples": """
            If it is a Qualitative Example, assess:
            1. Detail: Do zoomed-in patches highlight improvements?
            2. Artifacts: Are there visible generative artifacts?
            3. Alignment: Are comparisons side-by-side/fair?
        """
    }

    BASE_DESCRIPTION_PROMPT = (
        "1. Identify the type of figure (e.g., teaser image, Plots / Charts, Method / Architecture Diagram, Qualitative Examples, Other).\n"
        "2. Describe the overall layout (number of subplots or panels, their arrangement).\n"
        "3. List all visual elements present (e.g., lines, bars, points, images, annotations, text blocks).\n"
        "4. If some text or details are too small or unreadable, explicitly say 'unreadable' instead of guessing.\n"
    )

    # --- Main Loop ---
    for paper_id in tqdm(my_paper_dirs, desc=f"Shard {args.shard_id}"):
        paper_path = os.path.join(args.data_root, paper_id)
        figures_path = os.path.join(paper_path, "figures")
        
        if not os.path.exists(figures_path):
            continue

        image_paths = get_image_files(figures_path)
        if not image_paths:
            continue

        if paper_id not in results:
            results[paper_id] = {}

        for img_path in image_paths:
            rel_path = os.path.relpath(img_path, args.data_root)

            # Skip if already described
            if rel_path in results[paper_id]:
                continue

            try:
                base64_image = encode_image(img_path)

                # ===== 1) FIRST PASS: Classification =====
                classify_messages = [
                    {
                        "role": "system",
                        "content": "You are an expert in scientific figure design for top-tier AI conferences. Classify the figure type."
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "Look at this figure and decide its type.\n"
                                    "Respond with EXACTLY ONE of the following options (no extra words):\n"
                                    " - Teaser Image\n"
                                    " - Architecture Diagram\n"
                                    " - Plots / Charts\n"
                                    " - Qualitative Examples\n"
                                    " - Other"
                                )
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": base64_image}
                            }
                        ]
                    }
                ]
                
                raw_type = call_qwen_api(client, args.model_name, classify_messages, max_tokens=32)
                raw_type = raw_type.strip()

                # Normalize type
                raw_type_lower = raw_type.lower()
                figure_type = "Other"
                if "teaser" in raw_type_lower:
                    figure_type = "Teaser Image"
                elif "architecture" in raw_type_lower or "diagram" in raw_type_lower or "method" in raw_type_lower:
                    figure_type = "Architecture Diagram"
                elif "plot" in raw_type_lower or "chart" in raw_type_lower or "curve" in raw_type_lower or "graph" in raw_type_lower:
                    figure_type = "Plots / Charts"
                elif "qualitative" in raw_type_lower or "example" in raw_type_lower or "visual result" in raw_type_lower:
                    figure_type = "Qualitative Examples"

                print(f"[{paper_id}] {os.path.basename(rel_path)} -> classified as: {figure_type}")

                # Skip second pass if classified as "Other"
                if figure_type == "Other":
                    results[paper_id][rel_path] = {
                        "figure_type": figure_type,
                        "raw_type_response": raw_type,
                        "description": "Skipped - classified as Other",
                    }
                    
                    # Save immediately
                    with open(shard_output_file, "w") as f:
                        json.dump(results, f, indent=2)
                    
                    print(f"  -> Skipping second pass (Other)")
                    continue

                # ===== 2) SECOND PASS: Description =====
                type_specific_prompt = prompts.get(figure_type, "")
                
                eval_messages = [
                    {
                        "role": "system",
                        "content": (
                            "You are a senior, critical but honest reviewer for top-tier AI conferences (CVPR/ICLR/NeurIPS). "
                            "You are an expert in Scientific Visualization. OBJECTIVELY DESCRIBE what is visible."
                        )
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    f"The figure has been classified as: {figure_type}.\n\n"
                                    "First, briefly restate the figure in your own words.\n\n"
                                    "Second follow these generic instructions:\n"
                                    f"{BASE_DESCRIPTION_PROMPT}\n\n"
                                    # "Then apply the following additional criteria:\n"
                                    # f"{type_specific_prompt}"
                                    "Then give a strict quality score from **1-10** based on the overall quality of the figure. Most figures should receive mid-range scores (4–6), while very low (1-3) and very high (8-10) scores should be rare and reserved for exceptionally poor or exceptionally strong figures. No rationale is needed, just the score at the end.\n\n "
                                )
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": base64_image}
                            }
                        ]
                    }
                ]

                description = call_qwen_api(client, args.model_name, eval_messages, max_tokens=1024)

                # Store results
                results[paper_id][rel_path] = {
                    "figure_type": figure_type,
                    "raw_type_response": raw_type,
                    "description": description,
                }

                # Save immediately
                with open(shard_output_file, "w") as f:
                    json.dump(results, f, indent=2)

            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue

    print(f"Shard {args.shard_id} complete.")

if __name__ == "__main__":
    main()