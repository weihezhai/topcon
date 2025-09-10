#!/usr/bin/env python
# coding: utf-8
"""

Three modes are supported:
  ① full        : images + abstract + intro
  ② image_only  : images
  ③ text_only   : abstract + intro


python dataset_builder.py \
    --mode full \
    --paper_root /path/to/iclr_2025_papers \
    --labels_json /path/to/labels.json \
    --output_dir /path/to/out_dataset \
    --split_ratio 0.2 \
    --statistics_json /path/to/statistics.json

(updated 2025-07-16)
"""

import os
import re
import json
import glob
import argparse
from typing import List, Dict

from PIL import Image, UnidentifiedImageError
from datasets import Dataset

def get_label_from_status(status: str) -> int:
    """Map paper status string to 0 = reject, 1 = accept"""
    return 0 if status in {
        "Submitted to ICLR 2025",
        "ICLR 2025 Conference Withdrawn Submission",
    } else 1


def _remove_github_links(text: str) -> str:
    """Remove sentences containing GitHub links from text."""
    # Remove entire sentences containing HTTP/HTTPS links
    # Use a more robust approach to avoid splitting at periods within URLs
    
    # First, temporarily replace URLs with a placeholder to avoid splitting issues
    url_pattern = re.compile(r'https?://[^\s]+')
    url_placeholder = "___URL_PLACEHOLDER___"
    
    # Find all URLs and their positions
    urls_found = url_pattern.findall(text)
    text_with_placeholders = url_pattern.sub(url_placeholder, text)
    
    # Now split into sentences (won't split at periods in URLs since they're replaced)
    sentences = re.split(r'(?<=[.!?])\s+', text_with_placeholders)
    
    # Filter out sentences containing GitHub links (both github.com and github.io)
    github_pattern = r'https?://(?:www\.)?(?:github\.com|[^/\s]*\.github\.io)[^\s)]*'
    filtered_sentences = []
    
    for sentence in sentences:
        # Check if this sentence had a GitHub URL
        has_github = False
        for url in urls_found:
            if re.match(github_pattern, url, re.IGNORECASE):
                # If this sentence contains the placeholder, it had this GitHub URL
                if url_placeholder in sentence:
                    has_github = True
                    break
        
        if not has_github:
            # Restore any non-GitHub URLs in this sentence
            sentence_restored = sentence
            for url in urls_found:
                if not re.match(github_pattern, url, re.IGNORECASE):
                    # Replace placeholder with original URL (only first occurrence)
                    sentence_restored = sentence_restored.replace(url_placeholder, url, 1)
            
            # Only add if there are no remaining placeholders (meaning no GitHub URLs)
            if url_placeholder not in sentence_restored:
                filtered_sentences.append(sentence_restored)
    
    return ' '.join(filtered_sentences)


def count_references(text: str) -> int:
    """Count references by counting occurrences of 'et al.' in the text"""
    # Count both "et al." and "et al," patterns (case-insensitive)
    et_al_pattern = re.compile(r'\bet\s+al\.?(?:\s*,|\s*;|\s*\)|\s+|\s*$)', re.IGNORECASE)
    matches = et_al_pattern.findall(text)
    return len(matches)


def format_statistics(paper_id: str, stats_dict: Dict, reference_count: int = None) -> str:
    """Format statistics for a paper as a tagged string"""
    features = []
    
    # Add reference count if provided
    if reference_count is not None:
        features.append(f"reference_count: {reference_count}")
    
    # Extract features from stats_dict if available
    if paper_id in stats_dict:
        paper_stats = stats_dict[paper_id]
        
        # (1) total_words
        if 'basic_metrics' in paper_stats and 'total_words' in paper_stats['basic_metrics']:
            features.append(f"total_words: {paper_stats['basic_metrics']['total_words']}")
        
        # (2) total_pages
        if 'basic_metrics' in paper_stats and 'total_pages' in paper_stats['basic_metrics']:
            features.append(f"total_pages: {paper_stats['basic_metrics']['total_pages']}")
        
        # (3) header_count
        if 'basic_metrics' in paper_stats and 'header_count' in paper_stats['basic_metrics']:
            features.append(f"header_count: {paper_stats['basic_metrics']['header_count']}")
        
        # (4) table_count
        if 'visual_content' in paper_stats and 'table_count' in paper_stats['visual_content']:
            features.append(f"table_count: {paper_stats['visual_content']['table_count']}")
        
        # (5) image_count
        if 'visual_content' in paper_stats and 'image_count' in paper_stats['visual_content']:
            features.append(f"image_count: {paper_stats['visual_content']['image_count']}")
        
        # (6) equation_count
        if 'visual_content' in paper_stats and 'equation_count' in paper_stats['visual_content']:
            features.append(f"equation_count: {paper_stats['visual_content']['equation_count']}")
        
        # (7) avg_caption_length
        if 'visual_densities' in paper_stats and 'avg_caption_length' in paper_stats['visual_densities']:
            features.append(f"avg_caption_length: {paper_stats['visual_densities']['avg_caption_length']}")
    
    if features:
        return f"\n<statistics>\n{'; '.join(features)}\n</statistics>"
    else:
        return ""


def load_statistics(statistics_file: str) -> Dict:
    """Load statistics from JSON file"""
    if not statistics_file or not os.path.exists(statistics_file):
        return {}
    
    with open(statistics_file, 'r', encoding='utf-8') as f:
        stats_data = json.load(f)
    
    # Return the papers dictionary
    return stats_data.get('papers', {})


def extract_paper_content_from_json(json_file_path: str, paper_id: str = None, stats_dict: Dict = None, include_stats: bool = False) -> Dict[str, str]:
    """
    Extract paper content from a JSON file (similar to dataset_builder_new.py).
    Returns dict with 'intro' and 'abstract' keys for compatibility.
    
    Args:
        json_file_path: Path to the JSON file
        paper_id: Paper ID for statistics lookup
        stats_dict: Dictionary of statistics
        include_stats: Whether to append statistics to the text
    """
    # Load JSON data from file
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            paper_data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON from {json_file_path}: {e}")
        raise ValueError(f"Failed to load JSON: {e}")
    
    content_parts = []
    abstract_parts = []
    intro_parts = []
    
    # Find indices of key sections
    title_idx = None
    abstract_idx = None
    intro_idx = None
    acknowledgments_idx = None
    references_idx = None
    
    for i, entry in enumerate(paper_data):
        # Check for level 1 headers
        if entry.get("type") == "text" and entry.get("text_level") == 1:
            text = entry.get("text", "").strip()
            
            # First text_level=1 is the title
            if title_idx is None:
                title_idx = i
            # Look for abstract
            elif "ABSTRACT" in text.upper():
                abstract_idx = i
            # Look for introduction
            elif "INTRODUCTION" in text.upper():
                intro_idx = i
            # Look for acknowledgments (before references)
            elif acknowledgments_idx is None and ("ACKNOWLEDGMENT" in text.upper() or "ACKNOWLEDGEMENT" in text.upper()):
                acknowledgments_idx = i
            # Look for references
            elif "REFERENCES" in text.upper() or "REFERENCE" in text.upper():
                references_idx = i
                break
        # Also check if any regular text block starts with "ACKNOWLEDGMENTS" or "REFERENCES"
        elif entry.get("type") == "text":
            text = entry.get("text", "").strip()
            upper_text = text.upper()
            
            # Check if text starts with "ACKNOWLEDGMENTS" (case-insensitive)
            if acknowledgments_idx is None and (upper_text.startswith("ACKNOWLEDGMENT") or upper_text.startswith("ACKNOWLEDGEMENT")):
                acknowledgments_idx = i
            # Check if text starts with "REFERENCES" (case-insensitive)
            elif references_idx is None and upper_text.startswith("REFERENCES"):
                references_idx = i
                break
    
    # Extract abstract (from abstract header to introduction)
    if abstract_idx is not None and intro_idx is not None:
        for i in range(abstract_idx + 1, intro_idx):  # Skip the header itself
            entry = paper_data[i]
            if entry.get("type") == "text":
                text = entry.get("text", "").strip()
                if text:
                    abstract_parts.append(text)
            elif entry.get("type") == "equation":
                # Include equations in abstract if any
                eq_text = entry.get("text", "").strip()
                if eq_text:
                    abstract_parts.append(eq_text)
    
    # Extract introduction (from intro header to next section)
    if intro_idx is not None:
        # Find next text_level=1 after introduction
        next_section_idx = None
        for i in range(intro_idx + 1, len(paper_data)):
            if paper_data[i].get("type") == "text" and paper_data[i].get("text_level") == 1:
                next_section_idx = i
                break
        
        if next_section_idx is None:
            next_section_idx = acknowledgments_idx if acknowledgments_idx else references_idx
            if next_section_idx is None:
                next_section_idx = len(paper_data)
        
        for i in range(intro_idx + 1, next_section_idx):  # Skip the header itself
            entry = paper_data[i]
            if entry.get("type") == "text":
                text = entry.get("text", "").strip()
                if text:
                    intro_parts.append(text)
            elif entry.get("type") == "equation":
                eq_text = entry.get("text", "").strip()
                if eq_text:
                    intro_parts.append(eq_text)
    
    # Join parts and remove GitHub links
    abstract_text = " ".join(abstract_parts)
    intro_text = " ".join(intro_parts)
    
    abstract_text = _remove_github_links(abstract_text)
    intro_text = _remove_github_links(intro_text)
    
    # Add statistics if requested
    if include_stats and paper_id and stats_dict:
        # Count references in both abstract and intro
        reference_count = count_references(abstract_text + " " + intro_text)
        stats_text = format_statistics(paper_id, stats_dict, reference_count)
        
        # Append statistics to both abstract and intro
        if stats_text:
            abstract_text += stats_text
            intro_text += stats_text
    
    if not abstract_text:
        raise ValueError("No ABSTRACT found")
    if not intro_text:
        raise ValueError("No INTRODUCTION found")
    
    return {"intro": intro_text, "abstract": abstract_text}


def two_fig_paths(
    img_dir: str, min_required: int = 2, max_size: int = 800
) -> List[str]:
    """Pick the first two valid figure images and optionally resize"""
    if not os.path.isdir(img_dir):
        raise FileNotFoundError(f"{img_dir} not found")

    fnames = [
        f for f in os.listdir(img_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    def sort_key(n):
        m = re.findall(r"\d+", n)
        return (0, int(m[0])) if m else (1, n)

    fnames.sort(key=sort_key)
    good = []
    for fn in fnames:
        p = os.path.join(img_dir, fn)
        try:
            with Image.open(p) as im:
                im.verify()

            # ── Optional resize to avoid giant images ───────────────
            with Image.open(p) as im:
                im = im.convert("RGB")
                w, h = im.size
                if max(w, h) > max_size:
                    scale = max_size / max(w, h)
                    new_size = (int(w * scale), int(h * scale))
                    im = im.resize(new_size, resample=Image.LANCZOS)
                    im.save(p)

            good.append(p)
            if len(good) == min_required:
                break
        except (UnidentifiedImageError, OSError):
            continue
    if len(good) < min_required:
        raise FileNotFoundError(f"valid images < {min_required} in {img_dir}")
    return good


def build_dataset(
    paper_root: str,
    labels_json: str,
    mode: str = "full",
    max_samples: int | None = None,
    split_ratio: float = 0.2,
    seed: int = 42,
    statistics_json: str = None,
    include_stats: bool = False,
) -> Dict[str, Dataset]:
    """Return train / test Datasets as a dict"""
    with open(labels_json, "r", encoding="utf-8") as f:
        label_dict = json.load(f)
    
    # Load statistics if provided
    stats_dict = load_statistics(statistics_json) if statistics_json else {}

    # Process subdirectories (each subdirectory is a paper_id)
    paper_dirs = []
    for paper_id in sorted(os.listdir(paper_root)):
        paper_dir_path = os.path.join(paper_root, paper_id)
        if os.path.isdir(paper_dir_path):
            paper_dirs.append((paper_id, paper_dir_path))
    
    if max_samples:
        paper_dirs = paper_dirs[:max_samples]

    samples: List[Dict] = []
    for paper_id, paper_dir in paper_dirs:
        if paper_id not in label_dict:
            continue
        
        try:
            # Look for JSON content file
            json_file = None
            for filename in os.listdir(paper_dir):
                if filename.endswith('_content_list.json'):
                    json_file = os.path.join(paper_dir, filename)
                    break
            
            if not json_file:
                # Fallback to old text file if JSON not found
                txt_files = glob.glob(os.path.join(paper_dir, "*_text.txt"))
                if txt_files:
                    # Use old extraction method as fallback (without stats)
                    from feng import extract_intro_abstract as old_extract
                    sects = old_extract(txt_files[0])
                else:
                    raise ValueError("No content file found")
            else:
                # Use new JSON extraction with optional statistics
                sects = extract_paper_content_from_json(
                    json_file, 
                    paper_id=paper_id,
                    stats_dict=stats_dict,
                    include_stats=include_stats
                )
            
            # Image extraction remains the same
            img_paths = two_fig_paths(os.path.join(paper_dir, "figures"))
        except Exception as e:
            print(f"skip {paper_id}: {e}")
            continue
        
        samples.append(
            {
                "paper_id": paper_id,
                "intro": sects["intro"],
                "abstract": sects["abstract"],
                "img_paths": img_paths,
                "label": get_label_from_status(label_dict[paper_id]),
            }
        )

    if not samples:
        raise RuntimeError("No valid samples found!")

    ds = Dataset.from_list(samples)
    split = ds.train_test_split(test_size=split_ratio, seed=seed, shuffle=True)
    return split


def save_dataset(split: Dict[str, Dataset], output_dir: str) -> None:
    """Save train and test datasets to disk"""
    os.makedirs(output_dir, exist_ok=True)
    split["train"].save_to_disk(os.path.join(output_dir, "train"))
    split["test"].save_to_disk(os.path.join(output_dir, "test"))
    print(
        f"Dataset saved: train={len(split['train'])}, test={len(split['test'])}  ->  {output_dir}"
    )


def get_dataset_stats(dataset: Dataset) -> Dict:
    """Get dataset statistics"""
    import pandas as pd
    
    # Handle empty dataset
    if len(dataset) == 0:
        return {
            'total_samples': 0,
            'label_distribution': {
                'accepted (1)': 0,
                'rejected (0)': 0
            },
            'text_stats': {
                'avg_length_words': 0,
                'min_length_words': 0,
                'max_length_words': 0
            }
        }
    
    label_counts = pd.Series(dataset['label']).value_counts()
    
    # Calculate text lengths for both intro and abstract
    intro_lengths = [len(text.split()) for text in dataset['intro']]
    abstract_lengths = [len(text.split()) for text in dataset['abstract']]
    combined_lengths = [i + a for i, a in zip(intro_lengths, abstract_lengths)]
    
    stats = {
        'total_samples': len(dataset),
        'label_distribution': {
            'accepted (1)': label_counts.get(1, 0),
            'rejected (0)': label_counts.get(0, 0)
        },
        'text_stats': {
            'intro': {
                'avg_length_words': sum(intro_lengths) / len(intro_lengths) if intro_lengths else 0,
                'min_length_words': min(intro_lengths) if intro_lengths else 0,
                'max_length_words': max(intro_lengths) if intro_lengths else 0
            },
            'abstract': {
                'avg_length_words': sum(abstract_lengths) / len(abstract_lengths) if abstract_lengths else 0,
                'min_length_words': min(abstract_lengths) if abstract_lengths else 0,
                'max_length_words': max(abstract_lengths) if abstract_lengths else 0
            },
            'combined': {
                'avg_length_words': sum(combined_lengths) / len(combined_lengths) if combined_lengths else 0,
                'min_length_words': min(combined_lengths) if combined_lengths else 0,
                'max_length_words': max(combined_lengths) if combined_lengths else 0
            }
        }
    }
    
    return stats


def parse_args():
    parser = argparse.ArgumentParser(description="ICLR 2025 dataset builder")
    parser.add_argument(
        "--mode",
        default="full",
        choices=["full", "image_only", "text_only"],
        help="Input mode (unused during building but kept for compatibility)",
    )
    parser.add_argument("--paper_root", required=True, help="Root folder of papers")
    parser.add_argument("--labels_json", required=True, help="Path to labels.json")
    parser.add_argument("--output_dir", required=True, help="Where to save dataset")
    parser.add_argument(
        "--max_samples", type=int, default=None, help="Debug: limit number of samples"
    )
    parser.add_argument(
        "--split_ratio",
        type=float,
        default=0.2,
        help="Fraction of data used for the test split",
    )
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed")
    parser.add_argument(
        "--statistics_json",
        type=str,
        default=None,
        help="Path to statistics JSON file"
    )
    parser.add_argument(
        "--include_stats",
        action="store_true",
        help="Include statistics in the text output"
    )
    parser.add_argument(
        "--print_stats",
        action="store_true",
        help="Print dataset statistics after building"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print("Building dataset …")
    split = build_dataset(
        paper_root=args.paper_root,
        labels_json=args.labels_json,
        mode=args.mode,
        max_samples=args.max_samples,
        split_ratio=args.split_ratio,
        seed=args.seed,
        statistics_json=args.statistics_json,
        include_stats=args.include_stats,
    )
    
    # Print statistics if requested
    if args.print_stats:
        print("\n=== Dataset Statistics ===")
        print("Train set:")
        train_stats = get_dataset_stats(split["train"])
        print(json.dumps(train_stats, indent=2))
        print("\nTest set:")
        test_stats = get_dataset_stats(split["test"])
        print(json.dumps(test_stats, indent=2))
        print("=========================\n")
    
    save_dataset(split, args.output_dir)
    print("All done.")


if __name__ == "__main__":
    main()