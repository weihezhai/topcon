#!/usr/bin/env python3
import sys
from datasets import load_from_disk

def main(cache_dir: str, add_placeholder: bool = False):
    # Load your processed dataset
    ds = load_from_disk(cache_dir)
    
    print("Columns in dataset:", ds.column_names)
    
    if 'paper_id' in ds.column_names:
        print("✅ 'paper_id' column is present.")
    else:
        print("❌ 'paper_id' column is MISSING.")
        if add_placeholder:
            # Add a placeholder column of -1 (or any default) for each example
            placeholder = [-1] * len(ds)
            ds = ds.add_column('paper_id', placeholder)
            ds.save_to_disk(cache_dir)
            print(f"Added placeholder 'paper_id' column and saved back to {cache_dir}")
        else:
            sys.exit(1)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Check for paper_id in a HuggingFace dataset")
    p.add_argument("cache_dir", help="Path to your dataset cache (where you called save_to_disk)")
    p.add_argument(
        "--add-placeholder",
        action="store_true",
        help="If missing, add a dummy paper_id column of -1 and overwrite the cache"
    )
    args = p.parse_args()
    main(args.cache_dir, args.add_placeholder)