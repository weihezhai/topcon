import os
import shutil
from pathlib import Path

def merge_folders():
    # Define source folders
    source_folders = ['balanced_cv', 'balanced_llm', 'balanced_rl', 'balanced_theory']
    target_folder = 'all'
    
    # Create target folder if it doesn't exist
    os.makedirs(target_folder, exist_ok=True)
    
    copied_count = 0
    skipped_count = 0
    
    for source in source_folders:
        if not os.path.exists(source):
            print(f"Warning: {source} does not exist, skipping...")
            continue
        
        print(f"Processing {source}...")
        
        # Iterate through subfolders in source
        for subfolder in os.listdir(source):
            source_path = os.path.join(source, subfolder)
            
            # Skip if not a directory
            if not os.path.isdir(source_path):
                continue
            
            target_path = os.path.join(target_folder, subfolder)
            
            # Check if subfolder already exists
            if os.path.exists(target_path):
                print(f"  Skipped: {subfolder} (already exists)")
                skipped_count += 1
                continue
            
            # Copy the subfolder
            try:
                shutil.copytree(source_path, target_path)
                print(f"  Copied: {source}/{subfolder} -> {target_folder}/{subfolder}")
                copied_count += 1
            except Exception as e:
                print(f"  Error copying {source_path}: {e}")
    
    print(f"\nMerge complete!")
    print(f"Copied: {copied_count} subfolders")
    print(f"Skipped: {skipped_count} subfolders (already existed)")

if __name__ == "__main__":
    merge_folders()