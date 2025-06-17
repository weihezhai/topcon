import os
import shutil
from pathlib import Path

def move_txt_files_from_subfolders(base_dir):
    """Move all .txt files from subfolders to the main directory"""
    base_path = Path(base_dir)
    
    if not base_path.exists():
        print(f"Directory {base_dir} does not exist")
        return
    
    moved_count = 0
    
    # Iterate through all subdirectories
    for subfolder in base_path.iterdir():
        if subfolder.is_dir():
            # Look for .txt files in this subfolder
            txt_files = list(subfolder.glob("*.txt"))
            
            for txt_file in txt_files:
                # Create new path in the main directory
                new_path = base_path / txt_file.name
                
                # Handle name conflicts by adding a suffix
                counter = 1
                original_name = txt_file.stem
                extension = txt_file.suffix
                
                while new_path.exists():
                    new_name = f"{original_name}_{counter}{extension}"
                    new_path = base_path / new_name
                    counter += 1
                
                # Move the file
                try:
                    shutil.move(str(txt_file), str(new_path))
                    print(f"Moved: {txt_file.name} -> {new_path.name}")
                    moved_count += 1
                except Exception as e:
                    print(f"Error moving {txt_file}: {e}")
            
            # Remove empty subfolder
            try:
                if not any(subfolder.iterdir()):  # Check if folder is empty
                    subfolder.rmdir()
                    print(f"Removed empty folder: {subfolder.name}")
            except Exception as e:
                print(f"Error removing folder {subfolder.name}: {e}")
    
    print(f"\nTotal files moved: {moved_count}")

# Run the script
if __name__ == "__main__":
    txt_outputs_dir = "/data/scratch/mpx602/topcon-1/conference_data/iclr_2025_data/txt_outputs"
    move_txt_files_from_subfolders(txt_outputs_dir)