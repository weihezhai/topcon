import sys
import os

# Add the current directory to Python path if needed
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset_builder import TextDatasetBuilder

def debug_filename_and_labels():
    """Debug the filename and label matching issue"""
    
    DATA_FOLDER = "/data/scratch/mpx602/topcon-1/conference_data/iclr_2025_data/filtered_llm_papers/llm_papers_text"
    LABELS_FILE = "/data/scratch/mpx602/topcon-1/conference_data/iclr_2025_data/filtered_llm_papers/label_simple.json"
    
    dataset_builder = TextDatasetBuilder(DATA_FOLDER, LABELS_FILE)
    
    print("🔍 Investigating filename and label format...")
    
    # Check labels file
    labels_dict = dataset_builder.load_labels()
    print(f"\n📊 Labels file contains {len(labels_dict)} entries")
    
    # Show first few labels with their exact format
    print("\n📝 Sample labels (showing exact strings):")
    for i, (paper_id, status) in enumerate(list(labels_dict.items())[:10]):
        print(f"  {i+1}. Key: '{paper_id}' (len={len(paper_id)})")
        print(f"     Value: '{status}'")
        print()
    
    # Check text files
    txt_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith('.txt')]
    print(f"📁 Found {len(txt_files)} .txt files")
    
    print("\n📝 Sample filenames and extracted IDs:")
    for i, filename in enumerate(txt_files[:10]):
        paper_id = dataset_builder.extract_paper_id(filename)
        print(f"  {i+1}. Filename: '{filename}'")
        print(f"     Extracted ID: '{paper_id}' (len={len(paper_id) if paper_id else 0})")
        print(f"     In labels: {paper_id in labels_dict if paper_id else False}")
        print()
    
    # Try different extraction methods
    print("🔬 Trying different ID extraction methods on first filename:")
    if txt_files:
        test_filename = txt_files[0]
        print(f"Test filename: '{test_filename}'")
        
        # Method 1: Remove _sections.txt
        if test_filename.endswith('_sections.txt'):
            id1 = test_filename[:-12]
            print(f"  Method 1 (remove '_sections.txt'): '{id1}' -> In labels: {id1 in labels_dict}")
        
        # Method 2: Split by underscore, take first part
        if '_' in test_filename:
            id2 = test_filename.split('_')[0]
            print(f"  Method 2 (split by '_', take first): '{id2}' -> In labels: {id2 in labels_dict}")
        
        # Method 3: Remove .txt
        id3 = test_filename[:-4] if test_filename.endswith('.txt') else test_filename
        print(f"  Method 3 (remove '.txt'): '{id3}' -> In labels: {id3 in labels_dict}")
        
        # Method 4: Check if filename itself is in labels
        print(f"  Method 4 (full filename): '{test_filename}' -> In labels: {test_filename in labels_dict}")

if __name__ == "__main__":
    debug_filename_and_labels()