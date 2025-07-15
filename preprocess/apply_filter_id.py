import json
import glob
import os
import shutil

def select_txt_files_by_ids(paper_ids_file, txt_source_dir, output_dir):
    """
    Select text files from source directory based on paper IDs and copy to output directory.
    
    Args:
        paper_ids_file (str): Path to JSON file containing paper IDs
        txt_source_dir (str): Directory containing txt files named with paper IDs
        output_dir (str): Directory to copy selected txt files
    
    Returns:
        dict: Statistics about the selection process
    """
    # Load paper IDs
    try:
        with open(paper_ids_file, 'r', encoding='utf-8') as f:
            paper_ids = json.load(f)
        print(f"Loaded {len(paper_ids)} paper IDs from {paper_ids_file}")
    except Exception as e:
        print(f"Error loading paper IDs: {e}")
        return {}
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all txt files in source directory
    txt_pattern = os.path.join(txt_source_dir, "*.txt")
    all_txt_files = glob.glob(txt_pattern)
    print(f"Found {len(all_txt_files)} txt files in {txt_source_dir}")
    
    # Statistics tracking
    stats = {
        'total_paper_ids': len(paper_ids),
        'total_txt_files': len(all_txt_files),
        'matched_files': 0,
        'copied_files': 0,
        'missing_files': [],
        'error_files': [],
        'matched_paper_ids': []
    }
    
    # Create a set of paper IDs for faster lookup
    paper_ids_set = set(paper_ids)
    
    # Process each txt file
    for txt_file in all_txt_files:
        filename = os.path.basename(txt_file)
        
        # Extract paper ID from filename (everything before the first underscore)
        if '_' in filename:
            paper_id = filename.split('_')[0]
            
            # Check if this paper ID is in our filtered list
            if paper_id in paper_ids_set:
                stats['matched_files'] += 1
                stats['matched_paper_ids'].append(paper_id)
                
                # Copy file to output directory
                try:
                    dest_path = os.path.join(output_dir, filename)
                    shutil.copy2(txt_file, dest_path)
                    stats['copied_files'] += 1
                    print(f"Copied: {filename}")
                except Exception as e:
                    print(f"Error copying {filename}: {e}")
                    stats['error_files'].append(filename)
    
    # Check for missing files (paper IDs that don't have corresponding txt files)
    found_paper_ids = set()
    for txt_file in all_txt_files:
        filename = os.path.basename(txt_file)
        if '_' in filename:
            paper_id = filename.split('_')[0]
            found_paper_ids.add(paper_id)
    
    missing_ids = paper_ids_set - found_paper_ids
    stats['missing_files'] = list(missing_ids)
    
    return stats

def print_selection_stats(stats):
    """Print detailed statistics about the file selection process."""
    print("\n" + "="*60)
    print("TXT FILE SELECTION SUMMARY")
    print("="*60)
    print(f"Total paper IDs to match: {stats['total_paper_ids']:,}")
    print(f"Total txt files available: {stats['total_txt_files']:,}")
    print(f"Matched files found: {stats['matched_files']:,}")
    print(f"Files successfully copied: {stats['copied_files']:,}")
    
    if stats['error_files']:
        print(f"\nFiles with copy errors ({len(stats['error_files'])}):")
        for error_file in stats['error_files'][:10]:  # Show first 10
            print(f"  - {error_file}")
        if len(stats['error_files']) > 10:
            print(f"  ... and {len(stats['error_files']) - 10} more")
    
    if stats['missing_files']:
        print(f"\nPaper IDs without corresponding txt files ({len(stats['missing_files'])}):")
        for missing_id in stats['missing_files'][:10]:  # Show first 10
            print(f"  - {missing_id}")
        if len(stats['missing_files']) > 10:
            print(f"  ... and {len(stats['missing_files']) - 10} more")
    
    coverage = (stats['matched_files'] / stats['total_paper_ids'] * 100) if stats['total_paper_ids'] > 0 else 0
    print(f"\nCoverage: {coverage:.1f}% of filtered papers have txt files")
    
    # Show example matched files
    if stats['matched_paper_ids']:
        print(f"\nExample matched paper IDs:")
        for paper_id in stats['matched_paper_ids'][:5]:
            print(f"  - {paper_id}")

if __name__ == "__main__":
    import sys
    
    # Default configuration
    paper_ids_file = "/data/scratch/mpx602/topcon-1/conference_data/iclr_2025_data/filtered_theory_papers/matched_paper_ids.json"
    txt_source_dir = "/data/scratch/mpx602/topcon-1/conference_data/iclr_2025_data/txt_outputs"
    output_dir = "/data/scratch/mpx602/topcon-1/conference_data/iclr_2025_data/filtered_theory_papers/theory_papers_text"

    # Allow command line arguments
    if len(sys.argv) >= 2:
        paper_ids_file = sys.argv[1]
    if len(sys.argv) >= 3:
        txt_source_dir = sys.argv[2]
    if len(sys.argv) >= 4:
        output_dir = sys.argv[3]
    
    print("Starting txt file selection based on filtered paper IDs...")
    print(f"Paper IDs file: {paper_ids_file}")
    print(f"Source txt directory: {txt_source_dir}")
    print(f"Output directory: {output_dir}")
    print("-" * 60)
    
    # Select and copy files
    stats = select_txt_files_by_ids(paper_ids_file, txt_source_dir, output_dir)
    
    if stats:
        # Print statistics
        print_selection_stats(stats)
        
        # Save statistics to file
        stats_file = os.path.join(output_dir, "selection_stats.json")
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        print(f"\nDetailed statistics saved to: {stats_file}")
        
        print(f"\nSelected {stats['copied_files']} txt files copied to: {output_dir}")
    else:
        print("Failed to process txt file selection.")