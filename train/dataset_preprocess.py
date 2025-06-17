import json
import os
import glob

def extract_paper_venues(data_folder="."):
    """
    Extract paper venue information from JSON files starting with 'iclr_2025_chunk_'
    
    Args:
        data_folder: Path to folder containing JSON files
        
    Returns:
        dict: Dictionary with paper_id as key and venue info as value
    """
    venue_dict = {}
    
    # Find all JSON files starting with 'iclr_2025_chunk_'
    pattern = os.path.join(data_folder, "iclr_2025_chunk_*.json")
    json_files = glob.glob(pattern)
    
    print(f"Found {len(json_files)} JSON files to process")
    
    for file_path in json_files:
        print(f"Processing: {os.path.basename(file_path)}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Process each paper in the JSON file
            for item in data:
                if 'paper' in item and 'id' in item['paper']:
                    paper_id = item['paper']['id']
                    
                    # Extract venue information
                    venue_info = {}
                    if 'content' in item['paper']:
                        content = item['paper']['content']
                        
                        # Get venue and venueid if they exist
                        if 'venue' in content and 'value' in content['venue']:
                            venue_info['venue'] = content['venue']['value']
                        
                        # Also include other useful metadata
                        if 'title' in content and 'value' in content['title']:
                            venue_info['title'] = content['title']['value']
                    
                    # Only add if we have some venue information
                    if venue_info:
                        venue_dict[paper_id] = venue_info
                        
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON file {file_path}: {e}")
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
    
    return venue_dict

def save_venue_dict(venue_dict, output_file="paper_venues.json"):
    """Save the venue dictionary to a JSON file"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(venue_dict, f, indent=2, ensure_ascii=False)
    print(f"Venue dictionary saved to {output_file}")

def main():
    # Extract venue information from all matching JSON files
    input_dir = '/data/scratch/mpx602/topcon-1/conference_data/iclr_2025_data/'
    venue_dict = extract_paper_venues(input_dir)  # Current directory, change as needed
    
    print(f"\nExtracted venue information for {len(venue_dict)} papers")
    
    # Show sample entries
    if venue_dict:
        print("\nSample entries:")
        for i, (paper_id, venue_info) in enumerate(venue_dict.items()):
            if i >= 3:  # Show first 3 entries
                break
            print(f"Paper ID: {paper_id}")
            print(f"Venue Info: {venue_info}")
            print("-" * 50)
    
    # Save to file
    save_venue_dict(venue_dict)
    
    # Optional: Save a simplified version with just venue names
    simplified_dict = {}
    for paper_id, venue_info in venue_dict.items():
        if 'venue' in venue_info:
            simplified_dict[paper_id] = venue_info['venue']
    
    save_venue_dict(simplified_dict, "paper_venues_simple.json")
    print(f"Simplified venue dictionary saved with {len(simplified_dict)} entries")

if __name__ == "__main__":
    main()