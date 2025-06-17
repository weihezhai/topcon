import json
import re

def filter_papers_by_keywords(papers_data, candidates):
    """
    Filter papers based on string overlap with candidate keywords.
    
    Args:
        papers_data (list): List of paper dictionaries in the specified format
        candidates (list): List of candidate strings to match against
    
    Returns:
        list: Filtered list of papers that match at least one candidate
    """
    # Convert candidates to lowercase for case-insensitive matching
    candidates_lower = [candidate.lower() for candidate in candidates]
    
    selected_papers = []
    
    for paper_entry in papers_data:
        # Extract paper content
        paper = paper_entry.get('paper', {})
        content = paper.get('content', {})
        
        # Get title and keywords
        title = content.get('title', {}).get('value', '').lower()
        keywords = content.get('keywords', {}).get('value', [])
        
        # Convert keywords to lowercase
        keywords_lower = [keyword.lower() for keyword in keywords]
        
        # Check for overlap with any candidate
        match_found = False
        
        # Check title for overlap
        for candidate in candidates_lower:
            if candidate in title:
                match_found = True
                break
        
        # If no match in title, check keywords
        if not match_found:
            for keyword in keywords_lower:
                for candidate in candidates_lower:
                    if candidate in keyword or keyword in candidate:
                        match_found = True
                        break
                if match_found:
                    break
        
        # Add paper if match found
        if match_found:
            selected_papers.append(paper_entry)
    
    return selected_papers

def load_and_filter_papers(file_path, candidates):
    """
    Load papers from JSON file and filter based on candidates.
    
    Args:
        file_path (str): Path to the JSON file containing papers
        candidates (list): List of candidate strings to match against
    
    Returns:
        list: Filtered list of papers
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            papers_data = json.load(file)
        
        filtered_papers = filter_papers_by_keywords(papers_data, candidates)
        return filtered_papers
    
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return []
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {file_path}")
        return []

def save_filtered_papers(filtered_papers, output_path):
    """
    Save filtered papers to a new JSON file.
    
    Args:
        filtered_papers (list): List of filtered papers
        output_path (str): Path to save the filtered papers
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as file:
            json.dump(filtered_papers, file, indent=2, ensure_ascii=False)
        print(f"Filtered papers saved to {output_path}")
        print(f"Total papers selected: {len(filtered_papers)}")
    except Exception as e:
        print(f"Error saving file: {e}")

# Main execution
if __name__ == "__main__":
    # Define candidates
    candidates = ["LLM", "language model", "RAG", "hallucination", "jailbreaking", "agents", "agent", "agentic"]
    
    # Input and output file paths
    input_file = "/data/scratch/mpx602/topcon-1/conference_data/iclr_2025_data/iclr_2025_chunk_000_20250609_004723.json"
    output_file = "/data/scratch/mpx602/topcon-1/filtered_papers.json"
    
    # Load and filter papers
    print(f"Loading papers from: {input_file}")
    print(f"Filtering with candidates: {candidates}")
    
    filtered_papers = load_and_filter_papers(input_file, candidates)
    
    if filtered_papers:
        # Print some statistics
        print(f"\nFound {len(filtered_papers)} matching papers")
        
        # Show titles of first few matches for verification
        print("\nFirst few matching papers:")
        for i, paper in enumerate(filtered_papers[:5]):
            title = paper['paper']['content']['title']['value']
            keywords = paper['paper']['content']['keywords']['value']
            print(f"{i+1}. {title}")
            print(f"   Keywords: {keywords}")
            print()
        
        # Save filtered papers
        save_filtered_papers(filtered_papers, output_file)
    else:
        print("No matching papers found.")

# Additional utility function for more detailed matching info
def analyze_matches(papers_data, candidates):
    """
    Analyze and show detailed matching information.
    
    Args:
        papers_data (list): List of paper dictionaries
        candidates (list): List of candidate strings to match against
    
    Returns:
        dict: Dictionary with matching statistics
    """
    candidates_lower = [candidate.lower() for candidate in candidates]
    match_stats = {candidate: {"title_matches": 0, "keyword_matches": 0} for candidate in candidates}
    
    for paper_entry in papers_data:
        paper = paper_entry.get('paper', {})
        content = paper.get('content', {})
        
        title = content.get('title', {}).get('value', '').lower()
        keywords = content.get('keywords', {}).get('value', [])
        keywords_lower = [keyword.lower() for keyword in keywords]
        
        # Check matches for each candidate
        for i, candidate in enumerate(candidates_lower):
            # Check title
            if candidate in title:
                match_stats[candidates[i]]["title_matches"] += 1
            
            # Check keywords
            for keyword in keywords_lower:
                if candidate in keyword or keyword in candidate:
                    match_stats[candidates[i]]["keyword_matches"] += 1
                    break
    
    return match_stats