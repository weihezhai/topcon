import json
import re
from collections import defaultdict

def load_json_files():
    """Load matched paper IDs and paper labels"""
    with open('/data/scratch/mpx602/topcon-1/conference_data/iclr_2025_data/filtered_theory_papers/matched_paper_ids.json', 'r') as f:
        matched_ids = json.load(f)

    with open('/data/scratch/mpx602/topcon-1/conference_data/iclr_2025_data/filtered_llm_papers/paper_labels.json', 'r') as f:
        paper_labels = json.load(f)
    
    return matched_ids, paper_labels

def filter_papers_by_keywords(matched_ids, paper_labels, keywords):
    """
    Filter papers whose titles contain at least one of the keywords.
    Returns a dictionary mapping each keyword to list of paper IDs.
    """
    keyword_to_papers = defaultdict(list)
    matched_papers = set()  # Track papers that matched at least one keyword
    
    # Process each matched paper
    for paper_id in matched_ids:
        if paper_id not in paper_labels:
            continue
        
        title = paper_labels[paper_id]['title']
        title_lower = title.lower()
        
        # Check each keyword
        for keyword in keywords:
            keyword_lower = keyword.lower()
            # Use word boundary to match whole words
            if re.search(r'\b' + re.escape(keyword_lower) + r'\b', title_lower):
                keyword_to_papers[keyword].append(paper_id)
                matched_papers.add(paper_id)
    
    # Convert defaultdict to regular dict and sort lists
    result = {}
    for keyword in keywords:
        if keyword in keyword_to_papers:
            result[keyword] = sorted(keyword_to_papers[keyword])
        else:
            result[keyword] = []
    
    # Add 'other' category for papers that don't match any keyword
    other_papers = []
    for paper_id in matched_ids:
        if paper_id in paper_labels and paper_id not in matched_papers:
            other_papers.append(paper_id)
    
    result['other'] = sorted(other_papers)
    
    return result

def save_results(keyword_papers, output_file='theory_papers_by_keywords.json'):
    """Save results to JSON file"""
    with open(output_file, 'w') as f:
        json.dump(keyword_papers, f, indent=2)

def print_summary(keyword_papers):
    """Print summary statistics"""
    print("\nKeyword Summary:")
    print("-" * 50)
    print(f"{'Keyword':<20} {'Paper Count':<15}")
    print("-" * 50)
    
    total_papers = set()
    for keyword, papers in keyword_papers.items():
        if keyword != 'other':  # Separate handling for 'other'
            print(f"{keyword:<20} {len(papers):<15}")
            total_papers.update(papers)
    
    # Print 'other' category separately
    if 'other' in keyword_papers:
        print("-" * 50)
        print(f"{'other':<20} {len(keyword_papers['other']):<15}")
        total_papers.update(keyword_papers['other'])
    
    print("-" * 50)
    print(f"\nTotal unique papers: {len(total_papers)}")
    
    # Find papers that match multiple keywords (excluding 'other')
    paper_keyword_count = defaultdict(int)
    for keyword, papers in keyword_papers.items():
        if keyword != 'other':
            for paper in papers:
                paper_keyword_count[paper] += 1
    
    multi_keyword_papers = [p for p, count in paper_keyword_count.items() if count > 1]
    if multi_keyword_papers:
        print(f"Papers matching multiple keywords: {len(multi_keyword_papers)}")

def main():
    # Define candidate keywords (you can modify this list)
    # rl_candidate_keywords = ['policy', 'offline', 'agent', 'agents', 'online', 'reward','world','planning','human']
    # llm_candidate_keywords = ['agent','agents','reasoning','alignment','context','benchmark','retrieval', 'code', 'instruction']
    # cv_candidate_keywords = ['diffusion','video','3d','detection','multimodal','gaussian','segmentation','editing']
    candidate_keywords = ['optimization', 'gradient', 'bayesian','stochastic','convergence','objective','linear','combinatorial']
    # Load data
    print("Loading data...")
    matched_ids, paper_labels = load_json_files()
    print(f"Loaded {len(matched_ids)} matched paper IDs")
    print(f"Loaded {len(paper_labels)} paper labels")
    
    # Filter papers by keywords
    print(f"\nFiltering papers by {len(candidate_keywords)} keywords...")
    keyword_papers = filter_papers_by_keywords(matched_ids, paper_labels, candidate_keywords)
    
    # Save results
    save_results(keyword_papers)
    print(f"\nResults saved to 'papers_by_keywords.json'")
    
    # Print summary
    print_summary(keyword_papers)
    
    # Optional: Print some example papers for each keyword
    print("\n\nExample papers for selected keywords:")
    print("=" * 70)
    
    # Show examples for top 5 keywords by paper count (excluding 'other')
    sorted_keywords = sorted(
        [(k, v) for k, v in keyword_papers.items() if k != 'other'], 
        key=lambda x: len(x[1]), 
        reverse=True
    )[:5]
    
    for keyword, papers in sorted_keywords:
        if papers:
            print(f"\n{keyword.upper()} ({len(papers)} papers):")
            # Show first 3 papers
            for paper_id in papers[:3]:
                if paper_id in paper_labels:
                    print(f"  - {paper_id}: {paper_labels[paper_id]['title']}")
            if len(papers) > 3:
                print(f"  ... and {len(papers) - 3} more")
    
    # Also show some examples from 'other' category
    if 'other' in keyword_papers and keyword_papers['other']:
        other_papers = keyword_papers['other']
        print(f"\nOTHER (papers without any keywords) ({len(other_papers)} papers):")
        for paper_id in other_papers[:3]:
            if paper_id in paper_labels:
                print(f"  - {paper_id}: {paper_labels[paper_id]['title']}")
        if len(other_papers) > 3:
            print(f"  ... and {len(other_papers) - 3} more")

if __name__ == "__main__":
    main()
