import json
import glob
import os
from filter_paper import filter_papers_by_keywords

def analyze_candidate_matches(papers_data, candidates):
    """
    Analyze and show detailed matching information for each candidate.
    
    Args:
        papers_data (list): List of paper dictionaries
        candidates (list): List of candidate strings to match against
    
    Returns:
        dict: Dictionary with matching statistics for each candidate
    """
    candidates_lower = [candidate.lower() for candidate in candidates]
    match_stats = {candidate: {"title_matches": 0, "keyword_matches": 0, "total_matches": 0} for candidate in candidates}
    
    for paper_entry in papers_data:
        paper = paper_entry.get('paper', {})
        content = paper.get('content', {})
        
        title = content.get('title', {}).get('value', '').lower()
        keywords = content.get('keywords', {}).get('value', [])
        keywords_lower = [keyword.lower() for keyword in keywords]
        
        # Track which candidates match this paper to avoid double counting
        paper_matches = set()
        
        # Check matches for each candidate
        for i, candidate in enumerate(candidates_lower):
            # Check title
            title_match = candidate in title
            if title_match:
                match_stats[candidates[i]]["title_matches"] += 1
                paper_matches.add(candidates[i])
            
            # Check keywords (only if not already matched in title)
            if not title_match:
                keyword_match = False
                for keyword in keywords_lower:
                    if candidate in keyword or keyword in candidate:
                        keyword_match = True
                        break
                if keyword_match:
                    match_stats[candidates[i]]["keyword_matches"] += 1
                    paper_matches.add(candidates[i])
        
        # Update total matches (unique papers per candidate)
        for matched_candidate in paper_matches:
            match_stats[matched_candidate]["total_matches"] += 1
    
    return match_stats

def batch_filter_papers(input_dir, output_file, candidates):
    """
    Process multiple JSON files and collect all matched paper IDs.
    
    Args:
        input_dir (str): Directory containing the JSON files
        output_file (str): Output file to save matched paper IDs
        candidates (list): List of candidate strings to match against
    
    Returns:
        list: List of all matched paper IDs
    """
    # Find all JSON files starting with 'iclr_2025_chunk_'
    pattern = os.path.join(input_dir, "iclr_2025_chunk_*.json")
    json_files = glob.glob(pattern)
    
    if not json_files:
        print(f"No files found matching pattern: {pattern}")
        return []
    
    print(f"Found {len(json_files)} files to process:")
    for file in sorted(json_files):
        print(f"  - {os.path.basename(file)}")
    print()
    
    all_matched_ids = []
    total_papers = 0
    total_matched = 0
    file_stats = []
    
    # Initialize candidate statistics
    candidate_stats = {candidate: {"title_matches": 0, "keyword_matches": 0, "total_matches": 0} for candidate in candidates}
    
    for json_file in sorted(json_files):
        print(f"Processing: {os.path.basename(json_file)}")
        
        try:
            # Load papers from current file
            with open(json_file, 'r', encoding='utf-8') as file:
                papers_data = json.load(file)
            
            # Filter papers
            filtered_papers = filter_papers_by_keywords(papers_data, candidates)
            
            # Analyze matches for candidate statistics
            file_candidate_stats = analyze_candidate_matches(papers_data, candidates)
            
            # Aggregate candidate statistics
            for candidate in candidates:
                candidate_stats[candidate]["title_matches"] += file_candidate_stats[candidate]["title_matches"]
                candidate_stats[candidate]["keyword_matches"] += file_candidate_stats[candidate]["keyword_matches"]
                candidate_stats[candidate]["total_matches"] += file_candidate_stats[candidate]["total_matches"]
            
            # Extract paper IDs
            matched_ids = []
            for paper_entry in filtered_papers:
                paper_id = paper_entry.get('paper', {}).get('id', '')
                if paper_id:
                    matched_ids.append(paper_id)
            
            # Update statistics
            file_papers = len(papers_data)
            file_matched = len(matched_ids)
            total_papers += file_papers
            total_matched += file_matched
            
            file_stats.append({
                'file': os.path.basename(json_file),
                'total_papers': file_papers,
                'matched_papers': file_matched,
                'match_rate': (file_matched / file_papers * 100) if file_papers > 0 else 0,
                'candidate_stats': file_candidate_stats
            })
            
            all_matched_ids.extend(matched_ids)
            
            print(f"  - Papers: {file_papers}, Matched: {file_matched} ({file_matched/file_papers*100:.1f}%)")
            
        except Exception as e:
            print(f"  - Error processing {json_file}: {e}")
            file_stats.append({
                'file': os.path.basename(json_file),
                'total_papers': 0,
                'matched_papers': 0,
                'match_rate': 0,
                'error': str(e)
            })
    
    # Save results
    results = {
        'candidates': candidates,
        'total_files_processed': len(json_files),
        'total_papers': total_papers,
        'total_matched': total_matched,
        'overall_match_rate': (total_matched / total_papers * 100) if total_papers > 0 else 0,
        'candidate_statistics': candidate_stats,
        'matched_paper_ids': all_matched_ids,
        'file_statistics': file_stats
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Print summary statistics
    print("\n" + "="*60)
    print("BATCH PROCESSING SUMMARY")
    print("="*60)
    print(f"Files processed: {len(json_files)}")
    print(f"Total papers: {total_papers:,}")
    print(f"Total matched: {total_matched:,}")
    print(f"Overall match rate: {total_matched/total_papers*100:.2f}%")
    print(f"Unique matched paper IDs: {len(set(all_matched_ids)):,}")
    
    print(f"\nCandidates used: {candidates}")
    
    # Print candidate statistics
    print(f"\nCandidate Match Statistics:")
    print(f"{'Candidate':<20} {'Title':<8} {'Keyword':<8} {'Total':<8} {'% of Matched':<12} {'% of All':<10}")
    print("-" * 75)
    for candidate in candidates:
        stats = candidate_stats[candidate]
        match_percentage = (stats["total_matches"] / total_matched * 100) if total_matched > 0 else 0
        all_percentage = (stats["total_matches"] / total_papers * 100) if total_papers > 0 else 0
        print(f"{candidate:<20} {stats['title_matches']:<8} {stats['keyword_matches']:<8} {stats['total_matches']:<8} {match_percentage:<11.1f}% {all_percentage:<9.2f}%")
    
    print(f"\nPer-file statistics:")
    print(f"{'File':<40} {'Papers':<8} {'Matched':<8} {'Rate':<8}")
    print("-" * 70)
    for stat in file_stats:
        if 'error' in stat:
            print(f"{stat['file']:<40} {'ERROR':<8} {'ERROR':<8} {'ERROR':<8}")
        else:
            print(f"{stat['file']:<40} {stat['total_papers']:<8} {stat['matched_papers']:<8} {stat['match_rate']:<7.1f}%")
    
    print(f"\nResults saved to: {output_file}")
    
    return all_matched_ids

if __name__ == "__main__":
    # Configuration
    # llm_candidates = ["LLM", "language model", "RAG", "hallucination", "jailbreaking", "agents", "agent", "agentic"]
    candidates = ["computer vision", "CV", "vision", "image", "video", '3D', 'diffusion', 'gaussian splatting']
    input_directory = "/data/scratch/mpx602/topcon-1/conference_data/iclr_2025_data/"
    output_file = "/data/scratch/mpx602/topcon-1/conference_data/iclr_2025_data/filtered_cv_papers/batch_filtered_results.json"
    
    # Run batch processing
    matched_ids = batch_filter_papers(input_directory, output_file, candidates)
    
    # Also save just the IDs list for convenience
    ids_only_file = "/data/scratch/mpx602/topcon-1/conference_data/iclr_2025_data/filtered_cv_papers/matched_paper_ids.json"
    with open(ids_only_file, 'w', encoding='utf-8') as f:
        json.dump(matched_ids, f, indent=2)
    
    print(f"\nMatched paper IDs also saved to: {ids_only_file}")