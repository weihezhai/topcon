import json
import re
from collections import Counter
import string

# Define common stop words
STOP_WORDS = {
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
    'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
    'to', 'was', 'will', 'with', 'via', 'using', 'through', 'over',
    'under', 'into', 'between', 'across', 'during', 'without', 'within',
    'upon', 'towards', 'toward', 'against', 'among', 'throughout',
    'despite', 'concerning', 'regarding', 'beyond', 'behind', 'before',
    'after', 'above', 'below', 'around', 'about', 'than', 'vs', 'versus'
}

def load_json_files():
    """Load matched paper IDs and paper labels"""
    with open('/data/scratch/mpx602/topcon-1/conference_data/iclr_2025_data/filtered_theory_papers/matched_paper_ids.json', 'r') as f:
        matched_ids = json.load(f)
    
    with open('/data/scratch/mpx602/topcon-1/conference_data/iclr_2025_data/filtered_llm_papers/paper_labels.json', 'r') as f:
        paper_labels = json.load(f)
    
    return matched_ids, paper_labels

def extract_words_from_title(title):
    """Extract words from title, converting to lowercase and removing punctuation"""
    # Convert to lowercase
    title = title.lower()
    
    # Replace hyphens with spaces to split hyphenated words
    title = title.replace('-', ' ')
    
    # Remove punctuation except for numbers and letters
    title = re.sub(r'[^\w\s]', ' ', title)
    
    # Split into words
    words = title.split()
    
    # Filter out stop words and single characters
    words = [word for word in words if word not in STOP_WORDS and len(word) > 1]
    
    return words

def count_word_frequencies(matched_ids, paper_labels):
    """Count word frequencies in titles of matched papers"""
    word_counter = Counter()
    
    # Track missing IDs
    missing_ids = []
    
    for paper_id in matched_ids:
        if paper_id in paper_labels:
            title = paper_labels[paper_id]['title']
            words = extract_words_from_title(title)
            word_counter.update(words)
        else:
            missing_ids.append(paper_id)
    
    return word_counter, missing_ids

def save_results(word_counter, output_file='title_word_frequencies.json'):
    """Save word frequencies to a JSON file"""
    # Convert to sorted list of tuples
    sorted_frequencies = sorted(word_counter.items(), key=lambda x: x[1], reverse=True)
    
    # Convert to dictionary format
    freq_dict = {
        'total_unique_words': len(word_counter),
        'word_frequencies': [
            {'word': word, 'count': count} for word, count in sorted_frequencies
        ]
    }
    
    with open(output_file, 'w') as f:
        json.dump(freq_dict, f, indent=2)
    
    return sorted_frequencies

def print_top_words(sorted_frequencies, n=50):
    """Print top N most frequent words"""
    print(f"\nTop {n} most frequent words in matched paper titles:")
    print("-" * 50)
    print(f"{'Rank':<6} {'Word':<25} {'Count':<10}")
    print("-" * 50)
    
    for i, (word, count) in enumerate(sorted_frequencies[:n], 1):
        print(f"{i:<6} {word:<25} {count:<10}")

def main():
    # Load data
    print("Loading data...")
    matched_ids, paper_labels = load_json_files()
    print(f"Loaded {len(matched_ids)} matched paper IDs")
    print(f"Loaded {len(paper_labels)} paper labels")
    
    # Count word frequencies
    print("\nCounting word frequencies...")
    word_counter, missing_ids = count_word_frequencies(matched_ids, paper_labels)
    
    if missing_ids:
        print(f"\nWarning: {len(missing_ids)} paper IDs not found in labels:")
        for id in missing_ids[:10]:  # Show first 10
            print(f"  - {id}")
        if len(missing_ids) > 10:
            print(f"  ... and {len(missing_ids) - 10} more")
    
    # Save results
    sorted_frequencies = save_results(word_counter)
    print(f"\nResults saved to 'title_word_frequencies.json'")
    
    # Print statistics
    print(f"\nStatistics:")
    print(f"Total papers analyzed: {len(matched_ids) - len(missing_ids)}")
    print(f"Total unique words: {len(word_counter)}")
    print(f"Total word occurrences: {sum(word_counter.values())}")
    
    # Print top words
    print_top_words(sorted_frequencies)
    
    # Additional analysis
    print("\n\nWord frequency distribution:")
    print("-" * 30)
    freq_distribution = Counter(word_counter.values())
    for freq in sorted(freq_distribution.keys(), reverse=True)[:10]:
        print(f"Words appearing {freq} times: {freq_distribution[freq]}")

if __name__ == "__main__":
    main()
