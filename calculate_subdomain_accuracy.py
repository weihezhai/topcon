import json
from collections import defaultdict
from typing import Dict, List, Tuple

def load_json(filepath: str) -> dict:
    """Load JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)

def calculate_subdomain_accuracy(
    predictions_path: str,
    index_to_paper_id_path: str,
    subdomain_paper_ids_path: str
) -> Dict[str, Dict[str, float]]:
    """
    Calculate accuracy for each subdomain.
    
    Args:
        predictions_path: Path to predictions JSON file
        index_to_paper_id_path: Path to index->paper_id mapping JSON
        subdomain_paper_ids_path: Path to subdomain->paper_ids mapping JSON
    
    Returns:
        Dictionary with subdomain accuracies and statistics
    """
    # Load data
    predictions = load_json(predictions_path)
    index_to_paper_id = load_json(index_to_paper_id_path)
    subdomain_paper_ids = load_json(subdomain_paper_ids_path)
    
    # Create paper_id to subdomain mapping
    paper_id_to_subdomains = defaultdict(list)
    for subdomain, paper_ids in subdomain_paper_ids.items():
        for paper_id in paper_ids:
            paper_id_to_subdomains[paper_id].append(subdomain)
    
    # Group predictions by subdomain
    subdomain_predictions = defaultdict(list)
    unmapped_count = 0
    
    for index_key, pred_data in predictions.items():
        # Extract index number from key (e.g., "index0" -> "0")
        index = index_key.replace("index", "")
        
        # Get paper_id for this index
        if index in index_to_paper_id:
            paper_id = index_to_paper_id[index]
            
            # Find subdomains for this paper_id
            if paper_id in paper_id_to_subdomains:
                for subdomain in paper_id_to_subdomains[paper_id]:
                    subdomain_predictions[subdomain].append(pred_data)
            else:
                unmapped_count += 1
        else:
            print(f"Warning: Index {index} not found in index_to_paper_id mapping")
    
    # Calculate accuracy for each subdomain
    subdomain_accuracies = {}
    
    for subdomain, preds in subdomain_predictions.items():
        correct = sum(1 for p in preds if p['correctness'])
        total = len(preds)
        accuracy = correct / total if total > 0 else 0.0
        
        # Calculate additional statistics
        label_0_correct = sum(1 for p in preds if p['label'] == 0 and p['correctness'])
        label_0_total = sum(1 for p in preds if p['label'] == 0)
        label_1_correct = sum(1 for p in preds if p['label'] == 1 and p['correctness'])
        label_1_total = sum(1 for p in preds if p['label'] == 1)
        
        subdomain_accuracies[subdomain] = {
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'label_0_accuracy': label_0_correct / label_0_total if label_0_total > 0 else 0.0,
            'label_0_total': label_0_total,
            'label_1_accuracy': label_1_correct / label_1_total if label_1_total > 0 else 0.0,
            'label_1_total': label_1_total
        }
    
    # Add overall statistics
    all_predictions = []
    for preds in subdomain_predictions.values():
        all_predictions.extend(preds)
    
    # Remove duplicates (papers can belong to multiple subdomains)
    unique_predictions = {}
    for index_key, pred_data in predictions.items():
        unique_predictions[index_key] = pred_data
    
    overall_correct = sum(1 for p in unique_predictions.values() if p['correctness'])
    overall_total = len(unique_predictions)
    
    subdomain_accuracies['_overall'] = {
        'accuracy': overall_correct / overall_total if overall_total > 0 else 0.0,
        'correct': overall_correct,
        'total': overall_total,
        'unmapped_papers': unmapped_count
    }
    
    return subdomain_accuracies

def print_results(results: Dict[str, Dict[str, float]]):
    """Pretty print the results"""
    print("\n" + "="*60)
    print("SUBDOMAIN ACCURACY RESULTS")
    print("="*60)
    
    # Print overall statistics first
    if '_overall' in results:
        overall = results['_overall']
        print(f"\nOVERALL STATISTICS:")
        print(f"  Accuracy: {overall['accuracy']:.4f} ({overall['correct']}/{overall['total']})")
        print(f"  Unmapped papers: {overall.get('unmapped_papers', 0)}")
    
    print("\n" + "-"*60)
    print("SUBDOMAIN RESULTS:")
    print("-"*60)
    
    # Sort subdomains by name (excluding _overall)
    sorted_subdomains = sorted([k for k in results.keys() if k != '_overall'])
    
    for subdomain in sorted_subdomains:
        stats = results[subdomain]
        print(f"\n{subdomain.upper()}:")
        print(f"  Overall Accuracy: {stats['accuracy']:.4f} ({stats['correct']}/{stats['total']})")
        print(f"  Label 0 Accuracy: {stats['label_0_accuracy']:.4f} ({stats['label_0_total']} samples)")
        print(f"  Label 1 Accuracy: {stats['label_1_accuracy']:.4f} ({stats['label_1_total']} samples)")

def save_results(results: Dict[str, Dict[str, float]], output_path: str):
    """Save results to JSON file"""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

if __name__ == "__main__":
    # Example usage
    predictions_path = "/mnt/parscratch/users/acr24wz/etu/topcon/qwen3_1d7B/finetuned_model/theory/individual_predictions.json"  # Update with your actual path
    index_to_paper_id_path = "/mnt/parscratch/users/acr24wz/etu/topcon/processed_dataset/theory/theory_index_to_paper_id_mapping.json"  # Update with your actual path
    subdomain_paper_ids_path = "/mnt/parscratch/users/acr24wz/topcon/subdomain_keywords/theory_papers_by_keywords.json"

    # Calculate accuracies
    results = calculate_subdomain_accuracy(
        predictions_path,
        index_to_paper_id_path,
        subdomain_paper_ids_path
    )
    
    # Print results
    print_results(results)
    
    # Save results to file
    save_results(results, "rl_1.7_subdomain_accuracy_results.json")
