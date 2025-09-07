import random
from datasets import load_from_disk
import sys

def print_random_samples(dataset_path, num_samples=20, output_file=None):
    """Load cached dataset and print random samples"""
    
    # Optionally redirect output to file for full text
    if output_file:
        original_stdout = sys.stdout
        sys.stdout = open(output_file, 'w', encoding='utf-8')
    
    # Load the cached dataset
    print(f"Loading dataset from: {dataset_path}")
    dataset = load_from_disk(dataset_path)
    
    print(f"\nDataset info:")
    print(f"Total samples: {len(dataset)}")
    print(f"Features: {dataset.features}")
    print("\n" + "="*80 + "\n")
    
    # Get random indices
    num_samples = min(num_samples, len(dataset))  # Don't exceed dataset size
    random_indices = random.sample(range(len(dataset)), num_samples)
    
    # Print random samples
    for i, idx in enumerate(random_indices, 1):
        sample = dataset[idx]
        
        print(f"SAMPLE {i} (Index: {idx})")
        print("-" * 40)
        
        # Print label
        label_text = "ACCEPTED" if sample['labels'] == 1 else "REJECTED"
        print(f"Label: {sample['labels']} ({label_text})")
        
        # Print paper ID if available
        if 'paper_id' in sample:
            print(f"Paper ID: {sample['paper_id']}")
        
        # Print text - FULL TEXT, NO TRUNCATION
        text = sample['text']
        word_count = len(text.split())
        char_count = len(text)
        print(f"Word count: {word_count}")
        print(f"Character count: {char_count}")
        
        # Print FULL text
        print(f"\nFull Text:")
        print("-" * 20)
        print(text)  # Print entire text without truncation
        print("-" * 20)
        
        print("\n" + "="*80 + "\n")
    
    if output_file:
        sys.stdout.close()
        sys.stdout = original_stdout
        print(f"Full output saved to: {output_file}")

if __name__ == "__main__":
    # Update this path to your cached dataset location
    dataset_path = "/mnt/parscratch/users/acr24wz/etu/topcon/processed_dataset/llm_mineru_all_with_stats"
    
    # Option 1: Print to terminal (might be cut off by terminal buffer)
    print_random_samples(dataset_path, num_samples=20)
    
    # Option 2: Save to file to see complete text (uncomment to use)
    # print_random_samples(dataset_path, num_samples=20, output_file="sample_output.txt")