import random
from datasets import load_from_disk

def print_random_samples(dataset_path, num_samples=20):
    """Load cached dataset and print random samples"""
    
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
        
        # Print text (truncated for readability)
        text = sample['text']
        word_count = len(text.split())
        print(f"Word count: {word_count}")
        
        # Show first 500 characters of text
        if len(text) > 500:
            print(f"Text (first 500 chars):\n{text[:500]}...")
        else:
            print(f"Text:\n{text}")
        
        print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    # Update this path to your cached dataset location
    dataset_path = "/mnt/parscratch/users/acr24wz/etu/topcon/processed_dataset/llm"  # or wherever your dataset is saved
    
    # Print 20 random samples
    print_random_samples(dataset_path, num_samples=20)