import os
import hashlib
from typing import Optional, Dict, Any
from datasets import Dataset, load_from_disk

class DatasetCache:
    """Handles caching and loading of processed datasets."""
    
    def __init__(self, cache_dir: str = "/mnt/parscratch/users/acr24wz/etu/topcon/processed_dataset"):
        self.cache_dir = cache_dir
        
    def get_cache_path(self, 
                      data_path: str, 
                      labels_file: str,
                      statistics_file: Optional[str] = None,
                      max_length: int = 10000) -> str:
        """Generate a unique cache path based on dataset configuration."""
        # Create a unique identifier based on dataset parameters
        cache_key = f"{data_path}_{max_length}"
        cache_suffix = hashlib.md5(cache_key.encode()).hexdigest()[:8]
        
        cache_path = os.path.join(self.cache_dir, cache_suffix)
        os.makedirs(cache_path, exist_ok=True)
        return cache_path
    
    def load_cached_dataset(self, cache_path: str) -> Optional[Dataset]:
        """Load dataset from cache if it exists."""
        dataset_info_path = os.path.join(cache_path, "dataset_info.json")
        
        if os.path.exists(dataset_info_path) and os.path.isdir(cache_path):
            try:
                print(f"Loading cached processed dataset from {cache_path}...")
                dataset = load_from_disk(cache_path)
                print("Successfully loaded cached dataset!")
                return dataset
            except Exception as e:
                print(f"Failed to load cached dataset: {e}")
                return None
        return None
    
    def save_dataset_to_cache(self, dataset: Dataset, cache_path: str) -> None:
        """Save processed dataset to cache."""
        print(f"Saving processed dataset to {cache_path}")
        dataset.save_to_disk(cache_path)
        print("Dataset saved to cache successfully!")
    
    def load_or_process_dataset(self, 
                               dataset_builder,
                               use_ids: bool = False) -> Dataset:
        """Load dataset from cache or process it if not cached."""
        # Generate cache path based on builder configuration
        cache_path = self.get_cache_path(
            data_path=dataset_builder.data_folder,
            labels_file=dataset_builder.labels_file,
            statistics_file=dataset_builder.statistics_file,
            max_length=dataset_builder.max_length
        )
        
        # Try to load from cache
        dataset = self.load_cached_dataset(cache_path)
        
        if dataset is None:
            # Process dataset from scratch
            print("No cached dataset found or loading failed. Processing dataset from scratch...")
            if use_ids:
                dataset = dataset_builder.load_dataset_with_ids()
            else:
                dataset = dataset_builder.load_dataset()
            
            # Save to cache
            self.save_dataset_to_cache(dataset, cache_path)
        
        return dataset
