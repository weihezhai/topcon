import os
import json
from datasets import load_from_disk
from dataset_builder_abs_intro import TextDatasetBuilder

class DatasetIDRecovery:
    def __init__(self, data_folder, labels_file, dataset_path):
        self.builder = TextDatasetBuilder(data_folder, labels_file)
        self.dataset_path = dataset_path
        
    def load_cached_dataset(self):
        """Load the processed dataset from cache"""
        return load_from_disk(self.dataset_path)
    
    def extract_paper_id_from_filename(self, filename):
        """Extract paper ID from filename (format: 0A6f1b66pE_sections.txt)"""
        if filename.endswith('_sections.txt'):
            return filename.replace('_sections.txt', '')
        return None
    
    def recover_ids(self):
        """Recover IDs by reprocessing files in the same order"""
        dataset = self.load_cached_dataset()
        labels_dict = self.builder.load_labels()
        
        # Store ID-index mapping
        id_mapping = []
        dataset_idx = 0
        unmatched_files = []
        
        # Process files in alphabetical order (same as original)
        for filename in sorted(os.listdir(self.builder.data_folder)):
            if filename.endswith('.txt'):
                # Use our custom extraction method for _sections.txt files
                paper_id = self.extract_paper_id_from_filename(filename)
                if not paper_id:
                    # Fallback to original method
                    paper_id = self.builder.extract_paper_id(filename)
                
                if paper_id and paper_id in labels_dict:
                    filepath = os.path.join(self.builder.data_folder, filename)
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            text = f.read().strip()
                            if text:
                                # Apply same preprocessing
                                text = self.builder.clean_text(text)
                                text = self.builder.abs_intro(text)
                                
                                # Check if this matches the dataset entry
                                if dataset_idx < len(dataset):
                                    # Compare first 100 chars for quick match
                                    if text[:100] == dataset['text'][dataset_idx][:100]:
                                        id_mapping.append({
                                            'index': dataset_idx,
                                            'paper_id': paper_id,
                                            'filename': filename,
                                            'label': dataset['labels'][dataset_idx]
                                        })
                                        dataset_idx += 1
                                    else:
                                        unmatched_files.append((filename, paper_id))
                    except Exception as e:
                        print(f"Error processing {filename}: {e}")
                        continue
        
        # If we have unmatched entries, try a more thorough matching
        if dataset_idx < len(dataset):
            print(f"\nAttempting to match remaining {len(dataset) - dataset_idx} entries...")
            self.thorough_matching(dataset, id_mapping, dataset_idx, unmatched_files)
        
        return id_mapping
    
    def thorough_matching(self, dataset, id_mapping, start_idx, unmatched_files):
        """Perform more thorough matching for remaining entries"""
        labels_dict = self.builder.load_labels()
        
        for idx in range(start_idx, len(dataset)):
            target_text = dataset['text'][idx]
            target_label = dataset['labels'][idx]
            
            # Try all unmatched files
            for filename, paper_id in unmatched_files:
                filepath = os.path.join(self.builder.data_folder, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        text = f.read().strip()
                        if text:
                            text = self.builder.clean_text(text)
                            text = self.builder.abs_intro(text)
                            
                            # Check label match first
                            status = labels_dict.get(paper_id)
                            if status:
                                label = self.builder.get_label_from_status(status)
                                if label == target_label:
                                    # Compare more of the text
                                    if text[:500] == target_text[:500]:
                                        id_mapping.append({
                                            'index': idx,
                                            'paper_id': paper_id,
                                            'filename': filename,
                                            'label': label
                                        })
                                        break
                except Exception:
                    continue
    
    def save_mapping(self, id_mapping, output_file='dataset_id_mapping.json'):
        """Save the ID mapping to a JSON file"""
        with open(output_file, 'w') as f:
            json.dump(id_mapping, f, indent=2)
        print(f"Saved ID mapping to {output_file}")
        
    def verify_mapping(self, id_mapping):
        """Verify the mapping is correct"""
        dataset = self.load_cached_dataset()
        print(f"Dataset size: {len(dataset)}")
        print(f"Recovered IDs: {len(id_mapping)}")
        
        if len(id_mapping) != len(dataset):
            print("⚠️  Warning: Not all IDs were recovered!")
        
        # Show sample mappings
        print("\nSample mappings:")
        for entry in id_mapping[:5]:
            print(f"Index {entry['index']}: {entry['paper_id']} (label: {entry['label']})")

# Usage example
if __name__ == "__main__":
    # Update these paths to match your setup
    data_folder = "/mnt/parscratch/users/acr24wz/src/iclr/data/scratch/mpx602/topcon-1/conference_data/iclr_2025_data/filtered_cv_papers/cv_papers_text/"
    labels_file = "/mnt/parscratch/users/acr24wz/topcon/train/label_simple.json"
    dataset_path = "/mnt/parscratch/users/acr24wz/etu/topcon/processed_dataset/llm"
    
    recovery = DatasetIDRecovery(data_folder, labels_file, dataset_path)
    id_mapping = recovery.recover_ids()
    recovery.verify_mapping(id_mapping)
    recovery.save_mapping(id_mapping)
