import os
import json
from datasets import Dataset

class TextDatasetBuilder:
    def __init__(self, data_folder, labels_file, max_length=1024):
        self.data_folder = data_folder
        self.labels_file = labels_file
        self.max_length = max_length
        
    def load_labels(self):
        """Load labels from JSON file"""
        with open(self.labels_file, 'r', encoding='utf-8') as f:
            labels_dict = json.load(f)
        return labels_dict
        
    def extract_paper_id(self, filename):
        """Extract paper ID from filename (format: id_sections.txt)"""
        if filename.endswith('_sections.txt'):
            return filename[:-12]  # Remove '_sections.txt'
        elif filename.endswith('.txt') and '_' in filename:
            return filename.split('_')[0]  # Take part before first underscore
        else:
            return None
            
    def get_label_from_status(self, status):
        """Convert status to binary label"""
        rejected_statuses = [
            "Submitted to ICLR 2025",
            "ICLR 2025 Conference Withdrawn Submission"
        ]
        return 0 if status in rejected_statuses else 1
        
    def load_dataset(self):
        """Load txt files and create dataset with labels"""
        texts = []
        labels = []
        
        # Load labels dictionary
        labels_dict = self.load_labels()
        
        missing_labels = []
        processed_files = 0
        
        for filename in os.listdir(self.data_folder):
            if filename.endswith('.txt'):
                filepath = os.path.join(self.data_folder, filename)
                
                # Extract paper ID from filename
                paper_id = self.extract_paper_id(filename)
                if paper_id is None:
                    print(f"Could not extract paper ID from {filename}")
                    continue
                
                # Look up label in labels dictionary
                if paper_id in labels_dict:
                    status = labels_dict[paper_id]
                    label = self.get_label_from_status(status)
                    
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            text = f.read().strip()
                            if text:  # Only add non-empty texts
                                texts.append(text)
                                labels.append(label)
                                processed_files += 1
                    except Exception as e:
                        print(f"Error reading {filename}: {e}")
                        continue
                else:
                    missing_labels.append(paper_id)
        
        print(f"Processed {processed_files} files")
        if missing_labels:
            print(f"Warning: {len(missing_labels)} files had no corresponding labels")
            print(f"First few missing IDs: {missing_labels[:5]}")
        
        return Dataset.from_dict({
            'text': texts,
            'labels': labels
        })
    
    def get_dataset_stats(self, dataset):
        """Get dataset statistics"""
        import pandas as pd
        
        label_counts = pd.Series(dataset['labels']).value_counts()
        text_lengths = [len(text.split()) for text in dataset['text']]
        
        stats = {
            'total_samples': len(dataset),
            'label_distribution': {
                'accepted (1)': label_counts.get(1, 0),
                'rejected (0)': label_counts.get(0, 0)
            },
            'text_stats': {
                'avg_length_words': sum(text_lengths) / len(text_lengths),
                'min_length_words': min(text_lengths),
                'max_length_words': max(text_lengths)
            }
        }
        
        return stats