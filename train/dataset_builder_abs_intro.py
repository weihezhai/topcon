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
        if filename.endswith('.txt') and '_' in filename:
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
    
    def abs_intro(self, text):
        """Extract text before </introduction> tag to shorten content"""
        intro_end_tag = "</introduction>"
        if intro_end_tag in text:
            return text.split(intro_end_tag)[0].strip()
        else:
            # If no </introduction> tag found, return original text
            return text.strip()
    
    def debug_label_matching(self):
        """Debug function to check label matching issues"""
        print("\n=== DEBUGGING LABEL MATCHING ===")
        
        # Load labels
        labels_dict = self.load_labels()
        print(f"Total labels in JSON: {len(labels_dict)}")
        
        # Show sample labels from JSON
        print("\nSample labels from JSON:")
        sample_labels = list(labels_dict.items())[:5]
        for paper_id, status in sample_labels:
            print(f"  '{paper_id}': '{status}'")
        
        # Get txt files
        txt_files = [f for f in os.listdir(self.data_folder) if f.endswith('.txt')]
        print(f"\nTotal txt files: {len(txt_files)}")
        
        # Show sample filenames and extracted IDs
        print("\nSample filenames and extracted IDs:")
        for filename in txt_files[:10]:
            paper_id = self.extract_paper_id(filename)
            in_labels = paper_id in labels_dict if paper_id else False
            print(f"  '{filename}' -> '{paper_id}' -> In labels: {in_labels}")
        
        # Check if any IDs match
        matched_count = 0
        for filename in txt_files:
            paper_id = self.extract_paper_id(filename)
            if paper_id and paper_id in labels_dict:
                matched_count += 1
        
        print(f"\nMatched files: {matched_count} out of {len(txt_files)}")
        
        # Check for different filename patterns
        print("\nFilename patterns found:")
        patterns = {}
        for filename in txt_files[:20]:  # Check first 20 files
            if '_' in filename:
                parts = filename.split('_')
                pattern = f"{len(parts)} parts: " + "_".join(["ID" if i == 0 else part for i, part in enumerate(parts)])
                patterns[pattern] = patterns.get(pattern, 0) + 1
            else:
                patterns["no_underscore"] = patterns.get("no_underscore", 0) + 1
        
        for pattern, count in patterns.items():
            print(f"  {pattern}: {count} files")
        
        print("=== END DEBUGGING ===\n")
        
    def load_dataset(self):
        """Load txt files and create dataset with labels"""
        texts = []
        labels = []
        
        # Load labels dictionary
        labels_dict = self.load_labels()
        
        missing_labels = []
        processed_files = 0
        
        # Debug label matching if no files are processed initially
        txt_files = [f for f in os.listdir(self.data_folder) if f.endswith('.txt')]
        quick_check_matches = 0
        for filename in txt_files[:10]:  # Quick check first 10 files
            paper_id = self.extract_paper_id(filename)
            if paper_id and paper_id in labels_dict:
                quick_check_matches += 1
        
        if quick_check_matches == 0:
            print("⚠️  No matches found in first 10 files. Running debug...")
            self.debug_label_matching()
        
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
                                # Apply abs_intro to extract only text before </introduction>
                                text = self.abs_intro(text)
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
        
        # Handle empty dataset case
        if processed_files == 0:
            print("❌ No files were successfully processed!")
            print("Please check the filename format or label file.")
            return Dataset.from_dict({'text': [], 'labels': []})
        
        return Dataset.from_dict({
            'text': texts,
            'labels': labels
        })
    
    def get_dataset_stats(self, dataset):
        """Get dataset statistics"""
        import pandas as pd
        
        # Handle empty dataset
        if len(dataset) == 0:
            return {
                'total_samples': 0,
                'label_distribution': {
                    'accepted (1)': 0,
                    'rejected (0)': 0
                },
                'text_stats': {
                    'avg_length_words': 0,
                    'min_length_words': 0,
                    'max_length_words': 0
                }
            }
        
        label_counts = pd.Series(dataset['labels']).value_counts()
        text_lengths = [len(text.split()) for text in dataset['text']]
        
        stats = {
            'total_samples': len(dataset),
            'label_distribution': {
                'accepted (1)': label_counts.get(1, 0),
                'rejected (0)': label_counts.get(0, 0)
            },
            'text_stats': {
                'avg_length_words': sum(text_lengths) / len(text_lengths) if text_lengths else 0,
                'min_length_words': min(text_lengths) if text_lengths else 0,
                'max_length_words': max(text_lengths) if text_lengths else 0
            }
        }
        
        return stats