import os
import json
import re
import csv
from datasets import Dataset

class TextDatasetBuilder:
    def __init__(self, data_folder, labels_file, statistics_file=None, max_length=1024):
        self.data_folder = data_folder
        self.labels_file = labels_file
        self.statistics_file = statistics_file
        self.max_length = max_length
        
    def load_labels(self):
        """Load labels from JSON file"""
        with open(self.labels_file, 'r', encoding='utf-8') as f:
            labels_dict = json.load(f)
        return labels_dict
        
    def extract_paper_id(self, filename):
        """Extract paper ID from filename (format: paperid_content_list.json)"""
        if filename.endswith('_content_list.json'):
            return filename.replace('_content_list.json', '')
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
    
    def _remove_github_links(self, text: str) -> str:
        """Remove sentences containing GitHub links from text."""
        # Remove entire sentences containing HTTP/HTTPS links
        # Use a more robust approach to avoid splitting at periods within URLs
        
        # First, temporarily replace URLs with a placeholder to avoid splitting issues
        url_pattern = re.compile(r'https?://[^\s]+')
        url_placeholder = "___URL_PLACEHOLDER___"
        
        # Find all URLs and their positions
        urls_found = url_pattern.findall(text)
        text_with_placeholders = url_pattern.sub(url_placeholder, text)
        
        # Now split into sentences (won't split at periods in URLs since they're replaced)
        sentences = re.split(r'(?<=[.!?])\s+', text_with_placeholders)
        
        # Filter out sentences containing GitHub links (both github.com and github.io)
        github_pattern = r'https?://(?:www\.)?(?:github\.com|[^/\s]*\.github\.io)[^\s)]*'
        filtered_sentences = []
        
        for sentence in sentences:
            # Check if this sentence had a GitHub URL
            has_github = False
            for url in urls_found:
                if re.match(github_pattern, url, re.IGNORECASE):
                    # If this sentence contains the placeholder, it had this GitHub URL
                    if url_placeholder in sentence:
                        has_github = True
                        break
            
            if not has_github:
                # Restore any non-GitHub URLs in this sentence
                sentence_restored = sentence
                for url in urls_found:
                    if not re.match(github_pattern, url, re.IGNORECASE):
                        # Replace placeholder with original URL (only first occurrence)
                        sentence_restored = sentence_restored.replace(url_placeholder, url, 1)
                
                # Only add if there are no remaining placeholders (meaning no GitHub URLs)
                if url_placeholder not in sentence_restored:
                    filtered_sentences.append(sentence_restored)
        
        return ' '.join(filtered_sentences)
    
    def _extract_paper_content(self, json_file_path: str) -> str:
        """
        Extract paper content from a JSON file.
        
        Args:
            json_file_path: Path to the JSON file containing paper data
            
        Returns:
            Concatenated text from:
            1. Title (first text_level=1) wrapped in <title> tags
            2. Abstract section wrapped in <abstract> tags
            3. Introduction section wrapped in <introduction> tags
        """
        # Load JSON data from file
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                paper_data = json.load(f)
        except Exception as e:
            print(f"Error loading JSON from {json_file_path}: {e}")
            return ""
        
        content_parts = []
        
        # Find indices of key sections
        title_idx = None
        abstract_idx = None
        intro_idx = None
        
        for i, entry in enumerate(paper_data):
            # Check for level 1 headers
            if entry.get("type") == "text" and entry.get("text_level") == 1:
                text = entry.get("text", "").strip()
                
                # First text_level=1 is the title
                if title_idx is None:
                    title_idx = i
                # Look for abstract
                elif "ABSTRACT" in text.upper():
                    abstract_idx = i
                # Look for introduction
                elif "INTRODUCTION" in text.upper():
                    intro_idx = i
                    # We can stop searching once we find introduction
                    break
        
        # Extract title
        if title_idx is not None:
            title_text = paper_data[title_idx].get("text", "").strip()
            content_parts.append(f"<title>{title_text}</title>")
        
        # Extract abstract (from abstract header to introduction)
        if abstract_idx is not None and intro_idx is not None:
            abstract_parts = []
            for i in range(abstract_idx, intro_idx):
                entry = paper_data[i]
                if entry.get("type") == "text":
                    text = entry.get("text", "").strip()
                    # Skip the "ABSTRACT" header itself
                    if text and not "ABSTRACT" in text.upper():
                        abstract_parts.append(text)
                elif entry.get("type") == "equation":
                    # Include equations in abstract if any
                    eq_text = entry.get("text", "").strip()
                    if eq_text:
                        abstract_parts.append(eq_text)
            
            if abstract_parts:
                abstract_text = " ".join(abstract_parts)
                content_parts.append(f"<abstract>{abstract_text}</abstract>")
        
        # Extract introduction (from intro header to next section)
        if intro_idx is not None:
            # Find next text_level=1 after introduction (to know where intro ends)
            next_section_idx = None
            for i in range(intro_idx + 1, len(paper_data)):
                if paper_data[i].get("type") == "text" and paper_data[i].get("text_level") == 1:
                    next_section_idx = i
                    break
            
            # If no next section found, use end of document
            if next_section_idx is None:
                next_section_idx = len(paper_data)
            
            # Extract introduction content
            intro_parts = []
            for i in range(intro_idx, next_section_idx):
                entry = paper_data[i]
                if entry.get("type") == "text":
                    text = entry.get("text", "").strip()
                    # Skip the "INTRODUCTION" header itself
                    if text and not "INTRODUCTION" in text.upper():
                        intro_parts.append(text)
                elif entry.get("type") == "equation":
                    eq_text = entry.get("text", "").strip()
                    if eq_text:
                        intro_parts.append(eq_text)
            
            if intro_parts:
                intro_text = " ".join(intro_parts)
                content_parts.append(f"<introduction>{intro_text}</introduction>")
        
        # Join all parts and remove GitHub links
        full_text = " ".join(content_parts)
        full_text = self._remove_github_links(full_text)
        
        return full_text
    
    def _extract_full_paper_content(self, json_file_path: str) -> str:
        """
        Extract full paper content from a JSON file for reference counting.
        
        Args:
            json_file_path: Path to the JSON file containing paper data
            
        Returns:
            Concatenated text from all text and equation entries
        """
        # Load JSON data from file
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                paper_data = json.load(f)
        except Exception as e:
            print(f"Error loading JSON from {json_file_path}: {e}")
            return ""
        
        content_parts = []
        
        # Extract all text and equation content
        for entry in paper_data:
            if entry.get("type") == "text":
                text = entry.get("text", "").strip()
                if text:
                    content_parts.append(text)
            elif entry.get("type") == "equation":
                eq_text = entry.get("text", "").strip()
                if eq_text:
                    content_parts.append(eq_text)
        
        # Join all parts
        full_text = " ".join(content_parts)
        
        return full_text

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
        
        # Get subdirectories (which are paper IDs)
        subdirs = []
        for subdir in os.listdir(self.data_folder):
            subdir_path = os.path.join(self.data_folder, subdir)
            if os.path.isdir(subdir_path):
                subdirs.append(subdir)
        
        print(f"\nTotal subdirectories (paper IDs): {len(subdirs)}")
        
        # Show sample subdirectory names and label matching
        print("\nSample subdirectory names and label matching:")
        for subdir in subdirs[:10]:
            in_labels = subdir in labels_dict
            print(f"  '{subdir}' -> In labels: {in_labels}")
        
        # Check if any IDs match
        matched_count = 0
        for subdir in subdirs:
            if subdir in labels_dict:
                matched_count += 1
        
        print(f"\nMatched subdirectories: {matched_count} out of {len(subdirs)}")
        
        print("=== END DEBUGGING ===\n")
    
    def load_statistics(self):
        """Load statistics from JSON file"""
        if not self.statistics_file or not os.path.exists(self.statistics_file):
            return {}
        
        with open(self.statistics_file, 'r', encoding='utf-8') as f:
            stats_data = json.load(f)
        
        # Return the papers dictionary
        return stats_data.get('papers', {})
    
    def count_references(self, text):
        """Count references by counting occurrences of 'et al.' in the text"""
        # Count both "et al." and "et al," patterns (case-insensitive)
        et_al_pattern = re.compile(r'\bet\s+al\.?(?:\s*,|\s*;|\s*\)|\s+|\s*$)', re.IGNORECASE)
        matches = et_al_pattern.findall(text)
        return len(matches)
    
    def format_statistics(self, paper_id, stats_dict, reference_count=None):
        """Format statistics for a paper as a tagged string"""
        features = []
        
        # Add reference count if provided
        if reference_count is not None:
            features.append(f"reference_count: {reference_count}")
        
        # Extract features from stats_dict if available
        if paper_id in stats_dict:
            paper_stats = stats_dict[paper_id]
            
            # (1) total_words
            if 'basic_metrics' in paper_stats and 'total_words' in paper_stats['basic_metrics']:
                features.append(f"total_words: {paper_stats['basic_metrics']['total_words']}")
            
            # (2) total_pages
            if 'basic_metrics' in paper_stats and 'total_pages' in paper_stats['basic_metrics']:
                features.append(f"total_pages: {paper_stats['basic_metrics']['total_pages']}")
            
            # (3) header_count
            if 'basic_metrics' in paper_stats and 'header_count' in paper_stats['basic_metrics']:
                features.append(f"header_count: {paper_stats['basic_metrics']['header_count']}")
            
            # (4) table_count
            if 'visual_content' in paper_stats and 'table_count' in paper_stats['visual_content']:
                features.append(f"table_count: {paper_stats['visual_content']['table_count']}")
            
            # (5) image_count
            if 'visual_content' in paper_stats and 'image_count' in paper_stats['visual_content']:
                features.append(f"image_count: {paper_stats['visual_content']['image_count']}")
            
            # (6) equation_count
            if 'visual_content' in paper_stats and 'equation_count' in paper_stats['visual_content']:
                features.append(f"equation_count: {paper_stats['visual_content']['equation_count']}")
            
            # (7) avg_caption_length
            if 'visual_densities' in paper_stats and 'avg_caption_length' in paper_stats['visual_densities']:
                features.append(f"avg_caption_length: {paper_stats['visual_densities']['avg_caption_length']}")
        
        if features:
            return f"\n<statistics>\n{'; '.join(features)}\n</statistics>"
        else:
            return ""
    
    def load_dataset(self):
        """Load JSON files and create dataset with labels"""
        texts = []
        labels = []
        
        # Load labels dictionary
        labels_dict = self.load_labels()
        
        # Load statistics dictionary
        stats_dict = self.load_statistics()
        
        missing_labels = []
        processed_files = 0
        
        # Process subdirectories (subdirectory name IS the paper ID)
        for paper_id in sorted(os.listdir(self.data_folder)):
            subdir_path = os.path.join(self.data_folder, paper_id)
            if not os.path.isdir(subdir_path):
                continue
            
            # Look for JSON file in subdirectory
            json_file = None
            for filename in os.listdir(subdir_path):
                if filename.endswith('_content_list.json'):
                    json_file = filename
                    break
            
            if not json_file:
                continue
            
            filepath = os.path.join(subdir_path, json_file)
            
            # Look up label in labels dictionary using subdirectory name as paper ID
            if paper_id in labels_dict:
                status = labels_dict[paper_id]
                label = self.get_label_from_status(status)
                
                try:
                    # Extract title+abstract+intro content for the main text
                    original_text = self._extract_paper_content(filepath)
                    if original_text:  # Only add non-empty texts
                        # Extract full paper content for reference counting
                        full_paper_text = self._extract_full_paper_content(filepath)
                        
                        # Count references on full paper content
                        reference_count = self.count_references(full_paper_text)
                        
                        # Apply abs_intro to extract only text before </introduction> if needed
                        # text = self.abs_intro(original_text)
                        
                        # Append statistics if available
                        stats_text = self.format_statistics(paper_id, stats_dict, reference_count)
                        if stats_text:
                            text = original_text + stats_text
                        else:
                            text = original_text
                        
                        texts.append(text)
                        labels.append(label)
                        processed_files += 1
                except Exception as e:
                    print(f"Error processing {filepath}: {e}")
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
    
    def load_dataset_with_ids(self):
        """Load JSON files and create dataset with labels and IDs"""
        texts = []
        labels = []
        paper_ids = []
        
        # Load labels dictionary
        labels_dict = self.load_labels()
        
        # Load statistics dictionary
        stats_dict = self.load_statistics()
        
        # Process subdirectories in sorted order for consistency (subdirectory name IS the paper ID)
        for paper_id in sorted(os.listdir(self.data_folder)):
            subdir_path = os.path.join(self.data_folder, paper_id)
            if not os.path.isdir(subdir_path):
                continue
            
            # Look for JSON file in subdirectory
            json_file = None
            for filename in os.listdir(subdir_path):
                if filename.endswith('_content_list.json'):
                    json_file = filename
                    break
            
            if not json_file:
                continue
            
            filepath = os.path.join(subdir_path, json_file)
            
            # Look up label in labels dictionary using subdirectory name as paper ID
            if paper_id in labels_dict:
                status = labels_dict[paper_id]
                label = self.get_label_from_status(status)
                
                try:
                    # Extract title+abstract+intro content for the main text
                    original_text = self._extract_paper_content(filepath)
                    if original_text:  # Only add non-empty texts
                        # Extract full paper content for reference counting
                        full_paper_text = self._extract_full_paper_content(filepath)
                        
                        # Count references on full paper content
                        reference_count = self.count_references(full_paper_text)
                        
                        # Apply abs_intro to extract only text before </introduction> if needed
                        # text = self.abs_intro(original_text)
                        
                        # Append statistics if available
                        stats_text = self.format_statistics(paper_id, stats_dict, reference_count)
                        if stats_text:
                            text = original_text + stats_text
                        else:
                            text = original_text
                        
                        texts.append(text)
                        labels.append(label)
                        paper_ids.append(paper_id)
                except Exception as e:
                    continue
        
        return Dataset.from_dict({
            'text': texts,
            'labels': labels,
            'paper_id': paper_ids
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