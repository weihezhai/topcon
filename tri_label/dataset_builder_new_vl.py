import os
import json
import re
import csv
from datasets import Dataset

class TextDatasetBuilder:
    def __init__(self, data_folder, labels_file, statistics_file=None, img_desc_file=None, metadata_file=None, max_length=1024):
        self.data_folder = data_folder
        self.labels_file = labels_file
        self.statistics_file = statistics_file
        self.img_desc_file = img_desc_file
        self.metadata_file = metadata_file
        self.max_length = max_length
        
    def load_labels(self):
        """Load labels from JSON file"""
        with open(self.labels_file, 'r', encoding='utf-8') as f:
            labels_dict = json.load(f)
        return labels_dict
        
    def load_metadata(self):
        """Load metadata from JSON file (list of dicts with 'id' and 'rating_avg')"""
        if not self.metadata_file or not os.path.exists(self.metadata_file):
            return {}
        
        try:
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            # Convert list of dicts to dict of id->score
            return {item['id']: item.get('rating_avg', None) for item in data}
        except Exception as e:
            print(f"Error loading metadata: {e}")
            return {}

    def load_image_descriptions(self):
        """Load image descriptions from JSON file"""
        if not self.img_desc_file or not os.path.exists(self.img_desc_file):
            return {}
        
        try:
            with open(self.img_desc_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading image descriptions: {e}")
            return {}

    def _extract_figure_number(self, caption):
        """Extract figure number from caption string (e.g., 'Figure 1: ...' -> '1')"""
        if not caption:
            return None
        match = re.search(r"Figure\s+(\d+)", caption, re.IGNORECASE)
        return match.group(1) if match else None

    def _process_image_entry(self, entry, content_parts, paper_id, paper_img_descs):
        """Process an image entry and append description/caption to content_parts"""
        captions = entry.get("image_caption", [])
        # Handle case where image_caption might be None or empty list
        if not captions or not isinstance(captions, list):
            return
            
        caption = captions[0]
        
        # Extract figure number to find description
        fig_num = self._extract_figure_number(caption)
        description = ""
        
        if fig_num and paper_id and paper_img_descs:
            # Construct key: {paper_id}/figures/figure_{number}.jpg
            img_key = f"{paper_id}/figures/figure_{fig_num}.jpg"
            if img_key in paper_img_descs:
                description = paper_img_descs[img_key].get("description", "")
        
        # Format: Caption + Description
        text_part = f"{caption}"
        if description:
            text_part += f"\n[Image Description]: {description}"
        
        content_parts.append(text_part)

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
    
    def get_label_from_score(self, score):
        """Convert score to 3-class label"""
        if score is None:
            return None
            
        # Handle case where score is a list [mean, std]
        if isinstance(score, list):
            if len(score) > 0:
                score = score[0]
            else:
                return None
                
        if score < 5.4:
            return 0 # No / Reject
        elif score > 6.2:
            return 1 # Yes / Accept
        else:
            return 2 # May / Borderline

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
    
    def _extract_paper_content(self, json_file_path: str, paper_id: str = None, paper_img_descs: dict = None) -> str:
        """
        Extract paper content from a JSON file.
        
        Args:
            json_file_path: Path to the JSON file containing paper data
            paper_id: The ID of the paper (used for image lookup)
            paper_img_descs: Dictionary of image descriptions for this paper
            
        Returns:
            Concatenated text from:
            1. Title (first text_level=1)
            2. Abstract section
            3. Introduction section  
            4. Main body sections (until Acknowledgments or References)
            5. Equations
            6. Images (Captions + Descriptions)
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
        acknowledgments_idx = None
        references_idx = None
        
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
                # Look for acknowledgments (before references)
                elif acknowledgments_idx is None and ("ACKNOWLEDGMENT" in text.upper() or "ACKNOWLEDGEMENT" in text.upper()):
                    acknowledgments_idx = i
                # Look for references
                elif "REFERENCES" in text.upper() or "REFERENCE" in text.upper():
                    references_idx = i
                    break
            # Also check if any regular text block starts with "ACKNOWLEDGMENTS" or "REFERENCES"
            elif entry.get("type") == "text":
                text = entry.get("text", "").strip()
                upper_text = text.upper()
                
                # Check if text starts with "ACKNOWLEDGMENTS" (case-insensitive)
                if acknowledgments_idx is None and (upper_text.startswith("ACKNOWLEDGMENT") or upper_text.startswith("ACKNOWLEDGEMENT")):
                    acknowledgments_idx = i
                # Check if text starts with "REFERENCES" (case-insensitive)
                elif references_idx is None and upper_text.startswith("REFERENCES"):
                    references_idx = i
                    break
        
        # Determine the end index for content extraction
        # Stop at acknowledgments if it exists, otherwise stop at references
        content_end_idx = acknowledgments_idx if acknowledgments_idx is not None else references_idx
        
        # Extract title
        if title_idx is not None:
            title_text = paper_data[title_idx].get("text", "").strip()
            content_parts.append(title_text)
        
        # Extract abstract (from abstract header to introduction)
        if abstract_idx is not None and intro_idx is not None:
            for i in range(abstract_idx, intro_idx):
                entry = paper_data[i]
                if entry.get("type") == "text":
                    text = entry.get("text", "").strip()
                    if text:
                        content_parts.append(text)
                elif entry.get("type") == "equation":
                    # Include equations in abstract if any
                    eq_text = entry.get("text", "").strip()
                    if eq_text:
                        content_parts.append(eq_text)
                elif entry.get("type") == "image":
                    self._process_image_entry(entry, content_parts, paper_id, paper_img_descs)
        
        # Extract introduction (from intro header to next section)
        if intro_idx is not None:
            # Find next text_level=1 after introduction
            next_section_idx = None
            for i in range(intro_idx + 1, len(paper_data)):
                if paper_data[i].get("type") == "text" and paper_data[i].get("text_level") == 1:
                    next_section_idx = i
                    break
            
            if next_section_idx is None:
                next_section_idx = content_end_idx if content_end_idx else len(paper_data)
            
            for i in range(intro_idx, min(next_section_idx, content_end_idx if content_end_idx else len(paper_data))):
                entry = paper_data[i]
                if entry.get("type") == "text":
                    text = entry.get("text", "").strip()
                    if text:
                        content_parts.append(text)
                elif entry.get("type") == "equation":
                    eq_text = entry.get("text", "").strip()
                    if eq_text:
                        content_parts.append(eq_text)
                elif entry.get("type") == "image":
                    self._process_image_entry(entry, content_parts, paper_id, paper_img_descs)
        
        # Extract main body (from after introduction to acknowledgments/references)
        if intro_idx is not None:
            start_idx = intro_idx
            # Find next section after introduction
            for i in range(intro_idx + 1, len(paper_data)):
                if paper_data[i].get("type") == "text" and paper_data[i].get("text_level") == 1:
                    start_idx = i
                    break
            
            end_idx = content_end_idx if content_end_idx else len(paper_data)
            
            for i in range(start_idx, end_idx):
                entry = paper_data[i]
                if entry.get("type") == "text":
                    text = entry.get("text", "").strip()
                    if text:
                        content_parts.append(text)
                elif entry.get("type") == "equation":
                    eq_text = entry.get("text", "").strip()
                    if eq_text:
                        content_parts.append(eq_text)
                elif entry.get("type") == "image":
                    self._process_image_entry(entry, content_parts, paper_id, paper_img_descs)
        
        # Join all parts and remove GitHub links
        full_text = " ".join(content_parts)
        full_text = self._remove_github_links(full_text)
        
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
        
        # Load metadata dictionary
        metadata_dict = self.load_metadata()
        
        # Load statistics dictionary
        stats_dict = self.load_statistics()

        # Load image descriptions
        all_img_descs = self.load_image_descriptions()
        
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
            
            # Determine label
            label = None
            
            # Use metadata score if available (3-class)
            if metadata_dict and paper_id in metadata_dict:
                score = metadata_dict[paper_id]
                label = self.get_label_from_score(score)
            # Fallback to binary labels if metadata not available but labels are
            elif not metadata_dict and paper_id in labels_dict:
                status = labels_dict[paper_id]
                label = self.get_label_from_status(status)
            
            if label is not None:
                try:
                    # Get image descriptions for this paper
                    paper_img_descs = all_img_descs.get(paper_id, {})

                    # Extract content from JSON
                    text = self._extract_paper_content(filepath, paper_id, paper_img_descs)
                    if text:  # Only add non-empty texts
                        # Count references in the text
                        reference_count = self.count_references(text)
                        
                        # Add statistics at the end of the text
                        stats_str = self.format_statistics(paper_id, stats_dict, reference_count)
                        text_with_stats = text + stats_str
                        
                        texts.append(text_with_stats)
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
        
        # Load metadata dictionary
        metadata_dict = self.load_metadata()
        
        # Load statistics dictionary
        stats_dict = self.load_statistics()

        # Load image descriptions
        all_img_descs = self.load_image_descriptions()
        
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
            
            # Determine label
            label = None
            
            # Use metadata score if available (3-class)
            if metadata_dict and paper_id in metadata_dict:
                score = metadata_dict[paper_id]
                label = self.get_label_from_score(score)
            # Fallback to binary labels if metadata not available but labels are
            elif not metadata_dict and paper_id in labels_dict:
                status = labels_dict[paper_id]
                label = self.get_label_from_status(status)
            
            if label is not None:
                try:
                    # Get image descriptions for this paper
                    paper_img_descs = all_img_descs.get(paper_id, {})

                    # Extract content from JSON
                    text = self._extract_paper_content(filepath, paper_id, paper_img_descs)
                    if text:  # Only add non-empty texts
                        # Count references in the text
                        reference_count = self.count_references(text)
                        
                        # Add statistics at the end of the text
                        stats_str = self.format_statistics(paper_id, stats_dict, reference_count)
                        text_with_stats = text + stats_str
                        
                        texts.append(text_with_stats)
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
                    'rejected (0)': 0,
                    'maybe (2)': 0
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
                'rejected (0)': label_counts.get(0, 0),
                'maybe (2)': label_counts.get(2, 0)
            },
            'text_stats': {
                'avg_length_words': sum(text_lengths) / len(text_lengths) if text_lengths else 0,
                'min_length_words': min(text_lengths) if text_lengths else 0,
                'max_length_words': max(text_lengths) if text_lengths else 0
            }
        }
        
        return stats