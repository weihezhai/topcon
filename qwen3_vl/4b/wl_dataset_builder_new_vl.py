import os
import json
import re
import csv
from datasets import Dataset

class TextDatasetBuilder:
    def __init__(
        self,
        data_folder,
        labels_file,
        statistics_file=None,
        img_desc_file=None,  # kept for backward compat; no longer used when using real images
        max_length=1024,
        metadata_file=None,
        noisy_low=5.4,
        noisy_high=6.2,
        noisy_weight=0.5,
        images_root=None,          # NEW: optional override; defaults to each paper subdir
        max_images_per_paper=6,    # NEW: cap images to limit VRAM/sequence length
    ):
        self.data_folder = data_folder
        self.labels_file = labels_file
        self.statistics_file = statistics_file
        self.img_desc_file = img_desc_file
        self.max_length = max_length

        # NEW: real-image configuration
        self.images_root = images_root
        self.max_images_per_paper = max_images_per_paper

        # rating-based weighting
        self.metadata_file = metadata_file
        self.noisy_low = noisy_low
        self.noisy_high = noisy_high
        self.noisy_weight = noisy_weight
        self._rating_map = None  # lazy-loaded
    
    # Add near top of class
    FIGURE_MARKER = "<|FIGURE|>"  # unique marker used to interleave figures later

    def load_labels(self):
        """Load labels from JSON file"""
        with open(self.labels_file, 'r', encoding='utf-8') as f:
            labels_dict = json.load(f)
        return labels_dict
        
    def load_image_descriptions(self):
        """Deprecated in VLM mode; kept for backward compatibility."""
        return {}

    def _extract_figure_number(self, caption):
        """Extract figure number from caption string (e.g., 'Figure 1: ...' -> '1')"""
        if not caption:
            return None
        match = re.search(r"Figure\s+(\d+)", caption, re.IGNORECASE)
        return match.group(1) if match else None

    def _process_image_entry(self, entry, content_parts, paper_id, paper_img_descs):
        """Deprecated in VLM mode: we no longer inject image descriptions into text."""
        return

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

    def _figures_dir(self, paper_id: str) -> str:
        base = self.images_root if self.images_root else self.data_folder
        return os.path.join(base, paper_id, "figures")

    def _match_figure_path_from_caption(self, paper_id: str, caption: str):
        """
        Match caption like 'Figure 1: ...' to a local image filename like figure_1.jpg (case-insensitive).
        Returns full path or None.
        """
        fig_num = self._extract_figure_number(caption)
        if not fig_num:
            return None

        figures_dir = self._figures_dir(paper_id)
        if not os.path.isdir(figures_dir):
            return None

        exts = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
        # robust regex: fig/figure + optional separators + number boundary
        pat = re.compile(rf"(figure|fig)[\s_\-]*{re.escape(str(fig_num))}\b", re.IGNORECASE)

        matches = []
        for fn in os.listdir(figures_dir):
            if not fn.lower().endswith(exts):
                continue
            stem = os.path.splitext(fn)[0]
            if pat.search(stem):
                matches.append(fn)

        if not matches:
            return None

        matches.sort()
        return os.path.join(figures_dir, matches[0])
    
    def _extract_paper_content_and_images(self, json_file_path: str, paper_id: str = None, paper_img_descs: dict = None) -> str:
        """
        Extract paper content from a JSON file.
        
        Args:
            json_file_path: Path to the JSON file containing paper data
            paper_id: The ID of the paper (used for image lookup)
            
        Returns:
            Concatenated text from:
            1. Title (first text_level=1)
            2. Abstract section
            3. Introduction section  
            4. Main body sections (until Acknowledgments or References)
            5. Equations
            6. inserts FIGURE_MARKER into the text at the exact JSON position of each image entry
            7. appends the matched image path (caption->Figure N->file match) in the same order
        """
        # Load JSON data from file
        try:
            with open(json_file_path, "r", encoding="utf-8") as f:
                paper_data = json.load(f)
        except Exception as e:
            print(f"Error loading JSON from {json_file_path}: {e}")
            return "", []

        content_parts = []
        ordered_image_paths = []

        # --- same section indexing logic as your current _extract_paper_content ---
        title_idx = abstract_idx = intro_idx = acknowledgments_idx = references_idx = None

        for i, entry in enumerate(paper_data):
            if entry.get("type") == "text" and entry.get("text_level") == 1:
                text = entry.get("text", "").strip()
                if title_idx is None:
                    title_idx = i
                elif "ABSTRACT" in text.upper():
                    abstract_idx = i
                elif "INTRODUCTION" in text.upper():
                    intro_idx = i
                elif acknowledgments_idx is None and ("ACKNOWLEDGMENT" in text.upper() or "ACKNOWLEDGEMENT" in text.upper()):
                    acknowledgments_idx = i
                elif "REFERENCES" in text.upper() or "REFERENCE" in text.upper():
                    references_idx = i
                    break
            elif entry.get("type") == "text":
                text = entry.get("text", "").strip()
                upper_text = text.upper()
                if acknowledgments_idx is None and (upper_text.startswith("ACKNOWLEDGMENT") or upper_text.startswith("ACKNOWLEDGEMENT")):
                    acknowledgments_idx = i
                elif references_idx is None and upper_text.startswith("REFERENCES"):
                    references_idx = i
                    break

        content_end_idx = acknowledgments_idx if acknowledgments_idx is not None else references_idx

        def _handle_entry(entry):
            t = entry.get("type")
            if t == "text":
                txt = entry.get("text", "").strip()
                if txt:
                    content_parts.append(txt)
            elif t == "equation":
                eq = entry.get("text", "").strip()
                if eq:
                    content_parts.append(eq)
            elif t == "image":
                # grab caption from common keys (depends on your parser)
                caption = (
                    entry.get("caption")
                    or entry.get("caption_text")
                    or entry.get("text")   # some pipelines store caption in "text"
                    or ""
                ).strip()

                # enforce cap
                if self.max_images_per_paper is not None and len(ordered_image_paths) >= int(self.max_images_per_paper):
                    return

                img_path = self._match_figure_path_from_caption(paper_id, caption)

                # Only interleave if we can match a real file
                if img_path:
                    content_parts.append(self.FIGURE_MARKER)
                    if caption:
                        content_parts.append(caption)
                    ordered_image_paths.append(img_path)

        # Title
        if title_idx is not None:
            title_text = paper_data[title_idx].get("text", "").strip()
            if title_text:
                content_parts.append(title_text)

        # Abstract section: [abstract_idx, intro_idx)
        if abstract_idx is not None and intro_idx is not None:
            for i in range(abstract_idx, intro_idx):
                _handle_entry(paper_data[i])

        # Introduction section
        if intro_idx is not None:
            next_section_idx = None
            for i in range(intro_idx + 1, len(paper_data)):
                if paper_data[i].get("type") == "text" and paper_data[i].get("text_level") == 1:
                    next_section_idx = i
                    break
            if next_section_idx is None:
                next_section_idx = content_end_idx if content_end_idx else len(paper_data)

            end_cap = content_end_idx if content_end_idx else len(paper_data)
            for i in range(intro_idx, min(next_section_idx, end_cap)):
                _handle_entry(paper_data[i])

        # Main body
        if intro_idx is not None:
            start_idx = intro_idx
            for i in range(intro_idx + 1, len(paper_data)):
                if paper_data[i].get("type") == "text" and paper_data[i].get("text_level") == 1:
                    start_idx = i
                    break

            end_idx = content_end_idx if content_end_idx else len(paper_data)
            for i in range(start_idx, end_idx):
                _handle_entry(paper_data[i])

        full_text = " ".join(content_parts)
        full_text = self._remove_github_links(full_text)
        return full_text, ordered_image_paths
    
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
    
    def load_metadata_ratings(self):
        """Load metadata JSON (list[dict]) and build map: paper_id -> rating_avg (float)."""
        if not self.metadata_file or not os.path.exists(self.metadata_file):
            print(f"Warning: Metadata file not found or not set: {self.metadata_file}")
            return {}

        print(f"Loading metadata from {self.metadata_file}...")
        try:
            with open(self.metadata_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                print("Error: Metadata JSON is not a list.")
                return {}

            rating_map = {}
            noisy_count = 0
            for item in data:
                if not isinstance(item, dict):
                    continue
                pid = item.get("id")
                ravg = item.get("rating_avg")
                if not pid:
                    continue
                
                val = None
                try:
                    # Handle list format [mean, std] or scalar
                    if isinstance(ravg, list) and len(ravg) > 0:
                        val = float(ravg[0])
                    elif ravg is not None:
                        val = float(ravg)
                except (TypeError, ValueError):
                    val = None
                
                rating_map[str(pid)] = val
                if val is not None and self.noisy_low <= val <= self.noisy_high:
                    noisy_count += 1
            
            print(f"Loaded {len(rating_map)} ratings from metadata.")
            print(f"Amount of data in noisy range [{self.noisy_low}, {self.noisy_high}]: {noisy_count}")
            if rating_map:
                # Filter out None values for display
                valid_ratings = {k: v for k, v in rating_map.items() if v is not None}
                print(f"Valid ratings count: {len(valid_ratings)}")
                print(f"Sample metadata IDs: {list(valid_ratings.keys())[:20]}")
                print(f"Sample ratings: {list(valid_ratings.values())[:20]}")
            
            return rating_map
        except Exception as e:
            print(f"Error loading metadata ratings: {e}")
            return {}

    def get_sample_weight(self, paper_id: str) -> float:
        """Weight=0.5 if rating_avg in [noisy_low,noisy_high], else 1.0; missing -> 1.0."""
        if self._rating_map is None:
            self._rating_map = self.load_metadata_ratings()

        if not paper_id or not self._rating_map:
            return 1.0

        rating = self._rating_map.get(str(paper_id), None)
        if rating is None:
            return 1.0
        return self.noisy_weight if (self.noisy_low <= rating <= self.noisy_high) else 1.0

    def add_sample_weights_to_dataset(self, dataset: Dataset) -> Dataset:
        """Attach/overwrite `sample_weight` column using `paper_id` in an existing dataset."""
        if "paper_id" not in dataset.column_names:
            return dataset

        def _add(batch):
            pids = batch["paper_id"]
            return {"sample_weight": [self.get_sample_weight(pid) for pid in pids]}

        return dataset.map(_add, batched=True)

    def _list_paper_image_paths(self, paper_id: str) -> list:
        """
        Collect real figure image paths for a paper.
        Expected layout: {data_folder}/{paper_id}/figures/*.jpg (or png/jpeg).
        If `images_root` is set, use {images_root}/{paper_id}/figures/*.
        """
        base = self.images_root if self.images_root else self.data_folder
        figures_dir = os.path.join(base, paper_id, "figures")
        if not os.path.isdir(figures_dir):
            return []

        exts = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
        paths = [
            os.path.join(figures_dir, fn)
            for fn in sorted(os.listdir(figures_dir))
            if fn.lower().endswith(exts)
        ]
        if self.max_images_per_paper is not None:
            paths = paths[: int(self.max_images_per_paper)]
        return paths

    def load_dataset(self):
        """Load JSON files and create dataset with labels"""
        texts = []
        labels = []
        sample_weights = []
        image_paths_all = []  # NEW

        labels_dict = self.load_labels()
        stats_dict = self.load_statistics()

        missing_labels = []
        processed_files = 0

        # NEW: image-match counters (matched = caption->file matches actually interleaved)
        papers_with_no_matched_images = 0
        total_matched_images = 0

        for paper_id in sorted(os.listdir(self.data_folder)):
            subdir_path = os.path.join(self.data_folder, paper_id)
            if not os.path.isdir(subdir_path):
                continue

            json_file = None
            for filename in os.listdir(subdir_path):
                if filename.endswith('_content_list.json'):
                    json_file = filename
                    break
            if not json_file:
                continue

            filepath = os.path.join(subdir_path, json_file)

            if paper_id in labels_dict:
                status = labels_dict[paper_id]
                label = self.get_label_from_status(status)

                try:
                    text, ordered_paths = self._extract_paper_content_and_images(filepath, paper_id)
                    if text:
                        reference_count = self.count_references(text)
                        stats_str = self.format_statistics(paper_id, stats_dict, reference_count)
                        text_with_stats = text + stats_str

                        texts.append(text_with_stats)
                        labels.append(label)
                        sample_weights.append(self.get_sample_weight(paper_id))
                        image_paths_all.append(ordered_paths) # NEW
                        processed_files += 1

                        # NEW: counters
                        if not ordered_paths:
                            papers_with_no_matched_images += 1
                        total_matched_images += len(ordered_paths)
                except Exception:
                    continue
            else:
                missing_labels.append(paper_id)

        # NEW: report
        if processed_files > 0:
            avg = total_matched_images / processed_files
            print(f"[ImageMatch] {papers_with_no_matched_images}/{processed_files} processed papers have 0 matched images; avg matched images/paper = {avg:.2f}")

        if missing_labels:
            print(f"Warning: {len(missing_labels)} files had no corresponding labels")
            print(f"First few missing IDs: {missing_labels[:5]}")

        if processed_files == 0:
            print("❌ No files were successfully processed!")
            print("Please check the filename format or label file.")
            return Dataset.from_dict({'text': [], 'labels': []})

        return Dataset.from_dict({
            'text': texts,
            'labels': labels,
            'sample_weight': sample_weights,
            'image_paths': image_paths_all,   # NEW
        })

    def load_dataset_with_ids(self):
        """Load JSON files and create dataset with labels and IDs"""
        texts = []
        labels = []
        paper_ids = []
        sample_weights = []
        image_paths_all = []  # NEW

        labels_dict = self.load_labels()
        stats_dict = self.load_statistics()

        # NEW: image-match counters
        processed_files = 0
        papers_with_no_matched_images = 0
        total_matched_images = 0

        for paper_id in sorted(os.listdir(self.data_folder)):
            subdir_path = os.path.join(self.data_folder, paper_id)
            if not os.path.isdir(subdir_path):
                continue

            json_file = None
            for filename in os.listdir(subdir_path):
                if filename.endswith('_content_list.json'):
                    json_file = filename
                    break
            if not json_file:
                continue

            filepath = os.path.join(subdir_path, json_file)

            if paper_id in labels_dict:
                status = labels_dict[paper_id]
                label = self.get_label_from_status(status)

                try:
                    text, ordered_paths = self._extract_paper_content_and_images(filepath, paper_id)
                    if text:
                        reference_count = self.count_references(text)
                        stats_str = self.format_statistics(paper_id, stats_dict, reference_count)
                        text_with_stats = text + stats_str

                        texts.append(text_with_stats)
                        labels.append(label)
                        paper_ids.append(paper_id)
                        sample_weights.append(self.get_sample_weight(paper_id))
                        image_paths_all.append(ordered_paths)  # NEW

                        # NEW: counters
                        processed_files += 1
                        if not ordered_paths:
                            papers_with_no_matched_images += 1
                        total_matched_images += len(ordered_paths)
                except Exception:
                    continue

        # NEW: report
        if processed_files > 0:
            avg = total_matched_images / processed_files
            print(f"[ImageMatch] {papers_with_no_matched_images}/{processed_files} processed papers have 0 matched images; avg matched images/paper = {avg:.2f}")

        return Dataset.from_dict({
            'text': texts,
            'labels': labels,
            'paper_id': paper_ids,
            'sample_weight': sample_weights,
            'image_paths': image_paths_all,  # NEW
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

        # NEW: image stats if present
        if "image_paths" in dataset.column_names:
            img_counts = [len(xs) for xs in dataset["image_paths"]]
            stats["image_stats"] = {
                "papers_with_0_matched_images": int(sum(c == 0 for c in img_counts)),
                "avg_matched_images_per_paper": (sum(img_counts) / len(img_counts)) if img_counts else 0.0,
                "max_matched_images_per_paper": max(img_counts) if img_counts else 0,
            }

        return stats