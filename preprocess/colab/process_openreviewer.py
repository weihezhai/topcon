#!/usr/bin/env python3
import json
import re
import sys

def remove_github_links(text: str) -> str:
    """Remove sentences containing GitHub links from text."""
    url_pattern = re.compile(r'https?://[^\s]+')
    url_placeholder = "___URL_PLACEHOLDER___"
    
    # Find all URLs and their positions
    urls_found = url_pattern.findall(text)
    text_with_placeholders = url_pattern.sub(url_placeholder, text)
    
    # Now split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text_with_placeholders)
    
    # Filter out sentences containing GitHub links
    github_pattern = r'https?://(?:www\.)?(?:github\.com|[^/\s]*\.github\.io)[^\s)]*'
    filtered_sentences = []
    
    for sentence in sentences:
        has_github = False
        for url in urls_found:
            if re.match(github_pattern, url, re.IGNORECASE):
                if url_placeholder in sentence:
                    has_github = True
                    break
        
        if not has_github:
            sentence_restored = sentence
            for url in urls_found:
                if not re.match(github_pattern, url, re.IGNORECASE):
                    sentence_restored = sentence_restored.replace(url_placeholder, url, 1)
            
            if url_placeholder not in sentence_restored:
                filtered_sentences.append(sentence_restored)
    
    return ' '.join(filtered_sentences)

def extract_paper_content(paper_data: list) -> str:
    """
    Extract paper content from JSON data.
    
    Returns:
        Concatenated text from:
        1. Title (first text_level=1)
        2. Abstract section
        3. Introduction section  
        4. Main body sections (until Acknowledgments or References)
        5. Equations
    """
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
            
            if acknowledgments_idx is None and (upper_text.startswith("ACKNOWLEDGMENT") or upper_text.startswith("ACKNOWLEDGEMENT")):
                acknowledgments_idx = i
            elif references_idx is None and upper_text.startswith("REFERENCES"):
                references_idx = i
                break
    
    # Determine the end index for content extraction
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
                eq_text = entry.get("text", "").strip()
                if eq_text:
                    content_parts.append(eq_text)
    
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
    
    # Join all parts and remove GitHub links
    full_text = " ".join(content_parts)
    full_text = remove_github_links(full_text)
    
    return full_text

def count_references(text):
    """Count references by counting occurrences of 'et al.' in the text"""
    et_al_pattern = re.compile(r'\bet\s+al\.?(?:\s*,|\s*;|\s*\)|\s+|\s*$)', re.IGNORECASE)
    matches = et_al_pattern.findall(text)
    return len(matches)

def main():
    input_file = "openreviewer.json"
    output_file = "openreviewer_processed.txt"
    
    # Load JSON data
    print(f"Loading {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        paper_data = json.load(f)
    
    print(f"Found {len(paper_data)} entries in JSON")
    
    # Extract paper content
    print("Extracting paper content...")
    text = extract_paper_content(paper_data)
    
    # Count references
    ref_count = count_references(text)
    
    # Add statistics
    stats_str = f"\n<statistics>\nreference_count: {ref_count}\n</statistics>"
    text_with_stats = text + stats_str
    
    # Save to file
    print(f"Saving processed text to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(text_with_stats)
    
    # Print summary
    word_count = len(text.split())
    print(f"\n✓ Processing complete!")
    print(f"  - Word count: {word_count:,}")
    print(f"  - Reference count: {ref_count}")
    print(f"  - Output saved to: {output_file}")
    
    # Print first 500 characters as preview
    print(f"\n--- Preview (first 500 chars) ---")
    print(text[:500])
    print("...")
    print(f"\n--- Statistics section ---")
    print(stats_str)

if __name__ == "__main__":
    main()
