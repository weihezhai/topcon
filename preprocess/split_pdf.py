import fitz  # PyMuPDF
import os
import re
from typing import List, Dict, Tuple, Optional

def sanitize_filename(name: str) -> str:
    """Removes invalid characters from a string to make it a valid filename."""
    # Remove leading/trailing whitespace
    name = name.strip()
    # Replace invalid characters with an underscore
    name = re.sub(r'[\\/*?:"<>|]', "_", name)
    # Remove any leading numbers and periods that might be part of the ToC entry
    name = re.sub(r'^[0-9\.\s]+', '', name)
    return name

def sanitize_section_name(name: str) -> str:
    """Sanitizes section name for use in XML-like tags."""
    # Remove leading/trailing whitespace
    name = name.strip()
    # Replace spaces and special characters with underscores
    name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
    # Remove any leading numbers and underscores
    name = re.sub(r'^[0-9_]+', '', name)
    # Ensure it starts with a letter
    if name and not name[0].isalpha():
        name = 'section_' + name
    return name.lower()

def find_header_line_on_page(doc: fitz.Document, page_num: int, header_text: str) -> int:
    """Find the line number of a header on a given page."""
    if page_num < 0 or page_num >= doc.page_count:
        return -1
    
    page = doc[page_num]
    lines = page.get_text().split('\n')
    
    # Clean the header text for comparison
    clean_header = re.sub(r'^[0-9\.\s]+', '', header_text.strip()).lower()
    
    for i, line in enumerate(lines):
        clean_line = re.sub(r'^[0-9\.\s]+', '', line.strip()).lower()
        # Check for exact match or if the header is contained in the line
        if clean_header == clean_line or (len(clean_header) > 3 and clean_header in clean_line):
            return i
    
    return -1

def detect_references_section_heuristic(doc: fitz.Document, level1_toc: List) -> Optional[Tuple[int, int]]:
    """
    Detect the References section that might not be in the TOC.
    
    Args:
        doc: The PDF document
        level1_toc: List of level 1 TOC entries
    
    Returns:
        Tuple of (start_page, end_page) of references section, or None if not found
    """
    # Search for References section starting from page 8 (0-indexed page 7) to page 12 (0-indexed page 11)
    search_start_page = 7  # Page 8 in 1-indexed
    search_end_page = min(11, doc.page_count - 1)  # Page 12 in 1-indexed, or last page if shorter
    
    references_start_page = None
    
    for page_num in range(search_start_page, search_end_page + 1):
        page = doc[page_num]
        text = page.get_text()
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            # Look for standalone "REFERENCES" or "References" line, possibly with a number prefix
            if re.match(r'^(\d+\.?\s+)?(REFERENCES?|References?)\.?$', line):
                references_start_page = page_num
                print(f"Found potential non-TOC References header: '{line}' on page {page_num + 1}")
                break
        
        if references_start_page is not None:
            break
    
    if references_start_page is None:
        print("No non-TOC References header found")
        return None
    
    # Find the end page: it should be before the next Level 1 TOC entry that appears after references_start_page
    references_end_page = doc.page_count - 1  # Default to end of document
    
    for entry in level1_toc:
        toc_page = entry[2] - 1  # Convert to 0-indexed
        if toc_page > references_start_page:
            references_end_page = toc_page - 1
            print(f"References section ends at page {references_end_page + 1} (before next TOC entry '{entry[1]}' at page {toc_page + 1})")
            break
    
    print(f"Detected non-TOC References section: pages {references_start_page + 1}-{references_end_page + 1}")
    return (references_start_page, references_end_page)

def find_references_header_in_page_range(doc: fitz.Document, start_page: int, end_page: int) -> Optional[Tuple[int, int]]:
    """
    Find a References header within a specific page range.
    Returns (page_num, line_num) if found, None otherwise.
    """
    for page_num in range(start_page, min(end_page + 1, doc.page_count)):
        page = doc[page_num]
        lines = page.get_text().split('\n')
        
        for line_num, line in enumerate(lines):
            line = line.strip()
            # Look for standalone "REFERENCES" or "References" line, possibly with a number prefix
            if re.match(r'^(\d+\.?\s+)?(REFERENCES?|References?)\.?$', line):
                print(f"Found References header: '{line}' at page {page_num + 1}, line {line_num}")
                return (page_num, line_num)
    
    return None

def extract_title_and_abstract(doc: fitz.Document, first_toc_sec_page_0idx: int, first_toc_sec_line_0idx: int) -> str:
    """
    Extracts title and abstract from the beginning of the document,
    stopping before the first actual TOC section.
    Args:
        doc: The PDF document.
        first_toc_sec_page_0idx: 0-indexed page where the first TOC section (e.g., Introduction) starts.
        first_toc_sec_line_0idx: 0-indexed line where the first TOC section header starts on its page.
    Returns:
        Extracted text for title and abstract.
    """
    content = []
    # Max pages to consider for title/abstract if the first TOC section is very late or absent.
    # e.g., 1 means pages 0 and 1.
    max_pages_to_scan_for_abstract = 1 

    for page_num in range(doc.page_count):
        if page_num > max_pages_to_scan_for_abstract and page_num < first_toc_sec_page_0idx:
            # If we've scanned enough initial pages and haven't hit the first TOC page yet, stop.
            break
        
        if page_num >= first_toc_sec_page_0idx and first_toc_sec_line_0idx == 0 :
            # If the first TOC section starts at the very beginning of this page,
            # then there's no title/abstract content on this page to extract before it.
            if page_num == first_toc_sec_page_0idx: # Ensure we break if this is the exact page
                 break


        page = doc[page_num]
        text = page.get_text()
        lines = text.split('\n')
        
        # Determine the end line for processing on the current page
        current_page_end_line = len(lines) # Default to all lines
        if page_num == first_toc_sec_page_0idx:
            current_page_end_line = first_toc_sec_line_0idx # Stop before the first TOC section's header
            
        for line_idx, line_text in enumerate(lines):
            if line_idx >= current_page_end_line:
                break 
                
            line_text = line_text.strip()
            # Basic filtering for title/abstract
            if not line_text: # Skip empty lines
                continue
            if re.match(r'^\d{3,}$', line_text): # Skip what look like page numbers if they are alone on a line
                continue
            if re.match(r'^Under review as a conference paper at [A-Z]+ \d+', line_text): # Skip "Under review" lines
                print(f"  Filtering out review status from title/abstract: '{line_text}'")
                continue
            content.append(line_text)
            
        if page_num == first_toc_sec_page_0idx:
            # If we've processed the page where the first TOC section starts (up to its starting line),
            # then we are done with title/abstract extraction.
            break
            
    return '\n'.join(content)

def find_section_text_boundaries(doc: fitz.Document, level1_toc: List) -> List[Dict]:
    """
    Find precise text boundaries for each Level 1 TOC section by searching for headers.
    Also handles References sections that may not be in TOC.
    """
    sections = []
    
    for i, entry in enumerate(level1_toc):
        level, title, start_page = entry
        start_page -= 1  # Convert to 0-indexed
        
        # Skip references sections that are explicitly in TOC
        if re.search(r'\b(references?|bibliography|works?\s+cited)\b', title, re.IGNORECASE):
            print(f"Skipping TOC references section: '{title}'")
            continue
            
        print(f"Processing section: '{title}' starting at page {start_page + 1}")
        
        # Find where this section's text actually starts
        section_start_line = find_header_line_on_page(doc, start_page, title)
        if section_start_line == -1:
            print(f"  Warning: Header '{title}' not found on page {start_page + 1}. Defaulting to line 0.")
            section_start_line = 0
        
        # Determine end by finding the next section's header OR a References header
        section_end_page = doc.page_count - 1
        section_end_line = None
        
        # First, look for the next TOC section's header
        next_toc_boundary = None
        for j in range(i + 1, len(level1_toc)):
            next_level, next_title, next_page = level1_toc[j]
            next_page -= 1  # Convert to 0-indexed
            
            # Skip references sections
            if re.search(r'\b(references?|bibliography|works?\s+cited)\b', next_title, re.IGNORECASE):
                continue
                
            # print(f"  Looking for next TOC section '{next_title}' starting from page {next_page + 1}")
            
            # Search for the next header starting from its TOC page
            for search_page in range(next_page, min(next_page + 3, doc.page_count)): # Search on its page and next few
                next_header_line = find_header_line_on_page(doc, search_page, next_title)
                if next_header_line != -1:
                    next_toc_boundary = (search_page, next_header_line)
                    # print(f"  Found next TOC section header '{next_title}' at page {search_page + 1}, line {next_header_line}")
                    break
            
            if next_toc_boundary is not None:
                break
        
        # Now look for References header in the current section's potential range
        # The search for references should go from the current section's start up to where the next TOC item was found (or end of doc)
        references_search_start_page = start_page
        references_search_end_page = next_toc_boundary[0] if next_toc_boundary else doc.page_count - 1
        
        references_boundary = find_references_header_in_page_range(doc, references_search_start_page, references_search_end_page)
        
        # Choose the earlier boundary (References or next TOC section)
        final_end_page = doc.page_count -1
        final_end_line = None

        if references_boundary and next_toc_boundary:
            if (references_boundary[0] < next_toc_boundary[0] or 
                (references_boundary[0] == next_toc_boundary[0] and references_boundary[1] < next_toc_boundary[1])):
                final_end_page, final_end_line = references_boundary
                final_end_line -= 1  # End before the references header
                print(f"  Section '{title}' ends before References at page {final_end_page + 1}, line {final_end_line + 1 if final_end_line is not None else 'end of page'}")
            else:
                final_end_page, final_end_line = next_toc_boundary
                final_end_line -= 1  # End before the next section header
                print(f"  Section '{title}' ends before next TOC section at page {final_end_page + 1}, line {final_end_line + 1 if final_end_line is not None else 'end of page'}")
        elif references_boundary:
            final_end_page, final_end_line = references_boundary
            final_end_line -= 1  # End before the references header
            print(f"  Section '{title}' ends before References at page {final_end_page + 1}, line {final_end_line + 1 if final_end_line is not None else 'end of page'}")
        elif next_toc_boundary:
            final_end_page, final_end_line = next_toc_boundary
            final_end_line -= 1  # End before the next section header
            print(f"  Section '{title}' ends before next TOC section at page {final_end_page + 1}, line {final_end_line + 1 if final_end_line is not None else 'end of page'}")
        else: # Section extends to end of document
            final_end_page = doc.page_count - 1
            final_end_line = None # Signifies end of document
            print(f"  Section '{title}' extends to end of document")
        
        # Ensure end_line is not negative if page is same as start_page
        if final_end_page == start_page and final_end_line is not None and final_end_line < section_start_line:
            final_end_line = section_start_line # Avoid empty section if end is before start on same page

        sections.append({
            "level": level,
            "title": title,
            "section_name": sanitize_section_name(title),
            "start_page": start_page,
            "start_line": section_start_line,
            "end_page": final_end_page,
            "end_line": final_end_line
        })
    
    return sections

def extract_text_from_section_boundaries(doc: fitz.Document, section: Dict) -> str:
    """Extract text content respecting line boundaries within sections."""
    text_content = []
    start_page = section['start_page']
    end_page = section['end_page']
    start_line = section.get('start_line', 0)
    end_line = section.get('end_line', None) # Can be None for end of page/document
    
    for page_num in range(start_page, end_page + 1):
        if page_num < doc.page_count:
            page = doc[page_num]
            text = page.get_text()
            lines = text.split('\n')
            
            page_start_line = 0
            page_end_line_exclusive = len(lines)

            if page_num == start_page:
                page_start_line = start_line
            
            if page_num == end_page and end_line is not None:
                page_end_line_exclusive = end_line + 1
            
            # Slice the lines for the current page
            relevant_lines = lines[page_start_line:page_end_line_exclusive]
            
            cleaned_lines = []
            for line_text in relevant_lines:
                line_text = line_text.strip()
                # Skip empty lines and line numbers
                if not line_text or re.match(r'^\d{3,}$', line_text):
                    continue
                # Skip standalone References headers (unless this IS the references section, which is handled by find_section_text_boundaries skipping it)
                if re.match(r'^(\d+\.?\s+)?(REFERENCES?|References?)\.?$', line_text):
                    # This check might be too aggressive if a section legitimately contains "References" in its text
                    # For now, assume actual References sections are skipped by find_section_text_boundaries
                    # print(f"  Filtering out potential References header in text: '{line_text}'")
                    pass # Let's be less aggressive here, boundary detection should handle it.
                
                # Skip "Under review" lines
                if re.match(r'^Under review as a conference paper at [A-Z]+ \d+', line_text):
                    print(f"  Filtering out review status: '{line_text}'")
                    continue
                cleaned_lines.append(line_text)
            
            if cleaned_lines:
                text_content.append('\n'.join(cleaned_lines))
    
    return '\n\n'.join(text_content)

def extract_text_from_page_range(doc: fitz.Document, start_page: int, end_page: int) -> str:
    """Extract text from a range of pages in the document."""
    text_content = []
    for page_num in range(start_page, end_page + 1):
        if page_num < doc.page_count:
            page = doc[page_num]
            text = page.get_text()
            
            # Clean up the text by removing line numbers and other artifacts
            lines = text.split('\n')
            cleaned_lines = []
            
            for line in lines:
                line = line.strip()
                # Skip lines that are just numbers (line numbers)
                if re.match(r'^\d{3,}$', line):
                    continue
                # Skip empty lines
                if not line:
                    continue
                # Skip "Under review" lines
                if re.match(r'^Under review as a conference paper at [A-Z]+ \d+', line):
                    continue
                cleaned_lines.append(line)
            
            if cleaned_lines:
                text_content.append('\n'.join(cleaned_lines))
    
    return '\n\n'.join(text_content)

def split_pdf_by_level1_toc(input_pdf_path: str, output_dir: str) -> str:
    """
    Splits a PDF into multiple parts based on Level 1 sections from its Table of Contents metadata.
    Extracts text content and creates a single text file with section markers.

    Args:
        input_pdf_path (str): The path to the source PDF file.
        output_dir (str): The directory where the output files will be saved.
    
    Returns:
        str: Path to the generated text file.
    """
    os.makedirs(output_dir, exist_ok=True)
    doc = fitz.open(input_pdf_path)
    toc = doc.get_toc()

    if not toc:
        print(f"Could not find a Table of Contents (outline) in '{input_pdf_path}'.")
        # Try to extract first page as title/abstract if no TOC
        pdf_basename = os.path.splitext(os.path.basename(input_pdf_path))[0]
        output_text_path = os.path.join(output_dir, f"{pdf_basename}_sections.txt")
        with open(output_text_path, 'w', encoding='utf-8') as text_file:
            title_abstract_text = extract_title_and_abstract(doc, doc.page_count, 0) # Effectively read first few pages
            if title_abstract_text.strip():
                text_file.write("<title_and_abstract>\n")
                text_file.write("# Title and Abstract\n\n")
                text_file.write(title_abstract_text)
                text_file.write("\n</title_and_abstract>\n\n")
                print("  > Added title and abstract section (no TOC found)")
        doc.close()
        return output_text_path if title_abstract_text.strip() else None


    print(f"Found {len(toc)} entries in the Table of Contents.")
    level1_toc = [entry for entry in toc if entry[0] == 1]
    
    if not level1_toc:
        print("No Level 1 sections found in the Table of Contents.")
        # Try to extract first page as title/abstract if no Level 1 TOC
        pdf_basename = os.path.splitext(os.path.basename(input_pdf_path))[0]
        output_text_path = os.path.join(output_dir, f"{pdf_basename}_sections.txt")
        with open(output_text_path, 'w', encoding='utf-8') as text_file:
            title_abstract_text = extract_title_and_abstract(doc, doc.page_count, 0) # Effectively read first few pages
            if title_abstract_text.strip():
                text_file.write("<title_and_abstract>\n")
                text_file.write("# Title and Abstract\n\n")
                text_file.write(title_abstract_text)
                text_file.write("\n</title_and_abstract>\n\n")
                print("  > Added title and abstract section (no Level 1 TOC found)")
        doc.close()
        return output_text_path if title_abstract_text.strip() else None

    print(f"Found {len(level1_toc)} Level 1 sections.")
    print("\nLevel 1 TOC entries:")
    for i, entry in enumerate(level1_toc):
        level, title, page = entry
        print(f"  {i}: '{title}' starts at page {page}")

    toc_sections = find_section_text_boundaries(doc, level1_toc)
    
    references_range = detect_references_section_heuristic(doc, level1_toc)
    references_end_page = references_range[1] if references_range else -1
    
    for section in toc_sections:
        section['is_appendix'] = section['start_page'] > references_end_page if references_end_page >= 0 else False

    pdf_basename = os.path.splitext(os.path.basename(input_pdf_path))[0]
    output_text_path = os.path.join(output_dir, f"{pdf_basename}_sections.txt")
    
    print(f"\nExtracting text content and creating: {output_text_path}")
    
    with open(output_text_path, 'w', encoding='utf-8') as text_file:
        first_toc_sec_page_0idx = doc.page_count 
        first_toc_sec_line_0idx = 0
        if toc_sections:
            first_toc_sec_page_0idx = toc_sections[0]['start_page']
            first_toc_sec_line_0idx = toc_sections[0]['start_line']
        
        title_abstract_text = extract_title_and_abstract(doc, first_toc_sec_page_0idx, first_toc_sec_line_0idx)
        if title_abstract_text.strip():
            text_file.write("<title_and_abstract>\n")
            text_file.write("# Title and Abstract\n\n")
            text_file.write(title_abstract_text)
            text_file.write("\n</title_and_abstract>\n\n")
            print("  > Added title and abstract section")
        
        appendix_opened = False
        print(f"Total TOC sections to process: {len(toc_sections)}")
        for i, section in enumerate(toc_sections):
            title = section['title']
            section_name = section['section_name']
            is_appendix = section.get('is_appendix', False)

            print(f"  > Processing section {i+1}/{len(toc_sections)}: '{title}' (Pages {section['start_page'] + 1}-{section['end_page'] + 1}, Lines {section['start_line']}-{section.get('end_line', 'end')}, Appendix: {is_appendix})")
            
            if is_appendix and not appendix_opened:
                text_file.write("<appendix>\n")
                appendix_opened = True
                print("  > Opening appendix section")
            
            text_content = extract_text_from_section_boundaries(doc, section)
            
            text_file.write(f"<{section_name}>\n")
            text_file.write(f"# {title}\n\n")
            text_file.write(text_content)
            text_file.write(f"\n</{section_name}>\n\n")
            
            is_last_section = (i == len(toc_sections) - 1)
            next_is_appendix = False
            if not is_last_section:
                next_is_appendix = toc_sections[i + 1].get('is_appendix', False)
            
            if is_appendix and appendix_opened and (is_last_section or not next_is_appendix):
                text_file.write("</appendix>\n\n")
                appendix_opened = False
                print("  > Closing appendix section")

    doc.close()
    print(f"\nProcess complete. Text file created: {output_text_path}")
    return output_text_path

def split_pdf_by_toc_with_pdfs(input_pdf_path: str, output_dir: str):
    """
    Splits a PDF into multiple PDF files based on Level 1 sections from its Table of Contents metadata.

    Args:
        input_pdf_path (str): The path to the source PDF file.
        output_dir (str): The directory where the split PDFs will be saved.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Open the source PDF
    doc = fitz.open(input_pdf_path)

    # Get the Table of Contents (document outline)
    toc = doc.get_toc()

    if not toc:
        print(f"Could not find a Table of Contents (outline) in '{input_pdf_path}'.")
        doc.close()
        return

    print(f"Found {len(toc)} entries in the Table of Contents.")

    # Filter to only Level 1 sections
    level1_toc = [entry for entry in toc if entry[0] == 1]
    
    if not level1_toc:
        print("No Level 1 sections found in the Table of Contents.")
        doc.close()
        return

    print(f"Found {len(level1_toc)} Level 1 sections.")

    # Detect non-TOC references section
    references_range = detect_references_section_heuristic(doc, level1_toc)

    # A list to hold processed section information
    processed_sections_for_pdf_split = [] # Renamed to avoid confusion

    # --- Handle content before the first Level 1 ToC entry ---
    if level1_toc: # Ensure level1_toc is not empty
        first_entry_page_1idx = level1_toc[0][2]
        first_entry_0idx = first_entry_page_1idx -1
        if first_entry_0idx > 0:
            preamble_end = first_entry_0idx - 1
            
            # If references section starts within preamble, truncate preamble
            if references_range and references_range[0] <= preamble_end:
                preamble_end = references_range[0] - 1
                
            if preamble_end >= 0:
                processed_sections_for_pdf_split.append({
                    "level": 0,
                    "title": "00_Preamble",
                    "start_page": 0,
                    "end_page": preamble_end
                })

    # --- Process each Level 1 ToC entry to define its page range ---
    for i, entry in enumerate(level1_toc):
        level, title, start_page_1idx = entry
        
        # Skip references section that are explicitly in TOC
        if re.search(r'\b(references?|bibliography|works?\s+cited)\b', title, re.IGNORECASE):
            print(f"  > Skipping TOC references section for PDF split: '{title}'")
            continue
        
        # Adjust page number to be 0-indexed
        current_section_start_page_0idx = start_page_1idx - 1
        
        # Determine the end page
        current_section_end_page_0idx = doc.page_count - 1 # Default to end of doc
        if i < len(level1_toc) - 1:
            next_section_start_page_1idx = level1_toc[i+1][2]
            current_section_end_page_0idx = next_section_start_page_1idx - 2 # Page before next section starts
        
        # Avoid creating empty PDFs or invalid ranges
        if current_section_end_page_0idx < current_section_start_page_0idx:
            current_section_end_page_0idx = current_section_start_page_0idx


        # Handle references range overlap for PDF splitting
        section_to_add = {
            "level": level,
            "title": f"{i+1:02d}_{sanitize_filename(title)}",
            "start_page": current_section_start_page_0idx,
            "end_page": current_section_end_page_0idx
        }

        if references_range:
            ref_start_0idx, ref_end_0idx = references_range
            
            # Case 1: Section is entirely within references range - skip it for PDF split
            if section_to_add['start_page'] >= ref_start_0idx and section_to_add['end_page'] <= ref_end_0idx:
                print(f"  > Skipping PDF section '{title}' as it's entirely within detected references")
                continue
            
            # Case 2: Section starts before references and overlaps - adjust end_page
            elif section_to_add['start_page'] < ref_start_0idx and section_to_add['end_page'] >= ref_start_0idx:
                print(f"  > Adjusting PDF section '{title}' due to overlap with references (ending before refs)")
                section_to_add['end_page'] = ref_start_0idx - 1
                # Part after references (if any) - this logic is complex for PDF splitting, usually we just take the part before.
                # For simplicity, we might just take the part before references.
                # If you need to split into "before" and "after" PDFs, that's more involved.
            
            # Case 3: Section starts within references but extends beyond - adjust start_page
            elif section_to_add['start_page'] >= ref_start_0idx and section_to_add['start_page'] <= ref_end_0idx and section_to_add['end_page'] > ref_end_0idx:
                print(f"  > Adjusting PDF section '{title}' due to overlap with references (starting after refs)")
                section_to_add['start_page'] = ref_end_0idx + 1
        
        if section_to_add['end_page'] >= section_to_add['start_page']: # Ensure valid page range
             processed_sections_for_pdf_split.append(section_to_add)


    # --- Create a new PDF for each section ---
    print("\nSplitting PDF into sections...")
    for section_data in processed_sections_for_pdf_split:
        title = section_data['title']
        start_page = section_data['start_page']
        end_page = section_data['end_page']

        if end_page < start_page : # Double check for safety
            print(f"  > Skipping PDF for '{title}' due to invalid page range ({start_page+1}-{end_page+1})")
            continue

        output_path = os.path.join(output_dir, f"{title}.pdf")
        new_doc = fitz.open()
        print(f"  > Creating '{output_path}' (Pages {start_page + 1}-{end_page + 1})")
        new_doc.insert_pdf(doc, from_page=start_page, to_page=end_page)
        new_doc.save(output_path)
        new_doc.close()

    doc.close()
    print("\nProcess complete for PDF splitting.")


def process_pdf(input_pdf_path: str, output_dir: str, create_pdfs: bool = False, create_text: bool = True):
    """
    Main function to process a PDF file.
    
    Args:
        input_pdf_path (str): Path to the input PDF file
        output_dir (str): Directory for output files
        create_pdfs (bool): Whether to create separate PDF files for each section
        create_text (bool): Whether to create a text file with section markers
    """
    if not os.path.exists(input_pdf_path):
        print(f"Error: Input PDF file '{input_pdf_path}' does not exist.")
        return
    
    if create_text:
        text_file_path = split_pdf_by_level1_toc(input_pdf_path, output_dir)
        if text_file_path:
            print(f"Text extraction completed: {text_file_path}")
    
    if create_pdfs:
        split_pdf_by_toc_with_pdfs(input_pdf_path, output_dir)
        # print("PDF splitting completed.") # Already printed in the function

# --- --- --- USAGE EXAMPLE --- --- ---
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python split_pdf.py <input_pdf_path> [output_dir] [--pdfs] [--text-only]")
        print("  --pdfs: Also create separate PDF files for each section")
        print("  --text-only: Only create text file (default behavior)")
        sys.exit(1)
    
    input_pdf = sys.argv[1]
    
    # Set default output directory if not provided
    if len(sys.argv) >= 3 and not sys.argv[2].startswith('--'):
        output_folder = sys.argv[2]
    else:
        # Create output directory based on input PDF name
        pdf_name = os.path.splitext(os.path.basename(input_pdf))[0]
        output_folder = f"output_{pdf_name}"
    
    # Parse options
    create_pdfs = '--pdfs' in sys.argv
    # Default to creating text unless --no-text is specified (or if only --pdfs is given, still create text)
    # Simplification: if --pdfs is there, text is also created. If --text-only, only text.
    # If no flags, only text.
    
    # If --pdfs is present, create_text is true by default.
    # If --text-only is present, create_pdfs is false.
    
    if '--text-only' in sys.argv:
        create_text = True
        create_pdfs = False
    elif '--pdfs' in sys.argv:
        create_text = True # Also create text when creating PDFs
        create_pdfs = True
    else: # Default behavior: only text
        create_text = True
        create_pdfs = False

    print(f"Input PDF: {input_pdf}")
    print(f"Output directory: {output_folder}")
    print(f"Create PDFs: {create_pdfs}")
    print(f"Create text: {create_text}")
    
    process_pdf(input_pdf, output_folder, create_pdfs, create_text)