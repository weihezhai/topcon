# PDF Section Splitter

This module provides functionality to split PDF documents based on their Table of Contents (ToC) metadata and extract text content with section markers.

## Features

- **Level 1 Section Detection**: Automatically identifies Level 1 sections from PDF outline/bookmarks
- **Text Extraction with Section Markers**: Creates a plain text file with `<section_name>...</section_name>` markers
- **Optional PDF Splitting**: Can also create separate PDF files for each section
- **Intelligent Section Naming**: Sanitizes section names for valid filenames and XML tags

## Requirements

- Python 3.6+
- PyMuPDF (fitz) - already installed in your environment

## Usage

### Command Line Usage

```bash
# Basic usage - creates text file with section markers
python split_pdf.py input.pdf

# Specify output directory
python split_pdf.py input.pdf output_directory

# Also create separate PDF files for each section
python split_pdf.py input.pdf output_directory --pdfs

# Only create text file (explicit)
python split_pdf.py input.pdf output_directory --text-only
```

### Python Module Usage

```python
from split_pdf import process_pdf

# Process PDF and create text file with section markers
process_pdf("my_paper.pdf", "output_dir", create_pdfs=False, create_text=True)

# Process PDF and create both text file and separate PDFs
process_pdf("my_paper.pdf", "output_dir", create_pdfs=True, create_text=True)
```

## Output Format

### Text File Output

The generated text file will have the following structure:

```
<preamble>
# Preamble

[Content before first Level 1 section - abstract, title page, etc.]
</preamble>

<introduction>
# Introduction

[Introduction section content]
</introduction>

<methodology>
# Methodology

[Methodology section content]
</methodology>

<results>
# Results

[Results section content]
</results>

<conclusion>
# Conclusion

[Conclusion section content]
</conclusion>
```

### Section Name Sanitization

- Section names are converted to lowercase
- Special characters and spaces are replaced with underscores
- Leading numbers are removed
- Names are ensured to start with a letter

Example transformations:
- "1. Introduction" → "introduction"
- "2.1 Data Collection & Analysis" → "data_collection___analysis"
- "Results and Discussion" → "results_and_discussion"

## How It Works

1. **PDF Outline Reading**: Uses PyMuPDF to read the embedded document outline/bookmarks
2. **Level 1 Filtering**: Filters the outline to only include Level 1 sections (main sections)
3. **Page Range Calculation**: Determines the exact page range for each section
4. **Text Extraction**: Extracts text content from each page range
5. **Section Formatting**: Wraps each section in XML-like tags with sanitized names

## Error Handling

- Checks if PDF file exists
- Handles PDFs without embedded outline/bookmarks
- Gracefully handles empty sections
- Provides informative error messages

## Example

See `example_usage.py` for a demonstration of how to use the module with your conference data PDFs.
