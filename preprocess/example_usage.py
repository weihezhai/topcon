#!/usr/bin/env python3
"""
Example usage of the split_pdf.py module
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from split_pdf import process_pdf

def main():
    # Example usage
    example_pdf_dir = "/data/scratch/mpx602/topcon-1/conference_data/iclr_2023_data/pdfs"
    
    # Find a PDF file to process (just take the first one for demo)
    pdf_files = [f for f in os.listdir(example_pdf_dir) if f.endswith('.pdf')]
    
    if not pdf_files:
        print(f"No PDF files found in {example_pdf_dir}")
        return
    
    # Use the first PDF file as an example
    input_pdf = os.path.join(example_pdf_dir, pdf_files[0])
    output_dir = "/data/scratch/mpx602/topcon-1/preprocess/output"
    
    print(f"Processing PDF: {input_pdf}")
    print(f"Output directory: {output_dir}")
    
    # Process the PDF - create text file with section markers
    process_pdf(input_pdf, output_dir, create_pdfs=False, create_text=True)

if __name__ == "__main__":
    main()
