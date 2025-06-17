#!/bin/bash

# Configuration
OUTPUT_DIR="/data/scratch/mpx602/topcon-1/conference_data/iclr_2025_data/filtered_llm_paperss/"
FILTER_SCRIPT="/data/scratch/mpx602/topcon-1/train/batch_filter.py"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

echo "Starting batch processing of ICLR 2025 papers..."
echo "Started at: $(date)"
echo "================================"

# Run the Python batch processor with output directory as argument
cd "$(dirname "$FILTER_SCRIPT")"
python3 batch_filter.py "$OUTPUT_DIR"

echo ""
echo "================================"
echo "Batch processing completed at: $(date)"

# Check if files were generated in the output directory
if [ -f "$OUTPUT_DIR/batch_filtered_results.json" ]; then
    echo "Results generated at: $OUTPUT_DIR/batch_filtered_results.json"
fi

if [ -f "$OUTPUT_DIR/matched_paper_ids.json" ]; then
    echo "Paper IDs generated at: $OUTPUT_DIR/matched_paper_ids.json"
    
    # Also create a text file with just the IDs for convenience
    jq -r '.[]' "$OUTPUT_DIR/matched_paper_ids.json" > "$OUTPUT_DIR/unique_matched_ids.txt"
    echo "Paper IDs text file created: $OUTPUT_DIR/unique_matched_ids.txt"
fi

echo "All outputs saved to: $OUTPUT_DIR"
