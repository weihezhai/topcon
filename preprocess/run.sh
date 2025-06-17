#!/bin/bash
#$ -cwd
#$ -j y
#$ -pe smp 2      # 2 cores
#$ -l h_rt=12:0:0  # 24 hours runtime
#$ -l h_vmem=10G      # 10G RAM per core
#$ -m be
##$ -l gpu=1         # request GPUs (commented out)
## $ -l h=sbg4
#$ -l rocky
##$ -l cluster=andrena   
#$ -N iclr_2025_convert_pdfs

# Load environment
source /data/home/mpx602/projects/py311/bin/activate

# Configuration
PDF_DIR="/data/scratch/mpx602/topcon-1/conference_data/iclr_2025_data/pdfs"
OUTPUT_BASE_DIR="/data/scratch/mpx602/topcon-1/conference_data/iclr_2025_data/txt_outputs"
SCRIPT_PATH="/data/scratch/mpx602/topcon-1/preprocess/split_pdf.py"

# Create output base directory
mkdir -p "$OUTPUT_BASE_DIR"

# Get list of PDF files
PDF_FILES=("$PDF_DIR"/*.pdf)
TOTAL_FILES=${#PDF_FILES[@]}

# Check if any PDFs exist
if [ ! -f "${PDF_FILES[0]}" ]; then
    echo "No PDF files found in $PDF_DIR"
    exit 1
fi

echo "Found $TOTAL_FILES PDF files to process"
echo "Output directory: $OUTPUT_BASE_DIR"
echo "Starting conversion..."
echo

# Progress bar function
show_progress() {
    local current=$1
    local total=$2
    local percent=$((current * 100 / total))
    local filled=$((percent / 2))
    local empty=$((50 - filled))
    
    printf "\r["
    printf "%${filled}s" | tr ' ' '='
    printf "%${empty}s" | tr ' ' '-'
    printf "] %d%% (%d/%d)" "$percent" "$current" "$total"
}

# Process each PDF
PROCESSED=0
SUCCESSFUL=0
FAILED=0

for pdf_file in "${PDF_FILES[@]}"; do
    if [ -f "$pdf_file" ]; then
        # Extract filename without extension
        filename=$(basename "$pdf_file" .pdf)
        output_dir="$OUTPUT_BASE_DIR/output_$filename"
        
        # Run the Python script and capture result
        python3 "$SCRIPT_PATH" "$pdf_file" "$output_dir" --text-only > /dev/null 2>&1
        
        # Check if the output file was created
        if [ -f "$output_dir/${filename}_sections.txt" ]; then
            ((SUCCESSFUL++))
        else
            ((FAILED++))
        fi
        
        ((PROCESSED++))
        show_progress $PROCESSED $TOTAL_FILES
    fi
done

echo
echo
echo "Conversion complete!"
echo "Successfully processed: $SUCCESSFUL files"
echo "Failed: $FAILED files"
echo "Total processed: $PROCESSED files"

# Show some example output files
if [ $SUCCESSFUL -gt 0 ]; then
    echo
    echo "Sample output files:"
    find "$OUTPUT_BASE_DIR" -name "*_sections.txt" | head -3 | while read -r file; do
        echo "  $file"
    done
fi