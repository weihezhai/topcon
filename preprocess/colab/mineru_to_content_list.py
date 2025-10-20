import json
import argparse
from pathlib import Path
from typing import List, Dict


def extract_text_content(pdf_info: List[Dict]) -> List[Dict]:
    """
    Extract text content from MinerU's pdf_info structure.
    
    Args:
        pdf_info: List of page information dictionaries
        
    Returns:
        List of content items with text content
    """
    content_list = []
    
    for page in pdf_info:
        page_idx = page.get("page_idx", 0)
        para_blocks = page.get("para_blocks", [])
        
        for block in para_blocks:
            block_type = block.get("type", "")
            
            # Process text blocks
            if block_type == "text":
                lines = block.get("lines", [])
                text_content = []
                
                for line in lines:
                    spans = line.get("spans", [])
                    for span in spans:
                        if isinstance(span, dict):
                            text = span.get("content", span.get("text", ""))
                        else:
                            text = str(span)
                        if text.strip():
                            text_content.append(text)
                
                if text_content:
                    content_item = {
                        "type": "text",
                        "text": " ".join(text_content),
                        "page_idx": page_idx
                    }
                    content_list.append(content_item)
            
            # Process title blocks
            elif block_type == "title":
                lines = block.get("lines", [])
                text_content = []
                
                for line in lines:
                    spans = line.get("spans", [])
                    for span in spans:
                        if isinstance(span, dict):
                            text = span.get("content", span.get("text", ""))
                        else:
                            text = str(span)
                        if text.strip():
                            text_content.append(text)
                
                if text_content:
                    content_item = {
                        "type": "text",
                        "text": " ".join(text_content),
                        "text_level": 1,  # Mark as title/heading
                        "page_idx": page_idx
                    }
                    content_list.append(content_item)
            
            # Process equation blocks
            elif block_type == "interline_equation":
                lines = block.get("lines", [])
                text_content = []
                
                for line in lines:
                    spans = line.get("spans", [])
                    for span in spans:
                        if isinstance(span, dict):
                            text = span.get("content", span.get("text", ""))
                        else:
                            text = str(span)
                        if text.strip():
                            text_content.append(text)
                
                if text_content:
                    content_item = {
                        "type": "equation",
                        "text": " ".join(text_content),
                        "text_format": "latex",
                        "page_idx": page_idx
                    }
                    content_list.append(content_item)
            
            # Process image blocks (store metadata only, no actual image)
            elif block_type == "image":
                image_blocks = block.get("blocks", [])
                caption_text = []
                
                for img_block in image_blocks:
                    if img_block.get("type") == "image_caption":
                        lines = img_block.get("lines", [])
                        for line in lines:
                            spans = line.get("spans", [])
                            for span in spans:
                                if isinstance(span, dict):
                                    text = span.get("content", span.get("text", ""))
                                else:
                                    text = str(span)
                                if text.strip():
                                    caption_text.append(text)
                
                if caption_text:
                    content_item = {
                        "type": "image",
                        "image_caption": [" ".join(caption_text)],
                        "page_idx": page_idx
                    }
                    content_list.append(content_item)
            
            # Process table blocks
            elif block_type == "table":
                table_blocks = block.get("blocks", [])
                caption_text = []
                
                for tbl_block in table_blocks:
                    if tbl_block.get("type") == "table_caption":
                        lines = tbl_block.get("lines", [])
                        for line in lines:
                            spans = line.get("spans", [])
                            for span in spans:
                                if isinstance(span, dict):
                                    text = span.get("content", span.get("text", ""))
                                else:
                                    text = str(span)
                                if text.strip():
                                    caption_text.append(text)
                
                if caption_text:
                    content_item = {
                        "type": "table",
                        "table_caption": [" ".join(caption_text)],
                        "page_idx": page_idx
                    }
                    content_list.append(content_item)
            
            # Process list blocks
            elif block_type == "list":
                list_blocks = block.get("blocks", [])
                for list_item in list_blocks:
                    lines = list_item.get("lines", [])
                    text_content = []
                    
                    for line in lines:
                        spans = line.get("spans", [])
                        for span in spans:
                            if isinstance(span, dict):
                                text = span.get("content", span.get("text", ""))
                            else:
                                text = str(span)
                            if text.strip():
                                text_content.append(text)
                    
                    if text_content:
                        content_item = {
                            "type": "text",
                            "text": " ".join(text_content),
                            "page_idx": page_idx
                        }
                        content_list.append(content_item)
    
    return content_list


def process_mineru_json(input_file: Path, output_file: Path) -> None:
    """
    Process MinerU JSON output and extract text content.
    
    Args:
        input_file: Path to input JSON file
        output_file: Path to output JSON file
    """
    # Load input JSON
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Extract pdf_info
    pdf_info = data.get("pdf_info", [])
    
    # Extract text content
    content_list = extract_text_content(pdf_info)
    
    # Write output JSON
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(content_list, f, indent=4, ensure_ascii=False)
    
    print(f"Processed {len(pdf_info)} pages")
    print(f"Extracted {len(content_list)} content items")
    print(f"Output saved to: {output_file}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Extract text content from MinerU JSON output"
    )
    parser.add_argument(
        "--input",
        type=str,
        help="Path to input MinerU JSON file"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to output content list JSON file"
    )
    
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return
    
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Process the file
    process_mineru_json(input_path, output_path)


if __name__ == "__main__":
    main()