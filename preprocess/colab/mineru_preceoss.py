import json
import os
import shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
import re
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import time
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("processing.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


@dataclass
class ProcessingStats:
    """Track processing statistics"""

    total_files: int = 0
    processed_files: int = 0
    failed_files: int = 0
    text_sections: int = 0
    figures_extracted: int = 0
    tables_extracted: int = 0
    equations_extracted: int = 0


class PaperProcessor:
    """Main class for processing paper content files"""

    def __init__(self, mineru_root: str, output_root: str):
        self.mineru_root = Path(mineru_root)
        self.output_root = Path(output_root)
        self.stats = ProcessingStats()

        # Create output directories
        self.output_root.mkdir(parents=True, exist_ok=True)

    def sanitize_filename(self, filename: str) -> str:
        """Sanitize filename to be filesystem-safe"""
        # Remove or replace problematic characters
        filename = re.sub(r'[<>:"/\\|?*]', "_", filename)
        # Remove extra whitespace and periods
        filename = re.sub(r"\s+", " ", filename).strip()
        filename = filename.replace("..", ".")
        # Limit length
        if len(filename) > 200:
            filename = filename[:200]
        return filename

    def extract_reference_name(self, caption: str, content_type: str) -> str:
        """Extract reference name from caption (e.g., 'Figure 1', 'Table 2')"""
        if not caption:
            return f"unnamed_{content_type}"

        # Try to find figure/table/equation numbers
        patterns = [
            rf"{content_type.title()}\s*(\d+(?:\.\d+)?)",  # Figure 1, Table 2.1
            rf"{content_type.title()}\s*([A-Z]\d*)",  # Figure A1
            rf"{content_type.title()}\s*([IVX]+)",  # Figure I, II, III
        ]

        for pattern in patterns:
            match = re.search(pattern, caption, re.IGNORECASE)
            if match:
                return match.group(1)

        # Fallback: use first few words of caption
        words = caption.split()[:3]
        name = "_".join(words)
        return self.sanitize_filename(name)

    def process_content_file(self, content_file_path: Path) -> Dict:
        """Process a single content file"""
        try:
            # Extract paper ID from the file structure
            # Path is: mineru/<paper>/auto/<paper_name>_content_list.json
            paper_id = content_file_path.stem.replace(
                "_content_list_trimmed", ""
            ).replace("_content_list", "")

            # Also get the paper directory name as backup
            paper_dir_name = content_file_path.parent.parent.name

            # Use the more specific paper_id, fallback to directory name
            if not paper_id or paper_id == content_file_path.stem:
                paper_id = paper_dir_name

            # Create paper-specific output directory
            paper_output_dir = self.output_root / paper_id
            paper_output_dir.mkdir(parents=True, exist_ok=True)

            # Create subdirectories
            figures_dir = paper_output_dir / "figures"
            tables_dir = paper_output_dir / "tables"
            equations_dir = paper_output_dir / "equations"

            figures_dir.mkdir(exist_ok=True)
            tables_dir.mkdir(exist_ok=True)
            equations_dir.mkdir(exist_ok=True)

            # Load content
            with open(content_file_path, "r", encoding="utf-8") as f:
                content = json.load(f)

            # Initialize counters and dictionaries for captions
            text_sections = []
            figure_captions = {}
            table_captions = {}
            equation_captions = {}

            figures_count = 0
            tables_count = 0
            equations_count = 0

            # Find the source directory for images (should be in the same 'auto' directory)
            source_dir = content_file_path.parent

            # Process each content item
            for item in content:
                item_type = item.get("type", "")

                if item_type == "text":
                    # Extract text content
                    text = item.get("text", "").strip()
                    if text:
                        text_sections.append(text)

                elif item_type == "image":
                    # Process images (figures, tables only)
                    img_path = item.get("img_path", "")
                    caption = item.get("image_caption", [])

                    if img_path and caption:
                        caption_text = (
                            " ".join(caption)
                            if isinstance(caption, list)
                            else str(caption)
                        )

                        # Determine image type and process accordingly
                        if self.is_table(caption_text):
                            table_number = self.extract_reference_name(
                                caption_text, "table"
                            )
                            self.process_image(
                                source_dir, img_path, caption_text, tables_dir, "table"
                            )
                            table_captions[table_number] = caption_text
                            tables_count += 1
                        else:
                            # Default to figure for all other images
                            figure_number = self.extract_reference_name(
                                caption_text, "figure"
                            )
                            self.process_image(
                                source_dir,
                                img_path,
                                caption_text,
                                figures_dir,
                                "figure",
                            )
                            figure_captions[figure_number] = caption_text
                            figures_count += 1

                elif item_type == "table":
                    # Process table content
                    table_caption = item.get("table_caption", [])
                    img_path = item.get("img_path", "")

                    if table_caption:
                        caption_text = (
                            " ".join(table_caption)
                            if isinstance(table_caption, list)
                            else str(table_caption)
                        )

                        table_number = self.extract_reference_name(
                            caption_text, "table"
                        )

                        if img_path:
                            self.process_image(
                                source_dir, img_path, caption_text, tables_dir, "table"
                            )

                        table_captions[table_number] = caption_text
                        tables_count += 1

                elif item_type == "equation":
                    # Process equation content
                    img_path = item.get("img_path", "")
                    equation_text = item.get("text", "")
                    text_format = item.get("text_format", "")
                    page_idx = item.get("page_idx", 0)

                    if img_path:
                        # Create a meaningful caption from the LaTeX text or use a generic one
                        if equation_text and text_format == "latex":
                            # Use first part of LaTeX as caption, truncated for readability
                            caption_text = (
                                f"Equation {equations_count + 1}: {equation_text[:100]}..."
                                if len(equation_text) > 100
                                else f"Equation {equations_count + 1}: {equation_text}"
                            )
                        else:
                            caption_text = (
                                f"Equation {equations_count + 1} (Page {page_idx + 1})"
                            )

                        equation_number = str(equations_count + 1)

                        self.process_image(
                            source_dir,
                            img_path,
                            caption_text,
                            equations_dir,
                            "equation",
                        )

                        # Store just the caption text to match figures and tables format
                        equation_captions[equation_number] = caption_text
                        equations_count += 1

            # Write text file
            if text_sections:
                text_file = paper_output_dir / f"{paper_id}_text.txt"
                with open(text_file, "w", encoding="utf-8") as f:
                    f.write("\n\n".join(text_sections))

            # Write combined captions JSON file
            captions_data = {
                "figures": figure_captions,
                "tables": table_captions,
                "equations": equation_captions,
            }

            captions_file = paper_output_dir / "figure_caption.json"
            with open(captions_file, "w", encoding="utf-8") as f:
                json.dump(captions_data, f, indent=2, ensure_ascii=False)

            # Copy the original content list file to the paper output directory
            content_list_copy = paper_output_dir / content_file_path.name
            shutil.copy2(content_file_path, content_list_copy)

            return {
                "paper_id": paper_id,
                "success": True,
                "text_sections": len(text_sections),
                "figures": figures_count,
                "tables": tables_count,
                "equations": equations_count,
            }

        except Exception as e:
            logger.error(f"Error processing {content_file_path}: {str(e)}")
            return {
                "paper_id": content_file_path.parent.parent.name,  # Use directory name as fallback
                "success": False,
                "error": str(e),
            }

    def is_figure(self, caption: str) -> bool:
        """Determine if caption represents a figure"""
        caption_lower = caption.lower()
        return (
            caption_lower.startswith("figure")
            or "fig." in caption_lower
            or "illustration" in caption_lower
            or "diagram" in caption_lower
        )

    def is_table(self, caption: str) -> bool:
        """Determine if caption represents a table"""
        caption_lower = caption.lower()
        return caption_lower.startswith("table") or "tab." in caption_lower

    def is_equation(self, caption: str) -> bool:
        """Determine if caption represents an equation"""
        caption_lower = caption.lower()

        # Check for explicit equation keywords
        if (
            caption_lower.startswith("equation")
            or "eq." in caption_lower
            or "formula" in caption_lower
            or "expression" in caption_lower
        ):
            return True

        # Check for mathematical notation patterns (more comprehensive)
        math_symbols = [
            "=",
            "\\sum",
            "\\int",
            "\\partial",
            "\\alpha",
            "\\beta",
            "\\gamma",
            "\\theta",
            "\\lambda",
            "\\mu",
            "\\pi",
            "\\sigma",
            "\\phi",
            "\\psi",
            "\\omega",
            "\\delta",
            "\\epsilon",
            "\\zeta",
            "\\eta",
            "\\kappa",
            "\\nu",
            "\\xi",
            "\\rho",
            "\\tau",
            "\\upsilon",
            "\\chi",
            "\\in",
            "\\subset",
            "\\cup",
            "\\cap",
            "\\infty",
            "\\rightarrow",
            "\\leftarrow",
            "\\Rightarrow",
            "\\Leftarrow",
            "\\forall",
            "\\exists",
            "\\nabla",
            "\\times",
            "\\cdot",
            "\\div",
            "\\pm",
            "\\mp",
            "\\leq",
            "\\geq",
            "\\neq",
            "\\approx",
            "\\equiv",
            "\\propto",
            "$$",
            "\\mathbb",
            "\\mathcal",
            "\\mathrm",
            "\\text{",
            "\\left",
            "\\right",
            "\\frac",
            "\\sqrt",
            "\\log",
            "\\ln",
            "\\exp",
            "\\sin",
            "\\cos",
            "\\tan",
        ]

        if any(symbol in caption for symbol in math_symbols):
            return True

        # Check for common equation/mathematical keywords
        math_keywords = [
            "optimization",
            "loss function",
            "objective",
            "constraint",
            "minimization",
            "maximization",
            "probability",
            "likelihood",
            "gradient",
            "derivative",
            "integral",
            "matrix",
            "vector",
            "tensor",
            "norm",
            "distance",
            "similarity",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "metric",
            "function",
        ]

        if any(keyword in caption_lower for keyword in math_keywords):
            return True

        return False

    def process_image(
        self,
        source_dir: Path,
        img_path: str,
        caption: str,
        output_dir: Path,
        content_type: str,
    ):
        """Process and copy an image file"""
        try:
            source_image = source_dir / img_path
            if not source_image.exists():
                logger.warning(f"Image not found: {source_image}")
                return

            # Extract reference name and create filename
            ref_name = self.extract_reference_name(caption, content_type)
            file_extension = source_image.suffix
            target_filename = f"{content_type}_{ref_name}{file_extension}"
            target_path = output_dir / target_filename

            # Handle duplicate names
            counter = 1
            while target_path.exists():
                target_filename = f"{content_type}_{ref_name}_{counter}{file_extension}"
                target_path = output_dir / target_filename
                counter += 1

            # Copy the image
            shutil.copy2(source_image, target_path)

        except Exception as e:
            logger.error(f"Error processing image {img_path}: {str(e)}")

    def find_content_files(self) -> List[Path]:
        """Find all content list files in mineru/<paper>/auto/<paper_name>_content_list.json"""
        content_files = []

        # Search for content list files in the specific structure
        # mineru/<paper>/auto/<paper_name>_content_list.json
        auto_dirs = list(self.mineru_root.glob("*/auto"))
        print(self.mineru_root)
        print(f"Auto dirs: {auto_dirs}")

        for auto_dir in auto_dirs:
            # Find JSON files in each auto directory
            for pattern in ["*_content_list.json", "*_content_list_trimmed.json"]:
                content_files.extend(auto_dir.glob(pattern))

        logger.info(
            f"Found {len(content_files)} content files across {len(auto_dirs)} paper directories"
        )
        return content_files

    def process_batch(
        self, content_files: List[Path], max_workers: Optional[int] = None
    ) -> None:
        """Process files in parallel batches"""
        if max_workers is None:
            # Limit to 8 to avoid overwhelming the system
            max_workers = min(cpu_count(), 8)

        logger.info(f"Processing {len(content_files)} files with {max_workers} workers")

        start_time = time.time()

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(self.process_content_file, content_file): content_file
                for content_file in content_files
            }

            # Process completed tasks
            for future in as_completed(future_to_file):
                content_file = future_to_file[future]
                try:
                    result = future.result()
                    self.stats.processed_files += 1

                    if result["success"]:
                        self.stats.text_sections += result["text_sections"]
                        self.stats.figures_extracted += result["figures"]
                        self.stats.tables_extracted += result["tables"]
                        self.stats.equations_extracted += result["equations"]
                    else:
                        self.stats.failed_files += 1
                        logger.error(
                            f"Failed to process {result['paper_id']}: {result.get('error', 'Unknown error')}"
                        )

                    # Log progress every 100 files
                    if self.stats.processed_files % 100 == 0:
                        elapsed = time.time() - start_time
                        rate = self.stats.processed_files / elapsed
                        eta = (
                            (len(content_files) - self.stats.processed_files) / rate
                            if rate > 0
                            else 0
                        )
                        logger.info(
                            f"Progress: {self.stats.processed_files}/{len(content_files)} files "
                            f"({self.stats.processed_files/len(content_files)*100:.1f}%) "
                            f"Rate: {rate:.1f} files/sec, ETA: {eta/60:.1f} minutes"
                        )

                except Exception as e:
                    self.stats.failed_files += 1
                    logger.error(f"Error processing {content_file}: {str(e)}")

        # Final statistics
        total_time = time.time() - start_time
        logger.info(f"\nProcessing complete!")
        logger.info(f"Total time: {total_time/60:.1f} minutes")
        logger.info(f"Files processed: {self.stats.processed_files}")
        logger.info(f"Files failed: {self.stats.failed_files}")
        logger.info(f"Text sections extracted: {self.stats.text_sections}")
        logger.info(f"Figures extracted: {self.stats.figures_extracted}")
        logger.info(f"Tables extracted: {self.stats.tables_extracted}")
        logger.info(f"Equations extracted: {self.stats.equations_extracted}")
        logger.info(
            f"Average processing rate: {self.stats.processed_files/total_time:.1f} files/sec"
        )


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Process MinerU content files to extract text, figures, tables, and equations."
    )
    parser.add_argument(
        "mineru_root",
        type=str,
        help="Path to the root directory containing MinerU processed papers",
    )
    parser.add_argument(
        "output_root",
        type=str,
        help="Path to the output directory where processed files will be saved",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Maximum number of worker processes (default: min(CPU count, 8))",
    )

    args = parser.parse_args()

    # Configuration from command line arguments
    mineru_root = args.mineru_root
    output_root = args.output_root

    # Initialise processor
    processor = PaperProcessor(mineru_root, output_root)

    # Find content files
    content_files = processor.find_content_files()

    if not content_files:
        logger.error("No content files found!")
        return

    processor.stats.total_files = len(content_files)

    # Process files
    processor.process_batch(content_files, max_workers=args.max_workers)

    logger.info("Processing completed successfully!")


if __name__ == "__main__":
    main()
