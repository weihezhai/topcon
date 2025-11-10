import json
import csv
import argparse
from pathlib import Path

def extract_titles(json_path: Path):
    with json_path.open('r', encoding='utf-8') as f:
        data = json.load(f)
    results = data.get("results", [])
    return [r["title"] for r in results if "title" in r]

def write_csv(titles, csv_path: Path):
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(["title"])
        for t in titles:
            w.writerow([t])

def main():
    parser = argparse.ArgumentParser(description="Extract paper titles to CSV.")
    parser.add_argument("--input", type=Path, default=Path(__file__).with_name("95.json"),
                        help="Path to the JSON file (default: 95.json in same dir).")
    parser.add_argument("--output", type=Path, default=Path("titles.csv"),
                        help="Output CSV path (default: titles.csv).")
    args = parser.parse_args()

    titles = extract_titles(args.input)
    write_csv(titles, args.output)
    print(f"Extracted {len(titles)} titles to {args.output}")

if __name__ == "__main__":
    main()
