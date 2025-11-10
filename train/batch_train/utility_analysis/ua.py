import os
import json
import csv
import argparse
from typing import List, Dict, Any

def load_results(json_path: str) -> List[Dict[str, Any]]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and isinstance(data.get("results"), list):
        return data["results"]
    raise ValueError("Input JSON must be a list or a dict with key 'results' (list).")

def extract_titles(records: List[Dict[str, Any]]) -> List[str]:
    titles = []
    for r in records:
        t = r.get("title")
        if isinstance(t, str):
            t = t.strip()
            if t:
                titles.append(t)
    return titles

def save_titles_csv(titles: List[str], out_csv: str):
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["title"])
        for t in titles:
            writer.writerow([t])

def parse_args():
    ap = argparse.ArgumentParser(description="Extract paper titles from eval JSON and export to CSV.")
    ap.add_argument("--input_json", required=True, help="Path to eval JSON (list or dict with 'results').")
    ap.add_argument("--output_csv", required=True, help="Path to write CSV containing a single 'title' column.")
    ap.add_argument("--print_list", action="store_true", help="Print the titles list to stdout.")
    return ap.parse_args()

def main():
    args = parse_args()
    records = load_results(args.input_json)
    titles = extract_titles(records)
    save_titles_csv(titles, args.output_csv)
    if args.print_list:
        # Print as a Python-style list for quick copy/paste; change to json.dumps(titles) if preferred
        print(titles)
    print(f"Extracted {len(titles)} titles to: {args.output_csv}")

if __name__ == "__main__":
    main()
