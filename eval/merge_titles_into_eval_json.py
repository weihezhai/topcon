import os
import json
import csv
import argparse

def read_titles(csv_path, id_col="paper_id", title_col="title"):
    title_map = {}
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if id_col not in reader.fieldnames or title_col not in reader.fieldnames:
            raise ValueError(f"CSV must contain columns '{id_col}' and '{title_col}'. Found: {reader.fieldnames}")
        for row in reader:
            pid = row.get(id_col, "").strip()
            title = row.get(title_col, "").strip()
            if pid:
                # Keep the first seen title; ignore duplicates silently
                title_map.setdefault(pid, title)
    return title_map

def load_eval_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Support either {"results": [...]} or a plain list
    if isinstance(data, list):
        return {"results": data, "summary": {}}, True
    if not isinstance(data, dict) or "results" not in data:
        raise ValueError("Input JSON must be either a list of results or a dict with key 'results'.")
    return data, False

def save_eval_json(data, out_path):
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Saved enriched JSON to: {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Merge paper titles into eval JSON results.")
    parser.add_argument("--csv_file", required=True, help="Path to CSV containing 'paper_id' and 'title'.")
    parser.add_argument("--input_json", required=True, help="Path to eval JSON produced by eval_real_paper.py.")
    parser.add_argument("--output_json", default=None, help="Path to write enriched JSON. Defaults to input with _with_titles suffix.")
    parser.add_argument("--id_col", default="paper_id", help="CSV column name for paper id.")
    parser.add_argument("--title_col", default="title", help="CSV column name for title.")
    args = parser.parse_args()

    titles = read_titles(args.csv_file, id_col=args.id_col, title_col=args.title_col)
    data, was_list = load_eval_json(args.input_json)

    results = data.get("results", [])
    matched = 0
    for row in results:
        pid = row.get("paper_id")
        if pid in titles:
            row["title"] = titles[pid]
            matched += 1
        else:
            # Leave missing titles as None for clarity
            row.setdefault("title", None)

    # If input was a list, and we converted it, optionally restore list-only format
    out_data = results if was_list else data

    # Derive default output path if not provided
    out_path = args.output_json
    if not out_path:
        base, ext = os.path.splitext(args.input_json)
        out_path = f"{base}_with_titles{ext or '.json'}"

    save_eval_json(out_data, out_path)
    print(f"Results: {len(results)} rows processed, {matched} titles matched.")

if __name__ == "__main__":
    main()
