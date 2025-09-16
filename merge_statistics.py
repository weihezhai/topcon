import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def dump_json(data: Dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def _list_dedup_preserve_order(items: List[Any]) -> List[Any]:
    # De-duplicate heterogeneous lists while preserving order
    seen = set()
    out = []
    for x in items:
        try:
            key = json.dumps(x, sort_keys=True)
        except TypeError:
            # Fallback for non-serializable types (unlikely here)
            key = repr(x)
        if key in seen:
            continue
        seen.add(key)
        out.append(x)
    return out

def deep_merge(a: Any, b: Any, prefer: str = "other") -> Any:
    # prefer: "base" | "other"
    if isinstance(a, dict) and isinstance(b, dict):
        merged = dict(a)
        for k, v in b.items():
            if k in merged:
                merged[k] = deep_merge(merged[k], v, prefer)
            else:
                merged[k] = v
        return merged
    if isinstance(a, list) and isinstance(b, list):
        return _list_dedup_preserve_order(a + b)
    # Primitive or mismatched types: resolve by preference
    return a if prefer == "base" else b

def merge_papers(base_data: Dict[str, Any], other_data: Dict[str, Any], prefer: str) -> Dict[str, Any]:
    base_papers = base_data.get("papers", {}) or {}
    other_papers = other_data.get("papers", {}) or {}
    merged_papers = deep_merge(base_papers, other_papers, prefer=prefer)

    # Start from base_data to preserve unrelated fields
    merged = dict(base_data)
    merged["papers"] = merged_papers
    return merged

def recompute_summary(merged: Dict[str, Any]) -> None:
    count = len(merged.get("papers", {}))
    merged["summary"] = {
        "total_papers_processed": count,
        "total_papers_attempted": count,
        "processing_errors": 0,
        "error_details": None,
    }

def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge the 'papers' field from another statistics file into a base JSON.")
    p.add_argument("base", help="Path to base statistics_per_paper.json")
    p.add_argument("other", help="Path to other statistics JSON with the same schema")
    p.add_argument("-o", "--output", help="Path to write merged JSON (default: overwrite base)", default=None)
    p.add_argument("--prefer", choices=["base", "other"], default="other",
                   help="Which file wins when merging non-dict/list values (default: other)")
    p.add_argument("--recompute-summary", action="store_true",
                   help="Recompute summary from merged 'papers' (sets processed/attempted to number of merged papers)")
    return p.parse_args(argv)

def main(argv: List[str]) -> int:
    args = parse_args(argv)
    base_path = Path(args.base)
    other_path = Path(args.other)
    out_path = Path(args.output) if args.output else base_path

    base_data = load_json(base_path)
    other_data = load_json(other_path)

    merged = merge_papers(base_data, other_data, prefer=args.prefer)
    if args.recompute_summary:
        recompute_summary(merged)

    dump_json(merged, out_path)
    print(f"Merged papers: {len(merged.get('papers', {}))} -> {out_path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
