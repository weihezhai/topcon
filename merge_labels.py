#!/usr/bin/env python3
import argparse
import json
import sys
from typing import Dict, Any

MAP_OTHER_TO_EXISTING = {
    "reject": "Submitted to ICLR 2025",
    "withdraw": "ICLR 2025 Conference Withdrawn Submission",
    "poster": "ICLR 2025 Poster",
    "spotlight": "ICLR 2025 Spotlight",
    "oral": "ICLR 2025 Oral",
}

def normalize_other_label(label: str) -> str:
    key = label.strip().lower()
    return MAP_OTHER_TO_EXISTING.get(key, label.strip())

def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must be a JSON object (mapping).")
    return data

def merge_labels(base: Dict[str, str], other_raw: Dict[str, str], priority: str) -> Dict[str, str]:
    # Normalize the "other" labels to match the existing format
    other = {k: normalize_other_label(v) for k, v in other_raw.items()}

    merged = dict(base)
    for k, v in other.items():
        if k in merged:
            if priority == "other" and merged[k] != v:
                merged[k] = v
        else:
            merged[k] = v
    return merged

def main():
    parser = argparse.ArgumentParser(description="Merge two label JSON files with value mapping.")
    parser.add_argument("--base", required=True, help="Path to existing-format labels JSON (e.g., label_simple.json).")
    parser.add_argument("--other", required=True, help="Path to other labels JSON (Reject/Withdraw/Poster/Spotlight/Oral).")
    parser.add_argument("--out", required=True, help="Output path for merged JSON.")
    parser.add_argument(
        "--priority",
        choices=["base", "other"],
        default="base",
        help="Which file wins on conflicts for the same key (default: base).",
    )
    args = parser.parse_args()

    try:
        base = load_json(args.base)
        other = load_json(args.other)
        merged = merge_labels(base, other, args.priority)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(merged, f, ensure_ascii=False, indent=2, sort_keys=True)
        print(f"Merged {len(base)} + {len(other)} -> {len(merged)} entries into {args.out}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()