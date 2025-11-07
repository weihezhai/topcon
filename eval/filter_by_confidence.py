import os
import json
import argparse
import math

def load_results(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "results" in data and isinstance(data["results"], list):
        return data["results"]
    raise ValueError("Input JSON must be a list or a dict with key 'results' (list).")

def stable_sigmoid(x: float) -> float:
    # sigmoid(x) with numerical stability
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)

def get_yes_prob(row):
    if "yes_prob" in row and row["yes_prob"] is not None:
        return float(row["yes_prob"])
    y = row.get("yes_logit")
    n = row.get("no_logit")
    if y is not None and n is not None:
        # Softmax over two logits => p_yes = sigmoid(yes - no)
        return stable_sigmoid(float(y) - float(n))
    raise ValueError("Row missing yes_prob and logits; cannot compute yes_prob.")

def main():
    ap = argparse.ArgumentParser(description="Filter eval results by confidence = 2*yes_prob - 1 in a given range.")
    ap.add_argument("--input_json", required=True, help="Path to eval JSON (list or dict with 'results').")
    ap.add_argument("--output_json", default=None, help="Optional path to save filtered results JSON.")
    ap.add_argument("--lower", type=float, default=0.97, help="Lower bound (inclusive) for confidence.")
    ap.add_argument("--upper", type=float, default=1.0, help="Upper bound (inclusive) for confidence.")
    args = ap.parse_args()

    results = load_results(args.input_json)
    total = len(results)
    filtered = []

    for row in results:
        try:
            y = get_yes_prob(row)
        except Exception:
            continue
        conf = 2.0 * y - 1.0
        if conf >= args.lower and conf <= args.upper:
            out_row = dict(row)
            out_row["confidence"] = conf
            filtered.append(out_row)

    print(f"Total records: {total}")
    print(f"Filtered (confidence in [{args.lower}, {args.upper}]): {len(filtered)}")
    # Print a few lines for quick inspection
    for r in filtered[:20]:
        pid = r.get("paper_id", "-")
        title = r.get("title", "-")
        print(f"{pid} | conf={r['confidence']:.4f} | yes_prob={get_yes_prob(r):.4f} | pred_label={r.get('pred_label')} | {title}")

    if args.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        payload = {
            "range": {"lower": args.lower, "upper": args.upper, "definition": "confidence = 2*yes_prob - 1"},
            "count": len(filtered),
            "total": total,
            "results": filtered,
        };
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved filtered results to: {args.output_json}")

if __name__ == "__main__":
    main()
