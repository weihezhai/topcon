from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_json_tolerant(path: Path):
    s = path.read_text(encoding="utf-8")
    # tolerate //... and /*...*/ comments (common in jsonc)
    s = re.sub(r"//.*?$", "", s, flags=re.M)
    s = re.sub(r"/\*.*?\*/", "", s, flags=re.S)
    return json.loads(s)


def is_accepted(status: str) -> bool:
    return (status or "").strip() not in {"Reject", "Withdraw"}


def to_score(item) -> float | None:
    ra = item.get("rating_avg", None)
    if isinstance(ra, (list, tuple)) and ra:
        try:
            return float(ra[0])
        except Exception:
            return None
    try:
        return float(ra)  # fallback if already scalar
    except Exception:
        return None


def bin_index_1to9(score: float) -> int:
    # bins: 1-2, 2-3, ..., 9-10 (10 goes to 9-10)
    i = int(np.floor(score))
    i = max(1, min(9, i))
    return i - 1  # 0..8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path, required=True, help="Path to json file")
    ap.add_argument("--flip-threshold", type=float, default=6.0, help="T in flip definition")
    ap.add_argument("--out", type=Path, default=None, help="If set, save figure to this path")
    args = ap.parse_args()

    data = load_json_tolerant(args.json)

    scores, acc = [], []
    for it in data:
        s = to_score(it)
        if s is None or not np.isfinite(s):
            continue
        scores.append(float(s))
        acc.append(is_accepted(it.get("status", "")))

    if not scores:
        raise SystemExit("No valid rating_avg found.")

    scores = np.array(scores, dtype=float)
    acc = np.array(acc, dtype=bool)

    bins = np.array([bin_index_1to9(s) for s in scores], dtype=int)
    n_bins = 9
    labels = [f"{i}-{i+1}" for i in range(1, 10)]
    x = np.arange(n_bins)

    total = np.bincount(bins, minlength=n_bins).astype(float)
    accepted = np.bincount(bins, weights=acc.astype(float), minlength=n_bins)

    # flip: (high score rejected) OR (low score accepted)
    T = float(args.flip_threshold)
    flip = ((scores >= T) & (~acc)) | ((scores < T) & (acc))
    flip_cnt = np.bincount(bins, weights=flip.astype(float), minlength=n_bins)

    with np.errstate(divide="ignore", invalid="ignore"):
        accept_rate = np.where(total > 0, accepted / total, np.nan)
        flip_rate = np.where(total > 0, flip_cnt / total, np.nan)

    plt.figure(figsize=(9, 4))
    plt.plot(x, accept_rate, marker="o", linewidth=2, label="Acceptance rate")
    plt.plot(x, flip_rate, marker="o", linewidth=2, label=f"Flip rate (T={T:g})")
    plt.xticks(x, labels, rotation=0)
    plt.ylim(0, 1)
    plt.ylabel("Rate")
    plt.xlabel("Rating bin")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(args.out, dpi=200)
    else:
        plt.show()


if __name__ == "__main__":
    main()
