import argparse
import concurrent.futures as futures
import json
import os
import random
import shutil
import string
import sys
from typing import List, Tuple, Dict

def load_ids(json_path: str) -> List[str]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or not all(isinstance(x, str) for x in data):
        raise ValueError("IDs JSON must be a list of strings.")
    # Deduplicate while preserving order
    seen = set()
    out = []
    for x in data:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

def _rand_suffix(n: int = 6) -> str:
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=n))

def safe_copytree(src: str, dst: str, *, overwrite: bool, dry_run: bool, verbose: bool) -> str:
    if os.path.exists(dst):
        if overwrite:
            if verbose:
                print(f"[overwrite] removing {dst}", flush=True)
            if not dry_run:
                shutil.rmtree(dst, ignore_errors=False)
        else:
            return "exists"
    if dry_run:
        return "dryrun"
    tmp_dst = f"{dst}.tmp-{os.getpid()}-{_rand_suffix()}"
    try:
        shutil.copytree(src, tmp_dst, symlinks=True)
        os.replace(tmp_dst, dst)  # atomic
        return "copied"
    except Exception:
        # Best-effort cleanup
        try:
            if os.path.exists(tmp_dst):
                shutil.rmtree(tmp_dst, ignore_errors=True)
        finally:
            raise

def process_one(paper_id: str, source_root: str, dest_root: str, exists_policy: str, missing_policy: str, dry_run: bool, verbose: bool) -> Tuple[str, str]:
    src = os.path.join(source_root, paper_id)
    dst = os.path.join(dest_root, paper_id)
    if not os.path.isdir(src):
        if missing_policy == "fail":
            return (paper_id, "missing-fail")
        return (paper_id, "missing-skip")
    overwrite = exists_policy == "overwrite"
    if exists_policy == "fail" and os.path.exists(dst):
        return (paper_id, "exists-fail")
    try:
        result = safe_copytree(src, dst, overwrite=overwrite, dry_run=dry_run, verbose=verbose)
        if verbose:
            print(f"[{result}] {paper_id}", flush=True)
        return (paper_id, result)
    except Exception as e:
        if verbose:
            print(f"[error] {paper_id}: {e}", file=sys.stderr, flush=True)
        return (paper_id, "error")

def main():
    parser = argparse.ArgumentParser(description="Extract subfolders (by IDs) from a source directory into a destination directory.")
    parser.add_argument("--source", required=True, help="Source root directory containing subfolders named by IDs.")
    parser.add_argument("--dest", required=True, help="Destination root directory to create the subset.")
    parser.add_argument("--ids", required=True, help="Path to JSON file containing list of IDs (array of strings).")
    parser.add_argument("--exists-policy", choices=["skip", "overwrite", "fail"], default="skip", help="What to do if destination folder already exists.")
    parser.add_argument("--missing-policy", choices=["skip", "fail"], default="skip", help="What to do if a source ID folder is missing.")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers for copying.")
    parser.add_argument("--dry-run", action="store_true", help="Do not copy, just report what would happen.")
    parser.add_argument("--verbose", action="store_true", help="Verbose per-ID logging.")
    args = parser.parse_args()

    source_root = os.path.abspath(args.source)
    dest_root = os.path.abspath(args.dest)
    ids = load_ids(args.ids)

    if not os.path.isdir(source_root):
        print(f"Source root does not exist or is not a directory: {source_root}", file=sys.stderr)
        sys.exit(2)

    os.makedirs(dest_root, exist_ok=True)

    max_workers = max(1, min(args.workers, len(ids))) if ids else 1

    results: Dict[str, str] = {}
    with futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [
            ex.submit(
                process_one,
                pid,
                source_root,
                dest_root,
                args.exists_policy,
                args.missing_policy,
                args.dry_run,
                args.verbose,
            )
            for pid in ids
        ]
        for fut in futures.as_completed(futs):
            pid, status = fut.result()
            results[pid] = status

    # Summarize
    summary = {
        "copied": 0,
        "exists": 0,
        "dryrun": 0,
        "missing-skip": 0,
        "missing-fail": 0,
        "exists-fail": 0,
        "error": 0,
    }
    for status in results.values():
        summary[status] = summary.get(status, 0) + 1

    total = len(ids)
    print(f"Total IDs: {total}")
    print(f"Copied: {summary.get('copied',0)}")
    print(f"Dry-run marked: {summary.get('dryrun',0)}")
    print(f"Skipped (exists): {summary.get('exists',0)}")
    print(f"Skipped (missing): {summary.get('missing-skip',0)}")
    print(f"Failed (missing): {summary.get('missing-fail',0)}")
    print(f"Failed (exists): {summary.get('exists-fail',0)}")
    print(f"Errors: {summary.get('error',0)}")

if __name__ == "__main__":
    main()
